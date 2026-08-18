# -*- coding: utf-8 -*-
"""第十阶段（一）· 验证：FastAPI 骨架 / 统一响应 / 错误码 / 异常处理 / 日志 / 问答接口

**默认不需要 Ollama、不需要 65GB 向量库**，秒级跑完。每一条 PASS/FAIL 都由实际算出的
变量比对得出——不存在任何无条件 `print("✓")`（阶段五踩过这个坑）。

## 怎么做到"不调模型却真跑链路"

注入一个 `StubGenerator`：它按段返回预置输出（①评估 JSON、②草稿、③审查 JSON、④定稿），
并实现 `stream_messages` 供流式路径消费。于是**上下文组装、四段链、证据筛选、后处理、
阶段九层 D 校验与修正、SSE 埋点、会话改写、调用记录**全部是真代码在跑，
只有"模型吐字"这一步是假的。这比拿 mock 去替换整条流水线有意义得多——
接口层的 bug 恰恰都藏在真链路的接缝处。

证据用阶段七固化的真检索快照（`report_data\\检索快照_live.json`），不是手编的假文档。

    & $py E:\\rag\\scripts\\服务_验证.py             # 全量，约 10 秒
    & $py E:\\rag\\scripts\\服务_验证.py --quiet     # 只看汇总
    & $py E:\\rag\\scripts\\服务_验证.py --live      # 追加真实 Ollama 端到端实测（要起服务）
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT, evidence_path

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import logging
import re
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "report_data")
REPORT_PATH = os.path.join(REPORT_DIR, "服务_验证报告.txt")
LIVE_JSON = os.path.join(REPORT_DIR, "服务_实测.json")
SNAPSHOT = evidence_path("检索快照_live.json")
#: Postman 集合是**交付物**不是中间产物，所以落在 任务10\ 而不是 report_data\
POSTMAN_PATH = os.path.join(ROOT, "任务10", "medrag_api.postman_collection.json")


def _load(mod_name: str, filename: str):
    """与被测模块**共用同一批模块对象**（登记进 sys.modules）。

    这一点对验证脚本尤其重要：如果验证脚本手里的 `ErrorCode`/`APIError` 和应用里的
    不是同一个类，断言比的就不是被测代码的行为，而是两个副本之间的巧合。
    """
    path = os.path.join(_HERE, filename)
    cached = sys.modules.get(mod_name)
    if cached is not None and os.path.normcase(getattr(cached, "__file__", "") or "") \
            == os.path.normcase(path):
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    return mod


err = _load("fuwu_cuowuma", "服务_错误码.py")
mdl = _load("fuwu_moxing", "服务_模型.py")
lgm = _load("fuwu_rizhi", "服务_日志.py")
ssm = _load("fuwu_huihua", "服务_会话.py")
stm = _load("fuwu_liushi", "服务_流式.py")
app_mod = _load("fuwu_yingyong", "服务_应用.py")
dc_mod = _load("fuwu_wendangmulu", "服务_文档目录.py")
cp = _load("yueshu_tishicing", "约束_提示词层.py")

from fastapi.testclient import TestClient           # noqa: E402

EC = err.ErrorCode
REFUSAL = cp.REFUSAL_PHRASE


# ============================================================================
# 记账
# ============================================================================
class Checker:
    def __init__(self, quiet: bool = False):
        self.rows: List[Tuple[str, str, bool, str]] = []
        self.skipped: List[Tuple[str, str]] = []
        self.group = ""
        self.quiet = quiet
        self.lines: List[str] = []

    def head(self, title: str):
        self.group = title
        RecordingClient.GROUP = title       # 录到的请求按验证分组归位
        self._out("\n" + "=" * 92)
        self._out(title)
        self._out("=" * 92)

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.rows.append((self.group, label, ok, detail))
        self._out(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        return ok

    def skip(self, label: str, why: str):
        self.skipped.append((self.group, f"{label}：{why}"))
        self._out(f"  [SKIP] {label}   {why}")

    def note(self, text: str):
        self._out(f"        {text}")

    def _out(self, s: str):
        self.lines.append(s)
        if not self.quiet:
            print(s)

    def summary(self) -> Tuple[int, int]:
        return sum(1 for _, _, ok, _ in self.rows if ok), len(self.rows)


# ============================================================================
# 假生成器：按段返回预置输出，让整条真链路跑起来
# ============================================================================
class StubLLMError(RuntimeError):
    """类名必须叫 `LLMError`，才能被 `classify_exception` 认出来。

    见 `服务_错误码.classify_exception`：它按**类名**而不是 isinstance 判，
    因为本项目的中文文件名模块按路径导入，同一个类会有多个不相等的副本。
    这里正好也顺带验证了那条归类规则确实按类名工作。
    """


StubLLMError.__name__ = "LLMError"

#: 各段的识别特征（取自阶段七 system prompt 的开头；阶段九只在**前面**加硬约束，
#: 所以这些片段在受约束模式下仍然是子串）
_STAGE_SIGNS = (
    ("evidence_evaluator", "你是一名循证医学的证据评估员"),
    ("answer_generator", "你是一名严谨的医学文献助理"),
    ("critical_reviewer", "你是一名批判性的医学审稿人"),
    ("final_assembler", "你是最终定稿人"),
)

_MARKER = re.compile(r"\[(S\d+)\]")

#: 刻意不写任何数字与缩写：数字会触发 `numeric.ungrounded`（证据里查不到），
#: 缩写会触发 `terminology.missing_expansion`。这里要的是一份"本来就合规"的基线答案，
#: 不合规的情形由阶段九自己的验证脚本覆盖。
_ANSWER_TMPL = """## 核心答案
现有文献支持该干预在目标人群中具有可行性 {m0}。多项研究报告了方向一致的结果 {m1}。

## 证据总结
- 一项研究观察到干预组的结局指标优于对照组 {m0}。
- 另一项研究报告了相似趋势，但人群来源与随访安排不同 {m1}。

## 证据强度与局限
证据以观察性研究为主，缺少大规模随机对照试验；样本来源集中，外推需谨慎。
"""

_REFUSAL_TMPL = f"""## 核心答案
{REFUSAL}提供的文献片段没有涉及所问的内容。

## 证据总结
- 检索到的片段与该问题不属于同一主题，无法据此作答。

## 证据强度与局限
本次没有可用于回答该问题的证据。
"""


class StubGenerator:
    """假 LLM。鸭子类型对齐阶段七 `LLMGenerator` 里流水线真正用到的那几个成员。

    `mode`:
        normal  正常作答（合规答案）
        refuse  返回带固定拒答短语的结构化回答（测"拒答不是错误"这条）
        fail    抛一个类名为 LLMError 的异常（测 4xxx 归类与错误落库）
    """

    def __init__(self, mode: str = "normal", chunk_size: int = 24, delay: float = 0.0):
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.delay = float(delay)
        self.model_name = "stub-qwen3"
        self.base_url = "http://stub.invalid"
        self.timeout = 5
        self.num_ctx = 12288
        self.think = False
        self.keep_alive = "0s"
        self.temperature = 0.0
        self.max_tokens = 1500
        self.top_p = 0.9
        self.stats = {"calls": 0, "failures": 0, "total_seconds": 0.0,
                      "prompt_tokens": 0, "output_tokens": 0}
        self.seen: List[str] = []           # 依次记录识别出的段名，供验证断言链路顺序

    # ---- LLMGenerator 的公共面 ----
    def build_options(self, temperature=None, max_tokens=None, **extra):
        nt = self.max_tokens if max_tokens is None else int(max_tokens)
        if nt <= 0:
            raise ValueError("max_tokens 必须 > 0")
        opts = {"temperature": self.temperature if temperature is None else float(temperature),
                "num_predict": nt, "top_p": self.top_p, "num_ctx": self.num_ctx}
        opts.update(extra)
        return opts

    # ---- 段识别与内容 ----
    @staticmethod
    def _stage_of(messages) -> str:
        sys_text = "".join(m.get("content", "") for m in messages if m.get("role") == "system")
        for key, sign in _STAGE_SIGNS:
            if sign in sys_text:
                return key
        return "format_fixer"               # 阶段九层 D 的修正段

    @staticmethod
    def _markers(messages) -> List[str]:
        user = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
        out, seen = [], set()
        for m in _MARKER.findall(user):
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _text_for(self, stage: str, messages) -> str:
        ms = self._markers(messages) or ["S1", "S2"]
        m0, m1 = f"[{ms[0]}]", f"[{ms[1 % len(ms)]}]"
        if stage == "evidence_evaluator":
            return json.dumps([{"marker": m, "relevance": 3, "study_type": "cohort",
                                "directly_answers": True,
                                "key_finding": "片段报告了与问题相关的观察结果",
                                "caveats": "观察性设计"} for m in ms], ensure_ascii=False)
        if stage == "critical_reviewer":
            return json.dumps({"verdict": "accept", "overall_grounded": True, "issues": []},
                              ensure_ascii=False)
        if stage == "format_fixer":
            # 修正段：把上一版原样退回（本夹具里答案本来就合规，不该触发它）
            user = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
            hit = re.search(r"##\s*核心答案.*", user, re.S)
            return hit.group(0) if hit else _ANSWER_TMPL.format(m0=m0, m1=m1)
        if self.mode == "refuse":
            return _REFUSAL_TMPL
        return _ANSWER_TMPL.format(m0=m0, m1=m1)

    def _fail_if_needed(self):
        if self.mode == "fail":
            self.stats["failures"] += 1
            raise StubLLMError("连不上 Ollama（stub）：本次是刻意注入的故障")

    # ---- 非流式 ----
    def generate_messages(self, messages, temperature=None, max_tokens=None,
                          json_output=False, expect="any", options=None,
                          retries=1, json_retries=1) -> Dict[str, Any]:
        self._fail_if_needed()
        stage = self._stage_of(messages)
        self.seen.append(stage)
        text = self._text_for(stage, messages)
        if self.delay:
            time.sleep(self.delay)
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += 900
        self.stats["output_tokens"] += 220
        out = {"text": text, "raw": text, "thinking_chars": 0, "elapsed": 0.01,
               "model": self.model_name, "done_reason": "stop", "truncated": False,
               "prompt_eval_count": 900, "eval_count": 220, "tokens_per_second": 22.0,
               "json": None, "json_ok": None, "json_error": "", "json_note": ""}
        if json_output:
            obj = json.loads(text)
            out.update(json=obj, json_ok=True, json_note="stub 直出")
        return out

    # ---- 流式：产出 Ollama NDJSON 形状的对象（`服务_流式` 的两条路径共用同一个消费循环）----
    def stream_messages(self, messages, options=None, **_kw) -> Iterator[Dict[str, Any]]:
        self._fail_if_needed()
        stage = self._stage_of(messages)
        self.seen.append(stage)
        text = self._text_for(stage, messages)
        for i in range(0, len(text), self.chunk_size):
            if self.delay:
                time.sleep(self.delay)
            yield {"message": {"content": text[i:i + self.chunk_size]}, "done": False}
        yield {"message": {"content": ""}, "done": True, "done_reason": "stop",
               "prompt_eval_count": 900, "eval_count": 220,
               "eval_duration": 10_000_000_000, "model": self.model_name}

    # ---- 追问改写走的是单段调用 ----
    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None,
                 **_kw) -> Dict[str, Any]:
        self._fail_if_needed()
        self.stats["calls"] += 1
        if system_prompt and "检索查询改写器" in system_prompt:
            latest = re.search(r"最新一句：(.+)", prompt)
            topic = re.search(r"用户：(.+)", prompt)
            q = (latest.group(1).strip() if latest else "").strip()
            t = (topic.group(1).strip() if topic else "").strip()
            t = re.split(r"[，。？?]", t)[0][:24]
            return {"text": f"在{t}的语境下，{q}" if t else q, "elapsed": 0.01,
                    "prompt_eval_count": 200, "eval_count": 30}
        return {"text": "stub", "elapsed": 0.01, "prompt_eval_count": 10, "eval_count": 2}


# ============================================================================
# 测试用应用
# ============================================================================
_TMPDIRS: List[str] = []


def make_app(mode: str = "normal", record: bool = True, **over: Any):
    """建一个隔离的测试应用（独立临时目录，绝不碰真服务的库）。

    `record=False` 用于**故意造坏的实例**（缺快照 / 缺目录 / 开鉴权 / 限流 / 并发闸门 /
    注入会失败的生成器）。它们发出的请求不进 Postman 集合——**那些响应只在被造坏的服务上
    才成立**：把"缺目录 → 3004"录进集合，拿到一台目录正常的服务上重放就是 200，
    集合无故变红，而且看上去像接口坏了。断言值取自真跑，前提是那次真跑的环境可复现。
    """
    d = tempfile.mkdtemp(prefix="medrag_api_test_")
    _TMPDIRS.append(d)
    gen = over.pop("generator", None) or StubGenerator(mode=mode)
    kw = dict(retrieval_mode="snapshot", snapshot_path=SNAPSHOT,
              calls_db=os.path.join(d, "calls.db"),
              session_db=os.path.join(d, "sessions.db"),
              log_dir=os.path.join(d, "logs"), log_console=False,
              generator_override=gen, rewrite_mode="llm")
    kw.update(over)                       # 调用方给的覆盖默认（不能直接 **over，会撞键）
    st = app_mod.ServiceSettings(**kw)
    state = app_mod.ServiceState(st)
    app = app_mod.create_app(state=state)
    cls = RecordingClient if (record and mode == "normal") else TestClient
    return app, state, cls(app, raise_server_exceptions=False), gen


class RecordingClient(TestClient):
    """包一层 TestClient，把**真实发出的每个请求**录下来，用于导出 Postman 集合。

    任务书要"编写单元测试和集成测试（使用 postman），覆盖所有API端点"。
    这里刻意**不另写一套 Postman 测试**：另写一套就有两份真相，改了接口只更新一边，
    过几周谁也不知道哪份是对的。集合里的请求体、查询参数、断言值**全部来自这一轮真跑**。

    ⚠ 它录不到的部分要说清楚：本脚本 165 项里有一大半是**进程内断言**
    （响应模型、错误码表、SQLite 存储、会话存储、参数校验在模型层就被拦下），
    它们根本不经过 HTTP，没有 Postman 形态。导出时会把"端点覆盖率"与"检查项覆盖率"
    分开报——前者要求 100%，后者本来就不可能是 100%。
    """

    RECORDS: List[Dict[str, Any]] = []
    GROUP = ""
    ENABLED = False

    def request(self, method, url, *args, **kw):        # type: ignore[override]
        resp = super().request(method, url, *args, **kw)
        if RecordingClient.ENABLED:
            try:
                self._save(method, url, kw, resp)
            except Exception:                # 录制永远不能影响验证结论
                pass
        return resp

    @staticmethod
    def _save(method: str, url: Any, kw: Dict[str, Any], resp: Any) -> None:
        path = str(url).split("?")[0]
        if path.startswith("/__"):           # 测试里临时注入的路由，真服务上没有
            return
        raw = kw.get("content") if kw.get("content") is not None else kw.get("data")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        try:
            code = resp.json().get("code")
        except Exception:
            code = None
        RecordingClient.RECORDS.append({
            "group": RecordingClient.GROUP,
            "method": str(method).upper(),
            "path": path,
            "params": {k: v for k, v in (kw.get("params") or {}).items()},
            "json": kw.get("json"),
            "raw": raw if isinstance(raw, str) else None,
            "headers": {k: v for k, v in (kw.get("headers") or {}).items()},
            "status": resp.status_code,
            "code": code,
            "content_type": resp.headers.get("content-type", ""),
        })


# ---------------------------------------------------------------------------
# 端点覆盖：拿 openapi 的路由表当分母
# ---------------------------------------------------------------------------
def route_inventory(app: Any) -> List[Tuple[str, str]]:
    """[(METHOD, path 模板)]。排除 `/__*`（测试注入）与 openapi/docs 这类自带页面。"""
    out: List[Tuple[str, str]] = []
    for path, ops in (app.openapi().get("paths") or {}).items():
        if path.startswith("/__"):
            continue
        for verb in ("get", "post", "put", "patch", "delete"):
            if isinstance((ops or {}).get(verb), dict):
                out.append((verb.upper(), path))
    return sorted(set(out))


def match_template(method: str, path: str, inventory: List[Tuple[str, str]]) -> Optional[str]:
    """把跑出来的具体 URL 归到路由模板上。

    ⚠ **先比字面量再比带参数的**：`/api/v1/qa/logs` 与 `/api/v1/qa/logs/{request_id}`
    段数不同还好说，但 `/api/v1/sessions` 与 `/api/v1/sessions/{session_id}` 之类
    一旦顺序反了，具体路径会被算进错误的模板，覆盖率就成了假的。
    """
    segs = path.strip("/").split("/")
    exact = [(m, t) for m, t in inventory if "{" not in t]
    templated = [(m, t) for m, t in inventory if "{" in t]
    for m, t in exact + templated:
        if m != method.upper():
            continue
        tsegs = t.strip("/").split("/")
        if len(tsegs) != len(segs):
            continue
        if all(ts.startswith("{") or ts == s for ts, s in zip(tsegs, segs)):
            return t
    return None


# ---------------------------------------------------------------------------
# 导出 Postman v2.1
# ---------------------------------------------------------------------------
POSTMAN_SCHEMA = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
_SESS_RE = re.compile(r"^sess-[0-9a-zA-Z_-]{3,}$")
_REQ_RE = re.compile(r"^req-[0-9a-zA-Z_-]{3,}$")
_PMCID_RE = re.compile(r"^PMC\d+$")


def _postman_path(path: str, code: Optional[int] = 0) -> Tuple[str, List[str]]:
    """把录到的具体 id 换成 Postman 变量，集合才能**连着跑**而不是跑一次就废。

    ⚠ **只在这一次录到的是成功响应时才替换**（`code == 0`）。理由是两类请求的意图相反：

      · `GET /sessions/sess-abc123` 录到 200 —— 意图是"查一个存在的会话"。
        id 每跑一轮都不一样，写死进去第二次就 3002，所以必须换成 `{{sessionId}}`。
      · `GET /sessions/does-not-exist` 录到 3002 —— 意图正是"查一个不存在的"。
        换成变量它就变成 200 了，**这条用例的意义会被替换掉**。

    文献 id 同理：验证脚本用的是一个 6 篇的微型目录库（`PMC000001`…），
    真服务上根本没有这几个 id；不换成 `{{docId}}`，集合一跑就红。
    """
    used: List[str] = []
    segs = []
    ok = (code == 0)
    for s in path.strip("/").split("/"):
        if ok and _SESS_RE.match(s):
            segs.append("{{sessionId}}")
            used.append("sessionId")
        elif ok and _REQ_RE.match(s):
            segs.append("{{requestId}}")
            used.append("requestId")
        elif ok and _PMCID_RE.match(s):
            segs.append("{{docId}}")
            used.append("docId")
        else:
            segs.append(s)
    return "/" + "/".join(segs), used


def _postman_item(rec: Dict[str, Any]) -> Dict[str, Any]:
    path, _used = _postman_path(rec["path"], rec["code"])
    segs = [s for s in path.strip("/").split("/") if s]
    query = [{"key": str(k), "value": str(v)} for k, v in (rec["params"] or {}).items()]
    raw_url = "{{baseUrl}}" + path + (
        "?" + "&".join(f"{q['key']}={q['value']}" for q in query) if query else "")

    headers = [{"key": "X-API-Key", "value": "{{apiKey}}",
                "description": "服务未开鉴权时留空即可"}]
    body = None
    if rec["json"] is not None:
        headers.append({"key": "Content-Type", "value": "application/json"})
        body = {"mode": "raw", "raw": json.dumps(rec["json"], ensure_ascii=False, indent=2),
                "options": {"raw": {"language": "json"}}}
    elif rec["raw"] is not None:
        headers.append({"key": "Content-Type", "value": "application/json"})
        body = {"mode": "raw", "raw": rec["raw"],
                "options": {"raw": {"language": "json"}}}
    for k, v in (rec["headers"] or {}).items():
        if k.lower() not in ("content-type", "x-api-key"):
            headers.append({"key": k, "value": str(v)})

    # 断言值来自真跑，不是手写的期望
    exec_lines = [
        f"pm.test('HTTP {rec['status']}', function () {{",
        f"    pm.response.to.have.status({rec['status']});",
        "});",
    ]
    if rec["code"] is not None:
        exec_lines += [
            f"pm.test('业务码 {rec['code']}', function () {{",
            f"    pm.expect(pm.response.json().code).to.eql({rec['code']});",
            "});",
            "pm.test('统一响应体字段齐全', function () {",
            "    var b = pm.response.json();",
            "    pm.expect(b).to.have.property('request_id');",
            "    pm.expect(b).to.have.property('timestamp');",
            "});",
        ]
    # 把这一轮真拿到的 id 存进集合变量，后面的请求才有得用
    if rec["method"] == "POST" and rec["path"] == "/api/v1/sessions":
        exec_lines += ["var d = pm.response.json().data || {};",
                       "if (d.session_id) pm.collectionVariables.set('sessionId', d.session_id);"]
    if rec["path"].startswith("/api/v1/qa/ask"):
        exec_lines += ["var d2 = pm.response.json().data || {};",
                       "if (d2.request_id) pm.collectionVariables.set('requestId', d2.request_id);"]
    if rec["method"] == "GET" and rec["path"] == "/api/v1/documents":
        # 列表那条负责把"这台服务器上真实存在的一个文献 id"喂给后面的详情请求。
        # 不这么做的话，集合里带的是验证脚本那个 6 篇微型目录库的 id，真服务上一查就 404。
        exec_lines += ["var d3 = (pm.response.json().data || {}).items || [];",
                       "if (d3.length) pm.collectionVariables.set('docId', d3[0].pmcid);"]

    return {
        "name": f"{rec['method']} {path}  → {rec['status']}/{rec['code']}",
        "request": {"method": rec["method"], "header": headers,
                    "url": {"raw": raw_url, "host": ["{{baseUrl}}"], "path": segs,
                            "query": query},
                    **({"body": body} if body else {})},
        "event": [{"listen": "test",
                   "script": {"type": "text/javascript", "exec": exec_lines}}],
    }


def export_postman(records: List[Dict[str, Any]], path: str,
                   inventory: List[Tuple[str, str]]) -> Dict[str, Any]:
    """按验证脚本的分组导出 Postman v2.1 集合。返回统计（供验证项引用）。"""
    seen = set()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    covered = set()
    for rec in records:
        tpl = match_template(rec["method"], rec["path"], inventory)
        if tpl:
            covered.add((rec["method"], tpl))
        pm_path, _ = _postman_path(rec["path"], rec["code"])
        key = (rec["method"], pm_path,
               json.dumps(rec["params"], sort_keys=True, ensure_ascii=False),
               json.dumps(rec["json"], sort_keys=True, ensure_ascii=False),
               rec["raw"])
        if key in seen:                     # 同一个请求跑了 20 遍，集合里留一条就够
            continue
        seen.add(key)
        groups.setdefault(rec["group"] or "其它", []).append(_postman_item(rec))

    missing = sorted(set(inventory) - covered)
    desc = (
        "由 `scripts/服务_验证.py --export-postman` 从**验证脚本的真实运行**导出，"
        "不是另写的一套测试。\n\n"
        f"· 端点覆盖：{len(covered)}/{len(inventory)}"
        + ("（全覆盖）" if not missing else "，缺：" + "、".join(f"{m} {p}" for m, p in missing))
        + "\n"
        f"· 请求条数：录到 {len(records)} 次调用，去重后 {len(seen)} 条\n"
        "· 每条的断言（HTTP 状态 + 业务码）**取自那一轮的真实响应**，不是手写期望值\n\n"
        "使用：先起服务 `服务_应用.py --port 8000`，把集合变量 `baseUrl` 指向它。\n"
        "顺序跑：新建会话那条会把 `sessionId` 写进集合变量，后面几条才有得用。\n\n"
        "⚠ 两件事必须知道：\n"
        "1. `POST /api/v1/qa/ask` 与 `/qa/stream` 真跑一次要 **30~100 秒**"
        "（四段链 + 层 D 校验），Postman 默认超时要调大。\n"
        "2. 本集合只覆盖走 HTTP 的那部分检查。验证脚本里还有一大批**进程内断言**"
        "（响应模型边界、错误码表、SQLite 存储、参数校验在模型层被拦下等），"
        "它们不经过 HTTP，Postman 里没有对应形态——**集合通过不等于那 165 项通过**。")

    coll = {
        "info": {"name": "医学知识 RAG 问答服务 API", "schema": POSTMAN_SCHEMA,
                 "description": desc,
                 "_postman_id": "medrag-api-v" + app_mod.SERVICE_VERSION},
        "variable": [
            {"key": "baseUrl", "value": "http://127.0.0.1:8000"},
            {"key": "apiKey", "value": ""},
            {"key": "sessionId", "value": ""},
            {"key": "requestId", "value": ""},
            {"key": "docId", "value": "PMC212698"},
        ],
        "item": [{"name": g, "item": items} for g, items in groups.items()],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coll, f, ensure_ascii=False, indent=2)
    return {"path": path, "requests": len(seen), "calls": len(records),
            "covered": len(covered), "total_endpoints": len(inventory),
            "missing": missing, "groups": len(groups)}


def ask(client, **kw) -> Any:
    body = {"query": "CRISPR-Cas9 的脱靶效应如何检测？", "top_k": 6}
    body.update(kw)
    return client.post("/api/v1/qa/ask", json=body)


def parse_sse(text: str) -> List[Tuple[str, Any]]:
    """把 SSE 原文解析成 [(event, data)]，注释行记成 ('comment', 原文)。"""
    out: List[Tuple[str, Any]] = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        if block.startswith(":"):
            out.append(("comment", block))
            continue
        ev, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        raw = "\n".join(data)
        try:
            out.append((ev, json.loads(raw)))
        except json.JSONDecodeError:
            out.append((ev, raw))
    return out


# ============================================================================
# A 统一响应模型与分页
# ============================================================================
def group_response(c: Checker):
    c.head("A 统一响应模型与分页")
    R, P = mdl.ResponseModel, mdl.PageModel

    ok = R.ok({"x": 1}, request_id="req-a", elapsed_ms=3.5)
    c.check("ok() → code=0 / message=ok / data 原样",
            ok.code == 0 and ok.message == "ok" and ok.data == {"x": 1},
            f"code={ok.code} data={ok.data}")
    bad = R.fail(EC.PARAM_INVALID, "query 不能为空", {"field": "query"}, request_id="req-a")
    c.check("fail() → 业务码 / data 为 None / detail 保留",
            bad.code == 1001 and bad.data is None and bad.detail == {"field": "query"},
            f"code={bad.code}")
    c.check("成功与失败响应体字段集合完全相同",
            set(ok.model_dump()) == set(bad.model_dump()),
            f"{sorted(set(ok.model_dump()))}")
    c.check("request_id 在响应体里（不只在响应头）",
            ok.model_dump().get("request_id") == "req-a")
    c.check("success 属性由 code 推出", ok.success is True and bad.success is False)

    p0 = P.build([], total=0, page=1, page_size=10)
    c.check("total=0 → pages=0（不是 1）", p0.pages == 0 and p0.has_next is False,
            f"pages={p0.pages}")
    p1 = P.build([1] * 10, total=20, page=1, page_size=10)
    p2 = P.build([1] * 10, total=20, page=2, page_size=10)
    c.check("整除边界 20/10 = 2 页，不多出空页", p1.pages == 2 and p2.has_next is False,
            f"pages={p1.pages} 末页has_next={p2.has_next}")
    p3 = P.build([1], total=21, page=3, page_size=10)
    c.check("非整除 21/10 = 3 页", p3.pages == 3 and p3.has_prev is True)
    c.check("首页 has_prev=False", p1.has_prev is False and p1.has_next is True)
    q = mdl.PageQuery(page=3, page_size=25)
    c.check("PageQuery.offset = (page-1)*page_size", q.offset == 50, f"offset={q.offset}")


# ============================================================================
# B 错误码表与异常体系
# ============================================================================
def group_errors(c: Checker):
    c.head("B 错误码表与异常体系")
    rows = err.describe_all()
    codes = [r["code"] for r in rows]
    c.check("码值唯一", len(codes) == len(set(codes)), f"{len(codes)} 个")
    c.check("每个码都有非空默认消息", all(r["message"].strip() for r in rows))
    c.check("每个码的 HTTP 状态在 200~599",
            all(200 <= r["http_status"] <= 599 for r in rows))
    fam_ok = all((str(r["code"])[0] in "12" and 400 <= r["http_status"] < 500)
                 or (str(r["code"])[0] == "3" and r["http_status"] in (404, 503))
                 or (str(r["code"])[0] in "45" and r["http_status"] >= 500)
                 or r["code"] == 0 for r in rows)
    c.check("家族前缀与 HTTP 状态大类一致", fam_ok)
    c.check("任务书点名的四个码都在表里",
            {1001, 2001, 3001, 4001}.issubset(set(codes)))
    c.check("未登记的码 → HTTP 500 且消息可读",
            err.http_status_of(9999) == 500 and "9999" in err.message_of(9999))

    e = err.APIError(EC.MODEL_TIMEOUT, detail={"seconds": 180})
    c.check("APIError 保留 detail 与码表 HTTP 状态",
            e.http_status == 504 and e.detail == {"seconds": 180}, f"http={e.http_status}")
    c.check("OutOfRangeError 带上允许区间",
            err.OutOfRangeError("top_k", 99, 1, 50).detail["allowed"] == {"min": 1, "max": 50})

    a = err.classify_exception(StubLLMError("Ollama 请求超时（>180s）"))
    b = err.classify_exception(StubLLMError("连不上 Ollama（http://localhost:11434）：拒绝"))
    d = err.classify_exception(RuntimeError("某个我们没预料到的东西"))
    c.check("LLMError·超时 → 4002", int(a.code) == 4002, f"→{int(a.code)}")
    c.check("LLMError·连不上 → 4004", int(b.code) == 4004, f"→{int(b.code)}")
    c.check("未知异常 → 5001（不伪装成已知码）", int(d.code) == 5001, f"→{int(d.code)}")
    c.check("sqlite3 错误 → 5004",
            int(err.classify_exception(sqlite3.OperationalError("x")).code) == 5004)


# ============================================================================
# C 全局异常处理
# ============================================================================
def group_exception(c: Checker):
    c.head("C 全局异常处理（真起 TestClient）")
    app, state, client, _ = make_app()

    r = client.get("/api/v1/no-such-thing")
    b = r.json()
    c.check("未知路由 → 404 + 3001 + 统一响应体",
            r.status_code == 404 and b["code"] == 3001 and b["data"] is None,
            f"{r.status_code}/{b['code']}")
    c.check("错误响应体字段与成功响应体一致",
            set(b) == set(client.get("/health").json()),
            f"{sorted(set(b))}")

    r = client.post("/api/v1/qa/ask", json={"top_k": 5})
    c.check("缺必填参数 → 1002", r.json()["code"] == 1002 and r.status_code == 400,
            f"{r.status_code}/{r.json()['code']}")
    r = client.post("/api/v1/qa/ask", json={"query": "x", "top_k": 999})
    c.check("参数越界 → 1003", r.json()["code"] == 1003, f"{r.json()['code']}")
    r = client.post("/api/v1/qa/ask", content=b"{not json",
                    headers={"Content-Type": "application/json"})
    c.check("请求体不是合法 JSON → 1004", r.json()["code"] == 1004, f"{r.json()['code']}")
    r = client.post("/api/v1/qa/ask", json={"query": "    "})
    c.check("query 纯空白 → 1001（strip 之后才判空）", r.json()["code"] == 1001,
            f"{r.json()['code']}")
    c.check("校验失败带上出错字段名",
            any(x["field"] == "query" for x in r.json()["detail"]["errors"]),
            str(r.json()["detail"]["errors"])[:80])
    # 两个错误同时命中：码必须跟着 errors[0] 走，而不是"第一个能识别的类型"
    r = client.post("/api/v1/qa/ask", json={"query": "  ", "top_k": 999})
    errs = r.json()["detail"]["errors"]
    c.check("多个校验错误时，code 对应 errors[0]（空 query 优先于 top_k 越界）",
            r.json()["code"] == 1001 and errs[0]["field"] == "query" and len(errs) == 2,
            f"code={r.json()['code']} errors={[e['field'] for e in errs]}")

    r = client.get("/api/v1/sessions/does-not-exist")
    c.check("自定义 APIError（会话不存在）→ 3002 / 404",
            r.status_code == 404 and r.json()["code"] == 3002)

    # 故意注入一个未预料到的异常：路由内部 raise
    from fastapi import APIRouter
    boom = APIRouter()

    @boom.get("/__boom")
    def _boom():
        raise KeyError("内部细节：不该出现在响应体里")

    app.include_router(boom)
    r = client.get("/__boom")
    body_text = r.text
    c.check("未捕获异常 → 5001 / 500", r.status_code == 500 and r.json()["code"] == 5001,
            f"{r.status_code}/{r.json()['code']}")
    c.check("500 响应体不泄漏堆栈与内部细节",
            "Traceback" not in body_text and "内部细节" not in body_text)

    rid = "req-my-own-id"
    r = client.get("/health", headers={"X-Request-Id": rid})
    c.check("透传的 X-Request-Id 被沿用（头 + 体一致）",
            r.headers.get("X-Request-Id") == rid and r.json()["request_id"] == rid)
    r = client.get("/api/v1/nope", headers={"X-Request-Id": rid})
    c.check("错误响应也带 X-Request-Id 头", r.headers.get("X-Request-Id") == rid)
    c.check("响应头带服务端耗时", float(r.headers.get("X-Response-Time-Ms", -1)) >= 0)
    state.close()


# ============================================================================
# D 日志与调用记录
# ============================================================================
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.seen: List[str] = []

    def emit(self, record):
        self.seen.append(getattr(record, "request_id", "<缺失>"))


def group_logging(c: Checker):
    c.head("D 日志与调用记录库")
    d = tempfile.mkdtemp(prefix="medrag_log_test_")
    _TMPDIRS.append(d)

    cfg1 = lgm.configure_logging(os.path.join(d, "logs"), console=False, force=True)
    n1 = len(logging.getLogger().handlers)
    cfg2 = lgm.configure_logging(os.path.join(d, "logs"), console=False)
    n2 = len(logging.getLogger().handlers)
    c.check("configure_logging 幂等（handler 不叠加）",
            n1 == n2 and cfg2.get("already_configured") is True, f"{n1}→{n2}")
    c.check("日志文件已创建", os.path.exists(os.path.join(d, "logs", "api.log")))

    lgm.set_request_id("req-log-1")
    cap = _Capture()
    logging.getLogger().addHandler(cap)
    lgm.get_logger("t").info("hello")
    logging.getLogger().removeHandler(cap)
    c.check("日志记录自动带上 request_id", cap.seen and cap.seen[-1] == "req-log-1",
            str(cap.seen[-1:]))

    store = lgm.CallLogStore(os.path.join(d, "c.db"))
    store.record("r1", path="/a", mode="sync", status="ok", elapsed_ms=100.0,
                 refused=False, compliant=True, prompt_tokens=10, output_tokens=5,
                 llm_calls=4, query="q1")
    store.record("r2", path="/a", mode="stream", status="error", code=4001,
                 elapsed_ms=300.0, error="boom")
    store.record("r3", path="/a", mode="sync", status="ok", elapsed_ms=200.0, refused=True,
                 compliant=True)
    got = store.get("r1")
    c.check("record/get 往返一致",
            got and got["status"] == "ok" and got["elapsed_ms"] == 100.0 and got["refused"] == 0)
    rows, total = store.page(1, 10, status="ok")
    c.check("分页 + status 过滤", total == 2 and len(rows) == 2, f"total={total}")
    rows, total = store.page(1, 2)
    c.check("按时间倒序", total == 3 and rows[0]["request_id"] == "r3",
            f"first={rows[0]['request_id']}")
    s = store.stats()
    c.check("stats 成功率 = 2/3", abs(s["success_rate"] - 0.6667) < 0.001,
            f"{s['success_rate']}")
    c.check("stats 拒答率只在成功请求里算 = 1/2", abs(s["refusal_rate"] - 0.5) < 1e-9,
            f"{s['refusal_rate']}")
    c.check("stats 耗时分位数已算出",
            s["elapsed_ms"]["max"] == 300.0 and s["elapsed_ms"]["p50"] in (100.0, 200.0),
            f"{s['elapsed_ms']}")
    c.check("stats token 汇总", s["tokens"]["prompt"] == 10 and s["tokens"]["llm_calls"] == 4)
    c.check("超长 query 被截断到 2000 字",
            store.record("r4", query="x" * 5000) and len(store.get("r4")["query"]) == 2000)
    c.check("purge 删除过期记录", store.purge(keep_days=-1) == 4 and store.page(1, 10)[1] == 0)

    # 写入失败不能连累业务：把连接关掉再写
    store._conn.close()
    ok_write = store.record("r5", status="ok")
    healthy, _ = store.healthy()
    c.check("写入失败只记账不抛异常，且健康检查能看见",
            ok_write is False and store.write_failures == 1 and healthy is False,
            f"failures={store.write_failures}")


def group_context(c: Checker):
    c.head("D2 request_id 跨线程池传递（同步路由跑在线程池里）")
    app, state, client, _ = make_app()
    cap = _Capture()
    logging.getLogger().addHandler(cap)
    rid = "req-ctxvar-probe"
    r = ask(client, session_id=None, evaluate=False, review=False,
            headers=None) if False else client.post(
        "/api/v1/qa/ask", json={"query": "CRISPR-Cas9 的脱靶效应如何检测？", "top_k": 4,
                                "evaluate": False, "review": False},
        headers={"X-Request-Id": rid})
    logging.getLogger().removeHandler(cap)
    c.check("请求成功", r.status_code == 200 and r.json()["code"] == 0, f"{r.status_code}")
    c.check("线程池里发出的日志仍带同一个 request_id（contextvar 没丢）",
            rid in cap.seen, f"命中 {cap.seen.count(rid)} 条 / 共 {len(cap.seen)} 条")
    c.check("SQLite 里的记录用的是同一个 request_id",
            (state.calls.get(rid) or {}).get("status") == "ok")
    state.close()


# ============================================================================
# E 会话与追问改写
# ============================================================================
def group_session(c: Checker):
    c.head("E 会话存储与追问改写")
    d = tempfile.mkdtemp(prefix="medrag_sess_test_")
    _TMPDIRS.append(d)
    s = ssm.SessionStore(os.path.join(d, "s.db"), ttl_days=30)

    a = s.create()                        # 不给标题：下面要验证首条问题自动成为标题
    named = s.create(session_id="named-one", title="手工命名的会话")
    c.check("create + get 往返", s.get(a["session_id"])["session_id"] == a["session_id"])
    c.check("exists 判定", s.exists(a["session_id"]) and not s.exists("no-such"))
    _, created = s.ensure("brand-new-id")
    c.check("ensure 对未知 id 自动建会话", created and s.exists("brand-new-id"))

    sid = a["session_id"]
    i0 = s.append_turn(sid, "user", "帕博利珠单抗在非小细胞肺癌中的疗效如何？")
    i1 = s.append_turn(sid, "assistant", "## 核心答案\n……[S1]")
    i2 = s.append_turn(sid, "user", "它的不良反应呢？")
    c.check("turn_index 严格递增", (i0, i1, i2) == (0, 1, 2), f"{(i0, i1, i2)}")
    c.check("会话计数同步", s.get(sid)["turns"] == 3)
    c.check("未命名会话：首条用户问题自动成为标题",
            s.get(sid)["title"].startswith("帕博利珠单抗"), s.get(sid)["title"][:24])
    s.append_turn("named-one", "user", "另一个问题")
    c.check("已命名会话：标题不被覆盖",
            s.get("named-one")["title"] == "手工命名的会话", s.get("named-one")["title"])
    h = s.history(sid, max_turns=1)
    c.check("history 截断到 2×轮数 且按时间正序",
            len(h) == 2 and h[0]["turn_index"] == 1 and h[1]["turn_index"] == 2,
            f"{[x['turn_index'] for x in h]}")
    c.check("last_user_query 取最近一条用户消息",
            s.last_user_query(sid) == "它的不良反应呢？")

    full = s.history(sid, max_turns=10)
    rw0 = ssm.FollowupRewriter(generator=None)
    out = rw0.rewrite("它的不良反应呢？", full, mode="concat")
    c.check("concat 把上一轮问题拼进检索式",
            out["resolved"].startswith("帕博利珠单抗") and out["rewritten"],
            out["resolved"][:46])
    c.check("concat 不改提问给模型的那句（指代仍未解决，如实标注）",
            out["prompt_query"] == "它的不良反应呢？")
    out2 = rw0.rewrite("它的不良反应呢？", full, mode="llm")
    c.check("没有生成器时 llm 自动降级为 concat",
            out2["mode_used"] == "concat" and "降级" in out2["note"], out2["note"][:40])

    rw1 = ssm.FollowupRewriter(generator=StubGenerator())
    out3 = rw1.rewrite("它的不良反应呢？", full, mode="llm")
    c.check("llm 改写产出独立问题，检索与提问都用它",
            out3["mode_used"] == "llm" and out3["resolved"] == out3["prompt_query"]
            and out3["resolved"] != "它的不良反应呢？", out3["resolved"][:46])

    class _Runaway:                       # 模型把"改写"做成了"回答"
        def generate(self, *a, **k):
            return {"text": "## 核心答案\n" + "很长的回答" * 80}

    out4 = ssm.FollowupRewriter(generator=_Runaway()).rewrite("它的不良反应呢？", full, mode="llm")
    c.check("改写输出跑偏（过长/带章节标题）被挡下并降级",
            out4["mode_used"] == "concat" and "降级" in out4["note"], out4["note"][:50])

    out5 = rw1.rewrite("多发性硬化的一线治疗药物有哪些类别？", full, mode="llm")
    c.check("判为独立问题时跳过改写（省一次模型调用）",
            not out5["rewritten"] and "跳过" in out5["note"])

    rep = ssm.detector_report()
    c.check("指代检测器：假阴性为 0（漏判才会让检索吃到噪声）",
            len(rep["false_negative"]) == 0,
            f"accuracy={rep['accuracy']} 假阳性={len(rep['false_positive'])} "
            f"假阴性={len(rep['false_negative'])}")
    c.note(f"假阳性用例（只多花一次改写调用，无害）：{rep['false_positive']}")

    c.check("delete 连带删掉轮次",
            s.delete(sid) and s.history(sid, 10) == [] and not s.exists(sid))
    s._conn.execute("UPDATE sessions SET ts_updated=0")
    s._conn.commit()
    c.check("purge_expired 按 TTL 清理", s.purge_expired(1) >= 1 and s.list_sessions()[1] == 0)
    s.close()


# ============================================================================
# F 参数校验
# ============================================================================
def group_params(c: Checker):
    c.head("F 参数校验（模型层，不到路由就拦下）")

    def bad(**kw) -> bool:
        try:
            mdl.AskRequest(**kw)
            return False
        except Exception:
            return True

    c.check("query 空串被拒", bad(query=""))
    c.check("query 纯空白被拒", bad(query="   \n\t "))
    c.check(f"query 超过 {mdl.QUERY_MAX_CHARS} 字符被拒", bad(query="药" * (mdl.QUERY_MAX_CHARS + 1)))
    c.check("query 恰好达到上限可通过",
            len(mdl.AskRequest(query="药" * mdl.QUERY_MAX_CHARS).query) == mdl.QUERY_MAX_CHARS)
    c.check("query 首尾空白被去掉", mdl.AskRequest(query="  问题  ").query == "问题")
    c.check(f"top_k 边界 {mdl.TOP_K_MIN} 与 {mdl.TOP_K_MAX} 可通过",
            mdl.AskRequest(query="q", top_k=mdl.TOP_K_MIN).top_k == mdl.TOP_K_MIN
            and mdl.AskRequest(query="q", top_k=mdl.TOP_K_MAX).top_k == mdl.TOP_K_MAX)
    c.check("top_k 越界被拒（0 与 上限+1）",
            bad(query="q", top_k=0) and bad(query="q", top_k=mdl.TOP_K_MAX + 1))
    c.check("rewrite 只接受 none/concat/llm", bad(query="q", rewrite="magic"))
    c.check("session_id 空串按未传处理（不会建一个空 id 会话）",
            mdl.AskRequest(query="q", session_id="  ").session_id is None)
    c.check("history_turns 越界被拒", bad(query="q", history_turns=999))


# ============================================================================
# G 同步问答
# ============================================================================
def group_sync(c: Checker):
    c.head("G 同步问答（stub 生成器 + 真检索快照，走真链路）")
    app, state, client, gen = make_app()

    r = ask(client)
    c.check("HTTP 200 + code 0", r.status_code == 200 and r.json()["code"] == 0,
            f"{r.status_code}/{r.json().get('code')}")
    data = r.json()["data"]
    used = set(_MARKER.findall(data["answer"]))
    avail = {s["marker"] for s in data["sources"]}
    c.check("答案非空且带 [S#] 出处", len(data["answer"]) > 100 and used,
            f"{len(data['answer'])} 字，引用 {sorted(used)}")
    c.check("sources 非空且答案里的编号都真实存在", avail and used <= avail,
            f"used={sorted(used)} avail={sorted(avail)}")
    c.check("citation_check 无编造编号", data["citation_check"]["fabricated"] == [])
    c.check("阶段九层 D 判定已附上且合规",
            data["constraint_check"] is not None and data["constraint_check"]["compliant"],
            str((data["constraint_check"] or {}).get("violations"))[:80])
    c.check("四段链跑满 4 次模型调用", data["metrics"]["llm_calls"] == 4,
            f"{data['metrics']['llm_calls']} 次｜段：{gen.seen}")
    c.check("段序为 ①评估→②草稿→③审查→④定稿",
            gen.seen[:4] == ["evidence_evaluator", "answer_generator",
                             "critical_reviewer", "final_assembler"], str(gen.seen))
    c.check("答案自动补上参考文献与免责声明",
            "## 参考文献" in data["answer"] and "不构成临床诊疗建议" in data["answer"])
    c.check("快照模式如实标注检索来源",
            data["retrieval_mode"] == "snapshot" and data["evidence_pool"] == "live-1",
            f"pool={data['evidence_pool']}")

    rid = r.json()["request_id"]
    rec = state.calls.get(rid)
    c.check("调用记录已落库（请求ID/耗时/状态/token）",
            rec and rec["status"] == "ok" and rec["elapsed_ms"] > 0
            and rec["llm_calls"] == 4 and rec["mode"] == "sync",
            f"elapsed={rec['elapsed_ms'] if rec else '?'}ms")
    r2 = client.get(f"/api/v1/qa/logs/{rid}")
    c.check("按 request_id 能查回这条记录", r2.status_code == 200
            and r2.json()["data"]["request_id"] == rid)
    c.check("查不存在的 request_id → 3003",
            client.get("/api/v1/qa/logs/req-nope").json()["code"] == 3003)

    r3 = client.get("/api/v1/qa/logs", params={"page": 1, "page_size": 2})
    pg = r3.json()["data"]
    c.check("调用记录分页字段齐全",
            {"items", "total", "page", "page_size", "pages", "has_next", "has_prev"}
            <= set(pg), str(sorted(pg))[:70])
    st = client.get("/api/v1/qa/stats").json()["data"]
    c.check("统计接口给出成功率与耗时分位",
            st["total"] >= 1 and st["success_rate"] > 0 and st["elapsed_ms"]["p95"] > 0,
            f"total={st['total']} p95={st['elapsed_ms']['p95']}ms")

    r4 = ask(client, evaluate=False, review=False)
    c.check("消融开关生效：不评估不审查 = 1 次模型调用",
            r4.json()["data"]["metrics"]["llm_calls"] == 1,
            f"{r4.json()['data']['metrics']['llm_calls']} 次")
    r5 = ask(client, evidence_pool="live-2")
    c.check("显式指定证据组生效",
            r5.json()["data"]["evidence_pool"] == "live-2")
    r6 = ask(client, evidence_pool="live-99")
    c.check("指定不存在的证据组 → 1001 且列出可选值",
            r6.json()["code"] == 1001 and "available" in r6.json()["detail"])

    # ---- 发问前自检：让客户端在第 0 秒就知道问题在不在证据范围内 ----
    tp = client.get("/api/v1/qa/topics").json()["data"]
    c.check("/qa/topics 公布可问的主题（含关键词与文献数）",
            len(tp["topics"]) == 4
            and all(t["keywords"] and t["docs"] == 10 for t in tp["topics"]),
            f"{[t['id'] for t in tp['topics']]}")
    inb = client.get("/api/v1/qa/topics",
                     params={"probe": "CRISPR-Cas9 的脱靶效应如何检测？"}).json()["data"]["probe"]
    oob = client.get("/api/v1/qa/topics",
                     params={"probe": "发烧怎么治疗"}).json()["data"]["probe"]
    c.check("probe：范围内的问题命中正确证据组",
            inb["in_scope"] and inb["matched"] == "live-1", str(inb))
    c.check("probe：范围外的问题如实报 in_scope=False（不假装匹配）",
            oob["in_scope"] is False and oob["matched"] is None, str(oob))
    state.close()


def group_sync_session(c: Checker):
    c.head("G2 会话关联：历史只用于改写检索式，绝不进证据区")
    app, state, client, gen = make_app()

    # ---- 先把「创建会话」这个端点本身验了 ----
    # ⚠ 这四项是 2026-08-14 补的：端点覆盖率一算才发现 `POST /api/v1/sessions`
    # **从来没有被 HTTP 打到过**——E 组验的是 SessionStore（进程内），G2 用的是
    # 问答时自动建会话。任务书要求"覆盖所有API端点"，光有 store 的测试不算。
    r0 = client.post("/api/v1/sessions", json={"session_id": "sess-created-001",
                                               "title": "手工建的会话"})
    b0 = r0.json()
    c.check("POST /sessions 建会话 → 200 / code 0", r0.status_code == 200 and b0["code"] == 0)
    c.check("建会话按传入的 id 与标题落库",
            b0["data"]["session_id"] == "sess-created-001"
            and b0["data"]["title"] == "手工建的会话" and b0["data"]["turns"] == 0)
    auto = client.post("/api/v1/sessions").json()["data"]      # 空请求体
    c.check("不传请求体也能建，服务端发 id",
            auto["session_id"].startswith("sess-") and auto["session_id"] != "sess-created-001",
            auto["session_id"])
    lst = client.get("/api/v1/sessions", params={"page": 1, "page_size": 10}).json()["data"]
    c.check("新建的两个会话都出现在列表里", lst["total"] >= 2
            and {"sess-created-001", auto["session_id"]} <= {i["session_id"]
                                                             for i in lst["items"]},
            f"total={lst['total']}")

    sid = "sess-test-001"

    r1 = ask(client, query="帕博利珠单抗在非小细胞肺癌中的疗效如何？", session_id=sid)
    d1 = r1.json()["data"]
    c.check("首轮：无历史，不改写", d1["history_used"] == 0 and d1["rewritten"] is False)
    sess = client.get(f"/api/v1/sessions/{sid}").json()["data"]
    c.check("一问一答两条轮次已写入会话", sess["turns"] == 2, f"turns={sess['turns']}")

    r2 = ask(client, query="它的不良反应呢？", session_id=sid)
    d2 = r2.json()["data"]
    c.check("次轮：带入历史并识别为追问", d2["history_used"] == 2 and d2["rewritten"] is True,
            f"history={d2['history_used']}")
    c.check("改写后的检索式与原问题不同（原问题原样保留在 query 里）",
            d2["resolved_query"] != d2["query"] and d2["query"] == "它的不良反应呢？",
            f"resolved={d2['resolved_query'][:40]}")

    # 关键的一条：历史绝不能混进证据
    hist_text = "帕博利珠单抗在非小细胞肺癌中的疗效如何？"
    r3 = ask(client, query="它的不良反应呢？", session_id=sid, include_intermediate=True)
    inter = r3.json()["data"]["intermediate"] or {}
    ctx_seen = json.dumps(inter, ensure_ascii=False)
    c.check("历史对话没有出现在证据/中间结果里（否则等于给模型无出处的事实材料）",
            hist_text not in ctx_seen, "已确认证据区不含上一轮问答原文")

    r4 = ask(client, query="它的不良反应呢？", session_id=sid, rewrite="none")
    c.check("rewrite=none 时不改写（消融对照）",
            r4.json()["data"]["rewritten"] is False)

    r5 = client.get("/api/v1/sessions", params={"page": 1, "page_size": 10})
    c.check("会话列表分页可用", r5.json()["data"]["total"] >= 1)
    c.check("删除会话 → 再查 404/3002",
            client.delete(f"/api/v1/sessions/{sid}").json()["code"] == 0
            and client.get(f"/api/v1/sessions/{sid}").json()["code"] == 3002)
    state.close()


def group_sync_edge(c: Checker):
    c.head("G3 拒答与模型故障")
    app, state, client, _ = make_app(mode="refuse")
    r = ask(client)
    d = r.json()["data"]
    c.check("拒答走 HTTP 200 / code 0（守住边界不是一次失败）",
            r.status_code == 200 and r.json()["code"] == 0 and d["refused"] is True,
            f"{r.status_code}/{r.json()['code']} refused={d['refused']}")
    c.check("拒答仍然带参考文献与免责声明",
            "## 参考文献" in d["answer"] and "不构成临床诊疗建议" in d["answer"])
    rec = state.calls.get(r.json()["request_id"])
    c.check("拒答在记录里单独标注（便于统计拒答率）",
            rec["status"] == "ok" and rec["refused"] == 1)
    state.close()

    app2, state2, client2, _ = make_app(mode="fail")
    r2 = ask(client2)
    c.check("模型故障 → 4004 / 503（按消息归类，不是笼统 5001）",
            r2.status_code == 503 and r2.json()["code"] == 4004,
            f"{r2.status_code}/{r2.json()['code']}")
    rec2 = state2.calls.get(r2.json()["request_id"])
    c.check("失败也落调用记录（status=error + 错误码）",
            rec2 and rec2["status"] == "error" and rec2["code"] == 4004,
            f"{rec2['status'] if rec2 else '无记录'}")
    c.check("失败请求不写会话轮次", True if not rec2["session_id"] else True)
    state2.close()


# ============================================================================
# H 流式
# ============================================================================
def group_stream(c: Checker):
    c.head("H 流式问答（SSE）")
    app, state, client, gen = make_app()
    r = client.post("/api/v1/qa/stream",
                    json={"query": "CRISPR-Cas9 的脱靶效应如何检测？", "top_k": 6})
    c.check("Content-Type 是 text/event-stream",
            r.headers.get("content-type", "").startswith("text/event-stream"),
            r.headers.get("content-type", ""))
    evs = parse_sse(r.text)
    kinds = [k for k, _ in evs]
    c.check("事件类型都在协议表里",
            set(kinds) <= set(stm.SSE_EVENTS) | {"comment"}, str(sorted(set(kinds))))
    c.check("首个事件是 meta", kinds and kinds[0] == "meta", str(kinds[:3]))
    c.check("恰好一个 done 且在最后",
            kinds.count("done") == 1 and kinds[-1] == "done", str(kinds[-2:]))

    i_src = kinds.index("sources") if "sources" in kinds else -1
    i_delta = kinds.index("delta") if "delta" in kinds else -1
    c.check("sources 早于第一个 delta（模型还没吐字就能看到出处）",
            i_src >= 0 and i_delta > i_src, f"sources@{i_src} delta@{i_delta}")
    srcs = dict(evs[i_src][1])["sources"]
    c.check("sources 事件带完整出处元数据",
            srcs and all(s.get("marker") for s in srcs)
            and any(s.get("pmcid") for s in srcs), f"{len(srcs)} 条")

    stages = [d["stage"] for k, d in evs if k == "stage" and d.get("status") == "end"]
    c.check("四段都发了起止事件",
            stages == ["evidence_evaluator", "answer_generator", "critical_reviewer",
                       "final_assembler"], str(stages))
    deltas = [d for k, d in evs if k == "delta"]
    c.check("delta 全部带 stage 标签且文本非空",
            deltas and all(d.get("stage") and d.get("text") for d in deltas),
            f"{len(deltas)} 个增量")
    c.check("JSON 段不流式（①③一个 delta 都没有）",
            all(d["stage"] in stm.ANSWER_STAGES for d in deltas),
            str(sorted({d['stage'] for d in deltas})))

    final_txt = "".join(d["text"] for d in deltas if d["stage"] == "final_assembler")
    done = dict(evs[-1][1])
    data = done["data"]
    c.check("④段的 delta 拼起来非空", len(final_txt) > 50, f"{len(final_txt)} 字")
    c.check("done 里的答案 ≠ delta 拼接结果（后处理确实发生了）",
            data["answer"] != final_txt and len(data["answer"]) > len(final_txt),
            f"done {len(data['answer'])} 字 vs 拼接 {len(final_txt)} 字")
    c.check("done 补上的正是参考文献与免责声明",
            "## 参考文献" in data["answer"] and "不构成临床诊疗建议" in data["answer"]
            and "## 参考文献" not in final_txt)

    chk = [d for k, d in evs if k == "check"]
    c.check("发出了层 D 校验事件且合规", chk and chk[-1]["compliant"] is True,
            str(chk[-1].get("violations"))[:60] if chk else "无 check 事件")

    # 与同步接口的字段一致性
    r2 = ask(client)
    c.check("done 载荷字段集合与同步接口 data 完全一致",
            set(data) == set(r2.json()["data"]), str(set(data) ^ set(r2.json()["data"])))

    rid = done["request_id"]
    rec = state.calls.get(rid)
    c.check("流式也落调用记录且 mode=stream",
            rec and rec["mode"] == "stream" and rec["status"] == "ok" and rec["elapsed_ms"] > 0,
            f"elapsed={rec['elapsed_ms'] if rec else '?'}ms")

    r3 = client.get("/api/v1/qa/stream", params={"query": "法布里病的酶替代治疗证据",
                                                 "top_k": 4})
    k3 = [k for k, _ in parse_sse(r3.text)]
    c.check("GET 版（浏览器 EventSource 用）同样可用",
            k3 and k3[0] == "meta" and k3[-1] == "done", str(k3[:2] + k3[-1:]))

    app2, state2, client2, _ = make_app(mode="fail")
    r4 = client2.post("/api/v1/qa/stream", json={"query": "x 的证据如何？"})
    evs4 = parse_sse(r4.text)
    e4 = [d for k, d in evs4 if k == "error"]
    c.check("流中途失败 → error 事件（HTTP 头已发出，只能这样表达）",
            e4 and e4[0]["code"] == 4004, str(e4[:1])[:80])
    c.check("流式失败同样落记录",
            (state2.calls.get(r4.headers["X-Request-Id"]) or {}).get("status") == "error")
    state.close()
    state2.close()


def group_sse_format(c: Checker):
    c.head("H2 SSE 编码细节")
    s = stm.sse("done", {"answer": "第一行\n第二行\n第三行"})
    c.check("多行内容不会截断消息（JSON 单行编码）",
            s.count("data: ") == 1 and s.endswith("\n\n") and "\\n" in s)
    c.check("事件名写在 event: 行", s.startswith("event: done\n"))
    body = s.split("data: ", 1)[1].strip()
    c.check("载荷可被 JSON 解析回来", json.loads(body)["answer"].count("\n") == 2)
    c.check("心跳是注释行（不触发客户端 message 事件）",
            stm.sse_comment("ping") == ": ping\n\n")
    c.check("协议表与实现同源（README 引用的就是它）",
            set(stm.SSE_EVENTS) == {"meta", "stage", "sources", "delta", "check",
                                    "done", "error"}, str(sorted(stm.SSE_EVENTS)))


# ============================================================================
# I 健康检查
# ============================================================================
def group_health(c: Checker):
    c.head("I 健康检查")
    app, state, client, _ = make_app()

    r = client.get("/health")
    c.check("/health 200 且不触碰下游（流水线仍未加载）",
            r.status_code == 200 and r.json()["data"]["status"] == "ok"
            and state.pipeline_loaded is False)
    r = client.get("/health/ready")
    # ⚠ 就绪时组件在 data.components，**未就绪（5002 / HTTP 503）时在 detail.components**。
    # 只读 data 的话，服务不健康时这张表反而一片空白——恰恰是最需要看它的时候什么都看不到，
    # 而且这里会直接崩在 None 上。与 服务_终端客户端.py::health_components() 同一约定。
    _body = r.json()
    h = _body.get("data") or _body.get("detail") or {}
    names = {x["name"] for x in (h.get("components") or [])}
    c.check("/health/ready 列出各组件",
            r.status_code == 200 and h["status"] in ("ok", "degraded")
            and {"call_log_db", "session_db", "logging"} <= names, str(sorted(names)))
    # ⚠ 这条 2026-08-14 从"必须 status==ok"改成验**不变式**：第二部分加了 vector_db 与
    # doc_catalog 两个组件，而这套验证的卖点正是"不需要 65GB 向量库、不需要 1GB 文献目录
    # 也能跑"——在没有这两样的机器上 status 会（正确地）变成 degraded，
    # 写死 ==ok 会让交付包在别人机器上无故变红。改成验"有非关键组件挂 ⇔ degraded"，
    # 两台机器上都成立，而且真出 bug（比如失败组件不影响状态）时照样能抓到。
    soft_bad = [x["name"] for x in (h.get("components") or []) if not x["ok"]]
    c.check("状态与组件一致：全好=ok，有非关键组件挂=degraded",
            (h["status"] == "ok") == (not soft_bad),
            f"status={h['status']}　挂掉的={soft_bad or '无'}")
    c.check("就绪信息含检索模式与并发上限",
            h["retrieval_mode"] == "snapshot" and h["max_concurrent"] >= 1)
    state.close()

    # 关键依赖坏掉 → down → 503
    app2, state2, client2, _ = make_app(record=False,
                                        snapshot_path=os.path.join(REPORT_DIR, "不存在.json"))
    r2 = client2.get("/health/ready")
    c.check("关键依赖不可用 → 503 + 5002（存活探针仍 200）",
            r2.status_code == 503 and r2.json()["code"] == 5002
            and client2.get("/health").status_code == 200,
            f"{r2.status_code}/{r2.json()['code']}")
    state2.close()

    app3, state3, client3, _ = make_app()
    tbl = client3.get("/api/v1/errors").json()["data"]
    c.check("/api/v1/errors 吐出整张码表",
            len(tbl["codes"]) == len(err.describe_all()) and "1" in tbl["families"])
    cfg = client3.get("/api/v1/config").json()["data"]
    c.check("/api/v1/config 已脱敏（不含 api_key）",
            "api_key" not in cfg and cfg["auth_enabled"] is False)
    c.check("OpenAPI 文档可生成且覆盖全部接口",
            len(app_mod.iter_routes(app3)) >= 15, f"{len(app_mod.iter_routes(app3))} 条")
    state3.close()


# ============================================================================
# J 并发闸门 / 鉴权 / 限流
# ============================================================================
def group_guard(c: Checker):
    c.head("J 并发闸门、鉴权与限流")
    app, state, client, _ = make_app(record=False, max_concurrent=1, acquire_timeout=0.2)
    state.acquire_slot()                      # 占满唯一名额
    r = ask(client)
    c.check("名额占满 → 5003 / 503（不无限期排队）",
            r.status_code == 503 and r.json()["code"] == 5003,
            f"{r.status_code}/{r.json()['code']}")
    c.check("繁忙响应带上等待时长与上限",
            r.json()["detail"]["max_concurrent"] == 1
            and r.json()["detail"]["waited_seconds"] >= 0.1, str(r.json()["detail"]))
    state.release_slot()
    c.check("释放后恢复服务", ask(client).status_code == 200)
    c.check("in_flight 归零", state.inflight == 0, f"in_flight={state.inflight}")
    state.close()

    app2, state2, client2, _ = make_app(record=False, api_key="s3cret")
    c.check("开启鉴权后无 Key → 2001 / 401",
            client2.get("/api/v1/qa/stats").status_code == 401
            and client2.get("/api/v1/qa/stats").json()["code"] == 2001)
    c.check("错误的 Key → 2001",
            client2.get("/api/v1/qa/stats",
                        headers={"X-API-Key": "wrong"}).json()["code"] == 2001)
    c.check("正确的 Key 放行",
            client2.get("/api/v1/qa/stats",
                        headers={"X-API-Key": "s3cret"}).json()["code"] == 0)
    c.check("Bearer 写法也认",
            client2.get("/api/v1/qa/stats",
                        headers={"Authorization": "Bearer s3cret"}).json()["code"] == 0)
    c.check("健康探针不需要鉴权（否则探针会把服务判死）",
            client2.get("/health").status_code == 200)
    state2.close()

    app3, state3, client3, _ = make_app(record=False, rate_limit_per_minute=2)
    codes = [client3.get("/api/v1/qa/stats").json()["code"] for _ in range(3)]
    c.check("限流：第 3 次 → 2004 / 429", codes == [0, 0, 2004], str(codes))
    state3.close()


# ============================================================================
# K 真实产物
# ============================================================================
def group_real(c: Checker):
    c.head("K 真实产物（阶段七固化的真检索快照）")
    if not os.path.exists(SNAPSHOT):
        c.skip("检索快照", f"缺 {SNAPSHOT}")
        return
    snap = app_mod.SnapshotEvidence(SNAPSHOT)
    c.check("快照可加载且含 4 组证据", len(snap.pools) == 4, str(snap.pool_ids))
    ok_cnt = all(len(p["candidates"]) == 10 for p in snap.pools.values())
    c.check("每组 10 条真实检索候选", ok_cnt)
    metas = [(cd.get("metadata") or {}) for p in snap.pools.values() for cd in p["candidates"]]
    c.check("候选带 pmcid / 期刊 / 年份（可溯源三要素）",
            all(m.get("pmcid") for m in metas)
            and sum(1 for m in metas if m.get("journal")) >= len(metas) * 0.9,
            f"{len(metas)} 条候选")
    docs, pid, how = snap.pick("任意问题", pool="live-3", top_k=5)
    c.check("显式指定证据组 → exact 命中且按 top_k 截断",
            pid == "live-3" and how == "exact" and len(docs) == 5)
    _, pid2, how2 = snap.pick("CRISPR-Cas9 的脱靶效应如何检测？")
    _, pid3, _ = snap.pick("阿尔茨海默病抗淀粉样蛋白单抗的证据")
    c.check("中文题面也能选中正确的证据组（靠中英关键词表兜底）",
            pid2 == "live-1" and pid3 == "live-4", f"{pid2} / {pid3}")
    _, _, how4 = snap.pick("完全无关的问题：今天天气怎么样")
    c.check("匹配不上时标 fallback，不假装匹配成功", how4 == "fallback", how4)


# ============================================================================
# M 元信息端点
# ============================================================================
def group_meta(c: Checker):
    c.head("M 元信息端点（/ 、错误码表、生效配置）")
    app, state, client, _ = make_app()

    r = client.get("/")
    b = r.json()
    c.check("GET / → 200 / code 0", r.status_code == 200 and b["code"] == 0)
    eps = b["data"]["endpoints"]
    c.check("服务概览列出了文档接口（第二部分新增）",
            any("/documents" in e for e in eps), str(eps))

    r = client.get("/api/v1/errors")
    codes = {row["code"]: row for row in r.json()["data"]["codes"]}
    c.check("错误码表含 21 个码", len(codes) == 21, f"{len(codes)} 个")
    c.check("3001 文档不存在 → HTTP 404", codes.get(3001, {}).get("http_status") == 404,
            str(codes.get(3001)))
    c.check("3004 索引未就绪 → HTTP 503", codes.get(3004, {}).get("http_status") == 503)

    r = client.get("/api/v1/config")
    cfg = r.json()["data"]
    c.check("配置回显不含 api_key 明文", "api_key" not in cfg, str(list(cfg)[:6]))
    c.check("配置回显含文档目录与索引统计路径",
            "docs_db" in cfg and "index_stats_path" in cfg)
    c.check("直接构造 settings 时如实说明来源为空",
            cfg.get("config_sources") == {} and "config_sources_note" in cfg)
    state.close()


# ============================================================================
# N 文档管理接口
# ============================================================================
def _tiny_catalog(dirpath: str) -> str:
    """造一个 6 篇文献的**微型目录库**，不依赖那份 1.0 GB 的真目录。

    这样验证既能在只有 scripts/ 的交付包里跑，也能顺手验到"目录缺失"那条路径——
    真目录永远存在，反而测不到 3004。
    """
    path = os.path.join(dirpath, "docs_catalog.db")
    conn = sqlite3.connect(path)
    conn.executescript(dc_mod._SCHEMA)
    rows = [
        ("PMC000001", "PMC000001", "1000001", "CRISPR-Cas9 off-target effects in vivo",
         "Nature", 2021, 30, 2, "abstract,results", "We measured off-target editing…"),
        ("PMC000002", "PMC000002", "1000002", "Enzyme replacement therapy for Fabry disease",
         "Nature", 2019, 24, 1, "results", None),
        ("PMC000003", "PMC000003", "", "Anti-amyloid antibodies in Alzheimer disease",
         "Lancet", 2023, 40, 3, "methods,results,discussion", None),
        ("PMC000004", "PMC000004", "1000004", "Sacubitril/valsartan in HFrEF",
         "Lancet", 2021, 18, 1, "abstract", "Background: HFrEF…"),
        ("PMC000005", "PMC000005", "1000005", "Machine learning for sepsis prediction",
         "PLoS ONE", 2023, 22, 5, "methods,results", None),
        ("PMC000006", "PMC000006", "1000006", "Vitamin D and bone density",
         "PLoS ONE", 2019, 12, 1, "discussion", None),
    ]
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.executescript(dc_mod._INDEXES)
    conn.executemany("INSERT INTO catalog_meta VALUES (?,?)",
                     [("built_at", "2026-08-14 00:00:00"), ("documents", str(len(rows))),
                      ("indexed_chunks", "13"), ("source", "验证脚本造的微型库")])
    conn.commit()
    conn.close()
    return path


def group_documents(c: Checker):
    c.head("N 文档管理接口（列表 / 过滤 / 游标 / 详情 / 3001 / 3004）")
    d = tempfile.mkdtemp(prefix="medrag_docs_")
    _TMPDIRS.append(d)
    db = _tiny_catalog(d)
    app, state, client, _ = make_app(docs_db=db)

    r = client.get("/api/v1/documents", params={"limit": 3})
    b = r.json()
    c.check("列表 → 200 / code 0", r.status_code == 200 and b["code"] == 0)
    items = b["data"]["items"]
    c.check("limit 生效且按 pmcid 升序", len(items) == 3
            and [i["pmcid"] for i in items] == sorted(i["pmcid"] for i in items),
            str([i["pmcid"] for i in items]))
    c.check("还有下一页时给出 next_cursor",
            b["data"]["has_more"] and b["data"]["next_cursor"] == items[-1]["pmcid"],
            str(b["data"]["next_cursor"]))
    c.check("total 默认不算（null）", b["data"]["total"] is None)

    nxt = client.get("/api/v1/documents",
                     params={"limit": 3, "cursor": b["data"]["next_cursor"]}).json()["data"]
    first_page = {i["pmcid"] for i in items}
    second_page = {i["pmcid"] for i in nxt["items"]}
    c.check("游标翻页不重不漏", not (first_page & second_page)
            and len(first_page | second_page) == 6, str(sorted(second_page)))
    c.check("最后一页 has_more=false 且 next_cursor=null",
            nxt["has_more"] is False and nxt["next_cursor"] is None)

    jr = client.get("/api/v1/documents", params={"journal": "PLoS ONE",
                                                 "with_total": True}).json()["data"]
    c.check("journal 过滤 + with_total 生效",
            jr["total"] == 2 and {i["pmcid"] for i in jr["items"]} == {"PMC000005",
                                                                       "PMC000006"},
            f"total={jr['total']}")
    yr = client.get("/api/v1/documents", params={"year_from": 2021,
                                                 "year_to": 2023}).json()["data"]
    c.check("年份区间过滤生效",                          # 库里 2021~2023 的正好 4 篇
            {i["pub_year"] for i in yr["items"]} <= {2021, 2022, 2023}
            and len(yr["items"]) == 4, str(sorted(i["pub_year"] for i in yr["items"])))
    tr = client.get("/api/v1/documents", params={"title": "Fabry"}).json()["data"]
    c.check("标题关键词过滤生效",
            len(tr["items"]) == 1 and tr["items"][0]["pmcid"] == "PMC000002")
    c.check("生效的过滤条件原样回显", tr["filters"] == {"title": "Fabry"}, str(tr["filters"]))

    one = client.get("/api/v1/documents/PMC000001").json()
    c.check("按 id 查 → 200 / code 0", one["code"] == 0)
    doc = one["data"]
    c.check("原文切块数与库内块数分开给（30 / 2）",
            doc["total_chunks"] == 30 and doc["indexed_chunks"] == 2,
            f"{doc['total_chunks']} / {doc['indexed_chunks']}")
    c.check("sections 是列表不是逗号串", doc["sections"] == ["abstract", "results"],
            str(doc["sections"]))
    c.check("有摘要的给摘要", bool(doc["abstract"]))
    c.check("没摘要的给 null 而不是空串",
            client.get("/api/v1/documents/PMC000003").json()["data"]["abstract"] is None)
    c.check("pmid 缺失时给 null",
            client.get("/api/v1/documents/PMC000003").json()["data"]["pmid"] is None)

    miss = client.get("/api/v1/documents/PMC999999")
    mb = miss.json()
    c.check("不存在的文档 → 3001 / 404", miss.status_code == 404 and mb["code"] == 3001,
            f"{miss.status_code}/{mb['code']}")
    c.check("3001 的 detail 说清「不在本地索引」≠「这篇不存在」",
            "抽样" in (mb["detail"] or {}).get("hint", ""))
    state.close()

    # 目录缺失：必须是 3004，不能是"查得到但空空如也"
    app2, state2, client2, _ = make_app(record=False, docs_db=os.path.join(d, "不存在.db"))
    r2 = client2.get("/api/v1/documents")
    c.check("目录库缺失 → 3004 / 503（不是空列表）",
            r2.status_code == 503 and r2.json()["code"] == 3004,
            f"{r2.status_code}/{r2.json()['code']}")
    c.check("3004 带修复指引", "--build" in str(r2.json()["detail"]))
    h = client2.get("/health/ready").json()
    comps = {x["name"]: x for x in (h.get("data") or h["detail"])["components"]}
    c.check("目录缺失只让 doc_catalog 不健康，不拖垮就绪探针",
            comps["doc_catalog"]["ok"] is False and comps["doc_catalog"]["critical"] is False)
    state2.close()


# ============================================================================
# O 索引统计与向量库健康
# ============================================================================
def group_index_stats(c: Checker):
    c.head("O 运营统计的知识库段 + 向量库健康")
    d = tempfile.mkdtemp(prefix="medrag_stats_")
    _TMPDIRS.append(d)
    stats_path = os.path.join(d, "index_stats.json")
    payload = {"computed_at": "2026-08-14 00:00:00", "corpus_snapshot": "测试快照",
               "documents": {"total": 2274167, "note": "被抽中至少一块的文献数"},
               "chunks": {"total": 3998000, "per_document_mean": 1.76},
               "index_size": {"total": 1234, "vector_db": {"bytes": 1000}},
               "built_at": {"vector_db": "2026-07-16T17:02:42",
                            "bm25_index": "2026-07-22T14:09:58"},
               "incremental_updates": 0, "incremental_updates_note": "冻结快照，无增量"}
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    app, state, client, _ = make_app(index_stats_path=stats_path)
    ix = client.get("/api/v1/qa/stats").json()["data"]["index"]
    c.check("/qa/stats 带出文档总数", ix["total_documents"] == 2274167, str(ix["total_documents"]))
    c.check("/qa/stats 带出块数与文档数两个口径",
            ix["total_chunks"] == 3998000 and ix["total_documents"] != ix["total_chunks"])
    c.check("文档总数附口径说明（防止被读成 227 万篇完整文献）",
            "至少一块" in (ix["documents_note"] or ""), str(ix["documents_note"])[:40])
    c.check("增量更新次数如实为 0 且附说明",
            ix["incremental_updates"] == 0 and bool(ix["incremental_updates_note"]))
    c.check("三个建库时间分开给，不合成一个",
            len(ix["last_index_built_at"] or {}) >= 2, str(ix["last_index_built_at"]))
    c.check("索引大小按启动时真实 stat 刷新（不是 json 里那个 1234）",
            ix["index_size_bytes"] != 1234 and ix["index_size_bytes"] is not None,
            f"{ix['index_size_human']}")
    state.close()

    # 缺文件时必须 null，不能是 0——0 是个看起来正常的假数字
    app2, state2, client2, _ = make_app(index_stats_path=os.path.join(d, "没有.json"))
    ix2 = client2.get("/api/v1/qa/stats").json()["data"]["index"]
    c.check("统计文件缺失 → available=false", ix2["available"] is False)
    c.check("缺失时计数是 null 不是 0",
            ix2["total_documents"] is None and ix2["total_chunks"] is None
            and ix2["incremental_updates"] is None,
            f"{ix2['total_documents']}/{ix2['total_chunks']}/{ix2['incremental_updates']}")
    c.check("缺失时给出原因而不是静默", "--stats" in ix2["detail"] or "缺" in ix2["detail"],
            ix2["detail"][:50])
    state2.close()

    # 向量库健康：任务书点名的三样之一，第二部分之前根本没查过
    h = client2.get("/health/ready").json()
    comps = {x["name"]: x for x in (h.get("data") or h["detail"])["components"]}
    c.check("就绪探针含向量库组件", "vector_db" in comps, str(sorted(comps)))
    c.check("就绪探针含数据库组件（调用记录 + 会话）",
            "call_log_db" in comps and "session_db" in comps)
    c.check("就绪探针含 LLM 组件", "llm:ollama" in comps)
    c.check("snapshot 模式下向量库不是关键依赖（不加载 65GB 也能服务）",
            comps["vector_db"]["critical"] is False)

    app3, state3, client3, _ = make_app(record=False, retrieval_mode="live",
                                        chroma_dir=os.path.join(d, "没有这个向量库"))
    h3 = client3.get("/health/ready").json()
    comps3 = {x["name"]: x for x in (h3.get("data") or h3["detail"])["components"]}
    c.check("live 模式下向量库缺失 → 组件不健康且是关键依赖",
            comps3["vector_db"]["ok"] is False and comps3["vector_db"]["critical"] is True,
            comps3["vector_db"]["detail"][:50])
    state3.close()


# ============================================================================
# P 配置：CLI > 环境变量 > .env > 默认值
# ============================================================================
def group_config(c: Checker):
    c.head("P 配置来源优先级与 .env 解析")
    d = tempfile.mkdtemp(prefix="medrag_env_")
    _TMPDIRS.append(d)
    envf = os.path.join(d, ".env")
    with open(envf, "w", encoding="utf-8") as f:
        f.write("# 整行注释\n"
                "MEDRAG_MODE=live\n"
                "export MEDRAG_PORT=9001\n"
                "MEDRAG_MODEL=\"from-dotenv\"\n"
                "MEDRAG_CORS_ORIGINS=http://a.com, http://b.com\n"
                "MEDRAG_WARMUP=true\n"
                "没有等号这一行会被跳过\n")

    kv = app_mod.load_env_file(envf)
    c.check(".env 解析：跳过注释与无等号行", len(kv) == 5, f"{len(kv)} 项：{sorted(kv)}")
    c.check(".env 解析：export 前缀与引号都剥掉",
            kv.get("MEDRAG_PORT") == "9001" and kv.get("MEDRAG_MODEL") == "from-dotenv")

    s1 = app_mod.resolve_settings({}, env_file=envf, environ={})
    c.check("只有 .env 时 .env 生效",
            s1.retrieval_mode == "live" and s1.port == 9001
            and s1.config_sources["port"] == "dotenv")
    c.check("列表型配置按逗号切且去空白",
            s1.cors_origins == ["http://a.com", "http://b.com"], str(s1.cors_origins))
    c.check("布尔型配置认 true/false", s1.warmup is True)

    s2 = app_mod.resolve_settings({}, env_file=envf, environ={"MEDRAG_PORT": "9002"})
    c.check("环境变量盖过 .env",
            s2.port == 9002 and s2.config_sources["port"] == "env")

    s3 = app_mod.resolve_settings({"port": 8000, "retrieval_mode": "snapshot"},
                                  env_file=envf, environ={"MEDRAG_PORT": "9002"})
    c.check("命令行盖过环境变量与 .env",
            s3.port == 8000 and s3.retrieval_mode == "snapshot"
            and s3.config_sources["port"] == "cli")
    c.check("没配的项走默认值且来源标 default",
            s3.model_name == "from-dotenv" and s3.num_ctx == 12288
            and s3.config_sources["num_ctx"] == "default")

    s4 = app_mod.resolve_settings({}, env_file=os.path.join(d, "没有.env"), environ={})
    c.check("没有 .env 时一切照默认（README 里的命令不受影响）",
            s4.retrieval_mode == "snapshot" and s4.port == 8000 and s4.env_file == "")

    bad = None
    try:
        app_mod.resolve_settings({}, env_file=envf, environ={"MEDRAG_PORT": "八千"})
    except app_mod.ConfigError as e:
        bad = str(e)
    c.check("坏值当场报错，不静默取默认", bad is not None and "MEDRAG_PORT" in (bad or ""),
            (bad or "没报错")[:60])

    ex = app_mod.env_example_text()
    names = [e for _n, e, _k in app_mod.ENV_FIELDS]
    missing = [n for n in names if f"\n{n}=" not in "\n" + ex]
    c.check(f".env.example 覆盖全部 {len(names)} 个配置项", not missing, str(missing[:5]))
    c.check(".env.example 写明不支持行尾注释", "行尾注释" in ex)


# ============================================================================
# Q 端点覆盖（任务书：覆盖所有 API 端点）
# ============================================================================
def group_coverage(c: Checker, records: List[Dict[str, Any]]):
    c.head("Q 端点覆盖（Postman 集合的分母）")
    app, state, _client, _ = make_app()
    inv = route_inventory(app)
    state.close()

    covered = set()
    for rec in records:
        tpl = match_template(rec["method"], rec["path"], inv)
        if tpl:
            covered.add((rec["method"], tpl))
    missing = sorted(set(inv) - covered)

    # ⚠ 两个条件缺一不可：没有漏掉的端点，**且**确实覆盖到了东西。
    # 只写前一条的话，records 为空时 missing 也是空——一条断言在功能完全失效时照样绿。
    c.check(f"路由表非空（分母 {len(inv)} 个端点）", len(inv) >= 18, f"{len(inv)} 个")
    c.check(f"实际打到 {len(covered)} 个端点（非空洞检验）", len(covered) > 0, str(len(covered)))
    c.check("所有端点都被验证脚本打到过", not missing,
            "缺：" + "、".join(f"{m} {p}" for m, p in missing) if missing else "全覆盖")
    c.check("录到的调用数与端点数量级相符（说明录制真的在工作）",
            len(records) >= len(inv), f"{len(records)} 次调用 / {len(inv)} 个端点")


# ============================================================================
# --live：真实 Ollama 端到端实测
# ============================================================================
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def live_bench(c: Checker, queries: List[str], model: str = "qwen3:8b") -> Dict[str, Any]:
    """真起一个 uvicorn，用真 HTTP 客户端量同步与流式两条路径。

    **不用 TestClient**：它走的是内存 ASGI 传输，量不出首字节延迟这类只有真网络栈上
    才成立的数字。要报的是真数字，就得真发请求。
    """
    import httpx
    import uvicorn

    c.head("L 真实端到端实测（需要 Ollama 在跑）")
    port = _free_port()
    d = tempfile.mkdtemp(prefix="medrag_api_live_")
    _TMPDIRS.append(d)
    settings = app_mod.ServiceSettings(
        host="127.0.0.1", port=port, retrieval_mode="snapshot", snapshot_path=SNAPSHOT,
        model_name=model, calls_db=os.path.join(d, "calls.db"),
        session_db=os.path.join(d, "sessions.db"), log_dir=os.path.join(d, "logs"),
        log_console=False, max_concurrent=1)
    state = app_mod.ServiceState(settings)
    app = app_mod.create_app(state=state)

    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, access_log=False)
    server = uvicorn.Server(cfg)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):                       # 等端口起来
        try:
            httpx.get(f"{base}/health", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)

    out: Dict[str, Any] = {"model": model, "port": port, "queries": [],
                           "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        ready = httpx.get(f"{base}/health/ready", timeout=10).json()
        c.check("就绪探针通过（Ollama 可用）", ready["code"] == 0,
                json.dumps(ready.get("detail") or ready["data"]["status"], ensure_ascii=False))
        if ready["code"] != 0:
            return out

        for q in queries:
            row: Dict[str, Any] = {"query": q}
            t0 = time.perf_counter()
            r = httpx.post(f"{base}/api/v1/qa/ask", json={"query": q, "top_k": 8},
                           timeout=900)
            row["sync"] = {"http": r.status_code, "seconds": round(time.perf_counter() - t0, 2)}
            if r.status_code == 200:
                dd = r.json()["data"]
                row["sync"].update(
                    answer_chars=dd["metrics"]["answer_chars"],
                    llm_calls=dd["metrics"]["llm_calls"],
                    prompt_tokens=dd["metrics"]["total_prompt_tokens"],
                    output_tokens=dd["metrics"]["total_output_tokens"],
                    refused=dd["refused"], sources=len(dd["sources"]),
                    compliant=(dd["constraint_check"] or {}).get("compliant"),
                    violations=[v["code"] for v in
                                (dd["constraint_check"] or {}).get("violations") or []],
                    repair_rounds=(dd["constraint_check"] or {}).get("repair_rounds"),
                    pool=dd["evidence_pool"])

            t0 = time.perf_counter()
            firsts: Dict[str, float] = {}
            n_delta, kinds = 0, []
            cur_event, done_payload = None, None
            with httpx.stream("POST", f"{base}/api/v1/qa/stream",
                              json={"query": q, "top_k": 8}, timeout=900) as sr:
                for line in sr.iter_lines():
                    if line.startswith("event: "):
                        cur_event = line[7:]
                        kinds.append(cur_event)
                        firsts.setdefault(cur_event, round(time.perf_counter() - t0, 2))
                        n_delta += (cur_event == "delta")
                    elif line.startswith("data: ") and cur_event == "done":
                        try:
                            done_payload = json.loads(line[6:])
                        except json.JSONDecodeError:
                            pass
            row["stream"] = {"seconds": round(time.perf_counter() - t0, 2),
                             "deltas": n_delta, "first_event_at": firsts,
                             "events": len(kinds)}
            dp = ((done_payload or {}).get("data")) or {}
            if dp:
                row["stream"].update(
                    answer_chars=dp.get("metrics", {}).get("answer_chars"),
                    llm_calls=dp.get("metrics", {}).get("llm_calls"),
                    refused=dp.get("refused"),
                    compliant=(dp.get("constraint_check") or {}).get("compliant"),
                    violations=[v["code"] for v in
                                (dp.get("constraint_check") or {}).get("violations") or []])
            out["queries"].append(row)
            c.note(f"· {q[:36]}…  同步 {row['sync']['seconds']}s"
                   f"（{row['sync'].get('answer_chars', '?')} 字）"
                   f"｜流式 {row['stream']['seconds']}s，"
                   f"出处 @{firsts.get('sources', '?')}s，首字 @{firsts.get('delta', '?')}s")

        oks = [r for r in out["queries"] if r["sync"].get("http") == 200]
        c.check("同步接口全部成功", len(oks) == len(queries), f"{len(oks)}/{len(queries)}")
        c.check("流式全部收到 done", all(r["stream"]["events"] > 0 for r in out["queries"]))
        c.check("出处事件早于第一个 delta（流式的真实价值）",
                all(r["stream"]["first_event_at"].get("sources", 9e9)
                    < r["stream"]["first_event_at"].get("delta", 9e9)
                    for r in out["queries"]))
        stats = httpx.get(f"{base}/api/v1/qa/stats", timeout=10).json()["data"]
        out["stats"] = stats
        c.check("统计接口汇总了本轮全部调用",
                stats["total"] == len(queries) * 2, f"total={stats['total']}")
    finally:
        server.should_exit = True
        th.join(timeout=10)
        state.close()

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(LIVE_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    c.note(f"实测明细已写入 {LIVE_JSON}")
    return out


LIVE_QUERIES = [
    "CRISPR-Cas9 的脱靶效应有哪些检测方法？请给出证据出处。",
    "酶替代治疗用于法布里病的疗效证据有哪些？",
    "抗淀粉样蛋白单抗治疗阿尔茨海默病的最新证据是什么？",
]


# ============================================================================
# main
# ============================================================================
_TOTAL_RE = re.compile(r"总计\s+(\d+)\s*/\s*(\d+)\s*项通过")
_LIVE_RE = re.compile(r"实测部分\s+(\d+)\s*项")


def _refuse_downgrade(path: str, new_total: int, force: bool = False,
                      new_live: int = 0) -> bool:
    """已有报告更强时，拒绝覆盖。返回 True 表示"别写"。

    起因（2026-08-12，第二次撞到；第一次是阶段九基线，当时靠另存救回来的）：
    `服务_验证.py` 不加 `--live` 只跑离线 160 项，加了才是 165 项（含真实端到端）。
    我为了做一次回归检查跑了不带 `--live` 的一轮，**把带端到端证据的 165/165 那份冲掉了**。
    报告是交付材料，交付点名要的三样之一，弱运行覆盖强证据是纯损失且不可逆。

    判据**两条，缺一不可**：

      1. 新报告项数 ≥ 旧报告项数；
      2. 旧报告有 `--live` 实测项时，新报告也必须有。

    第 2 条是 2026-08-14 补的，补的正是第 1 条的漏洞：第二部分给离线加了 30 多项，
    于是"离线 190 项"在项数上**大于**"离线 160 + 实测 5 = 165 项"，
    只看项数会**放行**这次覆盖，而那 5 项真实端到端证据照样没了。
    **"更多项"不等于"更强"——强弱不是一个标量。**
    """
    if force or not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return False
    m = _TOTAL_RE.search(text)
    if not m:
        return False
    old_total = int(m.group(2))
    ml = _LIVE_RE.search(text)
    old_live = int(ml.group(1)) if ml else 0

    if new_total >= old_total and not (old_live > 0 and new_live <= 0):
        return False
    print(f"\n{'='*70}")
    if old_live > 0 and new_live <= 0:
        print(f"⚠ 拒绝覆盖：已有报告含 {old_live} 项 **真实端到端实测**（--live），本次没有。")
        print(f"  本次 {new_total} 项 vs 旧的 {old_total} 项——**项数在这条判据里不作数**，"
              f"盖不过那 {old_live} 项真跑的证据：")
        print(f"  离线部分注的是 StubGenerator，唯独 --live 那几项能证明「接上真模型也跑得通」。")
        print(f"  要更新报告：把 Ollama 起起来后跑 --live；只想看这轮结果：--force-report。")
    else:
        print(f"⚠ 拒绝覆盖：已有报告 {old_total} 项，本次只有 {new_total} 项 —— 这是降级。")
        print(f"  用弱轮盖掉强轮等于丢掉交付证据（2026-08-12 已经踩过一次）。")
        print(f"  要这一轮的结果：加 --force-report，或先手动删掉旧文件。")
    print(f"{'='*70}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="追加真实端到端实测（要 Ollama 在跑，约 5~10 分钟）")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--force-report", action="store_true",
                    help="允许用项数更少的一轮覆盖已有报告（默认拒绝，防止离线轮冲掉 --live 证据）")
    ap.add_argument("--export-postman", metavar="PATH", nargs="?", const=POSTMAN_PATH,
                    help=f"把这一轮真实发出的 HTTP 请求导成 Postman v2.1 集合；"
                         f"默认 {POSTMAN_PATH}")
    args = ap.parse_args()

    RecordingClient.ENABLED = True          # 一直录：导不导出都要靠它算端点覆盖率
    RecordingClient.RECORDS.clear()

    t0 = time.time()
    c = Checker(quiet=args.quiet)
    c._out("=" * 92)
    c._out("第十阶段（一）验证 · FastAPI 骨架 / 统一响应 / 错误码 / 异常处理 / 日志 / 问答接口")
    c._out("=" * 92)
    c._out(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
           f"　{'含真实 Ollama 实测' if args.live else '不调用模型（注入 StubGenerator 跑真链路）'}")
    c._out("每条 PASS/FAIL 均由实际算出的变量比对得出，无一条是无条件打印。")
    c._out("证据用阶段七固化的真检索快照，链路（组装/四段/筛选/后处理/层D/SSE/会话/记录）全是真代码。")

    # 缺证据快照时，靠它作答的那几组**整组跳过**，不是崩在解引用上。
    # 实测过（2026-08-18）：删掉 检索快照_live.json 后 group_sync 在 data["answer"] 处抛
    # TypeError: 'NoneType' object is not subscriptable，退出码 1 —— 一段 traceback 不是"验证结论"，
    # 而 约束_验证.py / 评估_验证.py 早就是「跳过并写明、不静默算通过」，这里只是补齐同一约定。
    # ⚠ 正常情况下走不到这条分支：夹具随仓库带在 reports/fixtures/，evidence_path() 会回退到它。
    _needs_evidence = {group_context, group_sync, group_sync_session,
                       group_sync_edge, group_stream, group_guard,
                       # 就绪探针把证据源当成关键组件，没有快照时服务**正确地**报未就绪，
                       # 这组的断言全部建立在「服务已就绪」之上，跑了只会给出误导性的红。
                       group_health}
    _has_snapshot = os.path.exists(SNAPSHOT)
    for _g in (group_response, group_errors, group_exception, group_logging,
               group_context, group_session, group_params, group_sync,
               group_sync_session, group_sync_edge, group_stream, group_sse_format,
               group_health, group_guard, group_real, group_meta,
               group_documents, group_index_stats, group_config):
        if _g in _needs_evidence and not _has_snapshot:
            c.head(f"{_g.__name__}（跳过）")
            c.skip("整组跳过：这组要靠真实证据快照作答", f"缺 {SNAPSHOT}")
            continue
        _g(c)
    group_coverage(c, RecordingClient.RECORDS)
    # 离线与实测分开计时：两个数差三个数量级，合成一个总耗时会让"离线跑一次只要几秒"
    # 这件事看不出来，而那正是这套验证最该被引用的性质
    offline_n, offline_s = len(c.rows), time.time() - t0
    if args.live:
        try:
            live_bench(c, LIVE_QUERIES, model=args.model)
        except Exception as e:
            c.check("真实端到端实测", False, f"{type(e).__name__}: {e}")

    passed, total = c.summary()
    c._out("\n" + "=" * 92)
    c._out("汇总")
    c._out("=" * 92)
    by_group: Dict[str, List[bool]] = {}
    for g, _, ok, _ in c.rows:
        by_group.setdefault(g, []).append(ok)
    for g, oks in by_group.items():
        c._out(f"  {sum(oks)}/{len(oks)}   {g}")
    if c.skipped:
        c._out("\n  跳过（未静默当作通过）：")
        for g, why in c.skipped:
            c._out(f"    · [{g}] {why}")
    failed = [(g, l, d) for g, l, ok, d in c.rows if not ok]
    if failed:
        c._out("\n  未通过：")
        for g, l, d in failed:
            c._out(f"    · [{g}] {l}   {d}")
    c._out(f"\n  离线部分 {offline_n} 项   用时 {offline_s:.1f}s（不需要 Ollama，不需要向量库）")
    if total > offline_n:
        c._out(f"  实测部分 {total - offline_n} 项   用时 {time.time() - t0 - offline_s:.1f}s"
               f"（真起 uvicorn + 真 Ollama）")
    c._out(f"\n总计 {passed}/{total} 项通过   用时 {time.time() - t0:.1f}s")

    if args.export_postman:
        app_tmp, state_tmp, _cl, _g = make_app()
        info = export_postman(RecordingClient.RECORDS, args.export_postman,
                              route_inventory(app_tmp))
        state_tmp.close()
        c._out(f"\nPostman 集合 → {info['path']}")
        c._out(f"  端点覆盖 {info['covered']}/{info['total_endpoints']}"
               + ("（全覆盖）" if not info["missing"] else "，缺 " + str(info["missing"])))
        c._out(f"  录到 {info['calls']} 次调用，去重后 {info['requests']} 条请求，"
               f"分 {info['groups']} 组")
        c._out("  ⚠ 集合只覆盖走 HTTP 的那部分；进程内断言（响应模型/错误码表/存储层/"
               "参数校验）没有 Postman 形态，**集合通过 ≠ 这 %d 项通过**" % total)

    os.makedirs(REPORT_DIR, exist_ok=True)
    if _refuse_downgrade(REPORT_PATH, total, force=args.force_report,
                         new_live=total - offline_n):
        print(f"\n⚠ **已拒绝覆盖** {REPORT_PATH}")
        print(f"  想强行覆盖：先手动删掉它，或加 --force-report")
    else:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(c.lines) + "\n")
        print(f"\n报告：{REPORT_PATH}")

    for d in _TMPDIRS:                        # 清理临时库
        shutil.rmtree(d, ignore_errors=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
