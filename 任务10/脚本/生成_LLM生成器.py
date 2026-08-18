# -*- coding: utf-8 -*-
"""第七阶段（二）· 本地 LLM 集成与生成

把本地 Ollama 上的 `qwen3:8b` 封装成一个可靠的生成后端，供四段医学提示词链调用。
"可靠"在这里有三层具体含义，每一层都对应一个踩过的坑：

  1. **上下文窗口必须显式设**。Ollama 的 `num_ctx` 默认只有 4096（`/api/ps` 实测确认），
     四段链装不下；更糟的是**超长不报错**——它静默丢掉最前面的内容，而那正是 system
     提示词和排序最靠前的证据。表现为"答得莫名其妙变差"，不查根本发现不了。
     本模块一律显式传 `num_ctx`（默认 12288，实测显存 6.33GB，10G 卡有余量）。
  2. **结构化阶段的 JSON 必须容错解析**。①证据评估器和③批判性审查器要 JSON，但模型会
     把它包进 ```json 代码块、前面加一句"好的，以下是评估结果："、偶尔漏掉最后一个 `]`。
     直接 `json.loads` 必失败。`extract_json()` 做的就是剥壳 → 定位 → 补符号 → 再解析。
  3. **`num_predict=0` 会让请求挂住不返回**（阶段七之一实测）。这里直接在入口拦掉，
     并给出可操作的报错，而不是让调用方等到超时。

用法：
    from 生成_LLM生成器 import LLMGenerator          # 中文名需按路径导入，见文件末尾示例
    gen = LLMGenerator(model_name="qwen3:8b", num_ctx=12288)
    r = gen.generate("一句话解释什么是随机对照试验", system_prompt="你是医学助理")
    r["text"] / r["elapsed"] / r["prompt_eval_count"] / r["eval_count"]

    # 结构化阶段：拿解析好的对象
    r = gen.generate(prompt, system_prompt=sys, json_output=True, expect="array")
    r["json"]        # 解析成功为 list/dict；失败为 None，原因在 r["json_error"]

CLI 自检（需要 Ollama 已启动）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_LLM生成器.py --smoke
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_LLM生成器.py --json-selftest
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

#: 与 `生成_提示词模板.py::RECOMMENDED_NUM_CTX` 保持一致。这里不 import 那个模块，
#: 是为了让本模块可以独立使用（它不依赖提示词，只依赖 Ollama）。
DEFAULT_NUM_CTX = 12288
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_BASE_URL = "http://localhost:11434"


class LLMError(RuntimeError):
    """Ollama 调用失败（连接不上 / 模型不存在 / HTTP 非 200 / 超时）。"""


# ============================================================================
# 一、JSON 抽取与修补
# ============================================================================
# qwen3 即使加了 /no_think，偶尔仍会吐 <think>…</think>；先整段剥掉
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# 未闭合的 <think>（被 num_predict 截断时会出现）：从它开始到结尾全丢
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
# ```json … ``` / ``` … ```
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_FENCE_OPEN = re.compile(r"```(?:json|JSON)?\s*(.*)\Z", re.DOTALL)
# 尾随逗号：{"a":1,}  [1,2,]
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# 中文全角引号 / 智能引号 → 直引号（模型偶尔会用它们包 key）
_SMART_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                 "\uff02": '"', "\u300c": '"', "\u300d": '"'}
# \u6570\u503c\u540e\u591a\u4e00\u4e2a\u5f15\u53f7\uff1a`{"relevance":3"}` \u800c\u4e0d\u662f `{"relevance":3}`
# \u2014\u2014 qwen3:8b \u5728\u8bc1\u636e\u8bc4\u4f30\u9636\u6bb5\u5b9e\u6d4b\u9ad8\u9891\uff084 \u9053\u9a8c\u6536\u9898\u91cc 2 \u9053\u4e2d\u62db\uff09\uff0c\u662f\u672c\u9879\u76ee\u89c1\u8fc7\u6700\u5e38\u89c1\u7684\u7578\u5f62\u3002
# \u5371\u5bb3\u4e0d\u6b62"\u8fd9\u4e00\u5904\u89e3\u6790\u5931\u8d25"\uff1a\u591a\u51fa\u6765\u7684\u5f15\u53f7\u4f1a\u628a\u5b83\u540e\u9762\u7684\u6587\u672c\u6574\u6bb5\u5f53\u6210\u5b57\u7b26\u4e32\uff0c\u5f15\u53f7\u5947\u5076\u5168\u4e71\uff0c
# \u5e73\u8861\u626b\u63cf\u7b97\u51fa\u7684\u5d4c\u5957\u6df1\u5ea6\u4e5f\u8ddf\u7740\u9519\uff08\u5b9e\u6d4b missing_depth \u88ab\u7b97\u6210 2\uff0c\u5176\u5b9e\u662f 0\uff09\u3002\u6240\u4ee5\u5fc5\u987b\u5728
# **\u626b\u63cf\u4e4b\u524d**\u4fee\u6389\uff0c\u800c\u4e0d\u662f\u7559\u5230\u6700\u540e\u4e00\u7ea7\u8865\u62ec\u53f7\u65f6\u518d\u5904\u7406\u3002
# \u8981\u6c42\u5192\u53f7\u540e\u7d27\u8ddf\u6570\u5b57\uff08\u6ca1\u6709\u524d\u5f15\u53f7\uff09\uff0c\u56e0\u6b64\u4e0d\u4f1a\u8bef\u4f24\u5408\u6cd5\u7684 `"3"` \u8fd9\u7c7b\u5b57\u7b26\u4e32\u503c\u3002
_STRAY_QUOTE_NUM = re.compile(r':(\s*-?\d+(?:\.\d+)?)"(?=\s*[,}\]])')


def normalize_quirks(s: str) -> str:
    """\u628a\u5df2\u77e5\u7684\u6a21\u578b\u4e66\u5199\u7656\u4fee\u6b63\u6389\u3002\u53ea\u505a**\u4e0d\u6539\u53d8\u8bed\u4e49**\u7684\u66ff\u6362\u3002"""
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    return _STRAY_QUOTE_NUM.sub(r":\1", s)


def strip_wrappers(text: str) -> str:
    """剥掉思考段与代码块围栏，留下最可能是 JSON 的裸文本。"""
    s = text or ""
    s = _THINK.sub("", s)
    s = _THINK_OPEN.sub("", s)
    m = _FENCE.search(s)
    if m:
        return m.group(1).strip()
    m = _FENCE_OPEN.search(s)          # 围栏只开不闭（被 num_predict 截断）
    if m:
        return m.group(1).strip()
    return s.strip()


def _scan_balanced(s: str, start: int) -> Tuple[str, int, bool]:
    """从 s[start]（'{' 或 '['）起扫到配对的收尾符号，正确跳过字符串与转义。

    Returns: (子串, 仍未闭合的层数, 结尾是否落在未闭合的字符串里)。
    后两项都 > 0 / True 时说明文本被 num_predict 截断了。
    """
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1], 0, False
    return s[start:], depth, in_str


def _drop_partial_tail(s: str) -> str:
    """截断落在字符串中间时，把这个**残缺的值连同它的键**一起丢掉。

    不这么做的话，`"verdict": "rev` 补个引号就成了看似合法的 `"verdict": "rev"`——
    一个不在枚举里的值伪装成正常值，比缺字段更难发现。宁可少一个键。
    """
    i = s.rfind('"')                     # 残缺字符串的起始引号
    if i < 0:
        return s
    s = s[:i].rstrip()
    if s.endswith(":"):                  # 连键一起丢
        s = s[:-1].rstrip()
        if s.endswith('"'):
            j = s.rfind('"', 0, len(s) - 1)
            if j >= 0:
                s = s[:j].rstrip()
    return s.rstrip().rstrip(",")


def _repair(fragment: str, missing_depth: int, str_open: bool) -> str:
    """补齐被截断的 JSON：丢残缺尾巴 → 去尾随逗号 → 补缺失的收尾括号。"""
    s = fragment.rstrip()
    if str_open:
        s = _drop_partial_tail(s)
    # 被截在 "key": 之后（值还没开始写）
    s = re.sub(r",?\s*\"[^\"]*\"\s*:\s*$", "", s).rstrip().rstrip(",")
    if missing_depth > 0 or str_open:
        # 按字符串外的括号净差补齐；JSON 只有两种括号，净差足以确定补什么、补几个
        need: List[str] = []
        in_str, esc = False, False
        for ch in s:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                need.append("}")
            elif ch == "[":
                need.append("]")
            elif ch in "}]" and need:
                need.pop()
        s += "".join(reversed(need))
    return _TRAILING_COMMA.sub(r"\1", s)


def extract_json(text: str, expect: str = "any") -> Tuple[Optional[Any], str]:
    """从模型输出里抽出 JSON 并解析。

    expect: "object" | "array" | "any" —— 决定优先找 '{' 还是 '['。
    Returns: (对象, 说明)。失败时对象为 None，说明里写清楚卡在哪一步。

    四道防线，逐级放宽（每一级都是实测见过的失败形态）：
      1. 直接 json.loads               —— 模型守规矩时最快
      2. 剥 <think> 与 ``` 围栏后再 loads
      3. 定位第一个 '{'/'[' 扫平衡子串  —— 处理"好的，以下是结果：{...}"这类前后缀
      4. 补尾随逗号与缺失括号后再 loads —— 处理被 num_predict 截断的输出
    """
    if text is None or not str(text).strip():
        return None, "空输出"

    def _try(s: str) -> Optional[Any]:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None

    raw = str(text)
    obj = _try(raw.strip())
    if obj is not None:
        return obj, "直接解析"

    body = normalize_quirks(strip_wrappers(raw))
    obj = _try(body)
    if obj is not None:
        return obj, "剥离包裹后解析"

    # 找起始符号：按 expect 决定优先级，找不到首选就用另一种
    order = {"object": "{[", "array": "[{"}.get(expect, "{[")
    starts = [(body.find(c), c) for c in order if body.find(c) >= 0]
    if not starts:
        return None, f"输出中找不到 JSON 起始符号（前 80 字符：{body[:80]!r}）"
    # 首选符号若存在就用它；都存在时取更靠前的那个同类
    starts.sort(key=lambda t: (order.index(t[1]), t[0]))
    pos = starts[0][0]

    fragment, missing, str_open = _scan_balanced(body, pos)
    obj = _try(fragment)
    if obj is not None:
        return obj, "定位平衡子串后解析"

    repaired = _repair(fragment, missing, str_open)
    obj = _try(repaired)
    if obj is not None:
        note = ("补齐截断的 JSON 后解析" if (missing > 0 or str_open)
                else "清理尾随逗号后解析")
        return obj, note

    return None, (f"四级解析全部失败（截断层数={missing}，串内截断={str_open}，"
                  f"片段前 120 字符：{fragment[:120]!r}）")


#: 追加到 user 提示词末尾的 JSON 硬性要求。提示词模板里已经写了"只输出 JSON"，
#: 但实测 qwen3:8b 仍有约 1/5 的概率加代码块围栏，这里再钉一次，成本只有十几个 token。
JSON_FORMAT_INSTRUCTION = (
    "\n\n严格输出要求：只输出 JSON 本身，不要写任何解释、前言或结语，"
    "不要用 ``` 代码块包裹，不要输出 <think> 段。第一个字符必须是 {} 或 [] 的起始符号。"
)


# ============================================================================
# 二、生成器
# ============================================================================
class LLMGenerator:
    """本地 Ollama 生成器。

    Args:
        model_name: Ollama 模型名称（如 "qwen3:8b"）
        base_url:   Ollama 服务地址
        timeout:    单次请求超时时间（秒）。四段链里最长的一段（④，max_tokens=2000）
                    本机实测数十秒，默认 180 留足余量。
        temperature / max_tokens: 默认采样参数；每次 generate 可覆盖
        num_ctx:    上下文窗口。**必须显式设**，见模块 docstring 第 1 条
        check_connection: 初始化时测连通性与模型是否存在（默认开）
    """

    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 timeout: int = 180,
                 temperature: float = 0.3,
                 max_tokens: int = 1500,
                 num_ctx: int = DEFAULT_NUM_CTX,
                 top_p: float = 0.9,
                 keep_alive: str = "10m",
                 think: bool = False,
                 check_connection: bool = True,
                 verbose: bool = False):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.num_ctx = int(num_ctx)
        self.top_p = float(top_p)
        self.keep_alive = keep_alive
        # qwen3 是思考模型：Ollama ≥0.9 把推理过程单独放进 message.thinking，content 要等
        # 思考结束才开始写。**只在提示词里写 /no_think 关不掉**——实测思考段会吃光整个
        # num_predict，content 全空、done_reason=length。必须用 API 层的 think 开关。
        self.think = bool(think)
        self.verbose = verbose

        #: 累计用量，供流水线写 generation_metrics
        self.stats = {"calls": 0, "failures": 0, "total_seconds": 0.0,
                      "prompt_tokens": 0, "output_tokens": 0}
        self.connection: Dict[str, Any] = {"checked": False, "ok": False}
        if check_connection:
            self.check_connection()

    # ---------------- 连接与模型检查 ----------------
    def _post(self, path: str, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"Ollama HTTP {e.code}：{detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"连不上 Ollama（{self.base_url}）：{e.reason}。"
                           f"请先启动服务：$env:OLLAMA_MODELS='E:\\rag\\ollama\\models'; ollama serve") from e
        except TimeoutError as e:
            raise LLMError(f"Ollama 请求超时（>{timeout or self.timeout}s）") from e

    def _get(self, path: str, timeout: int = 10) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMError(f"连不上 Ollama（{self.base_url}）：{e.reason}。"
                           f"请先启动服务：$env:OLLAMA_MODELS='E:\\rag\\ollama\\models'; ollama serve") from e

    def check_connection(self) -> Dict[str, Any]:
        """测连通性 + 确认模型已安装。失败抛 LLMError，信息里带可操作的下一步。"""
        t0 = time.time()
        tags = self._get("/api/tags")
        names = [m.get("name", "") for m in tags.get("models", [])]
        # Ollama 的 tag 省略 :latest，两种写法都算命中
        hit = self.model_name in names or f"{self.model_name}:latest" in names \
            or any(n.split(":")[0] == self.model_name for n in names)
        self.connection = {"checked": True, "ok": bool(hit), "base_url": self.base_url,
                           "model": self.model_name, "installed_models": names,
                           "latency_ms": round((time.time() - t0) * 1000, 1)}
        if not hit:
            raise LLMError(f"Ollama 上没有模型 {self.model_name!r}，已安装：{names}。"
                           f"请先 `ollama pull {self.model_name}`")
        if self.verbose:
            print(f"[LLM] 连接正常 {self.base_url} | 模型 {self.model_name} | "
                  f"{self.connection['latency_ms']}ms")
        return self.connection

    # ---------------- 单次生成 ----------------
    def build_options(self, temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      **extra: Any) -> Dict[str, Any]:
        """组装 Ollama options。num_ctx 一定在里面，这是本模块存在的主要理由之一。"""
        nt = self.max_tokens if max_tokens is None else int(max_tokens)
        if nt <= 0:
            # 实测：num_predict=0 会让 /api/generate 挂住不返回（要读 prompt_eval_count 请传 1）
            raise ValueError("max_tokens 必须 > 0：num_predict=0 会让 Ollama 挂住不返回")
        opts = {"temperature": self.temperature if temperature is None else float(temperature),
                "num_predict": nt,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx}
        opts.update(extra)
        return opts

    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 json_output: bool = False,
                 expect: str = "any",
                 options: Optional[Dict[str, Any]] = None,
                 retries: int = 1,
                 json_retries: int = 1) -> Dict[str, Any]:
        """生成一次。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（None 则不发 system 消息）
            json_output: 结构化模式——追加 JSON 硬性要求，并对输出做容错解析
            expect: json_output 时期望的顶层类型 "object"/"array"/"any"
            retries: 传输层失败（超时/连接断）的重试次数
            json_retries: JSON 解析失败后的重试次数（重试时温度压到 0，去掉随机性）

        Returns: dict —— text/json/json_ok/json_error/elapsed/prompt_eval_count/
                 eval_count/truncated/attempts/options
        """
        user = prompt + (JSON_FORMAT_INSTRUCTION if json_output else "")
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
            + [{"role": "user", "content": user}]
        return self.generate_messages(messages, temperature=temperature, max_tokens=max_tokens,
                                      json_output=json_output, expect=expect, options=options,
                                      retries=retries, json_retries=json_retries)

    def generate_messages(self,
                          messages: Sequence[Dict[str, str]],
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          json_output: bool = False,
                          expect: str = "any",
                          options: Optional[Dict[str, Any]] = None,
                          retries: int = 1,
                          json_retries: int = 1) -> Dict[str, Any]:
        """直接喂 chat messages —— 提示词模板的 `format_messages()` 可原样传进来。

        JSON 模式下如果模板本身没写 JSON 要求，这里会追加到最后一条 user 消息末尾。
        """
        msgs = [dict(m) for m in messages]
        if json_output and msgs and msgs[-1].get("role") == "user":
            if JSON_FORMAT_INSTRUCTION.strip()[:12] not in msgs[-1]["content"]:
                msgs[-1]["content"] += JSON_FORMAT_INSTRUCTION

        opts = self.build_options(temperature, max_tokens, **(options or {}))
        attempts: List[Dict[str, Any]] = []
        result: Dict[str, Any] = {}

        cur_msgs = msgs
        for json_try in range(json_retries + 1):
            cur = dict(opts)
            if json_try > 0:
                # ①③ 本来就是温度 0，原样重发只会拿到**一模一样**的坏输出（实测确认）。
                # 所以重试时必须改变输入：把上一次的具体错误回灌给模型，让它有机会改。
                cur["temperature"] = 0.0
                cur_msgs = [dict(m) for m in msgs]
                if cur_msgs and cur_msgs[-1].get("role") == "user":
                    cur_msgs[-1]["content"] += (
                        f"\n\n上一次的输出无法被 JSON 解析器接受，原因：{result.get('json_error', '')}。"
                        f"请重新输出**严格合法**的 JSON：数字值不要加引号，字符串值必须成对引号，"
                        f"不要有多余的引号或逗号。")
            result = self._call_with_retry(cur_msgs, cur, retries, attempts)
            if not json_output:
                break
            obj, note = extract_json(result["text"], expect=expect)
            result["json"], result["json_ok"] = obj, obj is not None
            result["json_error"] = "" if obj is not None else note
            result["json_note"] = note
            if obj is not None:
                break
            if self.verbose:
                print(f"[LLM] JSON 解析失败（{note}），重试 {json_try + 1}/{json_retries}")

        result["attempts"] = attempts
        result["options"] = opts
        return result

    def _call_with_retry(self, msgs: List[Dict[str, str]], opts: Dict[str, Any],
                         retries: int, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        last: Optional[Exception] = None
        for i in range(retries + 1):
            t0 = time.time()
            try:
                data = self._post("/api/chat", {"model": self.model_name, "messages": msgs,
                                                "stream": False, "options": opts,
                                                "think": self.think,
                                                "keep_alive": self.keep_alive})
            except LLMError as e:
                last = e
                self.stats["failures"] += 1
                attempts.append({"ok": False, "error": str(e), "seconds": round(time.time() - t0, 2)})
                if i < retries:
                    time.sleep(1.5 * (i + 1))
                    continue
                raise
            el = time.time() - t0
            msg = data.get("message") or {}
            text = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
            # done_reason == "length" 说明是被 num_predict 截停的，不是自然收尾
            out = {
                "text": text.strip(),
                "raw": text,
                "thinking_chars": len(thinking),
                "elapsed": round(el, 3),
                "model": data.get("model", self.model_name),
                "done_reason": data.get("done_reason", ""),
                "truncated": data.get("done_reason") == "length",
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "tokens_per_second": (round(data["eval_count"] / (data["eval_duration"] / 1e9), 1)
                                      if data.get("eval_count") and data.get("eval_duration") else None),
                "json": None, "json_ok": None, "json_error": "", "json_note": "",
            }
            self.stats["calls"] += 1
            self.stats["total_seconds"] += el
            self.stats["prompt_tokens"] += out["prompt_eval_count"] or 0
            self.stats["output_tokens"] += out["eval_count"] or 0
            attempts.append({"ok": True, "seconds": out["elapsed"],
                             "eval_count": out["eval_count"], "truncated": out["truncated"]})
            if self.verbose:
                print(f"[LLM] {out['elapsed']}s | in={out['prompt_eval_count']} "
                      f"out={out['eval_count']} tok"
                      + ("（被 num_predict 截断）" if out["truncated"] else ""))
            if not out["text"] and thinking:
                # 这个组合几乎只有一个成因，直接把结论写出来，省得下次再查一遍
                print(f"[LLM] ⚠ content 为空但产生了 {len(thinking)} 字符的思考段——"
                      f"think 未关闭（当前 think={self.think}），num_predict 被思考吃光")
            return out
        raise last or LLMError("未知失败")

    # ---------------- 批量生成 ----------------
    def batch_generate(self,
                       prompts: Sequence[Any],
                       system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       json_output: bool = False,
                       expect: str = "any",
                       stop_on_error: bool = False,
                       progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
                       ) -> List[Dict[str, Any]]:
        """批量生成，**串行执行**。

        不做并发：本机单卡 10G，qwen3:8b 已占约 6.3GB，Ollama 并行请求会争同一份权重，
        吞吐不升反降（还可能触发换出）。这里的价值在于统一的错误隔离与进度回报——
        单条失败不影响其余，失败项以 {"ok": False, "error": ...} 占位返回。

        prompts 每项可以是 str（当作 user prompt），也可以是 messages 列表。
        """
        out: List[Dict[str, Any]] = []
        total = len(prompts)
        for i, p in enumerate(prompts):
            try:
                if isinstance(p, str):
                    r = self.generate(p, system_prompt=system_prompt, temperature=temperature,
                                      max_tokens=max_tokens, json_output=json_output, expect=expect)
                else:
                    r = self.generate_messages(p, temperature=temperature, max_tokens=max_tokens,
                                               json_output=json_output, expect=expect)
                r["ok"], r["index"] = True, i
            except (LLMError, ValueError) as e:
                r = {"ok": False, "index": i, "error": str(e), "text": "", "elapsed": 0.0,
                     "json": None, "json_ok": False, "json_error": str(e)}
                if stop_on_error:
                    out.append(r)
                    break
            out.append(r)
            if progress:
                progress(i + 1, total, r)
            elif self.verbose:
                mark = "ok" if r.get("ok") else "FAIL"
                print(f"[批量] {i + 1}/{total} {mark} {r.get('elapsed', 0)}s")
        return out

    # ---------------- 辅助 ----------------
    def count_prompt_tokens(self, messages: Sequence[Dict[str, str]]) -> Optional[int]:
        """让 Ollama 自己报这批消息占多少 token（推理端口径，最权威）。

        ⚠ 必须传 num_predict=1 而不是 0——0 会让请求挂住不返回（阶段七之一实测）。
        """
        r = self._post("/api/chat", {"model": self.model_name, "messages": list(messages),
                                     "stream": False, "keep_alive": self.keep_alive,
                                     "options": {"num_predict": 1, "num_ctx": self.num_ctx}})
        return r.get("prompt_eval_count")

    def runtime_info(self) -> Dict[str, Any]:
        """/api/ps —— 看模型实际加载的 context 与显存占用，用来验证 num_ctx 真的生效了。"""
        try:
            data = self._get("/api/ps")
        except LLMError:
            return {}
        for m in data.get("models", []):
            if m.get("name", "").startswith(self.model_name.split(":")[0]):
                return {"name": m.get("name"), "context_length": m.get("context_length"),
                        "size_vram_bytes": m.get("size_vram"),
                        "size_vram_gb": round((m.get("size_vram") or 0) / 1024 ** 3, 2)}
        return {}


# ============================================================================
# CLI 自检
# ============================================================================
_JSON_CASES = [
    # (名称, 模型可能吐出的样子, 期望类型, 期望解析后的值)
    ("裸对象", '{"verdict": "accept", "n": 2}', "object", {"verdict": "accept", "n": 2}),
    ("代码块包裹", '```json\n[{"marker":"S1","relevance":3}]\n```', "array",
     [{"marker": "S1", "relevance": 3}]),
    ("前后缀寒暄", '好的，以下是评估结果：\n[{"marker":"S1"}]\n希望有帮助。', "array",
     [{"marker": "S1"}]),
    ("think 段", '<think>让我想想</think>\n{"ok": true}', "object", {"ok": True}),
    ("尾随逗号", '{"a": 1, "b": 2,}', "object", {"a": 1, "b": 2}),
    ("截断缺右括号", '[{"marker":"S1","relevance":3},{"marker":"S2","relevance":2}',
     "array", [{"marker": "S1", "relevance": 3}, {"marker": "S2", "relevance": 2}]),
    ("截断在半个键值对", '{"issues": [{"claim": "x", "severity": "high"}], "verdict": "rev',
     "object", {"issues": [{"claim": "x", "severity": "high"}]}),
    ("围栏只开不闭", '```json\n{"overall_grounded": false}', "object", {"overall_grounded": False}),
    ("全角引号", '{\u201cok\u201d: 1}', "object", {"ok": 1}),
    # ↓ 实测最高频的一种：qwen3 在数值后多写一个引号（4 道验收题里 2 道中招）
    ("数值后多引号", '[{"marker":"S1","relevance":3","study_type":"rct"}]', "array",
     [{"marker": "S1", "relevance": 3, "study_type": "rct"}]),
    ("数值后多引号·多处（会把括号深度算歪）", '[{"a":1"},{"a":2"},{"a":3"}]', "array",
     [{"a": 1}, {"a": 2}, {"a": 3}]),
    ("合法的数字字符串不被误伤", '{"relevance":"3","n":3}', "object",
     {"relevance": "3", "n": 3}),
    ("空输出", "", "any", None),
    ("根本不是 JSON", "抱歉，我无法评估这些证据。", "any", None),
]


def _json_selftest() -> int:
    print("=" * 78)
    print("JSON 容错解析自检（每条的通过与否由实际解析结果与期望值比对得出）")
    print("=" * 78)
    passed = 0
    for name, raw, expect, want in _JSON_CASES:
        got, note = extract_json(raw, expect=expect)
        ok = got == want
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<16} → {note}")
        if not ok:
            print(f"         期望 {want!r}\n         实得 {got!r}")
    print("-" * 78)
    print(f"JSON 自检：{passed}/{len(_JSON_CASES)} 通过")
    return 0 if passed == len(_JSON_CASES) else 1


def _smoke(args) -> int:
    gen = LLMGenerator(model_name=args.model, base_url=args.base_url,
                       num_ctx=args.num_ctx, verbose=True)
    print(f"已安装模型：{gen.connection['installed_models']}")

    print("\n--- ① 纯文本生成 ---")
    r = gen.generate("用一句话说明什么是随机对照试验（RCT）。",
                     system_prompt="你是严谨的医学文献助理，只用中文回答。",
                     temperature=0.2, max_tokens=120)
    print(r["text"])
    print(f"耗时 {r['elapsed']}s | in={r['prompt_eval_count']} out={r['eval_count']} "
          f"| {r['tokens_per_second']} tok/s")

    print("\n--- ② JSON 结构化生成 ---")
    r2 = gen.generate(
        '把下面这句话转成 JSON：药物 metformin 用于 type 2 diabetes，证据等级 high。'
        '字段：drug, indication, evidence_level。',
        system_prompt="你只输出 JSON。", temperature=0.0, max_tokens=200,
        json_output=True, expect="object")
    print(f"原始输出：{r2['text'][:160]!r}")
    print(f"解析结果：{r2['json']}（{r2['json_note']}）")

    print("\n--- ③ num_ctx 是否真的生效（/api/ps 实测）---")
    info = gen.runtime_info()
    print(f"  {info}")
    ctx_ok = info.get("context_length") == args.num_ctx

    print("\n--- ④ 批量生成 ---")
    rs = gen.batch_generate(["列出一个常见的降压药通用名，只回名字。",
                             "列出一个常见的降糖药通用名，只回名字。"],
                            temperature=0.0, max_tokens=40)
    for r3 in rs:
        print(f"  [{r3['index']}] ok={r3['ok']} {r3['text'][:60]!r}")

    print("\n--- ⑤ num_predict=0 保护 ---")
    try:
        gen.build_options(max_tokens=0)
        guard_ok = False
    except ValueError as e:
        guard_ok = True
        print(f"  已拦截：{e}")

    checks = {
        "文本生成非空": bool(r["text"]),
        "思考模式已关闭（无 thinking 段）": r["thinking_chars"] == 0,
        "报告了 token 用量": r["prompt_eval_count"] is not None and r["eval_count"] is not None,
        "JSON 解析成功": bool(r2["json_ok"]),
        f"num_ctx 生效（={args.num_ctx}）": ctx_ok,
        "批量两条全部成功": all(x.get("ok") for x in rs) and len(rs) == 2
                            and all(x.get("text") for x in rs),
        "num_predict=0 被拦截": guard_ok,
    }
    print("\n" + "=" * 78)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    n_ok = sum(1 for v in checks.values() if v)
    print(f"冒烟自检：{n_ok}/{len(checks)} 通过 | 累计用量 {gen.stats}")
    return 0 if n_ok == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="真调模型跑一遍（需 Ollama 已启动）")
    ap.add_argument("--json-selftest", action="store_true", help="只测 JSON 容错解析，不调模型")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    args = ap.parse_args()

    if args.json_selftest:
        return _json_selftest()
    if args.smoke:
        rc = _json_selftest()
        print()
        return max(rc, _smoke(args))
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
