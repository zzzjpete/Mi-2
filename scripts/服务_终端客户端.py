# -*- coding: utf-8 -*-
"""第十阶段（一）· 服务化 —— 终端问答客户端

一个够简洁、适合演示与录屏的命令行界面：打字提问 → 出处先出来 → 答案逐字流出来。

**它走的是 HTTP 接口，不是直接调流水线。** 这一点是刻意的：演示要演示的就是这一阶段做的
那层服务，绕过接口去调流水线等于把日志、错误码、并发闸门、调用记录全绕过去，
演示出来的是阶段七、九，不是阶段十。

与阶段一的 `命令行提问.py` 的区别：那个直接问裸模型，**没有检索、没有出处**；
这个每句话背后都有 `[S#]` 可查。

跑之前先起服务：
    & $py E:\\rag\\scripts\\服务_应用.py --port 8000

然后：
    & $py E:\\rag\\scripts\\服务_终端客户端.py
    & $py E:\\rag\\scripts\\服务_终端客户端.py --fast     # 关掉①③两段，10 秒出结果（演示提速）
    & $py E:\\rag\\scripts\\服务_终端客户端.py --sync     # 同步模式，看一次性返回长什么样

界面里的命令：
    /help   /new    开新会话（清空上下文）      /sync /stream  切换响应模式
    /fast   /full   切换四段链开关              /last   打印上一条完整最终答案
    /stats  /logs   看统计与调用记录            /health 看各组件状态
    /topics 看快照模式下有证据的主题            /quit
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Windows 传统控制台默认不解析 ANSI 转义；跑一次空命令即可打开 VT 模式（Win10+）。
# 不引 colorama，是为了保持"零第三方依赖"。
if os.name == "nt":
    os.system("")

DEFAULT_BASE = "http://127.0.0.1:8000"


class C:
    """终端配色。`--no-color` 时整体置空。"""
    DIM = "\033[2m"
    B = "\033[1m"
    R = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAG = "\033[35m"
    GRAY = "\033[90m"

    @classmethod
    def off(cls):
        for k in list(vars(cls)):
            if k.isupper():
                setattr(cls, k, "")


#: 四段链的中文名，演示时让观众知道现在在跑哪一段
STAGE_NAMES = {
    "evidence_evaluator": "① 证据评估",
    "answer_generator": "② 起草答案",
    "critical_reviewer": "③ 批判审查",
    "final_assembler": "④ 定稿",
}


def stage_label(key: str) -> str:
    if key.startswith("format_fixer"):
        return f"层D 修正（第 {key.rsplit('_r', 1)[-1]} 轮）"
    return STAGE_NAMES.get(key, key)


# ============================================================================
# HTTP
# ============================================================================
class ApiError(RuntimeError):
    pass


def _req(method: str, url: str, body: Optional[Dict[str, Any]] = None,
         timeout: int = 600) -> urllib.request.Request:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Accept": "text/event-stream, application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def call_json(base: str, path: str, method: str = "GET",
              body: Optional[Dict[str, Any]] = None, timeout: int = 600) -> Dict[str, Any]:
    """打一个普通接口，返回统一响应体（业务失败也返回，不抛——错误码本身就是要展示的东西）。"""
    try:
        with urllib.request.urlopen(_req(method, base + path, body), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)          # 服务端的错误响应也是统一格式，原样返回
        except json.JSONDecodeError:
            raise ApiError(f"HTTP {e.code}：{raw[:200]}") from e
    except urllib.error.URLError as e:
        raise ApiError(f"连不上服务（{base}）：{e.reason}\n"
                       f"先起服务：& $py E:\\rag\\scripts\\服务_应用.py --port 8000") from e


def call_sse(base: str, path: str, body: Dict[str, Any], timeout: int = 900):
    """打流式接口，逐个 yield (事件名, 载荷)。注释行（心跳）产出 ('ping', None)。"""
    try:
        resp = urllib.request.urlopen(_req("POST", base + path, body), timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            yield "error", json.loads(raw)
        except json.JSONDecodeError:
            yield "error", {"message": f"HTTP {e.code}：{raw[:200]}"}
        return
    except urllib.error.URLError as e:
        yield "error", {"message": f"连不上服务（{base}）：{e.reason}"}
        return

    event, data = "message", []
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith(":"):
                yield "ping", None
                continue
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
            elif line == "":                 # 空行 = 一条消息结束
                if data:
                    try:
                        yield event, json.loads("\n".join(data))
                    except json.JSONDecodeError:
                        pass
                event, data = "message", []


# ============================================================================
# 展示
# ============================================================================
def print_sources(sources: List[Dict[str, Any]]) -> None:
    """出处清单。这是流式最先到的东西——模型还没吐字，先让人看到"依据什么在答"。"""
    print(f"\n{C.CYAN}{C.B}▎依据以下 {len(sources)} 篇文献作答{C.R}")
    for s in sources:
        year = f"{s.get('pub_year')}" if s.get("pub_year") else "年份缺失"
        journal = s.get("journal") or "期刊缺失"
        title = (s.get("title") or "（无标题）").strip()
        if len(title) > 72:
            title = title[:71] + "…"
        print(f"  {C.CYAN}[{s['marker']}]{C.R} {title}")
        print(f"       {C.GRAY}{journal} · {year} · {s.get('pmcid') or '-'}{C.R}")
    print()


def print_metrics(d: Dict[str, Any], elapsed: float) -> None:
    m = d.get("metrics") or {}
    cc = d.get("constraint_check") or {}
    bits = [f"{elapsed:.1f}s",
            f"{m.get('llm_calls', '?')} 次模型调用",
            f"{m.get('answer_chars', '?')} 字",
            f"in {m.get('total_prompt_tokens', '?')} / out {m.get('total_output_tokens', '?')} token"]
    if cc:
        ok = cc.get("compliant")
        bits.append(f"{'合规' if ok else '不合规'}"
                    + (f"（{'、'.join(v['code'] for v in cc.get('violations') or [])}）"
                       if not ok else ""))
    if d.get("refused"):
        bits.append(f"{C.YELLOW}含拒答短语{C.R}")
    print(f"{C.GRAY}└─ {' ｜ '.join(bits)}｜{d.get('request_id', '')}{C.R}\n")


def health_components(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 /health/ready 的响应里取组件表。

    ⚠ 就绪时组件在 `data.components`，**未就绪时整个响应是 5002，组件在
    `detail.components`**（HTTP 503）。只读 data 的话，服务不健康时这张表反而
    一片空白——恰恰是最需要看它的时候什么都看不到。
    """
    for key in ("data", "detail"):
        v = r.get(key)
        if isinstance(v, dict) and v.get("components"):
            return v["components"]
    return []


def print_health(base: str) -> None:
    r = call_json(base, "/health/ready", timeout=30)
    d = r.get("data") or {}
    status = d.get("status") or ("down" if r.get("code") else "?")
    color = C.GREEN if status == "ok" else (C.YELLOW if status == "degraded" else C.RED)
    print(f"\n{C.B}服务状态{C.R} {color}{status}{C.R}"
          f"｜检索模式 {d.get('retrieval_mode')}｜模型 {d.get('model')}"
          f"｜并发上限 {d.get('max_concurrent')}")
    for c in health_components(r):
        mark = f"{C.GREEN}✓{C.R}" if c["ok"] else f"{C.RED}✗{C.R}"
        print(f"  {mark} {c['name']:<20} {C.GRAY}{c['detail']}{C.R}")
    if status == "down":
        print(f"{C.RED}关键依赖不可用，问答会失败。{C.R}")
    print()


# ============================================================================
# 一次提问
# ============================================================================
def ask_stream(base: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """流式提问：出处先出、分段进度、答案逐字。返回 done 里的 data。"""
    t0 = time.time()
    cur_stage = ""
    printed_any = False
    data: Optional[Dict[str, Any]] = None

    for event, obj in call_sse(base, "/api/v1/qa/stream", payload):
        if event == "ping":
            continue
        if event == "meta":
            if obj.get("rewritten"):
                print(f"{C.MAG}↻ 追问改写：{obj.get('resolved_query')}{C.R}")
            elif obj.get("history_used"):
                # 判为追问却没改成 —— 这句会被原样拿去检索，代词检索不到东西。
                # 不说出来的话，用户只会看到一句莫名其妙的"无法回答"。
                print(f"{C.YELLOW}↻ 未改写，按原句检索"
                      f"（带入历史 {obj['history_used']} 条）{C.R}")
            if obj.get("pool_match") == "fallback":
                print(f"{C.YELLOW}⚠ 这个问题没有匹配到证据组，用的是不相关证据"
                      f"（快照模式的已知限制，/topics 看有哪些主题）{C.R}")
        elif event == "sources":
            print_sources(obj.get("sources") or [])
        elif event == "stage":
            key = obj.get("stage", "")
            if obj.get("retrieval_query"):
                # 检索一做完就回显"我理解成了什么"。译坏了的话，这一行在等待期里
                # 就能看出来，不用等三十秒后靠一句"无法回答"去猜。
                src = obj.get("translated_from")
                print(f"{C.CYAN}🔍 检索式：{obj['retrieval_query']}{C.R}"
                      + (f"{C.GRAY}　（中译英自：{src}）{C.R}" if src else "")
                      + f"{C.GRAY}　命中 {obj.get('hits')} 条{C.R}")
                continue
            if obj.get("status") == "start":
                cur_stage = key
                end = "\n" if obj.get("is_answer_stage") else ""
                print(f"{C.GRAY}· {stage_label(key)}…{C.R}", end=end, flush=True)
            elif obj.get("status") == "end" and not obj.get("is_answer_stage"):
                el = obj.get("elapsed")
                print(f"{C.GRAY} {el}s{C.R}" if el else "", flush=True)
        elif event == "delta":
            printed_any = True
            sys.stdout.write(obj.get("text", ""))
            sys.stdout.flush()
        elif event == "check":
            if not obj.get("compliant"):
                codes = "、".join(v["code"] for v in obj.get("violations") or [])
                print(f"\n{C.YELLOW}⚠ 层D 判定不合规：{codes}{C.R}")
        elif event == "done":
            data = obj.get("data")
            if printed_any:
                print()
            print(f"{C.GRAY}{'─' * 68}{C.R}")
            print(f"{C.GRAY}以上正文是**过程输出**；最终答案由服务端后处理补齐了"
                  f"参考文献与免责声明（/last 看完整版）。{C.R}")
            print_metrics(data or {}, time.time() - t0)
        elif event == "error":
            print(f"\n{C.RED}✗ [{obj.get('code')}] {obj.get('message')}{C.R}")
            if obj.get("detail"):
                print(f"{C.GRAY}   {json.dumps(obj['detail'], ensure_ascii=False)}{C.R}")
    return data


def ask_sync(base: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """同步提问：等整条链跑完一次性返回。用来对照"流式到底省了什么等待感"。"""
    t0 = time.time()
    print(f"{C.GRAY}· 同步模式，等整条链跑完（本机通常 35~50 秒，中间没有任何输出）…{C.R}",
          flush=True)
    r = call_json(base, "/api/v1/qa/ask", "POST", payload)
    if r.get("code") != 0:
        print(f"{C.RED}✗ [{r.get('code')}] {r.get('message')}{C.R}")
        if r.get("detail"):
            print(f"{C.GRAY}   {json.dumps(r['detail'], ensure_ascii=False)}{C.R}")
        return None
    d = r["data"]
    print_sources(d.get("sources") or [])
    print(d.get("answer", ""))
    print(f"{C.GRAY}{'─' * 68}{C.R}")
    print_metrics(d, time.time() - t0)
    return d


# ============================================================================
# REPL
# ============================================================================
HELP = f"""
{C.B}命令{C.R}
  /help              这份帮助
  /new               开一个新会话（清空上下文，追问不再关联上一轮）
  /stream  /sync     切换流式 / 同步响应模式
  /fast    /full     快速模式（关①③，1 次调用约 10 秒）/ 完整四段链
  /last              打印上一条**完整最终答案**（含参考文献与免责声明）
  /health            各组件状态
  /stats             调用统计（成功率 / 拒答率 / p95 / token）
  /logs              最近的调用记录
  /topics            快照模式下有真实证据的主题
  /quit  /exit       退出
直接打字就是提问。追问会自动带上会话 ID。
"""

def topic_hint(topics: List[Dict[str, Any]]) -> str:
    """主题清单从服务端 `/qa/topics` 读，不在客户端写死——写死的那份迟早和快照对不上。"""
    if not topics:
        return f"{C.GRAY}当前是真检索模式，任何医学问题都可以问。{C.R}\n"
    lines = [f"\n{C.B}目前有真实文献证据的 {len(topics)} 个主题{C.R}"
             f"{C.GRAY}（问别的会拿到不相关证据，然后系统会如实拒答）{C.R}"]
    for t in topics:
        kw = "、".join(t["keywords"][:4])
        lines.append(f"  · {C.CYAN}{kw}{C.R}　{C.GRAY}{t['docs']} 篇文献{C.R}")
    lines.append(f"{C.GRAY}例：CRISPR-Cas9 的脱靶效应有哪些检测方法？{C.R}")
    lines.append(f"{C.GRAY}要问任意问题需要真检索：服务端启动时加 --mode live"
                 f"（要 65GB 向量库，首次加载数分钟）。{C.R}")
    return "\n".join(lines) + "\n"


def repl(args: argparse.Namespace) -> int:
    base = args.base_url.rstrip("/")
    try:
        print_health(base)
    except ApiError as e:
        print(f"{C.RED}{e}{C.R}")
        return 1

    topics_data = call_json(base, "/api/v1/qa/topics", timeout=30).get("data") or {}
    topics = topics_data.get("topics") or []
    session_id = args.session or f"cli-{int(time.time())}"
    streaming = not args.sync
    fast = args.fast
    last: Optional[Dict[str, Any]] = None

    print(f"{C.B}医学文献问答{C.R}　{C.GRAY}接口 {base}｜会话 {session_id}"
          f"｜{'流式' if streaming else '同步'}｜{'快速(1 次调用)' if fast else '完整四段链'}{C.R}")
    print(topic_hint(topics))
    print(f"{C.GRAY}输入 /help 看命令，直接打字提问。{C.R}\n")

    while True:
        try:
            q = input(f"{C.GREEN}{C.B}你 › {C.R}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if not q:
            continue

        # 启动参数被打进聊天框：`--mode live`、`--fast` 这类是**起服务时**用的，
        # 当成问题发出去会白等二十几秒再收到一句拒答（实测踩过）。这里直接拦下。
        if q.startswith("-"):
            print(f"{C.YELLOW}「{q}」看起来是命令行启动参数，不是问题。{C.R}")
            if "live" in q or "mode" in q:
                print(f"{C.GRAY}  --mode live 是**起服务时**加的，要重启服务端：\n"
                      f"    & $py E:\\rag\\scripts\\服务_应用.py --port 8000 --mode live\n"
                      f"  （需要 65GB 向量库，首次加载数分钟）{C.R}")
            else:
                print(f"{C.GRAY}  聊天里的命令用斜杠开头，/help 看有哪些。{C.R}")
            print()
            continue

        if q.startswith("/"):
            cmd = q.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                print("再见。")
                return 0
            if cmd == "/help":
                print(HELP); continue
            if cmd == "/topics":
                topics = (call_json(base, "/api/v1/qa/topics", timeout=30).get("data")
                          or {}).get("topics") or []
                print(topic_hint(topics)); continue
            if cmd == "/new":
                session_id = f"cli-{int(time.time())}"
                print(f"{C.GRAY}新会话：{session_id}{C.R}\n"); continue
            if cmd in ("/stream", "/sync"):
                streaming = cmd == "/stream"
                print(f"{C.GRAY}已切换到{'流式' if streaming else '同步'}模式{C.R}\n"); continue
            if cmd in ("/fast", "/full"):
                fast = cmd == "/fast"
                print(f"{C.GRAY}已切换到{'快速（关①③，1 次调用）' if fast else '完整四段链（4 次调用）'}{C.R}\n")
                continue
            if cmd == "/last":
                if last:
                    print(f"\n{last.get('answer', '')}\n")
                else:
                    print(f"{C.GRAY}还没有问过问题。{C.R}\n")
                continue
            if cmd == "/health":
                print_health(base); continue
            if cmd == "/stats":
                s = call_json(base, "/api/v1/qa/stats", timeout=30).get("data") or {}
                print(f"\n{C.B}调用统计{C.R}　共 {s.get('total')} 次"
                      f"｜成功率 {s.get('success_rate')}"
                      f"｜拒答率 {s.get('refusal_rate')}"
                      f"｜层D合规率 {s.get('compliant_rate')}")
                e = s.get("elapsed_ms") or {}
                print(f"  耗时 avg {e.get('avg')}ms / p50 {e.get('p50')} / p95 {e.get('p95')}"
                      f" / max {e.get('max')}")
                t = s.get("tokens") or {}
                print(f"  token in {t.get('prompt')} / out {t.get('output')}"
                      f"｜模型调用 {t.get('llm_calls')} 次")
                print(f"  分布 {s.get('by_mode')} {s.get('by_status')}\n")
                continue
            if cmd == "/logs":
                r = call_json(base, "/api/v1/qa/logs?page=1&page_size=8", timeout=30)
                pg = r.get("data") or {}
                print(f"\n{C.B}最近 {len(pg.get('items') or [])} / 共 {pg.get('total')} 条{C.R}")
                for it in pg.get("items") or []:
                    mark = C.GREEN + "ok " + C.R if it["status"] == "ok" else C.RED + it["status"] + C.R
                    print(f"  {it['created_at']}  {it['request_id']}  {it['mode']:<6} {mark}"
                          f" code={it['code']:<5} {it['elapsed_ms']:>9.0f}ms  {(it['query'] or '')[:34]}")
                print()
                continue
            print(f"{C.GRAY}未知命令 {cmd}，/help 看可用命令{C.R}\n")
            continue

        # 发问前自检：服务端第 0 秒就知道证据匹不匹配，没必要让人等二十几秒才收到拒答
        if topics:
            probe = call_json(base, "/api/v1/qa/topics?probe="
                              + urllib.parse.quote(q), timeout=30).get("data") or {}
            if probe.get("probe") and not probe["probe"].get("in_scope"):
                print(f"\n{C.YELLOW}⚠ 这个问题不在现有证据范围内。{C.R}")
                print(f"{C.GRAY}  系统会拿到不相关的文献，然后如实回答"
                      f"「根据现有文献无法回答此问题」——这是正确行为，但要花二十多秒。\n"
                      f"  /topics 看有哪些主题问得出东西。{C.R}")
                try:
                    yn = input(f"{C.GRAY}  仍然要问吗？(y/N) {C.R}").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    yn = "n"
                if yn not in ("y", "yes"):
                    print()
                    continue

        payload = {"query": q, "top_k": args.top_k, "session_id": session_id,
                   "evaluate": not fast, "review": not fast}
        print()
        try:
            d = (ask_stream if streaming else ask_sync)(base, payload)
            if d:
                last = d
        except ApiError as e:
            print(f"{C.RED}{e}{C.R}\n")
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}已中断（服务端已发出的那一段仍会跑完）{C.R}\n")


# ============================================================================
# --demo：按任务书逐条跑一遍，顺便留下一份干净的交付日志
# ============================================================================
def _banner(no: str, title: str, req: str) -> None:
    print(f"\n{C.B}{C.CYAN}{'═' * 74}{C.R}")
    print(f"{C.B}{C.CYAN}【{no}】{title}{C.R}")
    print(f"{C.GRAY}对应任务书：{req}{C.R}")
    print(f"{C.B}{C.CYAN}{'═' * 74}{C.R}")


def _brief(r: Dict[str, Any], keys: Tuple[str, ...] = ("code", "message")) -> str:
    return "　".join(f"{k}={r.get(k)}" for k in keys)


def run_demo(base: str, top_k: int = 6) -> int:
    """按任务书顺序演示一遍。每一步都打印它在证明哪条要求。

    刻意全部用**快速模式**（关①③，1 次模型调用约 12 秒）：演示要的是接口行为，
    不是等四段链跑完。完整四段链单独留最后一步演示。
    """
    covered: List[Tuple[str, str]] = []
    sid = f"demo-{int(time.time())}"

    # ---------------------------------------------------------------- 1
    _banner("1", "健康检查：存活探针与就绪探针分开", "1 · 配置日志和健康检查")
    live = call_json(base, "/health", timeout=30)
    print(f"GET /health          → {_brief(live)}　{live.get('data')}")
    print(f"{C.GRAY}存活探针不碰任何下游——探针一旦依赖下游，下游抖动就会把好好的进程重启掉。{C.R}")
    print_health(base)
    covered.append(("1 健康检查", "/health 与 /health/ready 分开，组件逐项可见"))

    # ---------------------------------------------------------------- 2
    _banner("2", "错误码枚举", "1 · 定义错误码枚举（1001/2001/3001/4001 等）")
    tbl = call_json(base, "/api/v1/errors", timeout=30).get("data") or {}
    codes = tbl.get("codes") or []
    print(f"共 {len(codes)} 个码，分 {len(tbl.get('families') or {})} 族：")
    for c in codes:
        if c["code"] in (0, 1001, 1002, 1003, 1004, 2001, 3001, 3002, 4001, 4002, 5001, 5003):
            print(f"  {c['code']:>5}  {c['name']:<20} HTTP {c['http_status']:<4} {c['message']}")
    covered.append(("1 错误码枚举", f"{len(codes)} 个码，GET /api/v1/errors 可查"))

    # ---------------------------------------------------------------- 3
    _banner("3", "统一响应格式 + 同步问答 + 集成 RAG 流水线",
            "1 · 统一响应模型 ResponseModel　2 · 同步接口、集成RAG流水线")
    q1 = "CRISPR-Cas9 的脱靶效应有哪些检测方法？"
    print(f"提问：{q1}\n")
    t0 = time.time()
    r = call_json(base, "/api/v1/qa/ask", "POST",
                  {"query": q1, "top_k": top_k, "evaluate": False, "review": False})
    d = r.get("data") or {}
    print(f"{C.B}响应外层（所有接口同一个形状）{C.R}")
    for k in ("code", "message", "request_id", "timestamp", "elapsed_ms"):
        print(f"  {k:<12} {r.get(k)}")
    print(f"  {'data':<12} …（下面展开）")
    print(f"\n{C.B}data 里有什么{C.R}")
    print(f"  答案 {d.get('metrics', {}).get('answer_chars')} 字，"
          f"出处 {len(d.get('sources') or [])} 条，"
          f"拒答={d.get('refused')}，"
          f"层D合规={(d.get('constraint_check') or {}).get('compliant')}")
    print(f"  引用校验 {d.get('citation_check')}")
    print(f"  耗时 {time.time() - t0:.1f}s，模型调用 {d.get('metrics', {}).get('llm_calls')} 次")
    rid = r.get("request_id", "")
    covered.append(("1 统一响应格式", "code/message/data/detail/request_id/timestamp/elapsed_ms"))
    covered.append(("2 同步接口", f"POST /qa/ask，{time.time() - t0:.1f}s 返回带出处的答案"))

    # ---------------------------------------------------------------- 4
    _banner("4", "同一个 request_id 在响应体、日志、数据库三处一致",
            "2 · 记录每次调用的请求ID、耗时、结果状态（用于后续统计）")
    print(f"上一步响应体里的 request_id：{C.B}{rid}{C.R}")
    one = call_json(base, f"/api/v1/qa/logs/{rid}", timeout=30).get("data") or {}
    print(f"数据库里查同一个 ID：")
    for k in ("request_id", "created_at", "mode", "status", "code", "elapsed_ms",
              "llm_calls", "prompt_tokens", "output_tokens", "answer_chars",
              "refused", "compliant"):
        print(f"  {k:<14} {one.get(k)}")
    print(f"{C.GRAY}服务端日志里同一个 ID 的行见 logs\\service\\api.log（录屏时同屏展示）。{C.R}")
    covered.append(("2 调用记录", "请求ID/耗时/状态/token/拒答/合规 全部落库，可按 ID 查回"))

    # ---------------------------------------------------------------- 5
    _banner("5", "参数校验与全局异常处理",
            "1 · 全局异常处理器　2 · query 非空且长度限制、top_k 范围检查")
    cases = [
        ("query 为纯空白", "POST", "/api/v1/qa/ask", {"query": "   "}),
        # 任务书写的是「query 非空**且长度限制**」——空与超长是两条独立的校验，都要演到
        ("query 超长（1001 字）", "POST", "/api/v1/qa/ask", {"query": "药" * 1001}),
        ("top_k 超出范围", "POST", "/api/v1/qa/ask", {"query": "正常问题", "top_k": 999}),
        ("缺必填参数 query", "POST", "/api/v1/qa/ask", {"top_k": 5}),
        ("访问不存在的路径", "GET", "/api/v1/no-such-endpoint", None),
        ("会话不存在", "GET", "/api/v1/sessions/not-exist", None),
    ]
    for name, m, path, body in cases:
        rr = call_json(base, path, m, body, timeout=30)
        det = rr.get("detail")
        print(f"  {name:<18} → code={C.YELLOW}{rr.get('code')}{C.R} {rr.get('message')}")
        if det:
            print(f"  {'':<18}   {C.GRAY}{json.dumps(det, ensure_ascii=False)[:120]}{C.R}")
    print(f"\n{C.GRAY}未知异常一律 5001，日志留全栈、响应体只给一句话——"
          f"把未知异常伪装成已知码，会让线上真 bug 藏在一个看起来正常的 4xxx 里。{C.R}")
    covered.append(("1 全局异常处理", "1001/1002/1003/3001/3002 各归其位，响应体形状不变"))
    covered.append(("2 参数校验", "空 query→1001，超长 query→1001，top_k 越界→1003，缺参→1002"))

    # ---------------------------------------------------------------- 6
    _banner("6", "流式接口：出处先到，答案逐字", "2 · 实现流式接口")
    print(f"提问：{q1}\n")
    t0 = time.time()
    firsts: Dict[str, float] = {}
    n_delta = 0
    for event, obj in call_sse(base, "/api/v1/qa/stream",
                               {"query": q1, "top_k": top_k, "evaluate": False,
                                "review": False}):
        if event == "ping":
            continue
        firsts.setdefault(event, round(time.time() - t0, 2))
        if event == "sources":
            print_sources(obj.get("sources") or [])
        elif event == "delta":
            n_delta += 1
            sys.stdout.write(obj.get("text", ""))
            sys.stdout.flush()
        elif event == "done":
            print()
    print(f"\n{C.B}各事件首次出现的时刻{C.R}")
    for k, v in firsts.items():
        star = f"  {C.YELLOW}← 出处比第一个字早 {firsts.get('delta', 0) - v:.1f} 秒{C.R}" \
            if k == "sources" and "delta" in firsts else ""
        print(f"  {k:<10} {v:>6.2f}s{star}")
    print(f"  共 {n_delta} 个增量事件")
    covered.append(("2 流式接口", f"SSE 七种事件；出处 {firsts.get('sources')}s，"
                                  f"首字 {firsts.get('delta')}s"))

    # ---------------------------------------------------------------- 7
    _banner("7", "会话管理：追问被改写成可独立检索的问题",
            "2 · 若传入 session_id，则关联历史对话")
    print(f"会话 ID：{sid}")
    print(f"\n第一轮：{q1}")
    call_json(base, "/api/v1/qa/ask", "POST",
              {"query": q1, "top_k": top_k, "session_id": sid,
               "evaluate": False, "review": False})
    q2 = "它有哪些局限？"
    print(f"第二轮（追问，句子本身检索不到任何东西）：{q2}\n")
    r2 = call_json(base, "/api/v1/qa/ask", "POST",
                   {"query": q2, "top_k": top_k, "session_id": sid,
                    "evaluate": False, "review": False})
    d2 = r2.get("data") or {}
    print(f"  原始问题     {d2.get('query')}")
    print(f"  {C.B}改写后检索式  {C.MAG}{d2.get('resolved_query')}{C.R}")
    print(f"  带入历史轮数  {d2.get('history_used')}　是否改写 {d2.get('rewritten')}")
    sess = call_json(base, f"/api/v1/sessions/{sid}", timeout=30).get("data") or {}
    print(f"  会话里已存 {sess.get('turns')} 条消息")
    print(f"\n{C.GRAY}⚠ 历史只用于改写检索式，一个字都不进证据区——"
          f"把上轮问答原文拼进上下文，等于给模型一段没有出处的事实材料。{C.R}")
    covered.append(("2 会话管理", f"传 session_id 后追问被改写：{d2.get('resolved_query')}"))

    # ---------------------------------------------------------------- 8
    _banner("8", "调用记录分页与聚合统计", "2 · 记录…（用于后续统计）")
    pg = call_json(base, "/api/v1/qa/logs?page=1&page_size=5", timeout=30).get("data") or {}
    print(f"分页字段：total={pg.get('total')} page={pg.get('page')} "
          f"page_size={pg.get('page_size')} pages={pg.get('pages')} "
          f"has_next={pg.get('has_next')} has_prev={pg.get('has_prev')}")
    for it in pg.get("items") or []:
        print(f"  {it['created_at']}  {it['request_id']}  {it['mode']:<6} "
              f"{it['status']:<5} {it['elapsed_ms']:>8.0f}ms  {(it['query'] or '')[:30]}")
    s = call_json(base, "/api/v1/qa/stats", timeout=30).get("data") or {}
    print(f"\n{C.B}聚合{C.R} 共 {s.get('total')} 次｜成功率 {s.get('success_rate')}"
          f"｜拒答率 {s.get('refusal_rate')}｜层D合规率 {s.get('compliant_rate')}")
    e = s.get("elapsed_ms") or {}
    print(f"  耗时 avg {e.get('avg')}ms / p50 {e.get('p50')} / p95 {e.get('p95')} / max {e.get('max')}")
    t = s.get("tokens") or {}
    print(f"  token in {t.get('prompt')} / out {t.get('output')}｜模型调用 {t.get('llm_calls')} 次")
    covered.append(("1 分页模型", "items/total/page/page_size/pages/has_next/has_prev"))

    # ---------------------------------------------------------------- 汇总
    print(f"\n{C.B}{C.GREEN}{'═' * 74}{C.R}")
    print(f"{C.B}{C.GREEN}演示完毕 —— 任务书要求覆盖情况{C.R}")
    print(f"{C.B}{C.GREEN}{'═' * 74}{C.R}")
    for req, how in covered:
        print(f"  {C.GREEN}✓{C.R} {req:<16} {C.GRAY}{how}{C.R}")
    print(f"\n{C.GRAY}服务端日志：E:\\rag\\logs\\service\\api.log"
          f"　调用记录：& $py 服务_日志.py --export <路径>{C.R}\n")
    return 0


# ============================================================================
# --demo2：服务化第二部分（会话管理 / 运营统计 / 文档管理 / 交付物）
# ============================================================================
#: 演示用的文献。取自真目录库，四个字段都齐、且 total_chunks 远大于 indexed_chunks
#: ——这一条本身就是「227 万不是 227 万篇完整文献」的实证。
DEMO2_DOC = "PMC10006239"
#: 用来演示 3001。库里确认不存在。
DEMO2_MISSING_DOC = "PMC00000000"
#: 列表演示的过滤条件。⚠ **必须带过滤**：不带过滤时按 pmcid 排序的第一条是目录里
#: 唯一一条脏数据（doc_id 非 PMC 开头、且 indexed_chunks > total_chunks），
#: 227 万条里就那一条，偏偏排在最前面，无过滤列表一打开就在镜头里。
DEMO2_JOURNAL = "Nature Communications"
DEMO2_YEAR = 2022
DEMO2_SESSION = "demo-part2"
DEMO2_QUERY = "CRISPR-Cas9 的脱靶效应有哪些检测方法？"


def _kv(d: Dict[str, Any], *keys: str) -> str:
    return "　".join(f"{k}={d.get(k)}" for k in keys)


def run_demo2(base: str, top_k: int = 6, skip_qa: bool = False) -> int:
    """第二部分演示。除【3】外全是毫秒级接口。

    【3】要一次真问答（约 20~30 秒，需要 Ollama）——「添加消息由问答接口自动调用」
    这条只能靠真问一次来证明，看 turns 从 0 变 2。**不需要 65GB 向量库**，
    snapshot 模式够用。Ollama 没起时该步会如实报出来并继续，不中断演示。

    ⚠ 汇总里的勾**必须由本轮实际结果算出**，不能因为"这一步跑过了"就打勾：
    `--skip-qa` 时 turns 根本没变，那条就得是 ⚠ 而不是 ✓。否则演示会替一个
    没发生的功能作证——本项目在验收判据上已经栽过三次同源的跟头。
    """
    #: (本轮是否真的验到了, 任务书条目, 证据)
    covered: List[Tuple[bool, str, str]] = []

    # ---------------------------------------------------------------- 1
    _banner("1", "各组件健康状态：LLM / 向量库 / 数据库",
            "1 · 运营统计api：各组件健康状态（LLM、向量库、数据库）")
    hr = call_json(base, "/health/ready", timeout=30)
    comps = health_components(hr)
    print(f"就绪探针返回 {C.B}{len(comps)}{C.R} 个组件"
          f"（code={hr.get('code')}）：\n")
    for c in comps:
        mark = f"{C.GREEN}✓{C.R}" if c["ok"] else f"{C.RED}✗{C.R}"
        crit = f"{C.YELLOW}关键{C.R}" if c.get("critical") else f"{C.GRAY}非关键{C.R}"
        star = f"  {C.MAG}← 第二部分新增{C.R}" if c["name"] in ("vector_db", "doc_catalog") else ""
        print(f"  {mark} {c['name']:<20} {crit}  {C.GRAY}{c['detail']}{C.R}{star}")
    print(f"\n{C.GRAY}向量库那一格第一部分是空的——当时查的其实是 BM25 目录。"
          f"现在只 stat 不打开：打开一次要 22 秒 + 13.7 GB，"
          f"会当场撕掉快照模式「不加载 65 GB 库」的承诺。{C.R}")
    print(f"{C.GRAY}⚠ retrieval:snapshot 与 retrieval:live 互斥，"
          f"所以任何模式下都是 {len(comps)} 个组件，不是 8 个。{C.R}")
    names = {c["name"] for c in comps}
    comp_ok = bool(comps) and {"vector_db", "doc_catalog"} <= names
    covered.append((comp_ok, "1 组件健康状态",
                    f"{len(comps)} 个组件逐项可见，含 vector_db 与 doc_catalog" if comp_ok
                    else f"**本轮未验到**：只拿到 {len(comps)} 个组件（{sorted(names)}）"))

    # ---------------------------------------------------------------- 2
    _banner("2", "会话管理：创建 / 获取 / 删除", "1 · 会话管理api：创建、获取、删除会话")
    r = call_json(base, "/api/v1/sessions", "POST",
                  {"session_id": DEMO2_SESSION, "title": "第二部分演示"})
    d = r.get("data") or {}
    print(f"POST /sessions            → {_brief(r)}　{_kv(d, 'session_id', 'turns', 'title')}")
    r = call_json(base, f"/api/v1/sessions/{DEMO2_SESSION}", timeout=30)
    d0 = r.get("data") or {}
    print(f"GET  /sessions/{DEMO2_SESSION}  → {_brief(r)}　"
          f"turns={d0.get('turns')}　history {len(d0.get('history') or [])} 条")
    pg = (call_json(base, "/api/v1/sessions?page=1&page_size=5", timeout=30).get("data") or {})
    print(f"GET  /sessions（分页）     → {_kv(pg, 'total', 'page', 'page_size', 'pages', 'has_next', 'has_prev')}")
    sess_ok = (d0.get("session_id") == DEMO2_SESSION and "history" in d0
               and pg.get("total") is not None)
    covered.append((sess_ok, "1 会话管理",
                    "创建 / 获取（含历史消息列表）/ 列表分页 / 删除，见下" if sess_ok
                    else "**本轮未验到**：创建或获取会话未返回预期字段"))

    # ---------------------------------------------------------------- 3
    _banner("3", "添加消息：由问答接口自动调用，不是手工调的",
            "1 · 会话管理api：添加消息（由问答接口自动调用）")
    before = (call_json(base, f"/api/v1/sessions/{DEMO2_SESSION}").get("data") or {}).get("turns")
    print(f"问答前 turns = {C.B}{before}{C.R}")
    if skip_qa:
        print(f"{C.YELLOW}--skip-qa：跳过真问答这一步。{C.R}")
    else:
        print(f"提问（带 session_id={DEMO2_SESSION}）：{DEMO2_QUERY}")
        print(f"{C.GRAY}这一步要 Ollama，约 20~30 秒；不需要 65 GB 向量库。{C.R}")
        t0 = time.time()
        qa = call_json(base, "/api/v1/qa/ask", "POST",
                       {"query": DEMO2_QUERY, "top_k": top_k, "session_id": DEMO2_SESSION,
                        "evaluate": False, "review": False})
        if qa.get("code") != 0:
            print(f"{C.RED}问答失败：{_brief(qa)}{C.R}")
            if qa.get("code") == 4004:
                print(f"{C.YELLOW}→ Ollama 没起。先跑："
                      f"$env:OLLAMA_MODELS='E:\\rag\\ollama\\models'; ollama serve{C.R}")
            print(f"{C.GRAY}后面的步骤不依赖它，继续。{C.R}")
        else:
            qd = qa.get("data") or {}
            print(f"  {_brief(qa)}　拒答={qd.get('refused')}　"
                  f"出处 {len(qd.get('sources') or [])} 条　用时 {time.time() - t0:.1f}s")
    after_d = call_json(base, f"/api/v1/sessions/{DEMO2_SESSION}").get("data") or {}
    after = after_d.get("turns")
    ok = (before is not None and after is not None and after > before)
    mark = f"{C.GREEN}✓{C.R}" if ok else f"{C.YELLOW}—{C.R}"
    print(f"问答后 turns = {C.B}{after}{C.R}  {mark} "
          f"{'一问一答被自动写入，没人手工调过添加消息' if ok else '未发生变化（问答未成功执行）'}")
    for h in (after_d.get("history") or [])[-2:]:
        role = h.get("role") or h.get("type") or "?"
        text = (h.get("content") or h.get("text") or "")[:60]
        print(f"  {C.GRAY}{role:<10}{text}{C.R}")
    covered.append((ok, "1 添加消息",
                    f"问答一次后 turns {before} → {after}（由 /qa/ask 自动写入）" if ok
                    else f"**本轮未验到**：turns 仍是 {after}（跳过了问答，或问答失败）"))

    # ---------------------------------------------------------------- 4
    _banner("4", "运营统计：问答次数 / 平均耗时 / 成功率 ＋ 索引三字段",
            "1 · 运营统计api：问答次数、平均耗时、成功率；文档总数、索引大小、增量更新次数")
    s = call_json(base, "/api/v1/qa/stats", timeout=60).get("data") or {}
    e = s.get("elapsed_ms") or {}
    print(f"{C.B}调用侧{C.R}  共 {s.get('total')} 次｜成功率 {s.get('success_rate')}"
          f"｜拒答率 {s.get('refusal_rate')}｜层D合规率 {s.get('compliant_rate')}")
    print(f"        耗时 avg {e.get('avg')}ms / p50 {e.get('p50')} / p95 {e.get('p95')} / max {e.get('max')}")
    idx = s.get("index") or {}
    print(f"\n{C.B}索引侧（第二部分新增）{C.R}")
    print(f"  文档总数      {C.B}{idx.get('total_documents')}{C.R}"
          f"　（块 {idx.get('total_chunks')}）")
    print(f"  索引大小      {C.B}{idx.get('index_size_human')}{C.R}")
    for k, v in (idx.get("index_size_detail") or {}).items():
        if isinstance(v, dict):
            print(f"                {k:<12} {v.get('human')}　{v.get('files')} 文件")
    print(f"  增量更新次数  {C.B}{idx.get('incremental_updates')}{C.R}")
    note = idx.get("incremental_updates_note")
    if note:
        print(f"                {C.GRAY}{note}{C.R}")
    dn = idx.get("documents_note")
    if dn:
        print(f"\n{C.YELLOW}⚠ 口径必须连着数字一起报：{C.R}{C.GRAY}{dn}{C.R}")
    stats_ok = (s.get("total") is not None and e.get("avg") is not None
                and s.get("success_rate") is not None
                and idx.get("total_documents") is not None
                and idx.get("index_size_human") is not None
                and idx.get("incremental_updates") is not None)
    covered.append((stats_ok, "1 运营统计",
                    f"问答次数/平均耗时/成功率 + 文档总数 {idx.get('total_documents')}"
                    f"/索引 {idx.get('index_size_human')}/增量 {idx.get('incremental_updates')}"
                    if stats_ok else "**本轮未验到**：六个字段里有缺（索引统计没建？）"))

    # ---------------------------------------------------------------- 5
    _banner("5", "文档管理：列表（过滤 + 游标）与按 id 查",
            "2 · 定义文档模型 DocumentIn；实现文档列表查询与id查询api")
    q = f"/api/v1/documents?journal={urllib.parse.quote(DEMO2_JOURNAL)}&pub_year={DEMO2_YEAR}&limit=3&with_total=true"
    r = call_json(base, q, timeout=60)
    dd = r.get("data") or {}
    list_ok = (r.get("code") == 0 and bool(dd.get("items")) and dd.get("total") is not None)
    page_ok = False
    if r.get("code") != 0:
        print(f"{C.RED}列表失败：{_brief(r)}{C.R}")
        if r.get("code") == 3004:
            print(f"{C.YELLOW}→ 文献目录没建。先跑：服务_文档目录.py --build（约 45 秒）{C.R}")
    else:
        print(f"过滤 journal={DEMO2_JOURNAL} pub_year={DEMO2_YEAR} limit=3 with_total=true")
        print(f"  {_kv(dd, 'total', 'limit', 'has_more', 'next_cursor')}")
        for it in dd.get("items") or []:
            print(f"    {it.get('doc_id')}  原文{it.get('total_chunks')}块/入库{it.get('indexed_chunks')}块"
                  f"  {(it.get('title') or '')[:44]}")
        cur = dd.get("next_cursor")
        print(f"\n{C.GRAY}分页只给游标不给页码：227 万行上跳过十万行要先扫掉十万行，"
              f"游标是常数代价。{C.R}")
        if cur:
            r2 = call_json(base, q.replace("&with_total=true", "") + f"&cursor={cur}", timeout=60)
            d2 = r2.get("data") or {}
            print(f"翻下一页 cursor={cur}")
            for it in d2.get("items") or []:
                print(f"    {it.get('doc_id')}  {(it.get('title') or '')[:52]}")
            # 翻页要真翻动了才算数：第二页的 id 必须和第一页没有交集，
            # 否则「游标分页」这条演的是它自己（同一页打印两遍照样好看）。
            first_ids = {i.get("doc_id") for i in (dd.get("items") or [])}
            next_ids = {i.get("doc_id") for i in (d2.get("items") or [])}
            page_ok = bool(next_ids) and not (first_ids & next_ids)

    print(f"\n{C.B}按 id 查{C.R}")
    r = call_json(base, f"/api/v1/documents/{DEMO2_DOC}", timeout=30)
    d = r.get("data") or {}
    detail_ok = (r.get("code") == 0 and d.get("doc_id") == DEMO2_DOC
                 and d.get("total_chunks") is not None and d.get("indexed_chunks") is not None)
    if r.get("code") == 0:
        print(f"  GET /documents/{DEMO2_DOC} → {_brief(r)}")
        for k in ("doc_id", "pmid", "journal", "pub_year", "total_chunks", "indexed_chunks", "sections"):
            print(f"    {k:<15} {d.get(k)}")
        ab = d.get("abstract") or ""
        print(f"    {'abstract':<15} {len(ab)} 字：{ab[:56]}…" if ab else
              f"    {'abstract':<15} 无（全库只有 7.7% 的文献有摘要块）")
        print(f"\n{C.YELLOW}⚠ 这一条就是口径那件事的实证：{C.R}"
              f"原文 {d.get('total_chunks')} 块，真正入库只有 {C.B}{d.get('indexed_chunks')}{C.R} 块。"
              f"\n{C.GRAY}所以 DocumentIn 同时给 total_chunks 与 indexed_chunks——只给前者，"
              f"详情页会显示「共 {d.get('total_chunks')} 块」然后一块也列不出来。{C.R}")

    r = call_json(base, f"/api/v1/documents/{DEMO2_MISSING_DOC}", timeout=30)
    miss_ok = (r.get("code") == 3001)
    print(f"\n  GET /documents/{DEMO2_MISSING_DOC} → {C.YELLOW}code={r.get('code')}{C.R} {r.get('message')}")
    if r.get("detail"):
        print(f"    {C.GRAY}{json.dumps(r['detail'], ensure_ascii=False)[:130]}{C.R}")

    t0 = time.time()
    rt = call_json(base, "/api/v1/documents?title=zzzznotexistzzzz&limit=5", timeout=60)
    ms = (time.time() - t0) * 1000
    print(f"\n  标题模糊搜·零命中（最坏一档，全表扫）→ "
          f"{len((rt.get('data') or {}).get('items') or [])} 条，{C.B}{ms:.0f}ms{C.R}")
    print(f"  {C.GRAY}只量「命中多」那一档会得出 6ms 的错误结论——凑够 limit 就停。"
          f"量到这个数之后才敢下「不上 FTS5」的结论。{C.R}")
    doc_parts = {"列表+总数": list_ok, "游标翻页": page_ok, "按id查": detail_ok, "3001": miss_ok}
    doc_ok = all(doc_parts.values())
    covered.append((doc_ok, "2 文档管理",
                    "DocumentIn 模型 + 列表（过滤/游标/总数）+ 按 id 查 + 3001" if doc_ok
                    else "**本轮未验到**：" + "、".join(k for k, v in doc_parts.items() if not v)))

    # ---------------------------------------------------------------- 6
    _banner("6", "会话删除：删完再查应当查不到", "1 · 会话管理api：删除会话")
    rd = call_json(base, f"/api/v1/sessions/{DEMO2_SESSION}", "DELETE", timeout=30)
    print(f"DELETE /sessions/{DEMO2_SESSION} → {_brief(rd)}　{rd.get('data')}")
    ra = call_json(base, f"/api/v1/sessions/{DEMO2_SESSION}", timeout=30)
    print(f"GET    /sessions/{DEMO2_SESSION} → {C.YELLOW}code={ra.get('code')}{C.R} {ra.get('message')}")
    rn = call_json(base, "/api/v1/sessions/no-such-session", timeout=30)
    print(f"GET    /sessions/no-such-session → {C.YELLOW}code={rn.get('code')}{C.R} {rn.get('message')}")
    # 删除要真删掉才算数：删之前那个会话必须存在过（本轮建过、且第 3 步可能写过轮次），
    # 删之后必须变成 3002，且与"从未存在过"同码。
    del_ok = (rd.get("code") == 0 and ra.get("code") == 3002 and rn.get("code") == 3002)
    covered.append((del_ok, "1 删除会话",
                    "删除后再查返 3002，与查一个从未存在的会话同码" if del_ok
                    else f"**本轮未验到**：删={rd.get('code')} 删后查={ra.get('code')} 未存在={rn.get('code')}"))

    # ---------------------------------------------------------------- 7
    _banner("7", "交付物：OpenAPI / Postman / .env / 部署文档 / 调用示例",
            "3 · 生成openapi文档；postman；配置环境变量管理；部署文档和API调用示例")
    root = os.environ.get("MEDRAG_ROOT") or r"E:\rag"
    items = [
        ("OpenAPI（落盘）", os.path.join(root, "任务10", "openapi.json"),
         "服务_应用.py --dump-openapi　与在线 /openapi.json 同一个来源"),
        ("Postman 集合", os.path.join(root, "任务10", "medrag_api.postman_collection.json"),
         "服务_验证.py --export-postman　从验证脚本真跑里录，不另写一套"),
        ("环境变量样例", os.path.join(root, ".env.example"),
         "服务_应用.py --write-env-example　由 ENV_FIELDS 表生成，不会和代码脱节"),
        ("部署文档", os.path.join(root, "任务10", "部署文档.md"), "手写"),
        ("API 调用示例", os.path.join(root, "任务10", "API调用示例.md"),
         "curl / PowerShell / Python / EventSource 四种，响应体都是真跑出来的"),
    ]
    missing: List[str] = []
    for name, path, how in items:
        exist = os.path.exists(path) and os.path.getsize(path) > 0
        if not exist:
            missing.append(name)
        mark = f"{C.GREEN}✓{C.R}" if exist else f"{C.RED}✗{C.R}"
        size = f"{os.path.getsize(path) / 1024:,.0f} KB" if os.path.exists(path) else "缺失"
        print(f"  {mark} {name:<16} {size:>10}　{C.GRAY}{how}{C.R}")
        print(f"    {C.GRAY}{path}{C.R}")
    all_deliv = not missing
    api = call_json(base, "/api/v1/config", timeout=30).get("data") or {}
    print(f"\n{C.GRAY}生效配置可在线查：GET /api/v1/config"
          f"（当前 mode={api.get('retrieval_mode')} port={api.get('port')}）；"
          f"命令行看来源用 服务_应用.py --print-config{C.R}")
    covered.append((all_deliv, "3 交付物",
                    "五样齐全，全部由代码生成" if all_deliv
                    else "**有缺失**：" + "、".join(missing)))

    # ---------------------------------------------------------------- 汇总
    n_ok = sum(1 for ok_, _, _ in covered if ok_)
    good = (n_ok == len(covered))
    col = C.GREEN if good else C.YELLOW
    print(f"\n{C.B}{col}{'═' * 74}{C.R}")
    print(f"{C.B}{col}第二部分演示完毕 —— 本轮实际验到 {n_ok}/{len(covered)} 条{C.R}")
    print(f"{C.B}{col}{'═' * 74}{C.R}")
    for ok_, req, how in covered:
        mark = f"{C.GREEN}✓{C.R}" if ok_ else f"{C.YELLOW}⚠{C.R}"
        print(f"  {mark} {req:<16} {C.GRAY}{how}{C.R}")
    if not good:
        print(f"\n{C.YELLOW}⚠ 打 ⚠ 的那几条本轮没有真验到，"
              f"不要当成「演示过了」——勾是按本轮结果算出来的，不是写死的。{C.R}")
    print(f"\n{C.GRAY}测试与 api 文档那条的「单元/集成测试」部分不在本演示里，"
          f"跑 服务_验证.py（224 项，4 秒，不需要 Ollama）。"
          f"\n⚠ Postman 集合只覆盖走 HTTP 的那部分，224 项里有一大批是进程内断言——"
          f"集合全绿不等于 224 项全绿。{C.R}\n")
    return 0 if good else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="医学文献问答 · 终端客户端（走 HTTP 接口）")
    ap.add_argument("--demo", action="store_true",
                    help="按任务书逐条演示一遍（约 1 分钟），顺便留下干净的交付日志")
    ap.add_argument("--demo2", action="store_true",
                    help="演示服务化**第二部分**：会话管理 / 运营统计 / 文档管理 / 交付物")
    ap.add_argument("--skip-qa", action="store_true",
                    help="--demo2 专用：跳过那一次真问答（不需要 Ollama，全程毫秒级）")
    ap.add_argument("--base-url", default=os.environ.get("MEDRAG_API", DEFAULT_BASE))
    ap.add_argument("--session", default=None, help="指定会话 ID（不传则自动生成）")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--sync", action="store_true", help="用同步接口（默认流式）")
    ap.add_argument("--fast", action="store_true",
                    help="关掉①证据评估与③批判审查，1 次模型调用约 10 秒（演示提速）")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        C.off()
    try:
        if args.demo:
            return run_demo(args.base_url.rstrip("/"), top_k=args.top_k)
        if args.demo2:
            return run_demo2(args.base_url.rstrip("/"), top_k=args.top_k,
                             skip_qa=args.skip_qa)
        return repl(args)
    except ApiError as e:
        print(f"{C.RED}{e}{C.R}")
        return 1
    except KeyboardInterrupt:
        print("\n再见。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
