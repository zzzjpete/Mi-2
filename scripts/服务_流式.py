# -*- coding: utf-8 -*-
"""第十阶段（一）· 服务化 —— 流式响应（SSE）与流水线埋点

## 为什么"流式"在这个系统里不等于"逐字吐答案"

阶段七的链是**四段**：①证据评估（JSON）→②写草稿→③批判审查（JSON）→④按审查定稿。
只有②④在写自然语言，①③是结构化中间产物；④之后还要跑后处理与阶段九的层 D 校验修正。
所以：

  · 直接把模型 token 转发给客户端，客户端会先看到一版草稿，再看到它被整段重写 —— 除非
    告诉它"新的一段开始了"。本协议因此**每个 delta 都带 stage**，并在段起止各发一个
    `stage` 事件；客户端在 `answer_generator` / `final_assembler` 开始时清空缓冲区即可。
  · **最终答案以 `done` 事件为准**，不是 delta 拼接的结果。参考文献列表、免责声明、
    越界编号删除、章节标题归一都发生在模型写完之后。把拼接结果当交付物就会漏掉这些。
    这不是实现偷懒，是这条链的事实：delta 是过程，done 才是产物。
  · ①③两段一个字都不会流出来，本机实测每段几十秒。**没有心跳，中间的静默会被反向代理
    掐断**，所以队列空转时发 SSE 注释行 `: ping`（注释行不触发客户端的 message 事件）。

## 埋点方式：包一层，不改上游

三个包装器共用一条**线程局部**的事件总线 `EmitBus`，阶段七/九的代码一行不动：

    _TappedAssembler   组装完上下文就把出处清单发出去 —— 用户在模型还要跑 100 秒时
                       就能看到"将依据这 8 篇文献作答"，这是流式在本系统里最实在的价值
    StreamTapGenerator 文本段走 `/api/chat` 的 `stream: true` 逐块转发；JSON 段原样透传
                       （①③要的是完整可解析的对象，流式对它们没有意义）
    TappedPipeline     覆写 `_run_stage` / `_run_fixer` / `postprocess` 发段落与校验事件

总线按线程存这一点不是可选项：流水线是并发共享的单例，把 emit 挂成实例属性，
两个并发请求会互相串台（同步请求还会收到别人的 delta）。

⚠ 跨模块的 `LLMError`：本项目的中文文件名模块按路径导入，同一个文件被不同调用方导入会得到
**两个不相等的类**（docs/工程笔记.md 三·8 记着这条）。流式失败时必须抛**流水线那一侧认识的**那个
LLMError，否则 `_run_stage` 的 `except` 接不住。取法见 `_llm_error_class`。

用法：
    st  = _load("st", r"E:\\rag\\scripts\\服务_流式.py")
    bus = st.EmitBus()
    tap = st.StreamTapGenerator(LLMGenerator(...), bus)
    Cls = st.make_tapped_pipeline(ConstrainedGenerationPipeline)
    pipe = Cls(generator=tap, assembler=st.tapped_assembler(ContextAssembler(), bus), bus=bus)

    for ev in st.stream_generate(pipe, bus, "…", gen_kwargs={...}):
        yield st.sse(ev["type"], ev)
"""
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence  # noqa: F401

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

#: 写自然语言的两段。客户端在这两段的 `stage/start` 上清空答案缓冲区。
ANSWER_STAGES = ("answer_generator", "final_assembler")
#: 阶段九层 D 的模型修正段，键名形如 format_fixer_r1
FIXER_PREFIX = "format_fixer"

#: SSE 事件全集（写在这里，README 与验证脚本都引用它，避免文档和实现各说各的）
SSE_EVENTS: Dict[str, str] = {
    "meta": "请求受理：request_id / session_id / 原始与改写后的问题 / 检索模式 / 链路开关",
    "stage": "某一段开始或结束：stage / status(start|end) / elapsed / token 数 / 是否成功",
    "sources": "出处清单，上下文组装完成即发出（早于模型的第一个字）",
    "delta": "文本增量：stage + text。**过程量**，最终答案以 done 为准",
    "check": "阶段九层 D 的校验与修正结果",
    "done": "最终结果，载荷与同步接口的 data 字段结构一致",
    "error": "失败：code / message / detail，之后流即结束",
}


# ============================================================================
# 一、SSE 编码
# ============================================================================
def sse(event: str, data: Any) -> str:
    """一条 SSE 消息。

    `data` 里的换行必须逐行加 `data:` 前缀，否则多行内容会把消息截断——这是 SSE 最常见的
    实现错误。这里统一 `json.dumps` 成单行，从源头避免（答案里全是 `\\n`）。
    """
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False,
                                                            default=str)
    body = "\n".join(f"data: {ln}" for ln in payload.split("\n"))
    return f"event: {event}\n{body}\n\n"


def sse_comment(text: str = "ping") -> str:
    """SSE 注释行：保活用。客户端不会当成事件，但足以让代理认为连接活着。"""
    return f": {text}\n\n"


# ============================================================================
# 二、事件总线
# ============================================================================
class EmitBus:
    """线程局部的事件总线。组装器、生成器、流水线共用一个实例。

    直接当函数用（`bus(ev)`）；没 bind 过的线程调用是无操作，所以同步请求走同一批
    包装器时不会有任何额外开销，也不会误发事件。
    """

    def __init__(self) -> None:
        self._tl = threading.local()

    def bind(self, emit: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self._tl.emit = emit

    def unbind(self) -> None:
        self._tl.emit = None

    def get(self) -> Optional[Callable[[Dict[str, Any]], None]]:
        return getattr(self._tl, "emit", None)

    @property
    def active(self) -> bool:
        return self.get() is not None

    def __call__(self, ev: Dict[str, Any]) -> None:
        fn = self.get()
        if fn is None:
            return
        try:
            fn(ev)
        except Exception:            # 埋点绝不能弄坏主流程
            pass


# ============================================================================
# 三、上下文组装器埋点
# ============================================================================
def _source_row(c: Dict[str, Any]) -> Dict[str, Any]:
    pmcid = c.get("pmcid") or ""
    return {"marker": c.get("marker", ""), "pmcid": pmcid, "pmid": c.get("pmid") or "",
            "title": c.get("title") or "", "journal": c.get("journal") or "",
            "pub_year": c.get("pub_year"), "section": c.get("section") or "",
            "chunk_id": c.get("chunk_id") or "",
            "relevance_score": c.get("relevance_score"),
            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else ""}


class _TappedAssembler:
    """透明代理：组装完就把出处清单发出去，其余全部转发给真组装器。

    ⚠ 不能给它加 `__len__`：阶段七写的是 `assembler or ContextAssembler(...)`，
    一个"空即为假"的对象会被 `or` 悄悄换掉（阶段八在 `CachedLLMGenerator` 上踩过同类坑）。
    这里没有 `__len__`，恒为真值，安全。
    """

    def __init__(self, inner: Any, bus: EmitBus):
        self.__dict__["inner"] = inner
        self.__dict__["bus"] = bus

    def __getattr__(self, name: str) -> Any:
        try:
            inner = self.__dict__["inner"]
        except KeyError:                       # __init__ 还没跑完就被访问
            raise AttributeError(name)
        return getattr(inner, name)

    def assemble_context(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        ctx = self.__dict__["inner"].assemble_context(*args, **kwargs)
        bus = self.__dict__["bus"]
        if bus.active:
            meta = ctx.get("metadata") or {}
            cits = meta.get("citations") or []
            bus({"type": "sources", "count": len(cits),
                 "sources": [_source_row(c) for c in cits],
                 "context_tokens": meta.get("estimated_tokens")})
        return ctx


def tapped_assembler(inner: Any, bus: EmitBus) -> Any:
    return _TappedAssembler(inner, bus)


class _TappedRetriever:
    """透明代理：检索一完成，就把**真正拿去检索的那句话**发出去。

    中文问题在检索层被译成英文。译坏了的表现是"证据全不相关"——在响应体里，
    它和"库里真没有"长得一模一样。把检索式发给客户端，等待期里人就能自己判断，
    不用等三十秒后读一句"根据现有文献无法回答"再猜是哪种。

    ⚠ 与 `_TappedAssembler` 同理：**不要给它加 `__len__`**。
    """

    def __init__(self, inner: Any, bus: EmitBus):
        self.__dict__["inner"] = inner
        self.__dict__["bus"] = bus

    def __getattr__(self, name: str) -> Any:
        try:
            inner = self.__dict__["inner"]
        except KeyError:                       # __init__ 还没跑完就被访问
            raise AttributeError(name)
        return getattr(inner, name)

    def search(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        out = self.__dict__["inner"].search(*args, **kwargs)
        bus = self.__dict__["bus"]
        if bus.active and isinstance(out, dict):
            eq = out.get("enhanced")
            bus({"type": "stage", "stage": "retrieval", "status": "end",
                 "is_answer_stage": False, "ok": bool(out.get("results")),
                 "hits": len(out.get("results") or []),
                 "retrieval_query": getattr(eq, "core_text", None),
                 "translated_from": getattr(eq, "translated_from", None)})
        return out


def tapped_retriever(inner: Any, bus: EmitBus) -> Any:
    return _TappedRetriever(inner, bus)


# ============================================================================
# 四、生成器埋点（文本段流式）
# ============================================================================
def _llm_error_class(inner: Any) -> type:
    """拿到 `inner` **所属那个模块**里的 LLMError。

    不能自己再 import 一遍：中文文件名按路径导入会产出不相等的类，流水线的
    `except LLMError` 就接不住我们抛的异常（docs/工程笔记.md 三·8 记过这条坑）。
    任意一个定义在该模块里的函数，其 `__globals__` 就是该模块的命名空间——从这里取最准。
    """
    try:
        cls = type(inner).__init__.__globals__.get("LLMError")
        return cls if isinstance(cls, type) else RuntimeError
    except Exception:
        return RuntimeError


class StreamTapGenerator:
    """包住阶段七的 `LLMGenerator`：文本段流式转发，JSON 段原样透传。

    同一个实例同时服务同步请求与流式请求：**总线没 bind 的线程走非流式**，
    所以流水线单例不必为两种模式各建一条。
    """

    def __init__(self, inner: Any, bus: Optional[EmitBus] = None, enable: bool = True):
        self.__dict__["inner"] = inner
        self.__dict__["bus"] = bus if bus is not None else EmitBus()
        self.__dict__["enable"] = bool(enable)
        self.__dict__["_tl"] = threading.local()
        self.__dict__["_err_cls"] = _llm_error_class(inner)
        self.__dict__["stream_stats"] = {"streamed_calls": 0, "delegated_calls": 0,
                                         "chunks": 0, "stream_failures": 0}

    # ---------------- 当前段（delta 要打这个标；按线程存）----------------
    @property
    def current_stage(self) -> str:
        return getattr(self.__dict__["_tl"], "stage", "") or ""

    @current_stage.setter
    def current_stage(self, v: str) -> None:
        self.__dict__["_tl"].stage = v or ""

    # ---------------- 透明代理 ----------------
    def __getattr__(self, name: str) -> Any:
        try:
            inner = self.__dict__["inner"]
        except KeyError:
            raise AttributeError(name)
        return getattr(inner, name)

    def generate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """单段调用（追问改写用的就是它）——不流式，直接透传。"""
        return self.__dict__["inner"].generate(*args, **kwargs)

    def generate_messages(self, messages: Sequence[Dict[str, str]],
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          json_output: bool = False, expect: str = "any",
                          options: Optional[Dict[str, Any]] = None,
                          retries: int = 1, json_retries: int = 1) -> Dict[str, Any]:
        inner = self.__dict__["inner"]
        bus = self.__dict__["bus"]
        # JSON 段不流式：①③要的是完整对象，逐块转发只会让客户端看到半截 JSON
        if json_output or not self.__dict__["enable"] or not bus.active:
            self.__dict__["stream_stats"]["delegated_calls"] += 1
            return inner.generate_messages(messages, temperature=temperature,
                                           max_tokens=max_tokens, json_output=json_output,
                                           expect=expect, options=options, retries=retries,
                                           json_retries=json_retries)
        objs = self._chunk_source(messages, temperature, max_tokens, options)
        return self._consume(objs, bus)

    # ---------------- 增量来源 ----------------
    def _chunk_source(self, messages: Sequence[Dict[str, str]],
                      temperature: Optional[float], max_tokens: Optional[int],
                      options: Optional[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """产出 Ollama NDJSON 形状的对象流。

        内层生成器若自带 `stream_messages`（返回同样形状的迭代器）就用它，否则走 HTTP。
        这个扩展点不是为测试硬凑的：它让"换一个生成后端"只需实现一个方法，
        顺带也让验证脚本能在**没有 Ollama** 的情况下把整条流式链路真跑一遍。
        """
        inner = self.__dict__["inner"]
        opts = inner.build_options(temperature, max_tokens, **(options or {}))
        own = getattr(inner, "stream_messages", None)
        if callable(own):
            return own([dict(m) for m in messages], options=opts)
        return self._http_objects([dict(m) for m in messages], opts)

    def _http_objects(self, messages: List[Dict[str, str]],
                      opts: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """`/api/chat` + `stream:true`，逐行 NDJSON。"""
        inner = self.__dict__["inner"]
        payload = {"model": inner.model_name, "messages": messages,
                   "stream": True, "options": opts, "think": inner.think,
                   "keep_alive": inner.keep_alive}
        req = urllib.request.Request(
            f"{inner.base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=inner.timeout) as resp:
                for raw in resp:                       # Ollama 每行一个 JSON 对象
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue                       # 半行/心跳，跳过
        except urllib.error.HTTPError as e:
            self.__dict__["stream_stats"]["stream_failures"] += 1
            detail = e.read().decode("utf-8", "replace")[:300]
            raise self.__dict__["_err_cls"](f"Ollama HTTP {e.code}（流式）：{detail}") from e
        except urllib.error.URLError as e:
            self.__dict__["stream_stats"]["stream_failures"] += 1
            raise self.__dict__["_err_cls"](
                f"连不上 Ollama（{inner.base_url}，流式）：{e.reason}") from e
        except TimeoutError as e:
            self.__dict__["stream_stats"]["stream_failures"] += 1
            raise self.__dict__["_err_cls"](f"Ollama 流式请求超时（>{inner.timeout}s）") from e

    # ---------------- 消费增量，拼回阶段七同构的返回值 ----------------
    def _consume(self, objs: Iterator[Dict[str, Any]], bus: EmitBus) -> Dict[str, Any]:
        """返回值与阶段七非流式调用**同构**，所以 `_run_stage` 不需要知道这次是不是流式。"""
        inner = self.__dict__["inner"]
        stage = self.current_stage
        t0 = time.time()
        parts: List[str] = []
        thinking_chars = 0
        first_token_at: Optional[float] = None
        tail: Dict[str, Any] = {}
        for obj in objs:
            msg = obj.get("message") or {}
            piece = msg.get("content") or ""
            thinking_chars += len(msg.get("thinking") or "")
            if piece:
                if first_token_at is None:
                    first_token_at = time.time()
                parts.append(piece)
                self.__dict__["stream_stats"]["chunks"] += 1
                bus({"type": "delta", "stage": stage, "text": piece})
            if obj.get("done"):
                tail = obj

        text = "".join(parts)
        el = time.time() - t0
        self.__dict__["stream_stats"]["streamed_calls"] += 1
        # 同步阶段七的累计用量：报告里的 token 合计要把流式这几段也算进去
        st = getattr(inner, "stats", None)
        if isinstance(st, dict):
            st["calls"] = st.get("calls", 0) + 1
            st["total_seconds"] = st.get("total_seconds", 0.0) + el
            st["prompt_tokens"] = st.get("prompt_tokens", 0) + (tail.get("prompt_eval_count") or 0)
            st["output_tokens"] = st.get("output_tokens", 0) + (tail.get("eval_count") or 0)
        return {
            "text": text.strip(), "raw": text, "thinking_chars": thinking_chars,
            "elapsed": round(el, 3), "model": tail.get("model", inner.model_name),
            "done_reason": tail.get("done_reason", ""),
            "truncated": tail.get("done_reason") == "length",
            "prompt_eval_count": tail.get("prompt_eval_count"),
            "eval_count": tail.get("eval_count"),
            "tokens_per_second": (round(tail["eval_count"] / (tail["eval_duration"] / 1e9), 1)
                                  if tail.get("eval_count") and tail.get("eval_duration") else None),
            "first_token_seconds": (round(first_token_at - t0, 3) if first_token_at else None),
            "streamed": True,
            "json": None, "json_ok": None, "json_error": "", "json_note": "",
        }


# ============================================================================
# 五、流水线埋点子类
# ============================================================================
_TAPPED_CACHE: Dict[type, type] = {}


def make_tapped_pipeline(base_cls: type) -> type:
    """按基类生成一个"会发事件"的流水线子类（同一基类只生成一次，缓存起来）。

    覆写三处，全部是"先/后加一句 emit"，业务逻辑一律交回 super()：
      `_run_stage`   段起止事件 + 告诉生成器当前在哪一段（delta 要打这个标）
      `_run_fixer`   阶段九层 D 的模型修正段（它不走 `_run_stage`，得单独埋）
      `postprocess`  校验结果事件
    """
    if base_cls in _TAPPED_CACHE:
        return _TAPPED_CACHE[base_cls]

    class TappedPipeline(base_cls):                        # type: ignore[misc, valid-type]
        """带 SSE 埋点的流水线。事件总线按线程存——单例被并发共享。"""

        def __init__(self, *a: Any, bus: Optional[EmitBus] = None, **kw: Any):
            self.bus = bus if bus is not None else EmitBus()
            super().__init__(*a, **kw)

        def _run_stage(self, key: str, values: Dict[str, Any],
                       metrics: Dict[str, Any]) -> Dict[str, Any]:
            gen = self.generator
            if hasattr(gen, "current_stage"):
                gen.current_stage = key
            self.bus({"type": "stage", "stage": key, "status": "start",
                      "is_answer_stage": key in ANSWER_STAGES})
            r = super()._run_stage(key, values, metrics)
            self.bus({"type": "stage", "stage": key, "status": "end",
                      "is_answer_stage": key in ANSWER_STAGES,
                      "ok": bool(r.get("ok")),
                      "elapsed": (metrics.get("stage_times") or {}).get(key),
                      "prompt_tokens": r.get("prompt_eval_count"),
                      "output_tokens": r.get("eval_count"),
                      "json_ok": r.get("json_ok"),
                      "error": (r.get("error") or "")[:200]})
            return r

        def _run_fixer(self, values: Dict[str, Any], n: int) -> Dict[str, Any]:
            """阶段九层 D 的模型修正段。基类没有这个方法时它永远不会被调到。"""
            key = f"{FIXER_PREFIX}_r{n}"
            gen = self.generator
            if hasattr(gen, "current_stage"):
                gen.current_stage = key
            self.bus({"type": "stage", "stage": key, "status": "start",
                      "is_answer_stage": False, "note": "层D 按违规清单重写"})
            r = super()._run_fixer(values, n)              # type: ignore[misc]
            self.bus({"type": "stage", "stage": key, "status": "end",
                      "is_answer_stage": False, "ok": bool(r.get("ok")),
                      "prompt_tokens": r.get("prompt_eval_count"),
                      "output_tokens": r.get("eval_count")})
            return r

        def postprocess(self, answer: str, citations: Any, reference_list: str) -> Dict[str, Any]:
            out = super().postprocess(answer, citations, reference_list)
            chk = out.get("constraint_check")
            if chk:
                self.bus({"type": "check", "compliant": bool(chk.get("compliant")),
                          "violations": [{"code": v.get("code"), "severity": v.get("severity"),
                                          "message": str(v.get("message", ""))[:200]}
                                         for v in (chk.get("violations") or [])],
                          "scores": chk.get("scores") or {}})
            return out

    TappedPipeline.__name__ = f"Tapped{base_cls.__name__}"
    TappedPipeline.__qualname__ = TappedPipeline.__name__
    _TAPPED_CACHE[base_cls] = TappedPipeline
    return TappedPipeline


# ============================================================================
# 六、跑一次流式生成
# ============================================================================
_SENTINEL = object()


def stream_generate(pipeline: Any, bus: EmitBus, query: str,
                    gen_kwargs: Optional[Dict[str, Any]] = None,
                    heartbeat_seconds: float = 10.0,
                    on_result: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
                    ) -> Iterator[Dict[str, Any]]:
    """在后台线程跑整条链，把埋点事件按发生顺序 yield 出来。

    最后一定会 yield 一个 `done` 或 `raised`；队列空转超过 `heartbeat_seconds`
    就 yield 一个 `heartbeat`（路由层把它编码成 SSE 注释行）。

    `on_result` 拿到流水线原始结果，返回要放进 `done` 的载荷——"结果怎么转成响应体"
    是路由层的事，本模块只管把事件按顺序送出来。
    """
    q: "queue.Queue[Any]" = queue.Queue()          # 不设上限：消费慢也不该卡住生成

    def emit(ev: Dict[str, Any]) -> None:
        q.put(ev)

    def work() -> None:
        try:
            bus.bind(emit)
            res = pipeline.generate(query, **(gen_kwargs or {}))
            q.put({"type": "result", "result": res})
        except BaseException as e:                 # 线程里任何异常都要变成事件送出去
            q.put({"type": "raised", "exc": e})
        finally:
            bus.unbind()
            q.put(_SENTINEL)

    th = threading.Thread(target=work, name="stream-generate", daemon=True)
    th.start()

    while True:
        try:
            ev = q.get(timeout=max(1.0, float(heartbeat_seconds)))
        except queue.Empty:
            yield {"type": "heartbeat", "ts": round(time.time(), 3)}
            continue
        if ev is _SENTINEL:
            break
        kind = ev.get("type")
        if kind == "result":
            payload = on_result(ev["result"]) if on_result else ev["result"]
            yield {"type": "done", **(payload if isinstance(payload, dict)
                                      else {"result": payload})}
            continue
        yield ev
    th.join(timeout=1.0)
