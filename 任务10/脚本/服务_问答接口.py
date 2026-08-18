# -*- coding: utf-8 -*-
"""第十阶段（一）· 服务化 —— 问答接口（同步 / 流式）、会话与调用记录

## 同步与流式共用同一条准备流程

    校验参数 → 关联会话（有 session_id 就取历史、判追问、改写）→ 选证据
    → 占并发名额 → 跑流水线 → 组装 AskData → 落库（会话轮次 + 调用记录）

两条路径的差别**只有输出方式**：同步等完再一次性返回，流式把中间事件按发生顺序推出去。
业务字段完全一致——`done` 事件的载荷就是同步接口 `data` 的那个结构。这不是巧合，
是刻意的：让客户端可以在两种模式间切换而不用改解析代码。

## 三个容易做错的地方

**1. 会话历史不能直接塞进提示词。**
把上一轮的问答原文拼进上下文，等于给模型一段"没有 [S#] 出处的事实材料"，
它会照着写——阶段九花一整个阶段压下去的编造，会从这个口子回来。
所以历史**只用来改写检索式**（见 `服务_会话.FollowupRewriter`），一个字都不进证据区。

**2. 拒答不是错误。**
证据不足时系统会返回一段带固定拒答短语的结构化回答，这是**正确行为**（阶段九量到
拒答率 0 → 1.0 就是靠它）。所以它走 HTTP 200 / code 0，只在 `data.refused` 标一个位，
并在调用记录里单独统计拒答率。把它做成 4xx 会让"守住边界"变成接口层的一次失败。

**3. 流式请求的耗时不能靠中间件量。**
中间件在响应头发出那一刻就停表了，SSE 正文还要再流一两分钟。所以流式的记录由流生成器
自己在 `finally` 里写——**并发名额也在那里释放**。

用法（被 `服务_应用.create_app()` 调用）：
    router = build_router(state)      # state 是 ServiceState
"""
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Path, Query, Request
from fastapi.responses import StreamingResponse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(mod_name: str, filename: str):
    """按路径导入并**登记进 sys.modules**。为什么必须登记见 `服务_模型.py` 里的同名函数：
    本文件 `raise SessionNotFound(...)` 与 `服务_应用` 注册的异常处理器必须用**同一个**
    APIError 类，否则处理器匹配不上，3002 会变成 5001。"""
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


_err = _load_by_path("fuwu_cuowuma", "服务_错误码.py")
_mdl = _load_by_path("fuwu_moxing", "服务_模型.py")
_lg = _load_by_path("fuwu_rizhi", "服务_日志.py")
_ss = _load_by_path("fuwu_huihua", "服务_会话.py")
_st = _load_by_path("fuwu_liushi", "服务_流式.py")
_cp = _load_by_path("yueshu_tishicing", "约束_提示词层.py")

ErrorCode = _err.ErrorCode
APIError = _err.APIError
SessionNotFound = _err.SessionNotFound
NotFoundError = _err.NotFoundError

ResponseModel = _mdl.ResponseModel
PageModel = _mdl.PageModel
AskRequest = _mdl.AskRequest
AskData = _mdl.AskData
SourceItem = _mdl.SourceItem
CitationCheck = _mdl.CitationCheck
ConstraintCheck = _mdl.ConstraintCheck
GenerationMetrics = _mdl.GenerationMetrics
SessionData = _mdl.SessionData
SessionTurn = _mdl.SessionTurn
SessionCreateRequest = _mdl.SessionCreateRequest
CallLogItem = _mdl.CallLogItem
CallStats = _mdl.CallStats

#: 阶段九那句固定拒答短语。`refused` 判定与阶段九的校验器口径一致（精确匹配）
REFUSAL_PHRASE = _cp.REFUSAL_PHRASE
PARTIAL_PHRASE = _cp.PARTIAL_PHRASE

log = _lg.get_logger("qa")


# ============================================================================
# 一、结果 → 响应模型
# ============================================================================
def _sources(res: Dict[str, Any]) -> List[SourceItem]:
    out = []
    for s in res.get("sources") or []:
        try:
            out.append(SourceItem(**{k: v for k, v in s.items()
                                     if k in SourceItem.model_fields}))
        except Exception:                       # 元数据残缺不该让整个响应失败
            out.append(SourceItem(marker=str(s.get("marker", "?"))))
    return out


def _constraint(res: Dict[str, Any]) -> Optional[ConstraintCheck]:
    chk = res.get("constraint_check")
    if not chk:
        return None
    rep = res.get("constraint_repair") or {}
    rounds = rep.get("rounds") or []
    return ConstraintCheck(
        compliant=bool(chk.get("compliant")),
        violations=[{"code": v.get("code"), "severity": v.get("severity"),
                     "message": str(v.get("message", ""))[:300]}
                    for v in (chk.get("violations") or [])],
        scores=chk.get("scores") or {},
        repaired=bool(rounds),
        repair_rounds=len(rounds))


def _metrics(res: Dict[str, Any]) -> GenerationMetrics:
    gm = res.get("generation_metrics") or {}
    return GenerationMetrics(
        total_time_seconds=float(gm.get("total_time_seconds") or 0.0),
        llm_calls=int(gm.get("llm_calls") or 0),
        total_prompt_tokens=int(gm.get("total_prompt_tokens") or 0),
        total_output_tokens=int(gm.get("total_output_tokens") or 0),
        context_tokens=int(gm.get("context_tokens") or 0),
        answer_chars=int(gm.get("answer_chars") or 0),
        stage_times=gm.get("stage_times") or {},
        stage_success=gm.get("stage_success") or {},
        chain=gm.get("chain") or {})


def build_ask_data(res: Dict[str, Any], *, request_id: str, prep: Dict[str, Any],
                   include_intermediate: bool) -> AskData:
    """流水线原始结果 → `AskData`。同步接口与流式 `done` 事件**共用这一个函数**。"""
    answer = res.get("answer") or ""
    cc = res.get("citation_check")
    # 拒答三态优先读阶段九校验器的判定；关掉约束层时退回按短语判（那时只有两态）。
    # ⚠ `refused` 只表示**完全拒答**：把"部分作答"并进来，统计里就再也看不出
    #   "首句说没答案、正文其实答了"这一类——而那正是最误导读者的形态。
    _state = ((res.get("constraint_check") or {}).get("refusal") or {}).get("state")
    refused = (_state == "full_refusal") if _state else (REFUSAL_PHRASE in answer)
    partial = (_state == "partial") if _state else (PARTIAL_PHRASE in answer)
    return AskData(
        request_id=request_id,
        session_id=prep.get("session_id"),
        query=prep["query"],
        resolved_query=prep["resolved_query"],
        retrieval_query=(res.get("generation_metrics") or {}).get("retrieval_query"),
        rewritten=bool(prep.get("rewritten")),
        history_used=int(prep.get("history_used") or 0),
        answer=answer,
        refused=refused,
        partial=partial,
        retrieval_mode=prep["retrieval_mode"],
        evidence_pool=prep.get("evidence_pool"),
        sources=_sources(res),
        citation_check=CitationCheck(**cc) if cc else None,
        constraint_check=_constraint(res),
        metrics=_metrics(res),
        intermediate=(res.get("intermediate_results") if include_intermediate else None))


# ============================================================================
# 二、共用准备流程
# ============================================================================
def prepare(state: Any, req: AskRequest, request_id: str) -> Dict[str, Any]:
    """关联会话 → 追问改写 → 选证据 → 组好 `pipeline.generate()` 的参数。

    返回的 dict 里 `gen_kwargs` 直接喂给流水线，其余字段进响应体。
    """
    s = state.settings
    prep: Dict[str, Any] = {
        "query": req.query, "resolved_query": req.query, "rewritten": False,
        "session_id": req.session_id, "history_used": 0,
        "retrieval_mode": s.retrieval_mode, "evidence_pool": None,
        "pool_match": None, "rewrite": None,
    }

    # ---- 会话：未知 id 自动建（理由见 服务_会话 模块 docstring）----
    history: List[Dict[str, Any]] = []
    if req.session_id:
        _, created = state.sessions.ensure(req.session_id)
        history = state.sessions.history(req.session_id, max_turns=req.history_turns)
        prep["history_used"] = len(history)
        prep["session_created"] = created

    # ---- 追问改写：只改检索式与提问，历史绝不进证据区 ----
    prompt_query = req.query
    if history:
        mode = req.rewrite or s.rewrite_mode
        rw = state.rewriter.rewrite(req.query, history, mode=mode)
        prep.update(resolved_query=rw["resolved"], rewritten=bool(rw["rewritten"]),
                    rewrite={"mode": rw["mode_used"], "note": rw["note"],
                             "seconds": rw["seconds"],
                             "detect": (rw.get("detect") or {}).get("reason")})
        prompt_query = rw["prompt_query"]
        if rw["rewritten"]:
            log.info("追问改写（%s）：%r → %r", rw["mode_used"], req.query, rw["resolved"])

    # ---- 证据 ----
    gen_kwargs: Dict[str, Any] = {"evaluate": req.evaluate, "review": req.review}
    if s.constrained:
        gen_kwargs["expect_refusal"] = req.expect_refusal
    if s.retrieval_mode == "snapshot":
        if state.snapshot is None:
            raise APIError(ErrorCode.INDEX_NOT_READY,
                           "快照证据不可用，且未启用 live 检索",
                           detail={"snapshot_path": s.snapshot_path})
        docs, pool_id, how = state.snapshot.pick(prep["resolved_query"],
                                                 req.evidence_pool, req.top_k)
        gen_kwargs["retrieved_docs"] = docs
        prep.update(evidence_pool=pool_id, pool_match=how)
    else:
        gen_kwargs["top_k"] = req.top_k

    prep["gen_kwargs"] = gen_kwargs
    prep["prompt_query"] = prompt_query
    return prep


def finish(state: Any, req: AskRequest, prep: Dict[str, Any], data: Optional[AskData],
           request_id: str, mode: str, elapsed_ms: float, *,
           status: str = "ok", code: int = 0, http_status: int = 200,
           error: str = "", path: str = "", method: str = "POST") -> None:
    """收尾：写会话轮次 + 写调用记录。**任何一步失败都不许连累已经产出的答案。**"""
    try:
        if req.session_id and data is not None:
            state.sessions.append_turn(req.session_id, "user", req.query,
                                       request_id=request_id)
            state.sessions.append_turn(req.session_id, "assistant", data.answer,
                                       request_id=request_id,
                                       meta={"refused": data.refused,
                                             "sources": len(data.sources)})
    except Exception as e:
        log.error("写会话轮次失败（不影响本次回答）：%s", e)

    m = data.metrics if data else None
    state.calls.record(
        request_id=request_id, method=method,
        path=path or ("/api/v1/qa/ask" if mode == "sync" else "/api/v1/qa/stream"),
        mode=mode,
        session_id=req.session_id, query=req.query, top_k=req.top_k,
        status=status, code=int(code), http_status=int(http_status),
        elapsed_ms=round(elapsed_ms, 2),
        llm_calls=(m.llm_calls if m else None),
        prompt_tokens=(m.total_prompt_tokens if m else None),
        output_tokens=(m.total_output_tokens if m else None),
        answer_chars=(m.answer_chars if m else None),
        sources=(len(data.sources) if data else None),
        refused=(data.refused if data else None),
        compliant=(data.constraint_check.compliant if data and data.constraint_check else None),
        error=error[:1000])


# ============================================================================
# 三、路由
# ============================================================================
def build_router(state: Any, guard: Optional[Any] = None) -> APIRouter:
    """`guard` 由 `服务_应用` 注入（鉴权 + 限流）。

    ⚠ 不能写成 `sys.modules.get("服务_应用")` 去反查：本项目的中文文件名模块是
    `spec_from_file_location` + `exec_module` 加载的，**根本不会注册进 sys.modules**，
    反查恒为 None——鉴权会静默失效而没有任何报错。依赖必须显式传进来。
    """
    router = APIRouter()

    def _guard(request: Request) -> None:
        if guard is not None:
            guard(request)

    def _rid(request: Request) -> str:
        return getattr(request.state, "request_id", "") or _lg.get_request_id()

    # ------------------------------------------------------------------
    # 3.1 同步问答
    # ------------------------------------------------------------------
    @router.post("/qa/ask", tags=["问答"], summary="同步问答",
                 response_model=ResponseModel[AskData],
                 description="等整条链跑完再一次性返回。本机四段链 + 层 D 校验，"
                             "**单次 100 秒以上是常态**，客户端超时请设到 300 秒以上。")
    def ask(request: Request, req: AskRequest = Body(...)) -> Any:
        _guard(request)
        rid = _rid(request)
        t0 = time.perf_counter()
        prep = prepare(state, req, rid)
        pipe = state.get_pipeline()             # 首次会加载，之后是常数时间

        with state.slot():                      # 并发闸门，见 服务_应用 模块 docstring
            try:
                res = pipe.generate(prep["prompt_query"], **prep["gen_kwargs"])
            except Exception as e:
                api = _err.classify_exception(e)
                el = (time.perf_counter() - t0) * 1000
                finish(state, req, prep, None, rid, "sync", el, status="error",
                       code=int(api.code), http_status=api.http_status,
                       error=f"{type(e).__name__}: {e}")
                raise api

        el = (time.perf_counter() - t0) * 1000
        data = build_ask_data(res, request_id=rid, prep=prep,
                              include_intermediate=req.include_intermediate)
        finish(state, req, prep, data, rid, "sync", el)
        log.info("同步问答完成｜%d 字｜%d 次调用｜拒答=%s｜%.1fms",
                 data.metrics.answer_chars, data.metrics.llm_calls, data.refused, el)
        return ResponseModel[AskData].ok(data, request_id=rid, elapsed_ms=round(el, 2))

    # ------------------------------------------------------------------
    # 3.2 流式问答
    # ------------------------------------------------------------------
    def _stream_response(request: Request, req: AskRequest) -> StreamingResponse:
        _guard(request)
        rid = _rid(request)
        t0 = time.perf_counter()
        prep = prepare(state, req, rid)
        pipe = state.get_pipeline()

        # **必须在开流之前抢名额**：SSE 一旦开始，状态行已发出，再"忙"就只能是流里的
        # 一个事件、HTTP 仍然 200。抢不到就在这里干干净净地返回 503。
        state.acquire_slot()

        def events():
            data: Optional[AskData] = None
            status, code, http_status, err = "ok", 0, 200, ""
            try:
                yield _st.sse("meta", {
                    "request_id": rid, "session_id": req.session_id,
                    "query": req.query, "resolved_query": prep["resolved_query"],
                    "rewritten": prep["rewritten"], "history_used": prep["history_used"],
                    "retrieval_mode": prep["retrieval_mode"],
                    "evidence_pool": prep.get("evidence_pool"),
                    "pool_match": prep.get("pool_match"),
                    "chain": {"evaluate": req.evaluate, "review": req.review,
                              "constrained": state.settings.constrained},
                    "model": state.settings.model_name,
                    "note": "最终答案以 done 事件为准；delta 是过程量",
                })
                for ev in _st.stream_generate(
                        pipe, state.bus, prep["prompt_query"],
                        gen_kwargs=prep["gen_kwargs"],
                        heartbeat_seconds=state.settings.heartbeat_seconds,
                        on_result=lambda r: {"data": json.loads(build_ask_data(
                            r, request_id=rid, prep=prep,
                            include_intermediate=req.include_intermediate).model_dump_json())}):
                    kind = ev.get("type")
                    if kind == "heartbeat":
                        yield _st.sse_comment("ping")
                    elif kind == "raised":
                        api = _err.classify_exception(ev["exc"])
                        status, code, http_status = "error", int(api.code), api.http_status
                        err = f"{type(ev['exc']).__name__}: {ev['exc']}"
                        log.error("流式生成失败：%s", err)
                        yield _st.sse("error", {"code": code, "message": api.message,
                                                "detail": api.detail, "request_id": rid})
                    elif kind == "done":
                        data = AskData(**ev["data"])
                        yield _st.sse("done", {"code": 0, "message": "ok", "data": ev["data"],
                                               "request_id": rid,
                                               "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)})
                    else:
                        yield _st.sse(kind, ev)
            except GeneratorExit:               # 客户端断开
                status, err = "error", "客户端断开连接"
                log.warning("流式请求被客户端中断：%s", rid)
                raise
            except Exception as e:              # 兜底：已经开流了，只能把错误写进流里
                api = _err.classify_exception(e)
                status, code, http_status = "error", int(api.code), api.http_status
                err = f"{type(e).__name__}: {e}"
                log.exception("流式接口异常")
                yield _st.sse("error", {"code": code, "message": api.message,
                                        "request_id": rid})
            finally:
                # 名额与记录都在这里收尾：流式的真实耗时中间件量不到（见模块 docstring）
                state.release_slot()
                el = (time.perf_counter() - t0) * 1000
                # ⚠ 这一行不能省：中间件记的那条 access 日志是**响应头发出的时刻**
                # （流式通常只有个位数毫秒），按 request_id grep 日志会看到一次"1.5ms 的问答"，
                # 完全不知道它实际跑了多久、结果如何。补一行收尾日志，让 grep 得到完整故事。
                if data is not None:
                    log.info("流式问答完成｜%d 字｜%d 次调用｜拒答=%s｜%.1fms",
                             data.metrics.answer_chars, data.metrics.llm_calls,
                             data.refused, el)
                else:
                    log.warning("流式问答结束但没有产出结果｜status=%s code=%s｜%.1fms",
                                status, code, el)
                try:
                    finish(state, req, prep, data, rid, "stream", el,
                           status=status, code=code, http_status=http_status, error=err,
                           path=str(request.url.path), method=request.method)
                except Exception as e2:
                    log.error("流式收尾记录失败：%s", e2)

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform",
                     "Connection": "keep-alive",
                     "X-Accel-Buffering": "no",          # 关掉 nginx 缓冲，否则流会攒着发
                     "X-Request-Id": rid})

    @router.post("/qa/stream", tags=["问答"], summary="流式问答（SSE）",
                 description="事件：" + "；".join(f"`{k}` {v}" for k, v in _st.SSE_EVENTS.items()))
    def ask_stream(request: Request, req: AskRequest = Body(...)) -> Any:
        return _stream_response(request, req)

    @router.get("/qa/stream", tags=["问答"], summary="流式问答（SSE，GET 版）",
                description="给浏览器 `EventSource` 用——它只会发 GET，不能带请求体。"
                            "参数语义与 POST 版完全一致。")
    def ask_stream_get(
            request: Request,
            query: str = Query(..., description="用户问题"),
            top_k: int = Query(10, ge=_mdl.TOP_K_MIN, le=_mdl.TOP_K_MAX),
            session_id: Optional[str] = Query(None, max_length=_mdl.SESSION_ID_MAX),
            evaluate: bool = Query(True), review: bool = Query(True),
            evidence_pool: Optional[str] = Query(None),
            history_turns: int = Query(4, ge=0, le=_mdl.HISTORY_TURNS_MAX)) -> Any:
        req = AskRequest(query=query, top_k=top_k, session_id=session_id,
                         evaluate=evaluate, review=review, evidence_pool=evidence_pool,
                         history_turns=history_turns)
        return _stream_response(request, req)

    @router.get("/qa/topics", tags=["问答"], summary="有真实证据的主题（快照模式）",
                description="快照模式下只有固定几组真实检索结果。把清单公布出去，"
                            "客户端可以在**发请求之前**判断这个问题问不问得出东西——"
                            "否则要等二十几秒才知道答不了，而服务端第 0 秒就知道了。"
                            "live 模式下返回空列表（任何问题都会现检索）。")
    def topics(request: Request,
               probe: Optional[str] = Query(None, description="试探一个问题会命中哪一组")) -> Any:
        _guard(request)
        snap = state.snapshot
        mode = state.settings.retrieval_mode
        data: Dict[str, Any] = {"retrieval_mode": mode,
                                "constrained": bool(state.settings.constrained),
                                "topics": snap.topics() if snap else []}
        if probe and snap:
            pool, score = snap.match_score(probe)
            data["probe"] = {"query": probe, "matched": pool, "score": score,
                             "in_scope": pool is not None}
        return ResponseModel.ok(data, request_id=_rid(request))

    # ------------------------------------------------------------------
    # 3.3 调用记录与统计
    # ------------------------------------------------------------------
    @router.get("/qa/logs", tags=["统计"], summary="调用记录（分页）",
                response_model=ResponseModel[PageModel[CallLogItem]])
    def logs(request: Request,
             page: int = Query(1, ge=_mdl.PAGE_MIN),
             page_size: int = Query(20, ge=_mdl.PAGE_SIZE_MIN, le=_mdl.PAGE_SIZE_MAX),
             status: Optional[str] = Query(None, description="ok / error / busy"),
             mode: Optional[str] = Query(None, description="sync / stream"),
             session_id: Optional[str] = Query(None),
             since_hours: Optional[float] = Query(None, ge=0)) -> Any:
        _guard(request)
        rows, total = state.calls.page(page, page_size, status=status, mode=mode,
                                       session_id=session_id, since_hours=since_hours)
        items = [CallLogItem(**{k: v for k, v in r.items() if k in CallLogItem.model_fields})
                 for r in rows]
        return ResponseModel[PageModel[CallLogItem]].ok(
            PageModel[CallLogItem].build(items, total, page, page_size),
            request_id=_rid(request))

    @router.get("/qa/logs/{request_id}", tags=["统计"], summary="按请求 ID 查单条记录",
                response_model=ResponseModel[CallLogItem])
    def log_one(request: Request, request_id: str = Path(..., max_length=64)) -> Any:
        _guard(request)
        row = state.calls.get(request_id)
        if not row:
            raise NotFoundError(ErrorCode.RECORD_NOT_FOUND,
                                f"没有 request_id={request_id} 的调用记录",
                                detail={"request_id": request_id})
        return ResponseModel[CallLogItem].ok(
            CallLogItem(**{k: v for k, v in row.items() if k in CallLogItem.model_fields}),
            request_id=_rid(request))

    @router.get("/qa/stats", tags=["统计"], summary="运营统计",
                response_model=ResponseModel[CallStats],
                description="**调用侧**：问答次数、平均耗时、成功率、拒答率、层 D 合规率、"
                            "耗时分位数与 token 花销（按 since_hours 开窗）。\n\n"
                            "**知识库侧**（`index` 段）：文档总数、索引大小、增量更新次数。"
                            "这一段是**启动时读一次**的快照，不随本次请求重算——"
                            "现算的实测代价是 221 秒。各组件健康状态在 `GET /health/ready`。")
    def stats(request: Request,
              since_hours: Optional[float] = Query(None, ge=0,
                                                   description="只统计最近 N 小时；不传=全部")) -> Any:
        _guard(request)
        data = CallStats(**state.calls.stats(since_hours))
        ix = getattr(state, "index_stats", None)
        if ix is not None:
            data.index = _mdl.IndexStatsData(**ix.as_stats_fields())
        return ResponseModel[CallStats].ok(data, request_id=_rid(request))

    # ------------------------------------------------------------------
    # 3.4 会话
    # ------------------------------------------------------------------
    @router.post("/sessions", tags=["会话"], summary="新建会话",
                 response_model=ResponseModel[SessionData])
    def create_session(request: Request,
                       req: Optional[SessionCreateRequest] = Body(None)) -> Any:
        _guard(request)
        req = req or SessionCreateRequest()      # 允许空请求体：POST /sessions 直接建一个
        s = state.sessions.create(req.session_id, req.title)
        return ResponseModel[SessionData].ok(_session_data(s), request_id=_rid(request))

    @router.get("/sessions", tags=["会话"], summary="会话列表（分页）",
                response_model=ResponseModel[PageModel[SessionData]])
    def list_sessions(request: Request,
                      page: int = Query(1, ge=_mdl.PAGE_MIN),
                      page_size: int = Query(20, ge=_mdl.PAGE_SIZE_MIN,
                                             le=_mdl.PAGE_SIZE_MAX)) -> Any:
        _guard(request)
        rows, total = state.sessions.list_sessions(page, page_size)
        return ResponseModel[PageModel[SessionData]].ok(
            PageModel[SessionData].build([_session_data(r) for r in rows],
                                         total, page, page_size),
            request_id=_rid(request))

    @router.get("/sessions/{session_id}", tags=["会话"], summary="会话详情（含历史）",
                response_model=ResponseModel[SessionData])
    def get_session(request: Request,
                    session_id: str = Path(..., max_length=_mdl.SESSION_ID_MAX),
                    turns: int = Query(20, ge=0, le=200)) -> Any:
        _guard(request)
        s = state.sessions.get(session_id, with_history=turns)
        if s is None:
            raise SessionNotFound(session_id)
        return ResponseModel[SessionData].ok(_session_data(s), request_id=_rid(request))

    @router.delete("/sessions/{session_id}", tags=["会话"], summary="删除会话")
    def delete_session(request: Request,
                       session_id: str = Path(..., max_length=_mdl.SESSION_ID_MAX)) -> Any:
        _guard(request)
        if not state.sessions.delete(session_id):
            raise SessionNotFound(session_id)
        return ResponseModel.ok({"session_id": session_id, "deleted": True},
                                request_id=_rid(request))

    return router


def _session_data(s: Dict[str, Any]) -> SessionData:
    return SessionData(
        session_id=s["session_id"], created_at=s.get("created_at", ""),
        updated_at=s.get("updated_at", ""), turns=int(s.get("turns") or 0),
        title=s.get("title") or "",
        history=[SessionTurn(**{k: v for k, v in h.items() if k in SessionTurn.model_fields})
                 for h in (s.get("history") or [])])
