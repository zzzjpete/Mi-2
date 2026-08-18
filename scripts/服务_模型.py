# -*- coding: utf-8 -*-
"""第十阶段（一）· 服务化 —— 统一响应模型、分页模型与请求校验

三件事，每件都有一个"为什么这么定"：

  1. **统一响应体** `{code, message, data, request_id, timestamp, elapsed_ms}`。
     `request_id` 放在**响应体里而不是只放响应头**：出问题时用户复制粘贴给你的是屏幕上的
     JSON，不是 F12 里的 header。它同时会写进日志与 SQLite，三处同一个值才能对得上。
  2. **分页模型**把 `pages / has_next / has_prev` 算好再给出去。让每个客户端自己
     `ceil(total/page_size)` 是在等着别人踩 `total=0` 与整除边界（`total=20, size=10`
     到底是 2 页还是 3 页）。这两个边界在验证里都有用例。
  3. **参数校验写在模型上，不写在处理函数里**。写在模型上才能同时得到：自动 422→1001
     转换、OpenAPI 文档里的约束说明、以及"改一处就全生效"。

关于 `query` 的长度上限（1000 字符）：这不是拍脑袋。问题原文会进**四段提示词的每一段**，
一个 1000 字的中文问题约 700 token，四段就是 2800 token，正好等于证据预算的全部
（`max_context_tokens=2800`）。再长就会开始挤压证据，而挤掉的证据正是这个系统的立身之本。

用法：
    ResponseModel.ok(data, request_id=rid, elapsed_ms=12.3)
    ResponseModel.fail(ErrorCode.PARAM_INVALID, "query 不能为空")
    PageModel.build(items, total=137, page=2, page_size=20)
"""
import importlib.util
import os
import sys
import time
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(mod_name: str, filename: str):
    """中文文件名模块按路径导入，并**登记进 sys.modules**。

    ⚠ 登记这一步不是省事的优化，是正确性要求。`module_from_spec` + `exec_module`
    每调一次就产出一个**新的模块对象**，里面的类彼此不相等。阶段十有五个模块都要
    import `服务_错误码.py`：不登记的话，问答接口 `raise SessionNotFound(...)` 抛的是
    B 副本，而 `@app.exception_handler(APIError)` 注册的是 A 副本——两者不在同一棵
    继承树上，异常处理器**静默匹配不上**，所有自定义业务异常都会掉进兜底处理器，
    3002/4004 这些精心分好的码统统退化成 5001。

    docs/工程笔记.md 三·8"跨模块 isinstance 必然失败"这条坑，这次是以"错误码全变 5001"的
    形式出现的（验证脚本 C 组一跑就抓到了）。阶段五~九的脚本没有这个问题，是因为
    它们跨模块传的是数据不是异常。
    """
    path = os.path.join(_HERE, filename)
    cached = sys.modules.get(mod_name)
    if cached is not None and os.path.normcase(getattr(cached, "__file__", "") or "") \
            == os.path.normcase(path):
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod          # 先登记再执行，循环引用时也能拿到半成品
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(mod_name, None)  # 半死不活的模块不能留在缓存里
        raise
    return mod


_err = _load_by_path("fuwu_cuowuma", "服务_错误码.py")
ErrorCode = _err.ErrorCode
message_of = _err.message_of

T = TypeVar("T")

# ---------------------------------------------------------------------------
# 校验边界：集中在这里，验证脚本与文档都读这几个常量，不各写一份
# ---------------------------------------------------------------------------
QUERY_MIN_CHARS = 1
QUERY_MAX_CHARS = 1000          # 见模块 docstring：再长会挤压 2800 token 的证据预算
TOP_K_MIN = 1
TOP_K_MAX = 50                  # 组装器按预算截断，超过这个数只是白花检索时间
PAGE_MIN = 1
PAGE_SIZE_MIN = 1
PAGE_SIZE_MAX = 200
SESSION_ID_MAX = 64
HISTORY_TURNS_MAX = 20


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# 一、统一响应体
# ============================================================================
class ResponseModel(BaseModel, Generic[T]):
    """所有接口的统一外层。`code == 0` 即成功，非 0 时 `data` 为 null、`detail` 说明原因。"""

    model_config = ConfigDict(json_schema_extra={
        "example": {"code": 0, "message": "ok", "data": {}, "request_id": "req-1a2b3c4d",
                    "timestamp": "2026-08-10 12:00:00", "elapsed_ms": 12.3}})

    code: int = Field(0, description="业务错误码，0 表示成功；码表见 GET /api/v1/errors")
    message: str = Field("ok", description="给人看的一句话说明")
    data: Optional[T] = Field(None, description="成功时的业务数据；失败时为 null")
    detail: Optional[Dict[str, Any]] = Field(None, description="失败时的结构化补充信息")
    request_id: str = Field("", description="本次请求的唯一 ID，与日志、SQLite 记录一一对应")
    timestamp: str = Field(default_factory=_now)
    elapsed_ms: Optional[float] = Field(None, description="服务端处理耗时（毫秒）")

    @classmethod
    def ok(cls, data: Any = None, request_id: str = "", elapsed_ms: Optional[float] = None,
           message: str = "ok") -> "ResponseModel":
        return cls(code=int(ErrorCode.OK), message=message, data=data,
                   request_id=request_id, elapsed_ms=elapsed_ms)

    @classmethod
    def fail(cls, code: Any = ErrorCode.INTERNAL_ERROR, message: Optional[str] = None,
             detail: Optional[Dict[str, Any]] = None, request_id: str = "",
             elapsed_ms: Optional[float] = None) -> "ResponseModel":
        return cls(code=int(code), message=message or message_of(code), data=None,
                   detail=detail or None, request_id=request_id, elapsed_ms=elapsed_ms)

    @property
    def success(self) -> bool:
        return self.code == int(ErrorCode.OK)


# ============================================================================
# 二、分页
# ============================================================================
class PageModel(BaseModel, Generic[T]):
    """分页结果。`pages / has_next / has_prev` 由服务端算好，客户端不必重算。"""

    items: List[T] = Field(default_factory=list)
    total: int = Field(0, ge=0, description="满足条件的总条数（不是本页条数）")
    page: int = Field(1, ge=PAGE_MIN)
    page_size: int = Field(20, ge=PAGE_SIZE_MIN, le=PAGE_SIZE_MAX)
    pages: int = Field(0, ge=0, description="总页数；total=0 时为 0，不是 1")
    has_next: bool = False
    has_prev: bool = False

    @classmethod
    def build(cls, items: List[Any], total: int, page: int, page_size: int) -> "PageModel":
        """两个边界都在这里定死，验证里有对应用例：

        · `total = 0` → `pages = 0`（不是 1）。"零条结果有一页"会让前端画出一个空页码。
        · `total` 恰为 `page_size` 整数倍 → 不多出一个空页（20/10 是 2 页，不是 3 页）。
        """
        page_size = max(PAGE_SIZE_MIN, int(page_size))
        total = max(0, int(total))
        pages = (total + page_size - 1) // page_size          # 向上取整；total=0 → 0
        page = max(PAGE_MIN, int(page))
        return cls(items=items, total=total, page=page, page_size=page_size, pages=pages,
                   has_next=page < pages, has_prev=page > 1 and pages > 0)


class PageQuery(BaseModel):
    """分页查询参数。"""
    page: int = Field(1, ge=PAGE_MIN, description="页码，从 1 开始")
    page_size: int = Field(20, ge=PAGE_SIZE_MIN, le=PAGE_SIZE_MAX)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


# ============================================================================
# 三、问答请求
# ============================================================================
def _clean_query(v: str) -> str:
    """去首尾空白后判空。

    ⚠ 只写 `min_length=1` 是不够的：`"   "` 长度为 3，能过 pydantic 的长度校验，
    却会让检索器拿到空查询、模型收到一个空问题。必须 strip 之后再判。
    """
    s = (v or "").strip()
    if len(s) < QUERY_MIN_CHARS:
        raise ValueError("query 不能为空或纯空白")
    if len(s) > QUERY_MAX_CHARS:
        raise ValueError(f"query 长度不能超过 {QUERY_MAX_CHARS} 字符，当前 {len(s)}")
    return s


class AskRequest(BaseModel):
    """问答请求。同步接口与流式接口共用同一份，两种模式的参数语义完全一致。"""

    model_config = ConfigDict(json_schema_extra={
        "example": {"query": "CRISPR-Cas9 的脱靶效应如何检测？", "top_k": 10,
                    "session_id": None, "evaluate": True, "review": True}})

    query: str = Field(..., description=f"用户问题，去空白后 1~{QUERY_MAX_CHARS} 字符")
    top_k: int = Field(10, ge=TOP_K_MIN, le=TOP_K_MAX, description="检索召回条数")
    session_id: Optional[str] = Field(
        None, max_length=SESSION_ID_MAX,
        description="传入则关联历史对话：用历史把指代问题改写成独立问题，并把本轮追加进会话")
    history_turns: int = Field(4, ge=0, le=HISTORY_TURNS_MAX,
                               description="改写时回看的历史轮数（一问一答算一轮）")
    rewrite: Optional[str] = Field(
        None, description="追问改写策略：none / concat / llm；缺省用服务端配置")

    # ---- 链路开关：与阶段七、九的消融口径一一对应 ----
    evaluate: bool = Field(True, description="①证据评估段（关掉可省一次模型调用）")
    review: bool = Field(True, description="③批判审查段（关掉则用②的草稿定稿，省两次调用）")
    constrained: bool = Field(True, description="阶段九强约束提示层 + 生成后校验修正")
    expect_refusal: Optional[bool] = Field(
        None, description="仅评测用：这道题是否应当拒答。只影响校验口径，绝不写进提示词")

    evidence_pool: Optional[str] = Field(
        None, description="快照检索模式下指定证据组 id（如 live-1）；live 模式忽略")
    include_intermediate: bool = Field(
        False, description="是否返回①③的原始中间结果（体积大，默认不返回）")

    @field_validator("query")
    @classmethod
    def _v_query(cls, v: str) -> str:
        return _clean_query(v)

    @field_validator("session_id")
    @classmethod
    def _v_session(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s or None            # 空串按"没传"处理，别去建一个 id 为空的会话

    @field_validator("rewrite")
    @classmethod
    def _v_rewrite(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip().lower()
        if s not in ("none", "concat", "llm"):
            raise ValueError("rewrite 只能是 none / concat / llm")
        return s


# ============================================================================
# 四、问答响应
# ============================================================================
class SourceItem(BaseModel):
    """一条可溯源的出处。`marker` 就是答案正文里的 [S#]。"""
    marker: str
    pmcid: str = ""
    pmid: str = ""
    title: str = ""
    journal: str = ""
    pub_year: Optional[int] = None
    section: str = ""
    chunk_id: str = ""
    relevance_score: Optional[float] = None
    url: str = ""


class CitationCheck(BaseModel):
    used: List[str] = Field(default_factory=list)
    available: List[str] = Field(default_factory=list)
    fabricated: List[str] = Field(default_factory=list, description="编造的编号，正常为空")
    has_citation: bool = False


class ConstraintCheck(BaseModel):
    """阶段九层 D 的判定结果（`constrained=False` 时为 null）。"""
    compliant: bool = False
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    scores: Dict[str, Any] = Field(default_factory=dict)
    repaired: bool = Field(False, description="是否发生过修正")
    repair_rounds: int = 0


class GenerationMetrics(BaseModel):
    total_time_seconds: float = 0.0
    llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    context_tokens: int = 0
    answer_chars: int = 0
    stage_times: Dict[str, float] = Field(default_factory=dict)
    stage_success: Dict[str, Any] = Field(default_factory=dict)
    chain: Dict[str, Any] = Field(default_factory=dict)


class AskData(BaseModel):
    """同步问答的业务数据。流式接口的 `done` 事件载荷与本模型字段一致。"""
    request_id: str
    session_id: Optional[str] = None
    query: str = Field(..., description="用户原始问题")
    resolved_query: str = Field("", description="用于检索的问题（追问改写后；未改写时与 query 相同）")
    retrieval_query: Optional[str] = Field(
        None, description="真正送进检索层的那句话（中文问题在这里被译成英文）；"
                          "快照模式或检索层没给出时为 null")
    rewritten: bool = False
    partial: bool = Field(
        False, description="部分作答：问题的一部分有证据、另一部分没有。"
                           "与 refused 互斥——refused 只表示**完全**拒答")
    history_used: int = Field(0, description="本次带入的历史轮数")

    answer: str
    refused: bool = Field(False, description="答案中出现了阶段九那句固定拒答短语")
    retrieval_mode: str = Field("", description="live=真实检索 / snapshot=固化快照（仅开发演示）")
    evidence_pool: Optional[str] = None

    sources: List[SourceItem] = Field(default_factory=list)
    citation_check: Optional[CitationCheck] = None
    constraint_check: Optional[ConstraintCheck] = None
    metrics: GenerationMetrics = Field(default_factory=GenerationMetrics)
    intermediate: Optional[Dict[str, Any]] = None


# ============================================================================
# 五、会话
# ============================================================================
class SessionTurn(BaseModel):
    turn_index: int
    role: str = Field(..., description="user / assistant")
    content: str
    request_id: str = ""
    created_at: str = ""


class SessionData(BaseModel):
    session_id: str
    created_at: str = ""
    updated_at: str = ""
    turns: int = 0
    title: str = ""
    history: List[SessionTurn] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(None, max_length=SESSION_ID_MAX,
                                      description="不传则服务端生成")
    title: str = Field("", max_length=200)


# ============================================================================
# 六、调用日志与统计
# ============================================================================
class CallLogItem(BaseModel):
    request_id: str
    created_at: str = ""
    path: str = ""
    mode: str = Field("", description="sync / stream")
    session_id: Optional[str] = None
    query: str = ""
    top_k: Optional[int] = None
    status: str = Field("", description="ok / error / busy")
    code: int = 0
    http_status: int = 200
    elapsed_ms: float = 0.0
    llm_calls: Optional[int] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    answer_chars: Optional[int] = None
    refused: Optional[bool] = None
    compliant: Optional[bool] = None
    error: str = ""


class IndexStatsData(BaseModel):
    """知识库侧的规模统计。**全部来自启动时读的 `data/index_stats.json`，请求路径零计算。**

    为什么不现算（都是实测的）：Chroma sqlite 上 `count(distinct pmcid)` 要 **221.2 秒**，
    `count(*)` 要 25.2 秒，`col.get(where=…)` 首次要 22 秒且把进程 RSS 顶到 13.68 GB。
    把这些放进 `/qa/stats`，一个毫秒级接口会掉到秒级甚至分钟级。

    ⚠ **文件缺失时各计数是 `null` 不是 `0`**：0 是个看起来正常的假数字，
    null 才能让人看出"这个数还没算"。`available=false` 时看 `detail` 里的原因。
    """

    available: bool = Field(False, description="index_stats.json 是否可用；false 时下面全为 null")
    total_documents: Optional[int] = Field(
        None, description="库内文献数（去重 pmcid）。⚠ 口径是**被抽中至少一块**的文献数——"
                          "4M 索引按块抽样，不等于这么多篇完整文献在库里，见 documents_note")
    total_chunks: Optional[int] = Field(None, description="库内文本块数")
    documents_note: Optional[str] = None
    index_size_bytes: Optional[int] = Field(None, description="向量库 + BM25 + 文献目录的占盘")
    index_size_human: str = ""
    index_size_detail: Dict[str, Any] = Field(default_factory=dict)
    incremental_updates: Optional[int] = Field(
        None, description="增量更新次数。当前语料是冻结快照、一次全量构建，如实为 0")
    incremental_updates_note: Optional[str] = None
    last_index_built_at: Optional[Dict[str, Optional[str]]] = Field(
        None, description="⚠ 有三个建库时间（向量库 / BM25 / 文献目录），不是一个")
    corpus_snapshot: Optional[str] = None
    computed_at: Optional[str] = Field(None, description="这份统计算于何时（不是查询时间）")
    detail: str = ""


class CallStats(BaseModel):
    """给后续统计用的聚合。分位数在 Python 侧算——这个量级下没必要为它上时序库。"""
    total: int = 0
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_code: Dict[str, int] = Field(default_factory=dict)
    by_mode: Dict[str, int] = Field(default_factory=dict)
    success_rate: float = 0.0
    refusal_rate: Optional[float] = Field(None, description="成功请求中拒答的比例")
    compliant_rate: Optional[float] = None
    elapsed_ms: Dict[str, float] = Field(default_factory=dict, description="avg/p50/p95/max")
    tokens: Dict[str, int] = Field(default_factory=dict)
    window: str = ""
    index: Optional[IndexStatsData] = Field(
        None, description="知识库规模（文档总数 / 索引大小 / 增量更新次数）。"
                          "调用量统计随请求变，这一段只随重建索引变，所以启动时读一次即可")


# ============================================================================
# 七、健康检查
# ============================================================================
class ComponentHealth(BaseModel):
    name: str
    ok: bool
    critical: bool = True
    detail: str = ""
    latency_ms: Optional[float] = None


class HealthData(BaseModel):
    status: str = Field(..., description="ok / degraded / down")
    version: str = ""
    uptime_seconds: float = 0.0
    retrieval_mode: str = ""
    model: str = ""
    pipeline_loaded: bool = False
    in_flight: int = 0
    max_concurrent: int = 0
    components: List[ComponentHealth] = Field(default_factory=list)


# ============================================================================
# 八、文档（第二部分）
# ============================================================================
DOC_LIMIT_DEFAULT = 20
DOC_LIMIT_MAX = 100
DOC_ID_MAX = 64
DOC_TITLE_KW_MAX = 100


class DocumentIn(BaseModel):
    """文献模型。**名字照任务书写的 `DocumentIn`**，不另起一个更"像响应模型"的名字，
    免得对照任务书时多一层翻译。

    与任务书字段清单的三处出入，都是语料决定的，不是省事：

    · **`pub_date` → `pub_year`**：切块阶段只保留了年份，库里 10 个元数据字段
      （doc_id/chunk_index/total_chunks/source_title/token_count/section/pmcid/pmid/
      journal/pub_year）里没有完整日期。放一个恒为 null 的 `pub_date` 只会让人以为
      是数据缺失而去查。
    · **`abstract` 只有 7.7% 有值**（2,274,167 篇里 175,664 篇）。4M 索引是按**块**
      抽样的，多数文献的摘要块根本没被抽中。字段保留（任务书点名了）但标可选，
      填充率写在这里而不是等人自己发现。
    · **`total_chunks` 与 `indexed_chunks` 必须并列**：前者是原文切块数（平均 28.6），
      后者是真正进了 4M 库的条数（平均 1.76）。只给前者，详情页会显示"共 26 块"
      然后一块也列不出来。
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {"doc_id": "PMC212698", "pmcid": "PMC212698", "pmid": "14551916",
                    "title": "Candidate Gene Association Study in Type 2 Diabetes…",
                    "journal": "PLoS Biology", "pub_year": 2003,
                    "total_chunks": 36, "indexed_chunks": 2,
                    "sections": ["results", "methods"], "abstract": None}})

    doc_id: str = Field(..., description="文档 ID。本语料里等于 pmcid（distinct 只差 17）")
    pmcid: str = Field(..., description="PMC 号，唯一主键")
    pmid: Optional[str] = Field(None, description="PubMed 号；**不唯一**（勘误与原文共用），"
                                                  "不能拿来当主键")
    title: str = Field("", description="文献标题（库内字段名 source_title）")
    journal: str = ""
    pub_year: Optional[int] = Field(None, description="发表年份。任务书写的是 pub_date，"
                                                      "但语料只保留到年份")
    abstract: Optional[str] = Field(None, description="摘要正文，由库内 section=abstract 的"
                                                      "块拼回；**仅 7.7% 的文献有**，其余为 null")
    total_chunks: int = Field(0, description="原文切块数（切块阶段记的，平均 28.6）")
    indexed_chunks: int = Field(0, description="真正进了 4M 向量库的块数（平均 1.76）")
    sections: List[str] = Field(default_factory=list,
                                description="本篇在库内出现过的规范段落名")


class DocumentPage(BaseModel):
    """文献列表结果。**用游标不用页码**——`PageModel` 那套 `page/offset` 在 227 万行上
    会掉进翻页深渊（`LIMIT 20 OFFSET 100000` 得先扫掉前十万行）。游标是上一页最后一条
    的 pmcid，翻到第几页都是常数代价。

    `total` 默认为 null：无过滤时 `COUNT(*)` 实测 24ms、最坏（标题全表扫）1.5s，
    而翻页并不需要它。要就显式传 `with_total=true`。
    """

    items: List[DocumentIn] = Field(default_factory=list)
    limit: int = DOC_LIMIT_DEFAULT
    has_more: bool = False
    next_cursor: Optional[str] = Field(None, description="下一页的 cursor；没有下一页时为 null")
    total: Optional[int] = Field(None, description="只在 with_total=true 时计算，否则为 null")
    filters: Dict[str, Any] = Field(default_factory=dict, description="本次真正生效的过滤条件")
    elapsed_ms: float = 0.0
