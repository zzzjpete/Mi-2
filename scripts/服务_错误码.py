# -*- coding: utf-8 -*-
"""第十阶段（一）· 服务化 —— 错误码枚举与异常体系

一个 RAG 服务的失败方式与普通 CRUD 服务很不一样，所以错误码不是照抄模板凑数的：

  · **模型不可用**和**模型答错**是两件事**。前者是 502/503，客户端重试有意义；
    后者根本不是错误——阶段九的系统在证据不足时会**正常返回**一段带拒答短语的答案，
    HTTP 200、code 0。把"检索不到证据"做成错误码是本项目最容易犯的设计错，
    它会让"边界守得住"这个花了一整个阶段做出来的能力，在接口层被表达成一次失败。
  · **超时要和调用失败分开**（4002 vs 4001）：本机四段链单次 100 秒以上是常态，
    超时往往意味着参数（num_ctx / top_k）不合适，而不是 Ollama 挂了。
  · **服务繁忙是本项目的常态错误**（5003）。单卡 10G，阶段八实测并发调本地模型只有
    约 1.36× 加速，放任并发只会让每个请求都变慢并可能触发换出。所以并发闸门是
    功能不是保护性摆设，它需要一个自己的错误码。

分组（前缀即家族，看一眼就知道该找谁）：

    1xxx  参数与请求        调用方能自己修 → 4xx
    2xxx  认证与配额        调用方要换凭据/降速 → 401/403/429
    3xxx  资源              找不到东西 → 404，索引没就绪 → 503
    4xxx  模型与外部依赖    上游的问题 → 502/503/504
    5xxx  服务内部          我们的问题 → 500/503

HTTP 状态码与业务码**同时**给：状态码让网关、探针、监控这些不读 body 的东西能工作，
业务码让客户端精确分支。不走"一律 200、错误写 body"那套——那会让反向代理与
可观测性设施全部失明。

用法：
    from 服务_错误码 import ErrorCode, APIError, ParamError      # 中文名按路径导入
    raise ParamError("query 不能为空", detail={"field": "query"})
    raise APIError(ErrorCode.MODEL_TIMEOUT, detail={"seconds": 180})

CLI：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\服务_错误码.py --table
"""
import argparse
import sys
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class ErrorCode(IntEnum):
    """业务错误码。0 = 成功；其余按家族分段，见模块 docstring。"""

    OK = 0

    # ---- 1xxx 参数与请求 ----
    PARAM_INVALID = 1001            # 参数值不合法（类型对、值不对）
    PARAM_MISSING = 1002            # 必填参数缺失
    PARAM_OUT_OF_RANGE = 1003       # 数值越界（top_k、page_size…）
    BODY_MALFORMED = 1004           # 请求体不是合法 JSON / 结构对不上

    # ---- 2xxx 认证与配额 ----
    AUTH_FAILED = 2001              # API Key 缺失或不正确
    TOKEN_EXPIRED = 2002            # 凭据过期（预留，本阶段用的是静态 Key）
    PERMISSION_DENIED = 2003        # 通过认证但无权访问该资源
    RATE_LIMITED = 2004             # 超过调用频率限制

    # ---- 3xxx 资源 ----
    DOC_NOT_FOUND = 3001            # 文献/文档不存在
    SESSION_NOT_FOUND = 3002        # 会话不存在或已过期
    RECORD_NOT_FOUND = 3003         # 调用记录不存在
    INDEX_NOT_READY = 3004          # 检索索引尚未加载完成

    # ---- 4xxx 模型与外部依赖 ----
    MODEL_CALL_FAILED = 4001        # Ollama 返回非 200 / 连接中断
    MODEL_TIMEOUT = 4002            # 单次生成超时
    MODEL_OUTPUT_INVALID = 4003     # 输出无法解析（JSON 四级容错仍失败）
    MODEL_UNAVAILABLE = 4004        # 连不上 Ollama，或模型未安装

    # ---- 5xxx 服务内部 ----
    INTERNAL_ERROR = 5001           # 未预料到的异常（兜底）
    DEPENDENCY_FAILED = 5002        # 依赖组件不可用（健康检查不通过）
    SERVICE_BUSY = 5003             # 并发闸门已满，见模块 docstring
    STORAGE_FAILED = 5004           # SQLite 读写失败


#: 码 → (默认中文消息, HTTP 状态码)。改这张表就等于同时改了文档、响应体与探针行为。
_META: Dict[ErrorCode, Tuple[str, int]] = {
    ErrorCode.OK:                   ("ok", 200),

    ErrorCode.PARAM_INVALID:        ("参数错误", 400),
    ErrorCode.PARAM_MISSING:        ("缺少必填参数", 400),
    ErrorCode.PARAM_OUT_OF_RANGE:   ("参数超出允许范围", 400),
    ErrorCode.BODY_MALFORMED:       ("请求体格式错误", 400),

    ErrorCode.AUTH_FAILED:          ("认证失败", 401),
    ErrorCode.TOKEN_EXPIRED:        ("凭据已过期", 401),
    ErrorCode.PERMISSION_DENIED:    ("无权访问该资源", 403),
    ErrorCode.RATE_LIMITED:         ("请求过于频繁，请稍后重试", 429),

    # 「文档或资源」：任务书点名的是"文档不存在"，实际还兜住了路由不存在
    # （Starlette 的 404 映射到这里，消息会被换成具体的 method + path）
    ErrorCode.DOC_NOT_FOUND:        ("文档或资源不存在", 404),
    ErrorCode.SESSION_NOT_FOUND:    ("会话不存在或已过期", 404),
    ErrorCode.RECORD_NOT_FOUND:     ("调用记录不存在", 404),
    ErrorCode.INDEX_NOT_READY:      ("检索索引尚未就绪", 503),

    ErrorCode.MODEL_CALL_FAILED:    ("模型调用失败", 502),
    ErrorCode.MODEL_TIMEOUT:        ("模型调用超时", 504),
    ErrorCode.MODEL_OUTPUT_INVALID: ("模型输出无法解析", 502),
    ErrorCode.MODEL_UNAVAILABLE:    ("模型服务不可用", 503),

    ErrorCode.INTERNAL_ERROR:       ("服务内部错误", 500),
    # 503 而不是 500：这个码只在**就绪探针**与依赖失联时出现，语义就是"暂时不可用，
    # 稍后再来"。给 500 会让编排系统把它当成代码 bug 而不是依赖抖动。
    ErrorCode.DEPENDENCY_FAILED:    ("依赖组件不可用", 503),
    ErrorCode.SERVICE_BUSY:         ("服务繁忙，已达并发上限", 503),
    ErrorCode.STORAGE_FAILED:       ("存储读写失败", 500),
}

#: 家族说明，供 `/api/v1/errors` 与文档使用
FAMILIES: Dict[str, str] = {
    "0": "成功",
    "1": "参数与请求（调用方可自行修正）",
    "2": "认证与配额",
    "3": "资源不存在或未就绪",
    "4": "模型与外部依赖",
    "5": "服务内部",
}

#: 本阶段**尚未接入**、仅先占位的码。列出来是为了不让人以为它们已经在跑。
#: 鉴权与限流虽然默认关闭，但开关一打开 2001/2003/2004 立刻生效，所以不算预留；
#: 2002 要接了动态令牌（本阶段是静态 API Key）才会用到，只有它是真占位。
RESERVED = {ErrorCode.TOKEN_EXPIRED}


def message_of(code: "ErrorCode | int") -> str:
    """错误码的默认消息。未登记的码不抛异常——错误处理路径上再抛异常最难排查。"""
    try:
        return _META[ErrorCode(code)][0]
    except (ValueError, KeyError):
        return f"未登记的错误码 {int(code)}"


def http_status_of(code: "ErrorCode | int") -> int:
    """错误码对应的 HTTP 状态码。未登记的一律 500（宁可报服务端问题，也别谎称成功）。"""
    try:
        return _META[ErrorCode(code)][1]
    except (ValueError, KeyError):
        return 500


def family_of(code: "ErrorCode | int") -> str:
    return FAMILIES.get(str(int(code))[0] if int(code) else "0", "未知")


def describe_all() -> list:
    """整张码表，供 `/api/v1/errors` 直接吐出来。"""
    return [{"code": int(c), "name": c.name, "message": msg, "http_status": st,
             "family": family_of(c), "reserved": c in RESERVED}
            for c, (msg, st) in _META.items()]


# ============================================================================
# 异常体系
# ============================================================================
class APIError(Exception):
    """所有可预期的业务失败都走这一个类型，由全局异常处理器统一转成标准响应体。

    Args:
        code:        `ErrorCode`
        message:     覆盖默认消息（**给人看的那一句**，不要塞堆栈）
        detail:      结构化补充信息（哪个字段、什么值、允许范围），给客户端分支用
        http_status: 覆盖码表里的默认状态码，一般不用
        cause:       原始异常，只写进服务端日志，**不进响应体**
    """

    def __init__(self,
                 code: "ErrorCode | int" = ErrorCode.INTERNAL_ERROR,
                 message: Optional[str] = None,
                 detail: Optional[Dict[str, Any]] = None,
                 http_status: Optional[int] = None,
                 cause: Optional[BaseException] = None):
        # ⚠ 不要写 `int(code) in set(ErrorCode)`：Enum 的 __hash__ 按**成员名**算，
        # 整数与成员的哈希不同，集合判定必然为假（登记过的码也会被当成未登记）。
        try:
            self.code: Any = ErrorCode(code)
        except ValueError:
            self.code = int(code)
        self.message = message or message_of(code)
        self.detail = detail or {}
        self.http_status = int(http_status) if http_status else http_status_of(code)
        self.cause = cause
        super().__init__(f"[{int(self.code)}] {self.message}")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"code": int(self.code), "message": self.message}
        if self.detail:
            d["detail"] = self.detail
        return d


class ParamError(APIError):
    def __init__(self, message: Optional[str] = None, **kw: Any):
        kw.setdefault("code", ErrorCode.PARAM_INVALID)
        super().__init__(message=message, **kw)


class MissingParamError(APIError):
    def __init__(self, field: str, message: Optional[str] = None, **kw: Any):
        kw.setdefault("code", ErrorCode.PARAM_MISSING)
        kw.setdefault("detail", {"field": field})
        super().__init__(message=message or f"缺少必填参数 {field}", **kw)


class OutOfRangeError(APIError):
    def __init__(self, field: str, value: Any, low: Any, high: Any, **kw: Any):
        kw.setdefault("code", ErrorCode.PARAM_OUT_OF_RANGE)
        kw.setdefault("detail", {"field": field, "value": value,
                                 "allowed": {"min": low, "max": high}})
        super().__init__(message=f"{field} 必须在 [{low}, {high}] 之间，收到 {value}", **kw)


class AuthError(APIError):
    def __init__(self, message: Optional[str] = None, **kw: Any):
        kw.setdefault("code", ErrorCode.AUTH_FAILED)
        super().__init__(message=message, **kw)


class NotFoundError(APIError):
    def __init__(self, code: "ErrorCode | int" = ErrorCode.DOC_NOT_FOUND,
                 message: Optional[str] = None, **kw: Any):
        super().__init__(code=code, message=message, **kw)


class SessionNotFound(NotFoundError):
    def __init__(self, session_id: str, **kw: Any):
        kw.setdefault("detail", {"session_id": session_id})
        super().__init__(code=ErrorCode.SESSION_NOT_FOUND, **kw)


class DocNotFound(NotFoundError):
    """3001。⚠ 「不在目录里」= 「不在这 400 万块的索引里」，**不等于 PMC 上没有这篇**：
    4M 是从 9,243 万块里抽样的子集。detail 里把这句说清楚，否则调用方会得出
    "这篇文献不存在"的错误结论。"""

    def __init__(self, doc_id: str, **kw: Any):
        kw.setdefault("detail", {
            "doc_id": doc_id,
            "hint": "该 ID 不在本地 4M 索引的文献目录中；本索引是 oa_comm 全量的抽样子集，"
                    "查不到不代表 PMC 上没有这篇文献"})
        super().__init__(code=ErrorCode.DOC_NOT_FOUND,
                         message=f"文档不存在：{doc_id}", **kw)


class ModelError(APIError):
    def __init__(self, code: "ErrorCode | int" = ErrorCode.MODEL_CALL_FAILED,
                 message: Optional[str] = None, **kw: Any):
        super().__init__(code=code, message=message, **kw)


class ServiceBusy(APIError):
    def __init__(self, waited_seconds: float, limit: int, **kw: Any):
        kw.setdefault("code", ErrorCode.SERVICE_BUSY)
        kw.setdefault("detail", {"waited_seconds": round(waited_seconds, 2),
                                 "max_concurrent": limit})
        super().__init__(message=f"并发生成已达上限（{limit}），等待 {waited_seconds:.1f}s 仍未排到", **kw)


class StorageError(APIError):
    def __init__(self, message: Optional[str] = None, **kw: Any):
        kw.setdefault("code", ErrorCode.STORAGE_FAILED)
        super().__init__(message=message, **kw)


# ---------------------------------------------------------------------------
# 上游异常 → 业务码
# ---------------------------------------------------------------------------
#: `生成_LLM生成器.LLMError` 的消息里已经写清了失败形态，按关键词分流即可。
#: 不用 `isinstance` 细分，是因为那个模块只有一个异常类型——它把区分信息放在了文案里。
_LLM_PATTERNS = (
    ("连不上", ErrorCode.MODEL_UNAVAILABLE),
    ("没有模型", ErrorCode.MODEL_UNAVAILABLE),
    ("超时", ErrorCode.MODEL_TIMEOUT),
    ("HTTP", ErrorCode.MODEL_CALL_FAILED),
)


def classify_exception(exc: BaseException) -> APIError:
    """把任意异常收敛成 `APIError`，供全局兜底处理器使用。

    只有**能确定含义**的才给具体码，其余一律 5001——把未知异常伪装成已知错误码，
    会让线上真正的 bug 藏在一个看起来很正常的 4xxx 里。
    """
    if isinstance(exc, APIError):
        return exc

    name = type(exc).__name__
    text = str(exc)

    # 生成_LLM生成器.LLMError（按路径导入，跨模块 isinstance 不可靠，按类名判）
    if name == "LLMError":
        for kw, code in _LLM_PATTERNS:
            if kw in text:
                return ModelError(code, message=f"{message_of(code)}：{text[:200]}", cause=exc)
        return ModelError(ErrorCode.MODEL_CALL_FAILED, message=text[:200], cause=exc)

    if isinstance(exc, TimeoutError):
        return APIError(ErrorCode.MODEL_TIMEOUT, cause=exc)
    if isinstance(exc, (ValueError, TypeError, KeyError)) and "retrieved_docs" in text:
        return APIError(ErrorCode.INDEX_NOT_READY,
                        message="没有检索器也没有传入证据，无法作答", cause=exc)
    import sqlite3
    if isinstance(exc, sqlite3.Error):
        return StorageError(f"SQLite：{text[:200]}", cause=exc)

    return APIError(ErrorCode.INTERNAL_ERROR, cause=exc)


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", help="打印整张错误码表")
    args = ap.parse_args()
    if not args.table:
        ap.print_help()
        return 0

    print("=" * 84)
    print(f"{'码':>6}  {'名称':<22} {'HTTP':>5}  {'消息':<24} 备注")
    print("=" * 84)
    last = ""
    for row in describe_all():
        fam = str(row["code"])[0] if row["code"] else "0"
        if fam != last:
            print(f"--- {fam}xxx  {FAMILIES[fam]} " + "-" * 40)
            last = fam
        print(f"{row['code']:>6}  {row['name']:<22} {row['http_status']:>5}  "
              f"{row['message']:<24} {'（预留，本阶段未接入）' if row['reserved'] else ''}")
    print("=" * 84)
    print(f"共 {len(describe_all())} 个码，其中预留 {len(RESERVED)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
