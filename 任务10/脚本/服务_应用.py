# -*- coding: utf-8 -*-
"""第十阶段（一）· 服务化 —— FastAPI 应用骨架

统一响应格式、错误码、全局异常处理、日志、健康检查，以及把整条 RAG 链接进来的依赖容器。

## 三个决定了整体形状的取舍

**1. 流水线懒加载，服务先起来。**
检索器首次加载约 15.8GB RSS + 数分钟（Chroma HNSW），模型另占 6.3GB 显存。
要是在 startup 里同步加载，服务几分钟起不来、健康检查探针会先把它判死。所以
`ServiceState.pipeline` 是**带锁的懒属性**：进程秒起，`/health` 立刻可用，
第一次问答才付加载代价（也可以 `--warmup` 显式预热）。

**2. 并发闸门是功能，不是保护性摆设。**
阶段八实测：单卡上并行调本地模型只有约 1.36× 加速，多个请求争同一份权重、Ollama 服务端
排队。放开并发只会让每个请求都变慢，还可能触发显存换出。所以有一个
`BoundedSemaphore(max_concurrent=2)`，排不上队就返回 **5003 服务繁忙**，
而不是让请求在里面无限期等。

**3. 路由一律写成同步 `def`，不写 `async def`。**
一次问答是 100 秒以上的 CPU/GPU 阻塞调用。写成 `async def` 会把事件循环钉死，
连 `/health` 都不响应。FastAPI 对同步路由自动丢进线程池，这正是我们要的；
`contextvars`（request_id）会随之复制过去，日志与记录仍然对得上。

## 检索模式

    live       阶段六 `RetrievalPipeline`（要 65GB 向量库 + 3.4GB BM25 索引 + 15.8GB 内存）
    snapshot   阶段七固化的 `检索快照_live.json`（4 组 × 10 条真实检索结果）
               —— **开发/演示用**。响应里的 `retrieval_mode` 与 `pool_match` 会如实标出来，
               不会让人把演示数据当成真检索。

启动：
    $py = "E:\\rag\\conda\\envs\\medrag\\python.exe"
    $env:OLLAMA_MODELS = "E:\\rag\\ollama\\models"      # 先起 Ollama
    & $py E:\\rag\\scripts\\服务_应用.py --port 8000                 # 快照证据（默认）
    & $py E:\\rag\\scripts\\服务_应用.py --port 8000 --mode live --warmup
    & $py E:\\rag\\scripts\\服务_应用.py --print-routes              # 不起服务，只看路由表

    交互式文档 http://127.0.0.1:8000/docs
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")        # 硬覆盖：用户级 HF_HOME 指向改名前的旧路径
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hmac
import importlib.util
import json
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "report_data")

SERVICE_NAME = "medrag-api"
SERVICE_VERSION = "0.10.2"          # 阶段十第二部分：文档管理 + 运营统计 + .env


def _load_by_path(mod_name: str, filename: str):
    """按路径导入并**登记进 sys.modules**。为什么必须登记见 `服务_模型.py` 里的同名函数：
    不登记会让异常类出现多个不相等的副本，全局异常处理器静默失配，错误码全退化成 5001。"""
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

ErrorCode = _err.ErrorCode
APIError = _err.APIError
AuthError = _err.AuthError
ServiceBusy = _err.ServiceBusy
classify_exception = _err.classify_exception

ResponseModel = _mdl.ResponseModel
HealthData = _mdl.HealthData
ComponentHealth = _mdl.ComponentHealth

_dc = _load_by_path("fuwu_wendangmulu", "服务_文档目录.py")
_qa = _load_by_path("fuwu_wenda", "服务_问答接口.py")
_doc = _load_by_path("fuwu_wendangjiekou", "服务_文档接口.py")

from fastapi import FastAPI, Request                       # noqa: E402
from fastapi.exceptions import RequestValidationError      # noqa: E402
from fastapi.responses import JSONResponse                 # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402
from starlette.middleware.cors import CORSMiddleware       # noqa: E402


# ============================================================================
# 一、配置
# ============================================================================
@dataclass
class ServiceSettings:
    """服务配置。全部有默认值，`create_app()` 不传参也能起。"""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    log_dir: str = _lg.LOG_DIR
    log_console: bool = True

    # ---- 检索 ----
    retrieval_mode: str = "snapshot"                 # snapshot | live
    snapshot_path: str = os.path.join(REPORT_DIR, "检索快照_live.json")
    bm25_dir: str = os.path.join(ROOT, "data", "bm25_index_4m")
    #: 中译英方式，只对 live 生效。dict = 阶段五手写词典（默认，零开销，但词表外的
    #: 病名/药名会被整段丢弃，剩下的泛词去查全库 399.8 万块 → 检索崩）；
    #: llm = 用 qwen3:8b 整句翻译（每次约 5s，首次约 60s）。
    translate: str = "dict"                          # off | dict | llm

    # ---- 生成 ----
    model_name: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    num_ctx: int = 12288
    llm_timeout: int = 180
    max_context_tokens: int = 2800
    constrained: bool = True                         # 阶段九层 A + 层 D
    max_repair_rounds: int = 1
    default_evaluate: bool = True
    default_review: bool = True

    # ---- 会话 ----
    session_db: str = _ss.DB_PATH
    session_ttl_days: float = 30.0
    history_turns: int = 4
    rewrite_mode: str = "llm"                        # none | concat | llm

    # ---- 调用记录 ----
    calls_db: str = _lg.DB_PATH

    # ---- 文档目录与索引统计（第二部分）----
    #: 离线建好的文献目录（`服务_文档目录.py --build`）。缺了不影响问答，
    #: 只让 /documents 返 3004、/qa/stats 的 index 段为 null。
    docs_db: str = _dc.DB_PATH
    index_stats_path: str = _dc.STATS_PATH
    #: 向量库目录。健康检查与占盘统计要它，**live 模式也要**——它会被传给
    #: `RetrievalPipeline(chroma_path=…)`。⚠ 曾经只喂给健康检查而不往下传，
    #: 于是 `MEDRAG_CHROMA_DIR` 配了不生效：/health 报的是你配的那个，真正打开的
    #: 却是 `检索_多路检索.py` 的模块默认值。字段名必须承诺它真实的语义。
    chroma_dir: str = _dc.CHROMA_DIR
    #: landmark collection 目录（P0）。同理必须往下传：缺这个目录**不报错**，
    #: 只把 landmark 那一路静默关掉——配错路径的表现就是「P0 无声消失」。
    #: ⚠ 直接由 ROOT 拼出来，不从 `检索_多路检索.py` 取常量——那个模块 import 阶段
    #: 就要 chromadb / torch，snapshot 模式本来根本不该加载它。
    landmark_dir: str = os.path.join(ROOT, "data", "chroma_landmark")
    #: P0 landmark 并行路（只对 live 模式生效）。关掉即为对照组，用来量「补语料值多少分」。
    use_landmark: bool = True

    # ---- 并发与保护 ----
    max_concurrent: int = 2                          # 见模块 docstring 第 2 条
    acquire_timeout: float = 30.0
    heartbeat_seconds: float = 10.0

    # ---- 安全（默认全关，单机离线场景不该强制）----
    api_key: str = ""                                # 非空即开启鉴权
    rate_limit_per_minute: int = 0                   # >0 即开启限流
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    docs_enabled: bool = True

    # ---- 测试注入：验证脚本用假生成器跑真链路，不需要 Ollama ----
    generator_override: Any = None
    warmup: bool = False

    #: 每个配置项的来源：cli / env / dotenv / default。由 `resolve_settings()` 填。
    #: 直接 `ServiceSettings(...)` 构造（验证脚本、进程内调用）时为空——那种情况下
    #: "来源"是调用方代码，写死一个 "default" 反而是撒谎。
    config_sources: Dict[str, str] = field(default_factory=dict)
    env_file: str = ""

    def sanitized(self) -> Dict[str, Any]:
        """能对外展示的配置（去掉密钥与不可序列化的注入对象）。"""
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("api_key", "generator_override")}
        d["auth_enabled"] = bool(self.api_key)
        d["rate_limit_enabled"] = self.rate_limit_per_minute > 0
        d["generator_override"] = type(self.generator_override).__name__ \
            if self.generator_override is not None else None
        if not self.config_sources:
            d["config_sources_note"] = ("本进程是直接构造 ServiceSettings 起的（未走 CLI），"
                                        "没有「哪一层生效」这回事，故为空")
        return d


# ---------------------------------------------------------------------------
# 一之二、配置来源：CLI > 环境变量 > .env > 默认值
# ---------------------------------------------------------------------------
#: `.env` 默认位置。也可用 `--env-file` 或环境变量 `MEDRAG_ENV_FILE` 指到别处。
DEFAULT_ENV_FILE = os.path.join(ROOT, ".env")

#: (设置项名, 环境变量名, 类型)。**这张表是唯一来源**：CLI 映射、`.env.example` 生成、
#: `/api/v1/config` 的来源回显三处都读它，不各写一份。
ENV_FIELDS: List[Tuple[str, str, str]] = [
    ("host", "MEDRAG_HOST", "str"),
    ("port", "MEDRAG_PORT", "int"),
    ("log_level", "MEDRAG_LOG_LEVEL", "str"),
    ("log_dir", "MEDRAG_LOG_DIR", "str"),
    ("retrieval_mode", "MEDRAG_MODE", "str"),
    ("snapshot_path", "MEDRAG_SNAPSHOT", "str"),
    ("bm25_dir", "MEDRAG_BM25_DIR", "str"),
    ("chroma_dir", "MEDRAG_CHROMA_DIR", "str"),
    ("landmark_dir", "MEDRAG_LANDMARK_DIR", "str"),
    ("translate", "MEDRAG_TRANSLATE", "str"),
    ("model_name", "MEDRAG_MODEL", "str"),
    ("base_url", "MEDRAG_BASE_URL", "str"),
    ("num_ctx", "MEDRAG_NUM_CTX", "int"),
    ("llm_timeout", "MEDRAG_LLM_TIMEOUT", "int"),
    ("max_context_tokens", "MEDRAG_MAX_CONTEXT_TOKENS", "int"),
    ("constrained", "MEDRAG_CONSTRAINED", "bool"),
    ("max_repair_rounds", "MEDRAG_REPAIR_ROUNDS", "int"),
    ("session_db", "MEDRAG_SESSION_DB", "str"),
    ("session_ttl_days", "MEDRAG_SESSION_TTL_DAYS", "float"),
    ("history_turns", "MEDRAG_HISTORY_TURNS", "int"),
    ("rewrite_mode", "MEDRAG_REWRITE", "str"),
    ("calls_db", "MEDRAG_CALLS_DB", "str"),
    ("docs_db", "MEDRAG_DOCS_DB", "str"),
    ("index_stats_path", "MEDRAG_INDEX_STATS", "str"),
    ("max_concurrent", "MEDRAG_MAX_CONCURRENT", "int"),
    ("acquire_timeout", "MEDRAG_ACQUIRE_TIMEOUT", "float"),
    ("api_key", "MEDRAG_API_KEY", "str"),
    ("rate_limit_per_minute", "MEDRAG_RATE_LIMIT", "int"),
    ("cors_origins", "MEDRAG_CORS_ORIGINS", "list"),
    ("docs_enabled", "MEDRAG_DOCS_ENABLED", "bool"),
    ("use_landmark", "MEDRAG_LANDMARK", "bool"),
    ("warmup", "MEDRAG_WARMUP", "bool"),
]

_BOOL_TRUE = {"1", "true", "yes", "on", "y"}
_BOOL_FALSE = {"0", "false", "no", "off", "n"}


class ConfigError(RuntimeError):
    """配置值不合法。**故意在启动时抛而不是静默取默认值**：`MEDRAG_PORT=八千` 被悄悄
    忽略掉，表现是"我明明配了却不生效"，比起不来还难查。"""


def _cast(raw: str, kind: str, where: str, key: str) -> Any:
    v = raw.strip()
    try:
        if kind == "int":
            return int(v)
        if kind == "float":
            return float(v)
        if kind == "bool":
            low = v.lower()
            if low in _BOOL_TRUE:
                return True
            if low in _BOOL_FALSE:
                return False
            raise ValueError(f"要 true/false，给的是 {v!r}")
        if kind == "list":
            return [x.strip() for x in v.split(",") if x.strip()]
        return v
    except ValueError as e:
        raise ConfigError(f"{where} 里 {key}={raw!r} 无法解析为 {kind}：{e}") from None


def load_env_file(path: str) -> Dict[str, str]:
    """极简 `.env` 解析。不引第三方库——规则就这么几条，自己写反而能把语义钉死：

      · `KEY=VALUE`，第一个 `=` 之前是键
      · 允许 `export KEY=VALUE`
      · 整行以 `#` 开头是注释
      · 值两端成对的引号会被剥掉

    ⚠ **不支持行尾注释**：`MEDRAG_MODE=live  # 真检索` 里的 `# 真检索` 会成为值的一部分。
    理由和 `.gitignore` 那条坑一样——一旦支持，值里带 `#` 的（比如密钥）就没法写了。
    注释请单独一行。这条也写在 `.env.example` 的抬头。
    """
    out: Dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8-sig") as f:      # -sig：吃掉记事本存的 BOM
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if key:
                out[key] = val
    return out


def resolve_settings(cli: Optional[Dict[str, Any]] = None,
                     env_file: Optional[str] = None,
                     environ: Optional[Dict[str, str]] = None,
                     **extra: Any) -> ServiceSettings:
    """按 **CLI > 环境变量 > .env > 默认值** 合成配置，并记下每一项的来源。

    优先级顺序是有理由的，不是习惯：`.env` 是"这台机器平时怎么跑"，CLI 是"这一次要怎么跑"。
    反过来的话，`--mode live` 会被 `.env` 里的 `MEDRAG_MODE=snapshot` 吃掉，
    而所有 README 里的复现命令都是 CLI 写法——那些命令会集体失效。
    """
    cli = {k: v for k, v in (cli or {}).items() if v is not None}
    environ = os.environ if environ is None else environ
    path = env_file or environ.get("MEDRAG_ENV_FILE") or DEFAULT_ENV_FILE
    dotenv = load_env_file(path)

    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for name, envname, kind in ENV_FIELDS:
        if name in cli:
            values[name], sources[name] = cli[name], "cli"
        elif environ.get(envname):
            values[name] = _cast(environ[envname], kind, "环境变量", envname)
            sources[name] = "env"
        elif dotenv.get(envname):
            values[name] = _cast(dotenv[envname], kind, path, envname)
            sources[name] = "dotenv"
        else:
            sources[name] = "default"
    values.update(extra)
    return ServiceSettings(config_sources=sources,
                           env_file=path if os.path.isfile(path) else "",
                           **values)


def env_example_text() -> str:
    """`.env.example` 的正文。由 `ENV_FIELDS` + 一份默认配置现生成，
    所以**不会和代码脱节**——手写一份就一定会。"""
    d = ServiceSettings()
    lines = [
        "# 医学知识 RAG 问答服务 —— 环境变量样例",
        "# 复制成 .env 后按需改；.env 不进 git（.gitignore 已排除），.env.example 进。",
        "#",
        "# 优先级：命令行参数 > 环境变量 > .env > 默认值。",
        "# 所以 README 里那些 `--mode live --translate llm` 的命令仍然有效，且优先。",
        "#",
        "# ⚠ 不支持行尾注释：`MEDRAG_MODE=live  # 真检索` 会把 `# 真检索` 读进值里。",
        "#   注释请单独一行（和 .gitignore 一个道理）。",
        "",
    ]
    for name, envname, kind in ENV_FIELDS:
        val = getattr(d, name)
        if isinstance(val, list):
            val = ",".join(val)
        if isinstance(val, bool):
            val = "true" if val else "false"
        if name == "api_key":
            lines.append("# 非空即开启鉴权；留空 = 不鉴权（单机离线默认）")
        lines.append(f"{envname}={val}")
    return "\n".join(lines) + "\n"


# ============================================================================
# 二、快照证据源（开发模式）
# ============================================================================
_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")

#: 中文题面 → 证据组的兜底提示。快照里的组问题是英文，中文问题与它们**没有字面交集**，
#: 纯词重叠必然全 0（这正是 docs/工程笔记.md 三·2"中文查询必须先中译英"那条坑的另一副面孔）。
#: 只在 snapshot 这个**开发模式**里用，live 模式走真检索，与此无关。
SNAPSHOT_ZH_HINTS: Dict[str, Tuple[str, ...]] = {
    "live-1": ("crispr", "cas9", "脱靶", "基因编辑", "sgrna"),
    "live-2": ("fabry", "法布里", "酶替代", "ert", "α-半乳糖苷酶"),
    "live-3": ("multiple sclerosis", "多发性硬化", "ms", "疾病修饰"),
    "live-4": ("alzheimer", "阿尔茨海默", "淀粉样", "amyloid", "单抗", "lecanemab"),
}


class SnapshotEvidence:
    """阶段七固化检索快照的只读证据源。

    有 `.search(query, top_k)`，所以在 live 之外它也能直接当检索器塞给流水线——
    两种模式走的是同一条代码路径，接口层不需要为演示模式开分支。
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
        self.created = snap.get("created", "")
        self.pools: Dict[str, Dict[str, Any]] = {}
        for q in snap.get("queries", []):
            pid = q["case"]["id"]
            cands = q.get("candidates") or []
            titles = " ".join(str((c.get("metadata") or {}).get("source_title") or "")
                              for c in cands)
            self.pools[pid] = {
                "id": pid, "query": q["case"].get("query", ""),
                "candidates": cands,
                "terms": set(_WORD.findall((q["case"].get("query", "") + " " + titles).lower()))
                | set(SNAPSHOT_ZH_HINTS.get(pid, ())),
            }
        if not self.pools:
            raise ValueError(f"快照里没有任何证据组：{path}")

    @property
    def pool_ids(self) -> List[str]:
        return sorted(self.pools)

    def topics(self) -> List[Dict[str, Any]]:
        """对外公布"有哪些主题问得出东西"。

        客户端拿它做**发问前的自检**：快照模式下问范围外的问题，系统要花二十几秒才能
        告诉你"答不了"，而在第 0 秒就已经知道证据不匹配了。把这份清单给出去，
        界面就能在发请求之前提醒，不白等。
        """
        return [{"id": p["id"], "query": p["query"],
                 "keywords": sorted(SNAPSHOT_ZH_HINTS.get(p["id"], ())),
                 "docs": len(p["candidates"])}
                for p in (self.pools[k] for k in self.pool_ids)]

    def match_score(self, query: str) -> Tuple[Optional[str], float]:
        """这个问题最像哪一组证据，以及匹配强度。`(None, 0)` 表示一组都不像。"""
        q = (query or "").lower()
        toks = set(_WORD.findall(q))
        best, score = None, 0.0
        for p in self.pools.values():
            hit = len(toks & p["terms"])
            hit += sum(2 for h in SNAPSHOT_ZH_HINTS.get(p["id"], ()) if h and h in q)
            if hit > score:
                best, score = p["id"], float(hit)
        return best, score

    def pick(self, query: str, pool: Optional[str] = None, top_k: int = 10
             ) -> Tuple[List[Dict[str, Any]], str, str]:
        """选一组证据。返回 (候选列表, 组 id, 命中方式 exact|keyword|fallback)。"""
        if pool:
            p = self.pools.get(pool)
            if p is None:
                raise APIError(ErrorCode.PARAM_INVALID,
                               f"快照里没有证据组 {pool!r}",
                               detail={"field": "evidence_pool", "available": self.pool_ids})
            return p["candidates"][:top_k], p["id"], "exact"

        best, _score = self.match_score(query)
        if best is None:
            # 匹配不上就用第一组，并在响应里标 fallback ——**绝不假装匹配成功**
            first = self.pools[self.pool_ids[0]]
            return first["candidates"][:top_k], first["id"], "fallback"
        p = self.pools[best]
        return p["candidates"][:top_k], p["id"], "keyword"


# ============================================================================
# 三、限流
# ============================================================================
class RateLimiter:
    """按标识的滑动窗口限流。默认不启用（`rate_limit_per_minute=0`）。"""

    def __init__(self, per_minute: int = 0):
        self.per_minute = int(per_minute)
        self._hits: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    def check(self, ident: str) -> Tuple[bool, float]:
        """返回 (是否放行, 建议重试秒数)。"""
        if not self.enabled:
            return True, 0.0
        now = time.time()
        with self._lock:
            xs = [t for t in self._hits.get(ident, []) if now - t < 60.0]
            if len(xs) >= self.per_minute:
                self._hits[ident] = xs
                return False, round(60.0 - (now - xs[0]), 1)
            xs.append(now)
            self._hits[ident] = xs
            return True, 0.0


# ============================================================================
# 四、服务状态（依赖容器）
# ============================================================================
class ServiceState:
    """进程内的单例容器：流水线、检索器、两个存储、并发闸门、健康探针。

    **懒加载 + 双检锁**：`pipeline` / `retriever` 只在第一次真正用到时构建。
    构建过程可能几分钟，锁保证并发请求不会同时建两份（32G 内存放不下两个检索器）。
    """

    def __init__(self, settings: Optional[ServiceSettings] = None):
        self.settings = settings or ServiceSettings()
        self.started_at = time.time()
        self.log_config = _lg.configure_logging(self.settings.log_dir, self.settings.log_level,
                                                console=self.settings.log_console)
        self.log = _lg.get_logger("app")

        self.calls = _lg.CallLogStore(self.settings.calls_db)
        self.sessions = _ss.SessionStore(self.settings.session_db,
                                         ttl_days=self.settings.session_ttl_days)
        self.limiter = RateLimiter(self.settings.rate_limit_per_minute)

        #: 文献目录（只读）与索引统计。**都在启动时办完**：目录是开一个 SQLite 连接，
        #: 统计是读一份小 JSON + stat 13 个文件。请求路径上一次都不会重算——
        #: 现算的代价实测是 221 秒（Chroma 上 count(distinct pmcid)）。
        self.docs = _dc.DocCatalog(self.settings.docs_db)
        self.index_stats = _dc.IndexStats.load(self.settings.index_stats_path)
        self.log.info("文献目录：%s｜索引统计：%s",
                      self.docs.healthy()[1], self.index_stats.detail)

        self._sema = threading.BoundedSemaphore(max(1, int(self.settings.max_concurrent)))
        self._inflight = 0
        self._inflight_lock = threading.Lock()

        self._pipeline: Any = None
        self._bus: Any = None
        self._tap: Any = None
        self._rewriter: Any = None
        self._retriever: Any = None
        self._build_lock = threading.RLock()
        self._build_error: str = ""

        self._snapshot: Any = None
        if self.settings.retrieval_mode == "snapshot":
            try:
                self._snapshot = SnapshotEvidence(self.settings.snapshot_path)
                self.log.info("已加载检索快照 %s（%d 组证据）",
                              self.settings.snapshot_path, len(self._snapshot.pools))
            except Exception as e:               # 缺快照不该让服务起不来，健康检查会报
                self._snapshot = None
                self._build_error = f"检索快照不可用：{e}"
                self.log.warning("检索快照加载失败：%s", e)

        #: Ollama 探活结果缓存，避免每次 /health 都打一次网络
        self._probe: Tuple[float, bool, str, float] = (0.0, False, "未探测", 0.0)

    # ---------------- 懒加载：检索器 ----------------
    @property
    def snapshot(self) -> Any:
        return self._snapshot

    def get_retriever(self) -> Any:
        """阶段六 `RetrievalPipeline`。首次约 15.8GB RSS + 数分钟。"""
        if self._retriever is not None:
            return self._retriever
        with self._build_lock:
            if self._retriever is None:
                self.log.warning("开始加载检索器（首次约 15.8GB 内存 + 数分钟）…")
                t0 = time.time()
                mp = _load_by_path("jiansuo_duolu", "检索_多路检索.py")
                self._retriever = mp.RetrievalPipeline(
                    bm25_dir=self.settings.bm25_dir,
                    chroma_path=self.settings.chroma_dir,
                    translate=self.settings.translate,
                    use_landmark=self.settings.use_landmark,
                    landmark_path=self.settings.landmark_dir)
                self.log.warning("检索器就绪，用时 %.1fs", time.time() - t0)
        return self._retriever

    # ---------------- 懒加载：流水线 ----------------
    @property
    def pipeline_loaded(self) -> bool:
        return self._pipeline is not None

    def get_pipeline(self) -> Any:
        """带 SSE 埋点的生成流水线（同步与流式共用同一个实例）。"""
        if self._pipeline is not None:
            return self._pipeline
        with self._build_lock:
            if self._pipeline is not None:
                return self._pipeline
            s = self.settings
            t0 = time.time()

            pm = _load_by_path("shengcheng_liushuixian", "生成_流水线.py")
            cm = _load_by_path("yueshu_shouxian", "约束_受限流水线.py")

            gen = s.generator_override
            if gen is None:
                gen = pm.LLMGenerator(model_name=s.model_name, base_url=s.base_url,
                                      timeout=s.llm_timeout, num_ctx=s.num_ctx, verbose=False)

            bus = _st.EmitBus()
            tap = _st.StreamTapGenerator(gen, bus)
            asm = _st.tapped_assembler(
                pm.ContextAssembler(max_context_tokens=s.max_context_tokens, verbose=False), bus)

            base = cm.ConstrainedGenerationPipeline if s.constrained \
                else pm.MedicalGenerationPipeline
            cls = _st.make_tapped_pipeline(base)
            kw: Dict[str, Any] = {"assembler": asm, "generator": tap, "bus": bus,
                                  "verbose": False}
            if s.retrieval_mode == "live":
                kw["retriever"] = _st.tapped_retriever(self.get_retriever(), bus)
            if s.constrained:
                kw["max_repair_rounds"] = s.max_repair_rounds

            self._pipeline = cls(**kw)
            self._bus, self._tap = bus, tap
            self._rewriter = _ss.FollowupRewriter(generator=tap,
                                                  history_turns=s.history_turns)
            self.log.warning("生成流水线就绪，用时 %.1fs（约束层 %s）",
                             time.time() - t0, "开" if s.constrained else "关")
        return self._pipeline

    @property
    def bus(self) -> Any:
        self.get_pipeline()
        return self._bus

    @property
    def rewriter(self) -> Any:
        self.get_pipeline()
        return self._rewriter

    def warmup(self) -> Dict[str, Any]:
        """显式预热：把懒加载的东西提前建好，让第一个真实请求不必等。

        ⚠ **必须连模型一起预热，不能只加载检索器**（实测踩过）：qwen3:8b 冷启动要把权重
        装进显存约十秒，而 live 模式下第一个 LLM 调用是**中译英**，它自带较短超时——
        于是重起后的第一条请求会静默降级成"不翻译"，检索拿中文去查英文库、回一句
        "无法回答"，而答案那次调用因为模型已被翻译那次装好了反而很快。
        表面看是"库里没有证据"，实际是冷启动吃掉了翻译。一次微型 generate 就能挡掉。
        """
        t0 = time.time()
        errs = []
        try:
            self.get_pipeline()
        except Exception as e:
            errs.append(f"pipeline: {e}")
        llm_seconds = None
        gen = getattr(self, "_tap", None)
        if gen is not None and self.settings.generator_override is None:  # 假生成器不必预热
            t1 = time.time()
            try:
                gen.generate("ok", max_tokens=1, temperature=0.0)
                llm_seconds = round(time.time() - t1, 2)
            except Exception as e:
                errs.append(f"llm: {e}")
        return {"seconds": round(time.time() - t0, 2), "pipeline_loaded": self.pipeline_loaded,
                "llm_warmup_seconds": llm_seconds, "errors": errs}

    # ---------------- 并发闸门 ----------------
    @property
    def inflight(self) -> int:
        return self._inflight

    def acquire_slot(self, timeout: Optional[float] = None) -> None:
        """占一个生成名额；排不上就抛 5003。见模块 docstring 第 2 条。

        与 `slot()` 分开成一对显式方法，是给**流式接口**用的：SSE 一旦开始 yield，
        状态行就已经发出去了，再抛 5003 只能变成流里的一个 error 事件、HTTP 仍是 200。
        所以流式必须在**返回 StreamingResponse 之前**抢到名额，才能把"忙"如实表达成 503。
        """
        wait = self.settings.acquire_timeout if timeout is None else float(timeout)
        t0 = time.time()
        if not self._sema.acquire(timeout=wait):
            raise ServiceBusy(time.time() - t0, self.settings.max_concurrent)
        with self._inflight_lock:
            self._inflight += 1

    def release_slot(self) -> None:
        with self._inflight_lock:
            self._inflight -= 1
        try:
            self._sema.release()
        except ValueError:          # BoundedSemaphore 多放一次会抛，吞掉但记下来
            self.log.error("并发名额重复释放（这是 bug，不该发生）")

    @contextmanager
    def slot(self, timeout: Optional[float] = None) -> Iterator[None]:
        """同步接口用的上下文管理器版本。"""
        self.acquire_slot(timeout)
        try:
            yield
        finally:
            self.release_slot()

    # ---------------- 健康检查 ----------------
    def probe_model(self, ttl: float = 10.0) -> Tuple[bool, str, float]:
        """探 Ollama 是否活着。结果缓存 `ttl` 秒——探针每秒来一次，不该每次都打网络。"""
        now = time.time()
        ts, ok, detail, ms = self._probe
        if now - ts < ttl:
            return ok, detail, ms
        if self.settings.generator_override is not None:
            self._probe = (now, True, "使用注入的生成器（测试模式），未探测 Ollama", 0.0)
            return self._probe[1], self._probe[2], self._probe[3]
        t0 = time.time()
        try:
            with urllib.request.urlopen(f"{self.settings.base_url}/api/tags", timeout=3) as r:
                names = [m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])]
            want = self.settings.model_name
            hit = any(n == want or n.split(":")[0] == want.split(":")[0] for n in names)
            ms = round((time.time() - t0) * 1000, 1)
            self._probe = (now, hit, (f"模型 {want} 已就绪" if hit
                                      else f"Ollama 在，但没有模型 {want}；已安装 {names}"), ms)
        except Exception as e:
            self._probe = (now, False, f"连不上 Ollama（{self.settings.base_url}）：{e}",
                           round((time.time() - t0) * 1000, 1))
        return self._probe[1], self._probe[2], self._probe[3]

    def health(self, deep: bool = True) -> Dict[str, Any]:
        """各组件状态。`critical=True` 的任一项不通过 → status=down（就绪探针返回 503）。"""
        comps: List[Dict[str, Any]] = []

        ok_l, det_l = self.calls.healthy()
        comps.append({"name": "call_log_db", "ok": ok_l, "critical": False, "detail": det_l})
        ok_s, det_s = self.sessions.healthy()
        comps.append({"name": "session_db", "ok": ok_s, "critical": False, "detail": det_s})
        comps.append({"name": "logging", "ok": True, "critical": False,
                      "detail": f"{self.settings.log_dir}（{self.settings.log_level}）"})

        # 任务书点名要「向量库」，而在第二部分之前这里**只查了 BM25 目录**——
        # 65 GB 的 Chroma 一次都没被体检过。这里只 stat 不打开：打开一个 4M 集合
        # 并真查一次要 22 秒、13.7 GB RSS，就绪探针绝不能付这笔钱。
        size, nfiles = _dc.dir_size(self.settings.chroma_dir)
        comps.append({
            "name": "vector_db", "ok": size > 0,
            "critical": self.settings.retrieval_mode == "live",
            "detail": (f"{self.settings.chroma_dir}｜{_dc.human_size(size)}／{nfiles} 文件"
                       f"｜检索器{'已' if self._retriever else '未'}加载"
                       if size > 0 else f"向量库不存在或为空：{self.settings.chroma_dir}")})

        ok_c, det_c = self.docs.healthy()
        comps.append({"name": "doc_catalog", "ok": ok_c, "critical": False, "detail": det_c})

        if self.settings.retrieval_mode == "snapshot":
            comps.append({"name": "retrieval:snapshot", "ok": self._snapshot is not None,
                          "critical": True,
                          "detail": (f"{len(self._snapshot.pools)} 组证据（开发模式，非真检索）"
                                     if self._snapshot else self._build_error)})
        else:
            ready = os.path.isdir(self.settings.bm25_dir)
            comps.append({"name": "retrieval:live", "ok": ready, "critical": True,
                          "detail": (f"BM25 索引 {self.settings.bm25_dir}"
                                     f"｜检索器{'已' if self._retriever else '未'}加载")
                          if ready else f"BM25 索引目录不存在：{self.settings.bm25_dir}"})

        if deep:
            ok_m, det_m, ms = self.probe_model()
            comps.append({"name": "llm:ollama", "ok": ok_m, "critical": True,
                          "detail": det_m, "latency_ms": ms})

        crit_bad = [c for c in comps if c["critical"] and not c["ok"]]
        soft_bad = [c for c in comps if not c["critical"] and not c["ok"]]
        status = "down" if crit_bad else ("degraded" if soft_bad else "ok")
        return {
            "status": status, "version": SERVICE_VERSION,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "retrieval_mode": self.settings.retrieval_mode,
            "model": self.settings.model_name,
            "pipeline_loaded": self.pipeline_loaded,
            "in_flight": self.inflight, "max_concurrent": self.settings.max_concurrent,
            "components": comps,
        }

    def close(self) -> None:
        try:
            self.calls.close()
        finally:
            self.sessions.close()


# ============================================================================
# 五、鉴权与限流依赖
# ============================================================================
def client_ident(request: Request) -> str:
    key = request.headers.get("X-API-Key") or ""
    if key:
        return f"key:{key[:8]}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def check_auth(state: ServiceState, request: Request) -> None:
    """静态 API Key 鉴权。`api_key` 为空即整体关闭（单机离线场景的默认）。"""
    want = state.settings.api_key
    if not want:
        return
    got = request.headers.get("X-API-Key") or ""
    if not got:
        auth = request.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            got = auth[7:].strip()
    if not got:
        raise AuthError("缺少 API Key", detail={"header": "X-API-Key"})
    # 定长比较：避免用字符串 == 泄漏前缀信息
    if not hmac.compare_digest(got, want):
        raise AuthError("API Key 不正确", detail={"header": "X-API-Key"})


def check_rate(state: ServiceState, request: Request) -> None:
    ok, retry = state.limiter.check(client_ident(request))
    if not ok:
        raise APIError(ErrorCode.RATE_LIMITED,
                       detail={"retry_after_seconds": retry,
                               "limit_per_minute": state.settings.rate_limit_per_minute})


def guard(state: ServiceState, request: Request) -> None:
    """每个业务接口的统一入口检查。"""
    check_auth(state, request)
    check_rate(state, request)


# ============================================================================
# 六、异常 → 标准响应
# ============================================================================
#: pydantic 的错误类型 → 业务码。让 1002/1003/1004 各归其位，
#: 而不是把所有校验失败一律塞进 1001（那样客户端分不出"少传了"和"传越界了"）。
_VALIDATION_CODE = {
    "json_invalid": ErrorCode.BODY_MALFORMED,
    "missing": ErrorCode.PARAM_MISSING,
    "greater_than": ErrorCode.PARAM_OUT_OF_RANGE,
    "greater_than_equal": ErrorCode.PARAM_OUT_OF_RANGE,
    "less_than": ErrorCode.PARAM_OUT_OF_RANGE,
    "less_than_equal": ErrorCode.PARAM_OUT_OF_RANGE,
    "too_long": ErrorCode.PARAM_OUT_OF_RANGE,
    "too_short": ErrorCode.PARAM_OUT_OF_RANGE,
}

#: HTTP 状态码 → 业务码（Starlette 自己抛的 404/405 这类）
_HTTP_CODE = {400: ErrorCode.PARAM_INVALID, 401: ErrorCode.AUTH_FAILED,
              403: ErrorCode.PERMISSION_DENIED, 404: ErrorCode.DOC_NOT_FOUND,
              405: ErrorCode.PARAM_INVALID, 422: ErrorCode.PARAM_INVALID,
              429: ErrorCode.RATE_LIMITED, 503: ErrorCode.DEPENDENCY_FAILED}


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "") or _lg.get_request_id()


def error_response(request: Request, code: Any, message: str,
                   detail: Optional[Dict[str, Any]] = None,
                   http_status: Optional[int] = None) -> JSONResponse:
    """所有失败路径的唯一出口——保证错误响应体与成功响应体是同一个形状。"""
    rid = _rid(request)
    body = ResponseModel.fail(code=code, message=message, detail=detail, request_id=rid)
    resp = JSONResponse(status_code=http_status or _err.http_status_of(code),
                        content=json.loads(body.model_dump_json()))
    # 兜底处理器跑在用户中间件**外面**，拿不到中间件加的头，所以这里自己补一次
    resp.headers["X-Request-Id"] = rid
    return resp


def install_exception_handlers(app: FastAPI, state: ServiceState) -> None:
    log = _lg.get_logger("error")

    @app.exception_handler(APIError)
    def _on_api_error(request: Request, exc: APIError) -> JSONResponse:
        log.warning("APIError %s %s → [%d] %s %s", request.method, request.url.path,
                    int(exc.code), exc.message, exc.detail or "")
        if exc.cause is not None:
            log.debug("原始异常：%r", exc.cause)
        return error_response(request, exc.code, exc.message, exc.detail, exc.http_status)

    @app.exception_handler(RequestValidationError)
    def _on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errs = exc.errors()
        # 业务码跟着 **errs[0]** 走，认不出来的类型归 1001。
        # 不能写成"扫到第一个能识别的类型"：`{"query":"  ","top_k":99}` 会先报 query 的
        # `value_error`（不在映射表里，被跳过）、再报 top_k 的 `less_than_equal`，
        # 于是返回 1003「参数超出范围」——而更根本的问题是 query 是空的。
        # 现在的规则可以一句话说清：**code 对应 detail.errors[0]**。
        code = ErrorCode.PARAM_INVALID
        if any(str(e.get("type")) == "json_invalid" for e in errs):
            code = ErrorCode.BODY_MALFORMED   # 请求体都没解析出来，其余报错都是噪声
        elif errs:
            code = _VALIDATION_CODE.get(str(errs[0].get("type")), ErrorCode.PARAM_INVALID)
        fields = [{"field": ".".join(str(x) for x in (e.get("loc") or [])[1:]) or "body",
                   "message": e.get("msg", ""), "type": e.get("type", "")}
                  for e in errs][:10]
        log.info("参数校验失败 %s → [%d] %s", request.url.path, int(code), fields)
        return error_response(request, code, _err.message_of(code), {"errors": fields})

    @app.exception_handler(StarletteHTTPException)
    def _on_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        msg = str(exc.detail) if exc.detail else _err.message_of(code)
        if exc.status_code == 404:
            msg = f"接口或资源不存在：{request.method} {request.url.path}"
        return error_response(request, code, msg, http_status=exc.status_code)

    @app.exception_handler(Exception)
    def _on_any(request: Request, exc: Exception) -> JSONResponse:
        api = classify_exception(exc)
        # 未知异常必须留全栈；**响应体里只给一句话**，不把内部细节漏给调用方
        if int(api.code) == int(ErrorCode.INTERNAL_ERROR):
            log.error("未处理异常 %s %s\n%s", request.method, request.url.path,
                      "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        else:
            log.warning("已归类异常 %s %s → [%d] %s", request.method, request.url.path,
                        int(api.code), api.message)
        return error_response(request, api.code, api.message, api.detail, api.http_status)


# ============================================================================
# 七、应用工厂
# ============================================================================
API_PREFIX = "/api/v1"

DESCRIPTION = """本地医学知识 RAG 问答服务。

先从 PubMed oa_comm 语料检索证据，再由本地 `qwen3:8b` 依据证据作答，
每个事实句都带 `[S#]` 出处，并在生成后逐条校验（阶段九强约束层）。

* **统一响应体**：`{code, message, data, detail, request_id, timestamp, elapsed_ms}`，
  成功与失败同一个形状；`code == 0` 为成功，码表见 `GET /api/v1/errors`。
* **两种响应模式**：`POST /api/v1/qa/ask` 同步；`POST /api/v1/qa/stream` 走 SSE。
* **一次问答 100 秒以上是常态**（四段链 + 层 D 校验），客户端超时请设足。
"""


def create_app(settings: Optional[ServiceSettings] = None,
               state: Optional[ServiceState] = None) -> FastAPI:
    """构建应用。`state` 可注入——验证脚本用它塞假生成器，跑真链路而不需要 Ollama。"""
    # 第一个位置参数是 settings 不是 state，很容易传错（转 word 那个脚本就传错过，
    # 报出来的是莫名其妙的 `'ServiceState' object has no attribute 'log_dir'`）。
    # 传错的意图是明确的，直接认下来，不给一个绕远路的报错。
    if state is None and isinstance(settings, ServiceState):
        state, settings = settings, None
    st = state or ServiceState(settings)
    s = st.settings

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        st.log.warning("%s v%s 启动｜检索模式 %s｜约束层 %s｜并发上限 %d｜鉴权 %s",
                       SERVICE_NAME, SERVICE_VERSION, s.retrieval_mode,
                       "开" if s.constrained else "关", s.max_concurrent,
                       "开" if s.api_key else "关")
        if s.warmup:
            st.log.warning("预热：%s", st.warmup())
        yield
        st.log.warning("服务停止，累计运行 %.1fs", time.time() - st.started_at)
        st.close()

    app = FastAPI(
        lifespan=lifespan,
        title="医学知识 RAG 问答服务",
        version=SERVICE_VERSION,
        description=DESCRIPTION,
        docs_url="/docs" if s.docs_enabled else None,
        redoc_url="/redoc" if s.docs_enabled else None,
        openapi_url="/openapi.json" if s.docs_enabled else None,
    )
    app.state.service = st

    if s.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins,
                           allow_credentials=False, allow_methods=["*"],
                           allow_headers=["*"],
                           expose_headers=["X-Request-Id", "X-Response-Time-Ms"])

    access_log = _lg.get_logger("access")

    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        """请求 ID + 计时 + 访问日志。

        ⚠ 流式响应的耗时**不在这里量**：中间件在响应头发出的那一刻就结束了，SSE 正文
        还要再流一两分钟。流式请求的真实耗时由流生成器收尾时自己写进 SQLite。
        """
        rid = (request.headers.get("X-Request-Id") or "").strip()[:64] or _lg.new_request_id()
        _lg.set_request_id(rid)
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-Id"] = rid
        response.headers["X-Response-Time-Ms"] = f"{dt:.1f}"
        if request.url.path not in ("/health", "/health/live"):     # 探针不刷屏
            access_log.info("%s %s → %d  %.1fms", request.method, request.url.path,
                            response.status_code, dt)
        return response

    install_exception_handlers(app, st)

    # ---------------- 基础路由 ----------------
    @app.get("/", tags=["元信息"], summary="服务概览")
    def root(request: Request) -> Any:
        return ResponseModel.ok({
            "service": SERVICE_NAME, "version": SERVICE_VERSION,
            "docs": "/docs" if s.docs_enabled else None,
            "endpoints": [f"{API_PREFIX}/qa/ask", f"{API_PREFIX}/qa/stream",
                          f"{API_PREFIX}/qa/logs", f"{API_PREFIX}/qa/stats",
                          f"{API_PREFIX}/sessions", f"{API_PREFIX}/documents",
                          f"{API_PREFIX}/errors", "/health", "/health/ready"],
            "retrieval_mode": s.retrieval_mode,
            "note": ("snapshot = 阶段七固化的真实检索快照，开发/演示用；"
                     "真检索请用 --mode live" if s.retrieval_mode == "snapshot" else ""),
        }, request_id=_rid(request))

    @app.get("/health", tags=["健康"], summary="存活探针（永远不碰下游）")
    def health_live(request: Request) -> Any:
        """只回答"进程还在不在"。**不探 Ollama、不碰磁盘**——存活探针一旦依赖下游，
        下游抖动就会把好好的进程重启掉。"""
        return ResponseModel.ok({"status": "ok", "version": SERVICE_VERSION,
                                 "uptime_seconds": round(time.time() - st.started_at, 1)},
                                request_id=_rid(request))

    @app.get("/health/ready", tags=["健康"], summary="就绪探针（含下游依赖）",
             response_model=ResponseModel[HealthData])
    def health_ready(request: Request) -> Any:
        h = st.health(deep=True)
        body = ResponseModel[HealthData].ok(HealthData(**h), request_id=_rid(request))
        if h["status"] == "down":
            bad = "、".join(c["name"] for c in h["components"]
                           if c["critical"] and not c["ok"])
            return error_response(request, ErrorCode.DEPENDENCY_FAILED,
                                  f"关键依赖不可用：{bad}",
                                  {"components": h["components"]})
        return body

    @app.get(f"{API_PREFIX}/errors", tags=["元信息"], summary="错误码表")
    def error_table(request: Request) -> Any:
        return ResponseModel.ok({"families": _err.FAMILIES, "codes": _err.describe_all()},
                                request_id=_rid(request))

    @app.get(f"{API_PREFIX}/config", tags=["元信息"], summary="生效中的配置（已脱敏）")
    def config(request: Request) -> Any:
        return ResponseModel.ok(s.sanitized(), request_id=_rid(request))

    # ---------------- 业务路由 ----------------
    app.include_router(_qa.build_router(st, guard=lambda r: guard(st, r)), prefix=API_PREFIX)
    app.include_router(_doc.build_router(st, guard=lambda r: guard(st, r)), prefix=API_PREFIX)

    return app


# ============================================================================
# CLI
# ============================================================================
def iter_routes(app: Any) -> List[Tuple[str, str, str]]:
    """展平路由表 → [(path, methods, 名称)]。

    ⚠ **不要遍历 `app.routes`**：fastapi 0.141 / starlette 1.6 的 `include_router` 不再把
    子路由摊平进去，而是塞进一个内部的 `_IncludedRouter` 包装对象（它连 `.routes` 属性都没有）。
    照着遍历会**一条业务路由都看不到**——第一次跑 `--print-routes` 就是这样，只列出了
    5 条骨架路由，看起来像"路由没注册成功"。改从 OpenAPI 文档读：那是对外发布的真实接口面，
    也不依赖框架内部结构。
    """
    out: List[Tuple[str, str, str]] = []
    verbs = ("get", "post", "put", "patch", "delete", "head", "options")
    try:
        paths = (app.openapi() or {}).get("paths", {})
    except Exception:
        paths = {}
    for path in sorted(paths):
        ops = paths[path] or {}
        for verb in verbs:
            op = ops.get(verb)
            if isinstance(op, dict):
                out.append((path, verb.upper(),
                            op.get("summary") or op.get("operationId") or ""))
    return out


#: CLI 参数名 → 设置项名。只列名字对不上的；同名的（host/port/translate…）自动映射。
_CLI_ALIAS = {"mode": "retrieval_mode", "snapshot": "snapshot_path", "model": "model_name",
              "repair_rounds": "max_repair_rounds", "rate_limit": "rate_limit_per_minute",
              "rewrite": "rewrite_mode"}


#: 静态 OpenAPI 落盘位置。在线 `/docs` 已经有了，但交付包里要有一份**不依赖服务在跑**
#: 的版本——拿到压缩包的人不会先起 uvicorn 才去看接口有哪些。
OPENAPI_DEFAULT_PATH = os.path.join(ROOT, "任务10", "openapi.json")


def dump_openapi(app: Any, path: str = OPENAPI_DEFAULT_PATH) -> str:
    """把 `app.openapi()` 落盘。**与 /openapi.json 同一个来源**，不另写一份描述——
    手写的那份第一次改接口就会过期，而且没人会发现。"""
    spec = app.openapi()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2, sort_keys=False)
    return path


def build_settings(args: argparse.Namespace) -> ServiceSettings:
    """把 argparse 结果交给 `resolve_settings()`。

    ⚠ **所有 CLI 默认值都是 `None`**，这是这套优先级能成立的关键：argparse 分不清
    "用户显式传了 --port 8000" 和 "没传所以用默认 8000"。默认值一旦写在 argparse 上，
    每个参数就都是"显式传过"，`.env` 与环境变量**永远轮不到**——配了也不生效，
    而且不报任何错。真正的默认值在 `ServiceSettings` 的字段上（单一来源）。
    """
    d = vars(args)
    cli: Dict[str, Any] = {}
    for key, val in d.items():
        if val is None or key in ("print_routes", "dump_openapi", "env_file",
                                  "print_config", "write_env_example"):
            continue
        name = _CLI_ALIAS.get(key, key)
        if key == "no_constrained":
            cli["constrained"] = not val
        elif key == "no_landmark":
            cli["use_landmark"] = not val
        elif key == "no_docs":
            cli["docs_enabled"] = not val
        elif any(name == f for f, _e, _k in ENV_FIELDS):
            cli[name] = val
    return resolve_settings(cli, env_file=args.env_file)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="医学 RAG 问答服务（配置优先级：命令行 > 环境变量 > .env > 默认值）")
    # ⚠ 下面**一律不写 default=**，理由见 build_settings 的 docstring。
    # 默认值只在 ServiceSettings 的字段上写一份；这里的 help 里标注方便查。
    ap.add_argument("--host", help="默认 127.0.0.1")
    ap.add_argument("--port", type=int, help="默认 8000")
    ap.add_argument("--mode", choices=["snapshot", "live"],
                    help="snapshot=阶段七固化检索快照（默认，开发用）｜"
                         "live=阶段六真检索（要 65GB 库）")
    ap.add_argument("--snapshot", help=f"默认 {os.path.join(REPORT_DIR, '检索快照_live.json')}")
    ap.add_argument("--bm25-dir", help=f"默认 {os.path.join(ROOT, 'data', 'bm25_index_4m')}")
    ap.add_argument("--translate", choices=["off", "dict", "llm"],
                    help="live 模式的中译英方式：dict=阶段五词典（默认）｜"
                         "llm=qwen3 整句翻译（每次约 5s，词表外的病名药名才查得准）")
    ap.add_argument("--model", help="默认 qwen3:8b")
    ap.add_argument("--base-url", help="默认 http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, help="默认 12288")
    ap.add_argument("--no-constrained", action="store_true", default=None,
                    help="关掉阶段九强约束层（默认开）")
    ap.add_argument("--repair-rounds", type=int, help="默认 1")
    ap.add_argument("--max-concurrent", type=int, help="默认 2")
    ap.add_argument("--rate-limit", type=int, help="每分钟请求上限，默认 0 = 不限")
    ap.add_argument("--api-key", help="非空即开启鉴权（也可用 MEDRAG_API_KEY 或 .env）")
    ap.add_argument("--rewrite", choices=["none", "concat", "llm"], help="默认 llm")
    ap.add_argument("--history-turns", type=int, help="默认 4")
    ap.add_argument("--log-level", help="默认 INFO")
    ap.add_argument("--no-docs", action="store_true", default=None,
                    help="关掉 /docs 与 /openapi.json")
    ap.add_argument("--warmup", action="store_true", default=None,
                    help="启动即加载流水线与模型")
    ap.add_argument("--no-landmark", action="store_true", default=None,
                    help="关掉 P0 的 landmark 并行路（默认开）。关掉即为对照组，用来量补语料值多少分")
    ap.add_argument("--env-file", help=f"配置文件位置，默认 {DEFAULT_ENV_FILE}")
    ap.add_argument("--print-routes", action="store_true", help="只打印路由表，不起服务")
    ap.add_argument("--print-config", action="store_true",
                    help="只打印生效配置与每一项的来源，不起服务")
    ap.add_argument("--dump-openapi", metavar="PATH", nargs="?", const=OPENAPI_DEFAULT_PATH,
                    help=f"把 OpenAPI(Swagger) 文档落盘，不起服务；默认 {OPENAPI_DEFAULT_PATH}")
    ap.add_argument("--write-env-example", metavar="PATH", nargs="?",
                    const=os.path.join(ROOT, ".env.example"),
                    help="按 ENV_FIELDS 生成 .env.example，不起服务")
    args = ap.parse_args()

    if args.write_env_example:
        with open(args.write_env_example, "w", encoding="utf-8") as f:
            f.write(env_example_text())
        print(f"已写 {args.write_env_example}（{len(ENV_FIELDS)} 项）")
        return 0

    settings = build_settings(args)

    if args.print_config:
        src = settings.config_sources
        print(f"{SERVICE_NAME} v{SERVICE_VERSION}  生效配置"
              f"（.env：{settings.env_file or '未使用'}）")
        print("=" * 92)
        by = {}
        for name, envname, _kind in ENV_FIELDS:
            val = "***" if name == "api_key" and getattr(settings, name) else \
                getattr(settings, name)
            s = src.get(name, "default")
            by[s] = by.get(s, 0) + 1
            print(f"  {name:<22}{str(val)[:44]:<46}{s:<8}{envname}")
        print("\n  " + "　".join(f"{k}={v}" for k, v in sorted(by.items())))
        return 0

    app = create_app(settings)

    if args.dump_openapi:
        path = dump_openapi(app, args.dump_openapi)
        print(f"OpenAPI → {path}")
        return 0

    if args.print_routes:
        print(f"{SERVICE_NAME} v{SERVICE_VERSION}  路由表")
        print("=" * 92)
        for path, method, name in iter_routes(app):
            print(f"  {method:<8} {path:<34} {name}")
        print(f"\n共 {len(iter_routes(app))} 条接口"
              + ("；交互文档 /docs" if app.docs_url else "；文档已关闭"))
        return 0

    import uvicorn
    host, port = settings.host, settings.port
    print(f"启动 {SERVICE_NAME} v{SERVICE_VERSION} → http://{host}:{port}"
          + (f"　文档 http://{host}:{port}/docs" if settings.docs_enabled else "")
          + (f"　配置 {settings.env_file}" if settings.env_file else ""))
    # log_config=None：保留我们自己配好的 handler，不让 uvicorn 覆盖
    uvicorn.run(app, host=host, port=port, log_config=None, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
