# -*- coding: utf-8 -*-
"""第十阶段（二）· 服务化 —— 文档管理接口

两个接口，都只读：

    GET  /api/v1/documents            列表（过滤 + 游标分页）
    GET  /api/v1/documents/{doc_id}   按 id 查单篇，不存在 → 3001

## 为什么不直接查向量库

任务书说的是"文档列表查询"，最自然的写法是拿 Chroma 的 `where` 去筛。实测三个数字
否掉了这条路（详见 `服务_文档目录.py` 的模块 docstring）：

    where 下推在 4M 集合上退化到 ~108s（阶段五就记过这条坑）
    col.get(where={'pmcid': …}) 首次 22.0s，且进程 RSS 涨到 13.68 GB
    Chroma sqlite 上 count(distinct pmcid) 221.2s

第二条尤其致命：服务默认是 snapshot 模式，卖点就是"不加载 65 GB 库"。
只要有人点一次文献详情，这个承诺就没了——而且是在**第一次请求**上，最像 bug 的时候。

所以列表与详情全部走 `服务_文档目录.DocCatalog`（离线建好的 1.0 GB SQLite）。
实测：详情 0.0ms、列表 0.0~1.0ms、标题模糊搜最坏 824ms（全表扫且零命中）。

## 分页用游标，不用页码

227 万行上 `LIMIT 20 OFFSET 100000` 要先扫掉前十万行，页码越深越慢。所以这里**没有
page 参数**，只有 `cursor`（上一页最后一条的 pmcid）。翻到第几页都是常数代价。

这也是本文件唯一不复用 `PageModel` 的地方——那个模型要求 `total`，而无过滤的
`COUNT(*)` 在这个量级上是笔真开销（实测 24ms，标题过滤时最坏 1.5s）。
`total` 改成显式索取（`with_total=true`），默认 null。

用法（被 `服务_应用.create_app()` 调用）：
    router = build_router(state, guard=…)
"""
import importlib.util
import os
import sys
from typing import Any, Optional

from fastapi import APIRouter, Path, Query, Request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(mod_name: str, filename: str):
    """按路径导入并登记进 `sys.modules`。理由见 `服务_模型.py` 里的同名函数——
    不登记的话本文件 `raise DocNotFound(...)` 抛的类与异常处理器注册的不是同一个，
    3001 会静默退化成 5001。"""
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
_dc = _load_by_path("fuwu_wendangmulu", "服务_文档目录.py")

ErrorCode = _err.ErrorCode
APIError = _err.APIError
DocNotFound = _err.DocNotFound

ResponseModel = _mdl.ResponseModel
DocumentIn = _mdl.DocumentIn
DocumentPage = _mdl.DocumentPage


def _to_model(d: dict) -> Any:
    return DocumentIn(**{k: v for k, v in d.items() if k in DocumentIn.model_fields})


def build_router(state: Any, guard: Optional[Any] = None) -> APIRouter:
    router = APIRouter()

    def _guard(request: Request) -> None:
        if guard is not None:
            guard(request)

    def _rid(request: Request) -> str:
        return getattr(request.state, "request_id", "") or ""

    def _catalog() -> Any:
        cat = getattr(state, "docs", None)
        if cat is None or not cat.ready:
            detail = cat.healthy()[1] if cat is not None else "未初始化"
            # 3004 → 503。不返空列表：空结果与"库没建"在响应体里长得一模一样，
            # 那正是 live 模式那条坑（"检索不到"与"证据不存在"分不开）的同一种错。
            raise APIError(ErrorCode.INDEX_NOT_READY,
                           "文献目录尚未就绪",
                           detail={"reason": detail,
                                   "fix": "跑一次 scripts\\服务_文档目录.py --build（约 45 秒）"})
        return cat

    # ------------------------------------------------------------------
    # 列表
    # ------------------------------------------------------------------
    @router.get("/documents", tags=["文档"], summary="文献列表（过滤 + 游标分页）",
                response_model=ResponseModel[DocumentPage],
                description="按期刊 / 年份 / 标题关键词过滤，游标分页。\n\n"
                            "· **没有 page 参数**：227 万行上的深翻页要扫掉前面所有行，"
                            "所以只提供 `cursor`（传上一页返回的 `next_cursor`）。\n"
                            "· `total` 默认不算（null），要就传 `with_total=true`："
                            "无过滤时约 24ms，配标题模糊搜最坏约 1.5s。\n"
                            "· 标题搜是 `LIKE %kw%`，命中少时要全表扫，实测最坏 824ms。")
    def list_documents(
            request: Request,
            journal: Optional[str] = Query(None, max_length=200, description="期刊全名，精确匹配"),
            pub_year: Optional[int] = Query(None, ge=1500, le=2100, description="发表年份，精确"),
            year_from: Optional[int] = Query(None, ge=1500, le=2100),
            year_to: Optional[int] = Query(None, ge=1500, le=2100),
            title: Optional[str] = Query(None, max_length=_mdl.DOC_TITLE_KW_MAX,
                                         description="标题包含该关键词（大小写不敏感）"),
            cursor: Optional[str] = Query(None, max_length=_mdl.DOC_ID_MAX,
                                          description="上一页返回的 next_cursor"),
            limit: int = Query(_mdl.DOC_LIMIT_DEFAULT, ge=1, le=_mdl.DOC_LIMIT_MAX),
            with_total: bool = Query(False, description="是否额外算总数（有代价，见上）"),
    ) -> Any:
        _guard(request)
        cat = _catalog()
        out = cat.list_documents(journal=journal, pub_year=pub_year,
                                 year_from=year_from, year_to=year_to,
                                 title_contains=title, cursor=cursor,
                                 limit=limit, with_total=with_total)
        filters = {k: v for k, v in (("journal", journal), ("pub_year", pub_year),
                                     ("year_from", year_from), ("year_to", year_to),
                                     ("title", title)) if v is not None}
        page = DocumentPage(items=[_to_model(d) for d in out["items"]],
                            limit=limit, has_more=out["has_more"],
                            next_cursor=out["next_cursor"], total=out["total"],
                            filters=filters, elapsed_ms=out["elapsed_ms"])
        return ResponseModel[DocumentPage].ok(page, request_id=_rid(request))

    # ------------------------------------------------------------------
    # 单篇
    # ------------------------------------------------------------------
    @router.get("/documents/{doc_id}", tags=["文档"], summary="按 id 查文献",
                response_model=ResponseModel[DocumentIn],
                description="`doc_id` 传 PMCID（如 `PMC212698`）；本语料里 doc_id 与 pmcid "
                            "相同，两种都认。不在目录里 → **3001**（注意：那只表示"
                            "不在这份 4M 抽样索引里，不代表 PMC 上没有这篇）。")
    def get_document(request: Request,
                     doc_id: str = Path(..., max_length=_mdl.DOC_ID_MAX,
                                        description="PMCID 或 doc_id")) -> Any:
        _guard(request)
        cat = _catalog()
        doc = cat.get(doc_id)
        if doc is None:
            raise DocNotFound(doc_id)
        return ResponseModel[DocumentIn].ok(_to_model(doc), request_id=_rid(request))

    return router
