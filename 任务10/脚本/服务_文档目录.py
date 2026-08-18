# -*- coding: utf-8 -*-
"""第十阶段（二）· 服务化 —— 文献目录（离线建） + 索引统计（离线算）

任务书要两样东西，它们都**不能在请求路径上算**，也不能在启动时算：

  · 文档管理接口的「列表查询」——按 journal / pub_year / 标题过滤。
  · 运营统计里的「文档总数、索引大小、增量更新次数」。

实测过为什么不能现算（数字见 `任务10/README.md` 十五节）：

    Chroma sqlite  count(distinct pmcid)          221.2 s
    Chroma sqlite  count(*) where key='pmcid'      25.2 s
    col.get(where={'pmcid': ...}) 首次              22.0 s  且 RSS 涨到 13.68 GB
    col.get(where=...) 之后每次                      0.44 s
    merged_4m.parquet 只读 pmcid 一列算 distinct     0.7 s   ← 结果与上面第一行**逐位相同**

最后一行是本模块成立的前提：`merged_4m.parquet` 就是 Chroma 与 BM25 的共同建库输入
（`data/bm25_index_4m/index_meta.json` 的 `source` 字段写着它），行数 3,998,000 与
`collection.count()` 一致，distinct pmcid 2,274,167 与 221 秒那趟一致。
**两边都真跑过，不是"应该相等"的推断。**

⚠ 那个 13.68 GB 是本模块存在的第二个理由：文献详情如果走 Chroma 的 `where`，
默认的 snapshot 模式会在**第一次请求**时被拖进 22 秒 + 13.7 GB——服务号称"默认不加载
65 GB 库"就成了空话。所以列表与详情**全部走本模块的 SQLite**，不碰 Chroma。

---------------------------------------------------------------------------
**「文档总数」这个数必须带着口径一起报**，否则会被读成"库里有 227 万篇完整文献"：

    库内 chunk 数        3,998,000
    去重后文献数         2,274,167      ← 被抽中**至少一块**的文献数
    每篇平均入库块数     1.76
    每篇原文平均块数     28.64（中位 26）
    只有 1 块入库的文献  54.2%
    完整入库的文献        0.6%（13,878 篇）

因为 4M 是从 92,432,502 块里**按块**分层抽样的，不是按文献抽的。所以：

  · `total_documents` 的 description 必须写明"被抽中至少一块"；
  · `DocumentIn` 必须同时给 `total_chunks`（原文切块数）与 `indexed_chunks`（库内实际条数），
    只给前者会让详情页显示"共 26 块"然后一块都列不出来。

`abstract` 同理：只有 7.7% 的文献（175,836 篇）有摘要块被抽中，其余为 null。
任务书点名了这个字段，所以**保留但标为可选并写明填充率**，不假装它总是有值。

---------------------------------------------------------------------------
用法（服务侧只用读的那半边，不需要 pyarrow）：

    dc = _load("dc", r"E:\\rag\\scripts\\服务_文档目录.py")
    cat = dc.DocCatalog()                       # data\\docs_catalog.db
    cat.get("PMC212698")                        # pmcid 或 doc_id 都认
    cat.list_documents(journal="PLoS ONE", pub_year=2020, limit=20)
    st = dc.IndexStats.load()                   # data\\index_stats.json，启动读一次

CLI（建库要 pyarrow，约 3~6 分钟）：
    ... 服务_文档目录.py --build              建目录 + 写 index_stats.json
    ... 服务_文档目录.py --stats              只重算 index_stats.json（不动目录）
    ... 服务_文档目录.py --show PMC212698     查一篇
    ... 服务_文档目录.py --bench              量列表/详情/COUNT 的真实耗时
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB_PATH = os.path.join(ROOT, "data", "docs_catalog.db")
STATS_PATH = os.path.join(ROOT, "data", "index_stats.json")

MERGED_PARQUET = os.path.join(ROOT, "data", "vectors", "merged_4m.parquet")
CORPUS_META = os.path.join(ROOT, "data", "dict", "corpus_meta.json")
CHROMA_DIR = os.path.join(ROOT, "data", "chroma_db_4m")
BM25_DIR = os.path.join(ROOT, "data", "bm25_index_4m")
VECTOR_STATS = os.path.join(ROOT, "任务4", "向量库统计_medrag_bge_base.json")

#: 语料是一次性冻结快照，没有增量更新机制。这个 0 是**如实报**，不是"还没实现所以填 0"。
CORPUS_SNAPSHOT = "PubMed Central oa_comm baseline 2026-06-18"
INCREMENTAL_NOTE = ("当前语料是 2026-06-18 冻结快照，一次全量构建，没有增量更新机制，"
                    "因此该计数恒为 0；重建索引请跑 恢复_重建库.py / 检索_构建BM25索引.py")

LIST_LIMIT_DEFAULT = 20
LIST_LIMIT_MAX = 100

#: 段落规范名。与 `data/dict/corpus_meta.json` 的 `section.canonical_to_raw` 同一套，
#: 映射不到的原始值统一归 `other`（原始值有 39 万种，不可能穷举）。
CANONICAL_SECTIONS = ("abstract", "introduction", "methods", "results",
                      "discussion", "conclusion", "_nonbody", "other")


# ============================================================================
# 一、目录库
# ============================================================================
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    pmcid           TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    pmid            TEXT DEFAULT '',
    title           TEXT DEFAULT '',
    journal         TEXT DEFAULT '',
    pub_year        INTEGER,
    total_chunks    INTEGER DEFAULT 0,
    indexed_chunks  INTEGER DEFAULT 0,
    sections        TEXT DEFAULT '',
    abstract        TEXT
);
CREATE TABLE IF NOT EXISTS catalog_meta (key TEXT PRIMARY KEY, value TEXT);
"""

#: 索引最后建，建库时插入 227 万行会快很多。
#: 每个索引都以 pmcid 收尾——游标分页按 (排序列, pmcid) 走全序，不靠 rowid 的副产品。
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_docs_journal ON documents(journal, pmcid);
CREATE INDEX IF NOT EXISTS idx_docs_year    ON documents(pub_year, pmcid);
CREATE INDEX IF NOT EXISTS idx_docs_jy      ON documents(journal, pub_year, pmcid);
CREATE INDEX IF NOT EXISTS idx_docs_docid   ON documents(doc_id);
"""


class CatalogUnavailable(RuntimeError):
    """目录库不存在或建到一半。**故意不静默降级成空结果**——空列表和"库没建"
    在响应体里长得一模一样，那正是本项目 live 模式踩过的那类坑。"""


class DocCatalog:
    """文献目录的只读访问。纯标准库，不 import pyarrow / chromadb。

    ⚠ 只读：服务进程绝不写这个库。写只发生在 `--build`（离线、单进程）。
    """

    def __init__(self, db_path: str = DB_PATH, timeout: float = 10.0):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ready = False
        self._detail = ""
        try:
            if not os.path.isfile(db_path):
                self._detail = f"目录库不存在：{db_path}（跑 服务_文档目录.py --build 建）"
                return
            self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                                         check_same_thread=False, timeout=timeout)
            self._conn.row_factory = sqlite3.Row
            row = self._conn.execute(
                "SELECT value FROM catalog_meta WHERE key='built_at'").fetchone()
            n = self._conn.execute("SELECT value FROM catalog_meta "
                                   "WHERE key='documents'").fetchone()
            if row is None or n is None:
                self._detail = "目录库缺 catalog_meta，可能建到一半，请重跑 --build"
                return
            self._ready = True
            self._detail = f"{int(n['value']):,} 篇，建于 {row['value']}"
        except Exception as e:                      # noqa: BLE001
            self._detail = f"{type(e).__name__}: {e}"

    # ---------------- 状态 ----------------
    @property
    def ready(self) -> bool:
        return self._ready

    def healthy(self) -> Tuple[bool, str]:
        return self._ready, self._detail

    def _require(self) -> sqlite3.Connection:
        if not self._ready or self._conn is None:
            raise CatalogUnavailable(self._detail or "目录库不可用")
        return self._conn

    def meta(self) -> Dict[str, str]:
        conn = self._require()
        with self._lock:
            return {r["key"]: r["value"]
                    for r in conn.execute("SELECT key, value FROM catalog_meta")}

    # ---------------- 单篇 ----------------
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """按 pmcid 查；查不到再按 doc_id 查一次。

        两者在本语料里 99.999% 相同（doc_id 的 distinct 只比 pmcid 多 17），
        但任务书说的是 "id 查询"，两种 id 都认才不会让调用方猜。
        """
        conn = self._require()
        key = (doc_id or "").strip()
        if not key:
            return None
        with self._lock:
            row = conn.execute("SELECT * FROM documents WHERE pmcid=?", (key,)).fetchone()
            if row is None:
                row = conn.execute("SELECT * FROM documents WHERE doc_id=? LIMIT 1",
                                   (key,)).fetchone()
        return _row_to_doc(row) if row is not None else None

    # ---------------- 列表 ----------------
    def list_documents(self, journal: Optional[str] = None,
                       pub_year: Optional[int] = None,
                       year_from: Optional[int] = None,
                       year_to: Optional[int] = None,
                       title_contains: Optional[str] = None,
                       cursor: Optional[str] = None,
                       limit: int = LIST_LIMIT_DEFAULT,
                       with_total: bool = False) -> Dict[str, Any]:
        """游标分页。**没有 offset**——227 万行上 `LIMIT n OFFSET 100000` 要扫掉前十万行，
        页码越深越慢，这就是所谓翻页深渊。游标是上一页最后一条的 pmcid，恒定代价。

        `with_total` 默认关：无过滤时 `COUNT(*)` 要全表扫（实测见 `--bench`），
        而它对翻页并不必要。要总数就显式要，代价写在接口 description 里。
        """
        conn = self._require()
        limit = max(1, min(int(limit), LIST_LIMIT_MAX))

        where: List[str] = []
        args: List[Any] = []
        if journal:
            where.append("journal = ?")
            args.append(journal)
        if pub_year is not None:
            where.append("pub_year = ?")
            args.append(int(pub_year))
        if year_from is not None:
            where.append("pub_year >= ?")
            args.append(int(year_from))
        if year_to is not None:
            where.append("pub_year <= ?")
            args.append(int(year_to))
        if title_contains:
            where.append("title LIKE ? ESCAPE '\\'")
            args.append("%" + _like_escape(title_contains) + "%")

        filters = list(where)
        filter_args = list(args)
        if cursor:
            where.append("pmcid > ?")
            args.append(cursor)

        sql = "SELECT * FROM documents"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY pmcid LIMIT ?"
        args.append(limit + 1)                      # 多取一条判断还有没有下一页

        t0 = time.time()
        with self._lock:
            rows = conn.execute(sql, args).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = [_row_to_doc(r) for r in rows]

        total = None
        if with_total:
            csql = "SELECT COUNT(*) AS n FROM documents"
            if filters:
                csql += " WHERE " + " AND ".join(filters)
            with self._lock:
                total = int(conn.execute(csql, filter_args).fetchone()["n"])

        return {
            "items": items,
            "next_cursor": items[-1]["pmcid"] if (items and has_more) else None,
            "has_more": has_more,
            "total": total,
            "elapsed_ms": round((time.time() - t0) * 1000, 2),
        }


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def _row_to_doc(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["sections"] = [s for s in (d.get("sections") or "").split(",") if s]
    d["pmid"] = d.get("pmid") or None
    d["abstract"] = d.get("abstract") or None
    return d


# ============================================================================
# 二、索引统计
# ============================================================================
def dir_size(path: str) -> Tuple[int, int]:
    """(字节数, 文件数)。只 stat 不读内容——Chroma 目录 6 个文件、BM25 7 个，
    整个 65 GB 目录量一遍是微秒级。"""
    total, n = 0, 0
    if os.path.isfile(path):
        return os.path.getsize(path), 1
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return total, n


def human_size(n: Optional[int]) -> str:
    if not n:
        return "0 B"
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024
    return f"{v:.1f} TB"


class IndexStats:
    """`data/index_stats.json` 的读侧。服务**启动时**读一次，请求路径上零计算。

    尺寸例外：13 个文件的 `getsize` 是微秒级，且比缓存值更真（建完库后目录还会长），
    所以 `load(refresh_sizes=True)` 会在启动时重新 stat 一遍。

    ⚠ 文件缺失时 `available=False`、各计数为 **None 而不是 0**。
    0 是一个看起来正常的假数字，null 才能让调用方看出"这个数还没算"。
    """

    def __init__(self, data: Optional[Dict[str, Any]], path: str = STATS_PATH,
                 detail: str = ""):
        self.path = path
        self.data = data or {}
        self.available = bool(data)
        self.detail = detail

    @classmethod
    def load(cls, path: str = STATS_PATH, refresh_sizes: bool = True) -> "IndexStats":
        if not os.path.isfile(path):
            return cls(None, path, f"缺 {path}（跑 服务_文档目录.py --stats 生成）")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:                      # noqa: BLE001
            return cls(None, path, f"{type(e).__name__}: {e}")
        st = cls(data, path, f"{path}（算于 {data.get('computed_at', '?')}）")
        if refresh_sizes:
            try:
                st.data["index_size"] = measure_sizes()
                st.data["index_size"]["refreshed_at"] = _now()
            except Exception:                       # noqa: BLE001
                pass
        return st

    # ---- 给 /qa/stats 用的扁平视图 ----
    def as_stats_fields(self) -> Dict[str, Any]:
        d = self.data
        docs = (d.get("documents") or {})
        chunks = (d.get("chunks") or {})
        size = (d.get("index_size") or {})
        built = (d.get("built_at") or {})
        return {
            "available": self.available,
            "total_documents": docs.get("total"),
            "total_chunks": chunks.get("total"),
            "documents_note": docs.get("note"),
            "index_size_bytes": size.get("total"),
            "index_size_human": human_size(size.get("total")),
            "index_size_detail": {k: v for k, v in size.items()
                                  if k not in ("total", "refreshed_at")},
            "incremental_updates": d.get("incremental_updates") if self.available else None,
            "incremental_updates_note": d.get("incremental_updates_note"),
            "last_index_built_at": built or None,
            "corpus_snapshot": d.get("corpus_snapshot"),
            "computed_at": d.get("computed_at"),
            "detail": self.detail,
        }


def measure_sizes() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    total = 0
    for name, path in (("vector_db", CHROMA_DIR), ("bm25_index", BM25_DIR),
                       ("doc_catalog", DB_PATH)):
        b, n = dir_size(path)
        out[name] = {"bytes": b, "human": human_size(b), "files": n, "path": path,
                     "exists": b > 0}
        total += b
    out["total"] = total
    out["total_human"] = human_size(total)
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:                               # noqa: BLE001
        return {}


def write_index_stats(db_path: str = DB_PATH, stats_path: str = STATS_PATH,
                      quiet: bool = False) -> Dict[str, Any]:
    """算一次并落盘。**所有计数都从已建好的目录库读**（227 万行的聚合，毫秒级），
    不再去碰 parquet 或 Chroma。"""
    cat = DocCatalog(db_path)
    meta = cat.meta() if cat.ready else {}
    vec = _read_json(VECTOR_STATS)
    bm = _read_json(os.path.join(BM25_DIR, "index_meta.json"))

    def _i(key: str) -> Optional[int]:
        v = meta.get(key)
        return int(v) if v not in (None, "") else None

    docs_total, chunks_total = _i("documents"), _i("indexed_chunks")
    data: Dict[str, Any] = {
        "computed_at": _now(),
        "corpus_snapshot": CORPUS_SNAPSHOT,
        "documents": {
            "total": docs_total,
            "note": "被抽中至少一块的文献数。4M 索引是从 92,432,502 块里按**块**分层抽样"
                    "得到的，不是按文献抽——所以这不等于「227 万篇完整文献在库里」",
            "with_abstract": _i("docs_with_abstract"),
            "single_chunk_only": _i("docs_single_chunk"),
            "fully_indexed": _i("docs_fully_indexed"),
        },
        "chunks": {
            "total": chunks_total,
            "per_document_mean": (round(chunks_total / docs_total, 2)
                                  if docs_total and chunks_total else None),
            "original_per_document_mean": float(meta["orig_chunks_mean"])
            if meta.get("orig_chunks_mean") else None,
        },
        "index_size": measure_sizes(),
        "built_at": {
            "vector_db": vec.get("index_built_at"),
            "bm25_index": bm.get("built_at"),
            "doc_catalog": meta.get("built_at"),
        },
        "incremental_updates": 0,
        "incremental_updates_note": INCREMENTAL_NOTE,
        "sources": {
            "catalog_db": db_path,
            "vector_stats": VECTOR_STATS,
            "bm25_meta": os.path.join(BM25_DIR, "index_meta.json"),
            "build_input": meta.get("source"),
        },
    }
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if not quiet:
        print(f"索引统计 → {stats_path}")
        print(f"  文献 {docs_total:,} 篇 / 块 {chunks_total:,} 条"
              if docs_total and chunks_total else "  ⚠ 目录库不可用，计数为 null")
        print(f"  索引占盘 {data['index_size']['total_human']}")
    return data


# ============================================================================
# 三、建目录（离线，要 pyarrow）
# ============================================================================
def _canonical_map() -> Dict[str, str]:
    """原始 section 值 → 规范名。取自阶段五扫出来的 `corpus_meta.json`。"""
    meta = _read_json(CORPUS_META)
    out: Dict[str, str] = {}
    for canon, raws in ((meta.get("section") or {}).get("canonical_to_raw") or {}).items():
        for raw in raws:
            out[str(raw)] = canon
    return out


def build_catalog(parquet_path: str = MERGED_PARQUET, db_path: str = DB_PATH,
                  with_abstract: bool = True, quiet: bool = False) -> Dict[str, Any]:
    """从建库输入 parquet 聚出文献级目录。

    ⚠ **必须用 `merged_4m.parquet`，不能用 `subset_4000000_s42.parquet`**：
    后者有 4,002,283 行 / 2,276,496 篇，比真正入库的多 4,283 块 / 2,329 篇。
    拿它建目录会凭空多出两千多篇「查得到、但检索不到」的幽灵文献，
    而 `3001 文档不存在` 本该是准的。
    """
    import numpy as np                              # 只有建库才需要，服务侧不 import
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    t_all = time.time()
    say = (lambda *a: None) if quiet else print

    say(f"读 {parquet_path} …")
    t0 = time.time()
    cols = ["pmcid", "doc_id", "pmid", "source_title", "journal",
            "pub_year", "total_chunks", "section"]
    tb = pq.read_table(parquet_path, columns=cols)
    n_rows = tb.num_rows
    say(f"  {n_rows:,} 行   {time.time() - t0:.1f}s")

    # ---- 按 pmcid 分组：字典编码后全用 numpy，避免 227 万个 Python dict 项 ----
    t0 = time.time()
    pm = tb["pmcid"].combine_chunks().dictionary_encode()
    idx = pm.indices.to_numpy()
    n_docs = int(idx.max()) + 1
    indexed_chunks = np.bincount(idx, minlength=n_docs)

    # 每篇取"首次出现"那一行的字符串字段（同一篇内这些字段本来就一致）
    _uniq, first_pos = np.unique(idx, return_index=True)
    take = np.sort(first_pos)                       # 按行号排 → dictionary 顺序对得上
    order = idx[take]                               # take 行对应的 doc 序号
    import pyarrow as pa
    take_arr = pa.array(take)

    def col_at(name: str) -> List[Any]:
        return tb[name].combine_chunks().take(take_arr).to_pylist()

    doc_ids = col_at("doc_id")
    pmids = col_at("pmid")
    titles = col_at("source_title")
    journals = col_at("journal")
    years = col_at("pub_year")
    orig_chunks = col_at("total_chunks")
    pmcids = col_at("pmcid")
    say(f"  分组：{n_docs:,} 篇   {time.time() - t0:.1f}s")

    # ---- 段落集合：规范名编码成位图，np.bitwise_or.at 一遍出结果 ----
    t0 = time.time()
    cmap = _canonical_map()
    sec_codes = {name: i for i, name in enumerate(CANONICAL_SECTIONS)}
    other = sec_codes["other"]
    sec_dict = tb["section"].combine_chunks().dictionary_encode()
    raw_vals = sec_dict.dictionary.to_pylist()
    lut = np.array([1 << sec_codes.get(cmap.get(v or "", "other"), other)
                    for v in raw_vals], dtype=np.int32)
    sec_idx = sec_dict.indices
    if sec_idx.null_count:                          # section 为空的块也要有归属，不能整篇丢
        sec_idx = sec_idx.fill_null(len(raw_vals))
        lut = np.append(lut, np.int32(1 << other))
    bits = np.zeros(n_docs, dtype=np.int32)
    np.bitwise_or.at(bits, idx, lut[sec_idx.to_numpy(zero_copy_only=False)])
    say(f"  段落位图   {time.time() - t0:.1f}s")

    # ---- 摘要正文：只对 section 规范名 = abstract 的块流式读 text ----
    abstracts: Dict[str, str] = {}
    if with_abstract:
        t0 = time.time()
        abs_bit = 1 << sec_codes["abstract"]
        pf = pq.ParquetFile(parquet_path)
        parts: Dict[str, List[Tuple[int, str]]] = {}
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["pmcid", "section", "chunk_index", "text"]):
            sec = batch.column("section")
            keep = pc.is_in(sec, value_set=pa.array(
                [v for v in raw_vals if cmap.get(v or "", "other") == "abstract"]))
            if not pc.any(keep).as_py():
                continue
            sub = batch.filter(keep)
            for p, ci, tx in zip(sub.column("pmcid").to_pylist(),
                                 sub.column("chunk_index").to_pylist(),
                                 sub.column("text").to_pylist()):
                parts.setdefault(p, []).append((int(ci), tx or ""))
        for p, lst in parts.items():
            lst.sort()
            abstracts[p] = "\n".join(t for _, t in lst).strip()
        say(f"  摘要 {len(abstracts):,} 篇（{100 * len(abstracts) / n_docs:.1f}%）"
            f"   {time.time() - t0:.1f}s   [abs_bit={abs_bit}]")

    # ---- 写库 ----
    t0 = time.time()
    tmp_path = db_path + ".building"
    for p in (tmp_path, tmp_path + "-wal", tmp_path + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(tmp_path)
    conn.executescript(_SCHEMA)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")

    sec_names = list(CANONICAL_SECTIONS)

    def sections_of(mask: int) -> str:
        return ",".join(sec_names[i] for i in range(len(sec_names)) if mask & (1 << i))

    pos_of_doc = np.empty(n_docs, dtype=np.int64)
    pos_of_doc[order] = np.arange(len(order))       # doc 序号 → 上面几个列表里的下标

    rows: List[Tuple] = []
    written = 0
    for d in range(n_docs):
        i = int(pos_of_doc[d])
        pmcid = pmcids[i]
        rows.append((pmcid, doc_ids[i] or pmcid, pmids[i] or "", titles[i] or "",
                     journals[i] or "", int(years[i] or 0), int(orig_chunks[i] or 0),
                     int(indexed_chunks[d]), sections_of(int(bits[d])),
                     abstracts.get(pmcid)))
        if len(rows) >= 50000:
            conn.executemany("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                             rows)
            written += len(rows)
            rows.clear()
            if written % 500000 == 0:
                say(f"    写入 {written:,} / {n_docs:,}")
    if rows:
        conn.executemany("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        written += len(rows)
    conn.commit()
    say(f"  写入 {written:,} 行   {time.time() - t0:.1f}s")

    t0 = time.time()
    conn.executescript(_INDEXES)
    conn.execute("ANALYZE")
    conn.commit()
    say(f"  建索引   {time.time() - t0:.1f}s")

    orig = np.array([int(x or 0) for x in orig_chunks], dtype=np.int64)
    orig_by_doc = np.empty(n_docs, dtype=np.int64)
    orig_by_doc[order] = orig
    fully = int((indexed_chunks >= np.maximum(orig_by_doc, 1)).sum())
    single = int((indexed_chunks == 1).sum())
    meta_rows = {
        "built_at": _now(),
        "source": parquet_path,
        "documents": str(n_docs),
        "indexed_chunks": str(int(n_rows)),
        "orig_chunks_mean": f"{orig_by_doc.mean():.2f}",
        "docs_with_abstract": str(len(abstracts)),
        "docs_single_chunk": str(single),
        "docs_fully_indexed": str(fully),
        "corpus_snapshot": CORPUS_SNAPSHOT,
        "builder": os.path.basename(__file__),
    }
    conn.executemany("INSERT OR REPLACE INTO catalog_meta VALUES (?,?)",
                     list(meta_rows.items()))
    conn.commit()
    conn.close()

    for p in (db_path, db_path + "-wal", db_path + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    os.replace(tmp_path, db_path)                   # 换名是原子的：不会留下半个库

    say(f"目录 → {db_path}   {human_size(os.path.getsize(db_path))}"
        f"   总用时 {time.time() - t_all:.1f}s")
    return meta_rows


# ============================================================================
# 四、CLI
# ============================================================================
def bench(db_path: str = DB_PATH) -> None:
    """量真实耗时。列表/详情要毫秒级才配放在请求路径上；COUNT 慢是预期的，
    所以它在接口上是可选项而不是默认项。"""
    cat = DocCatalog(db_path)
    if not cat.ready:
        print("目录库不可用：" + cat.healthy()[1])
        return
    cases = [
        ("详情 get(pmcid)", lambda: cat.get("PMC212698")),
        ("列表 无过滤 首页", lambda: cat.list_documents(limit=20)),
        ("列表 游标第 5 页", lambda: cat.list_documents(cursor="PMC5000000", limit=20)),
        ("列表 journal 过滤", lambda: cat.list_documents(journal="PLoS ONE", limit=20)),
        ("列表 年份过滤", lambda: cat.list_documents(pub_year=2020, limit=20)),
        ("列表 journal+年份", lambda: cat.list_documents(journal="PLoS ONE",
                                                       pub_year=2020, limit=20)),
        ("标题 LIKE %kw%", lambda: cat.list_documents(title_contains="CRISPR", limit=20)),
        ("COUNT 无过滤", lambda: cat.list_documents(limit=1, with_total=True)),
        ("COUNT journal 过滤", lambda: cat.list_documents(journal="PLoS ONE", limit=1,
                                                        with_total=True)),
    ]
    print(f"{'用例':<22}{'耗时':>10}   结果")
    for name, fn in cases:
        t = time.time()
        out = fn()
        ms = (time.time() - t) * 1000
        if isinstance(out, dict) and "items" in out:
            got = f"{len(out['items'])} 条" + (f"   total={out['total']:,}"
                                              if out["total"] is not None else "")
        else:
            got = "命中" if out else "无"
        print(f"{name:<22}{ms:>8.1f}ms   {got}")


def main() -> int:
    ap = argparse.ArgumentParser(description="文献目录（离线建）与索引统计")
    ap.add_argument("--build", action="store_true", help="从 merged_4m.parquet 建目录")
    ap.add_argument("--parquet", default=MERGED_PARQUET)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--no-abstract", action="store_true", help="不抽摘要正文（库小约 300MB）")
    ap.add_argument("--stats", action="store_true", help="只重算 index_stats.json")
    ap.add_argument("--show", metavar="ID", help="按 pmcid / doc_id 查一篇")
    ap.add_argument("--list", action="store_true", help="列前几篇")
    ap.add_argument("--journal", default=None)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--bench", action="store_true", help="量列表/详情/COUNT 耗时")
    args = ap.parse_args()

    if args.build:
        build_catalog(args.parquet, args.db, with_abstract=not args.no_abstract)
        write_index_stats(args.db)
        return 0
    if args.stats:
        write_index_stats(args.db)
        return 0
    if args.bench:
        bench(args.db)
        return 0

    cat = DocCatalog(args.db)
    ok, detail = cat.healthy()
    if not ok:
        print("目录库不可用：" + detail)
        return 1
    if args.show:
        doc = cat.get(args.show)
        print(json.dumps(doc, ensure_ascii=False, indent=2) if doc else f"没有 {args.show}")
        return 0 if doc else 1
    if args.list or args.journal or args.year or args.title:
        out = cat.list_documents(journal=args.journal, pub_year=args.year,
                                 title_contains=args.title, limit=10, with_total=False)
        for d in out["items"]:
            print(f"  {d['pmcid']:<14}{(d['pub_year'] or '?')}  "
                  f"{(d['journal'] or '')[:28]:<30}{(d['title'] or '')[:60]}")
        print(f"  —— {len(out['items'])} 条   {out['elapsed_ms']}ms"
              f"   next_cursor={out['next_cursor']}")
        return 0
    print(detail)
    print(json.dumps(cat.meta(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
