# -*- coding: utf-8 -*-
"""第六阶段 · 多路检索 + 多准则重排 + 完整检索流水线

在上周的查询理解层（检索_查询理解.py）之上，接一条真正的混合检索链路：

    原始查询 ─ MedicalQueryProcessor.process_query ─▶ EnhancedQuery
                                                        │
        ┌───────────────────────────────────────────────┤
        ▼(向量：BGE+Chroma，多变体加权)        ▼(关键词：BM25/bm25s)
     稠密候选                               稀疏候选
        └────────────── 融合(simple/rrf/weighted) ──────────────┐
                                                                ▼
                                                   MultiPathRetriever 候选池
                                                                ▼
                              MultiCriteriaReranker(bge-reranker-base + 时效 + 权威)
                                                                ▼
                                                         最终证据列表

三个类：
  · MultiPathRetriever   —— 向量检索 + BM25 检索 + 三种融合策略
  · MultiCriteriaReranker —— 交叉编码器相关性 × 时效性 × 权威性 的加权重排
  · RetrievalPipeline    —— 把查询理解、多路检索、重排串成一条 search() 调用

设计取舍见各类 docstring；实测口径与阶段五保持一致（加权 RRF 的权重来自查询理解层）。

用法（需要 4M Chroma + 已构建的 BM25 索引）：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_多路检索.py \
      --bm25 E:\\rag\\data\\bm25_index_500k --query "heart attack prevention with aspirin, recent studies"
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")       # 硬覆盖：用户级 HF_HOME 指向旧路径
os.environ["HF_HUB_OFFLINE"] = "1"               # bge-base / bge-reranker-base 均已缓存
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pyarrow.parquet as pq                      # 必须早于 torch（本机先 torch 再 pyarrow 会段错误）

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 默认资源路径 ----
CHROMA_PATH = os.path.join(ROOT, "data", "chroma_db_4m")
COLLECTION = "medrag_bge_base"
CORPUS_META = os.path.join(ROOT, "data", "dict", "corpus_meta.json")
RERANKER_MODEL = "BAAI/bge-reranker-base"

#: P0 landmark 补充集合（独立 collection，与主库并行检索、结果合并）。
#: 语料是 oa_comm 子集，NEJM/JAMA/Lancet 的原始 RCT 与学会指南许可证不满足 CC BY，
#: **物理上不在主库里**（2026-08-15 按 PMID 逐条查证：10 篇全部查无此条）。
#: 缺这个目录不报错，只是这一路自动关掉——它是增强，不是依赖。
LANDMARK_PATH = os.path.join(ROOT, "data", "chroma_landmark")
LANDMARK_COLLECTION = "medrag_landmark"
LANDMARK_QUOTA = 2                       # 保底进最终结果的条数，P0 建议值


# ============================================================================
# 按路径导入中文文件名模块（查询理解 / 建库嵌入器 / BM25 分词公共约定）
# ============================================================================
def _load_by_path(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_qu = _load_by_path("chaxun_lijie", "检索_查询理解.py")
MedicalQueryProcessor = _qu.MedicalQueryProcessor
EnhancedQuery = _qu.EnhancedQuery

_jk = _load_by_path("jianku", "向量化_建库.py")
BGEEmbedder = _jk.BGEEmbedder

_bc = _load_by_path("bm25_common", "检索_BM25公共.py")
bm25_tokenize = _bc.bm25_tokenize
BM25_TOKENIZER_META = _bc.BM25_TOKENIZER_META

import bm25s


# ============================================================================
# 候选数据结构
# ============================================================================
@dataclass
class Candidate:
    """一条检索候选，贯穿融合与重排两个阶段，字段按需填充。"""
    chunk_id: str
    text: str = ""
    metadata: dict = field(default_factory=dict)

    # 两路各自的原始信号（命中该路才有值）
    cos_sim: Optional[float] = None       # 向量路：最高变体余弦相似度
    vec_rank: Optional[int] = None        # 向量路：聚合后最好排名（1-based）
    bm25_score: Optional[float] = None    # 关键词路：BM25 分
    bm25_rank: Optional[int] = None       # 关键词路：BM25 排名（1-based）
    sources: Set[str] = field(default_factory=set)   # {'vector','keyword'}

    # 融合
    fused_score: float = 0.0
    fusion_strategy: str = ""

    # 重排（多准则）
    rel_score: Optional[float] = None         # 交叉编码器相关性 [0,1]
    recency_score: Optional[float] = None     # 时效性 [0,1]
    authority_score: Optional[float] = None   # 权威性 [0,1]
    rerank_score: Optional[float] = None       # 加权总分（tiebreak 模式下是"档内展示分"，不用于排序）
    rerank_key: Optional[tuple] = None         # **排序的唯一来源**，见 MultiCriteriaReranker.sort_key

    def source_tag(self):
        v = "V" if "vector" in self.sources else "·"
        k = "K" if "keyword" in self.sources else "·"
        return v + k

    def brief(self, width=90):
        m = self.metadata or {}
        head = (f"{self.chunk_id}  [{self.source_tag()}]  "
                f"{m.get('journal','?')} {m.get('pub_year','?')} · {m.get('section','?')}")
        t = (self.text or "").replace("\n", " ")
        if len(t) > width:
            t = t[:width] + " …"
        return head, t


# ============================================================================
# Chroma where / post_filter 的 Python 复现
#   向量路命中已被 Chroma 强制满足 filters；BM25 路命中没有，需要在这里补齐，
#   而 section 这类后置过滤对两路都要在检索后统一施加（与查询理解层的语义一致）。
# ============================================================================
def match_where(meta: dict, where: dict) -> bool:
    """支持查询理解层会产出的形状：$and / $gte / $lte / $gt / $lt / $in / $eq / $ne，
    以及裸值等值（如 {"journal": "PLoS ONE"}）。"""
    if not where:
        return True
    if "$and" in where:
        return all(match_where(meta, c) for c in where["$and"])
    if "$or" in where:
        return any(match_where(meta, c) for c in where["$or"])
    for field_name, cond in where.items():
        if field_name in ("$and", "$or"):
            continue
        val = meta.get(field_name)
        if isinstance(cond, dict):
            for op, tgt in cond.items():
                if op == "$gte" and not (val is not None and val >= tgt):
                    return False
                if op == "$lte" and not (val is not None and val <= tgt):
                    return False
                if op == "$gt" and not (val is not None and val > tgt):
                    return False
                if op == "$lt" and not (val is not None and val < tgt):
                    return False
                if op == "$in" and val not in tgt:
                    return False
                if op == "$nin" and val in tgt:
                    return False
                if op == "$eq" and val != tgt:
                    return False
                if op == "$ne" and val == tgt:
                    return False
        else:
            if val != cond:
                return False
    return True


class SectionPostFilter:
    """section 后置过滤：复用查询理解层用于下推 $in 的同一份 canonical_to_raw，
    对候选的原始 section 做「是否属于该规范类」判断——保证与下推路径同一套归一化规则。"""

    def __init__(self, corpus_meta_path=CORPUS_META):
        self.c2r: Dict[str, Set[str]] = {}
        if os.path.exists(corpus_meta_path):
            with open(corpus_meta_path, encoding="utf-8") as f:
                cm = json.load(f)
            for canon, variants in cm.get("section", {}).get("canonical_to_raw", {}).items():
                self.c2r[canon] = set(variants)

    def match(self, meta: dict, post: dict) -> bool:
        if not post:
            return True
        canon = post.get("section_canon")
        if canon is None:
            return True
        allowed = self.c2r.get(canon)
        if allowed is None:          # 没有该类的写法表，无法判定，不误杀
            return True
        return meta.get("section") in allowed


# ============================================================================
# 一、多路检索器
# ============================================================================
class MultiPathRetriever:
    """向量检索 + 关键词检索(BM25) + 融合。

    输入是查询理解层的 EnhancedQuery：向量一路直接消费 `vector_queries`（已带 BGE
    指令前缀、含消歧变体与权重），关键词一路用 `keyword_groups` 拼成词袋喂给 BM25。

    融合策略（fusion_strategy）：
      · 'simple'   并集去重。定序 = 命中路数优先，再看最好排名。实现最简单，但基本
                   忽略了各路内部的排名/分数差异——rank1 和 rank50 只作为次序 tiebreak。
      · 'rrf'      倒数排名融合。score = Σ_list w_list / (rrf_k + rank)。学术检索常用，
                   跨路只看排名不看分数量纲，稳健；向量各变体按查询理解层给的权重参与，
                   关键词路按 keyword_weight 参与（阶段五实测：k=60 时 2× 权重变化几乎
                   不影响结果，所以默认等权，权重只是留出调节口）。
      · 'weighted' 加权分数融合。对候选集内的余弦相似度与 BM25 分各自 min-max 归一化后
                   加权求和，score = vw·norm(cos) + (1-vw)·norm(bm25)，默认 vw=0.7 让
                   向量路权重更高。缺一路的分量记 0。

    融合后统一从 Chroma 按 id 取正文+元数据，并对两路候选一致地施加 filters + 后置过滤，
    保证整份结果都满足查询理解层解析出的年份/章节/期刊约束。
    """

    def __init__(self, collection, embedder: BGEEmbedder,
                 bm25_retriever, bm25_doc_ids: List[str],
                 section_filter: Optional[SectionPostFilter] = None,
                 rrf_k: int = 60, default_vector_weight: float = 0.7,
                 keyword_weight: float = 1.0,
                 vector_filter_mode: str = "postfilter",
                 filter_oversample: int = 10, filter_oversample_cap: int = 500,
                 verbose: bool = False):
        self.col = collection
        self.embedder = embedder
        self.bm25 = bm25_retriever
        self.doc_ids = bm25_doc_ids
        self.n_bm25 = len(bm25_doc_ids)
        self.section_filter = section_filter or SectionPostFilter()
        self.rrf_k = rrf_k
        self.default_vector_weight = default_vector_weight
        self.keyword_weight = keyword_weight
        # 过滤策略：'postfilter'（默认，快）不给 Chroma 下推 where，改无过滤检索+过量取样+Python 后过滤；
        #           'where'（精确，慢）把 filters 下推给 Chroma。见 retrieve() docstring 里的实测原因。
        self.vector_filter_mode = vector_filter_mode
        self.filter_oversample = filter_oversample
        self.filter_oversample_cap = filter_oversample_cap
        self.verbose = verbose

    # ---------------- 向量检索 ----------------
    def _vector_search(self, eq: EnhancedQuery, top_k: int, push_where: bool):
        """对每个向量变体查一次 Chroma，返回 [(weight, [(id, rank, cos_sim), ...]), ...]。

        vector_queries 已含 BGE 指令前缀，所以直接走 _encode（不能再调 embed_query，会二次加前缀）。
        push_where=False 时不给 Chroma 下推过滤（无过滤 HNSW ~1ms，带过滤 ~100s，见 retrieve）。
        """
        queries = eq.vector_queries or [eq.vector_query_expanded or (self.embedder and "")]
        queries = [q for q in queries if q]
        if not queries:
            return []
        weights = list(eq.vector_query_weights) + [1.0] * (len(queries) - len(eq.vector_query_weights))
        vecs = self.embedder._encode(queries, batch_size=max(1, len(queries)))
        where = (eq.filters or None) if push_where else None
        res = self.col.query(
            query_embeddings=[v.tolist() for v in vecs],
            n_results=top_k, where=where, include=["distances"],
        )
        out = []
        for i in range(len(queries)):
            ids = res["ids"][i]
            dists = res["distances"][i]
            lst = [(cid, rank, 1.0 - d) for rank, (cid, d) in enumerate(zip(ids, dists), 1)]
            out.append((weights[i], lst))
        return out

    # ---------------- 关键词检索（BM25） ----------------
    def _bm25_query_text(self, eq: EnhancedQuery) -> str:
        """用实体+同义词+残余实词组成的词袋（keyword_groups 的平铺）作为 BM25 查询；
        BM25 是词袋模型，布尔算符对它没意义，同义词平铺反而能扩大字面命中。"""
        terms = []
        for g in eq.keyword_groups:
            terms.extend(g)
        text = " ".join(terms).strip()
        return text or eq.core_text or eq.cleaned

    def _keyword_search(self, eq: EnhancedQuery, top_k: int):
        """BM25 检索，返回 [(id, rank, bm25_score), ...]。"""
        qtext = self._bm25_query_text(eq)
        if not qtext.strip():
            return []
        toks = bm25_tokenize([qtext])
        k = min(top_k, self.n_bm25)
        if k <= 0:
            return []
        results, scores = self.bm25.retrieve(toks, k=k, show_progress=False)
        out = []
        for rank, (idx, sc) in enumerate(zip(results[0], scores[0]), 1):
            idx = int(idx)
            if 0 <= idx < self.n_bm25:
                out.append((self.doc_ids[idx], rank, float(sc)))
        return out

    # ---------------- 融合 ----------------
    @staticmethod
    def _minmax(vals: Dict[str, float]):
        if not vals:
            return {}
        lo = min(vals.values())
        hi = max(vals.values())
        if hi <= lo:
            return {k: 1.0 for k in vals}      # 全相等 → 都记满分（不制造虚假区分）
        return {k: (v - lo) / (hi - lo) for k, v in vals.items()}

    def _fuse(self, vec_lists, kw_list, strategy: str):
        cands: Dict[str, Candidate] = {}
        rrf = defaultdict(float)

        def get(cid):
            c = cands.get(cid)
            if c is None:
                c = cands[cid] = Candidate(chunk_id=cid)
            return c

        for w, lst in vec_lists:
            for cid, rank, cos in lst:
                c = get(cid)
                c.sources.add("vector")
                c.cos_sim = cos if c.cos_sim is None else max(c.cos_sim, cos)
                c.vec_rank = rank if c.vec_rank is None else min(c.vec_rank, rank)
                rrf[cid] += w / (self.rrf_k + rank)
        for cid, rank, sc in kw_list:
            c = get(cid)
            c.sources.add("keyword")
            c.bm25_score = sc if c.bm25_score is None else max(c.bm25_score, sc)
            c.bm25_rank = rank if c.bm25_rank is None else min(c.bm25_rank, rank)
            rrf[cid] += self.keyword_weight / (self.rrf_k + rank)

        if strategy == "rrf":
            for cid, c in cands.items():
                c.fused_score = rrf[cid]
        elif strategy == "weighted":
            vw = self.default_vector_weight
            ncos = self._minmax({cid: c.cos_sim for cid, c in cands.items() if c.cos_sim is not None})
            nbm = self._minmax({cid: c.bm25_score for cid, c in cands.items() if c.bm25_score is not None})
            for cid, c in cands.items():
                c.fused_score = vw * ncos.get(cid, 0.0) + (1 - vw) * nbm.get(cid, 0.0)
        elif strategy == "simple":
            # 并集去重：命中两路 > 命中一路；同档内谁的最好排名靠前谁优先。
            for cid, c in cands.items():
                best_rank = min(r for r in (c.vec_rank, c.bm25_rank) if r is not None)
                c.fused_score = len(c.sources) + 1.0 / (1.0 + best_rank)
        else:
            raise ValueError(f"未知融合策略：{strategy}（可选 simple/rrf/weighted）")

        for c in cands.values():
            c.fusion_strategy = strategy
        return cands

    # ---------------- 命中后取正文+元数据，并施加过滤 ----------------
    def _hydrate_and_filter(self, cands: Dict[str, Candidate], eq: EnhancedQuery):
        if not cands:
            return {}
        ids = list(cands.keys())
        got = self.col.get(ids=ids, include=["documents", "metadatas"])
        by_id = {cid: (doc, meta) for cid, doc, meta
                 in zip(got["ids"], got["documents"], got["metadatas"])}
        out = {}
        dropped = 0
        for cid, c in cands.items():
            if cid not in by_id:
                dropped += 1
                continue
            doc, meta = by_id[cid]
            c.text, c.metadata = doc or "", meta or {}
            if not match_where(c.metadata, eq.filters):
                dropped += 1
                continue
            if not self.section_filter.match(c.metadata, eq.post_filters):
                dropped += 1
                continue
            out[cid] = c
        if self.verbose and dropped:
            print(f"    [融合] 过滤丢弃 {dropped} 条（不满足 filters/后置过滤或未在库中）")
        return out

    # ---------------- 对外主入口 ----------------
    def retrieve(self, query_info: EnhancedQuery,
                 top_k_vector: int = 50, top_k_keyword: int = 50,
                 fusion_strategy: str = "rrf",
                 final_k: Optional[int] = None) -> List[Candidate]:
        """执行多路检索并融合。

        过滤策略（实测驱动，是本阶段的关键发现）：4M 集合上给 Chroma 下推 `where` 过滤会把
        单次向量查询从 ~1ms（纯 HNSW）拖到 ~100s——带过滤的 HNSW 在 4M 规模退化，慢约 5 个
        数量级。因此默认 vector_filter_mode='postfilter'：向量路不下推 where，改为无过滤检索、
        把 top_k 过量取样（×filter_oversample，上限 filter_oversample_cap），再在 hydrate 阶段
        用 Python（match_where + section 后过滤）对两路候选统一过滤。BM25 侧同样过量取样。
        对高选择性过滤（如 pub_year>=2026，命中占比极低），过量取样窗口可能兜不住，此时可传
        vector_filter_mode='where' 换精确但慢的下推。

        Args:
            query_info:      查询理解层输出的 EnhancedQuery
            top_k_vector:    向量检索每个变体返回数量（有过滤时内部会过量取样）
            top_k_keyword:   关键词检索返回数量（有过滤时内部会过量取样）
            fusion_strategy: 'simple' | 'rrf' | 'weighted'
            final_k:         融合后截断数量（None=全部；一般给重排留一个候选池，如 50）
        Returns:
            按 fused_score 降序的 Candidate 列表（已带正文+元数据、已过滤）
        """
        t0 = time.time()
        has_filter = bool(query_info.filters) or bool(query_info.post_filters)
        push_where = (self.vector_filter_mode == "where") and bool(query_info.filters)
        # postfilter 模式且有过滤：过量取样，保证过滤后仍有足够候选
        if has_filter and not push_where:
            eff_kv = min(self.filter_oversample_cap, top_k_vector * self.filter_oversample)
            eff_kk = min(self.filter_oversample_cap, top_k_keyword * self.filter_oversample)
        else:
            eff_kv, eff_kk = top_k_vector, top_k_keyword

        vec_lists = self._vector_search(query_info, eff_kv, push_where)
        t_vec = time.time()
        kw_list = self._keyword_search(query_info, eff_kk)
        t_kw = time.time()
        cands = self._fuse(vec_lists, kw_list, fusion_strategy)
        cands = self._hydrate_and_filter(cands, query_info)
        ranked = sorted(cands.values(), key=lambda c: c.fused_score, reverse=True)
        if final_k:
            ranked = ranked[:final_k]
        if self.verbose:
            n_v = sum(len(l) for _, l in vec_lists)
            mode = "where下推" if push_where else ("后过滤×%d" % self.filter_oversample if has_filter else "无过滤")
            print(f"    [多路检索] 向量 {len(vec_lists)}变体×{eff_kv}(共{n_v}) {t_vec-t0:.3f}s | "
                  f"BM25 top{eff_kk} {t_kw-t_vec:.3f}s | 融合({fusion_strategy},{mode}) "
                  f"→ {len(cands)} 候选 / 返回 {len(ranked)}  总 {time.time()-t0:.3f}s")
        return ranked


# ============================================================================
# 二、多准则重排器
# ============================================================================
# 期刊权威性：短名易被子串误伤，用精确全名小写映射；特异全名可安全子串匹配（特异在前）。
# 这是一份可调的启发式清单，不是绝对排名；未知期刊落到 default_authority。
JOURNAL_EXACT = {
    "science": 1.00, "nature": 1.00, "cell": 0.95, "blood": 0.90,
    "circulation": 0.90, "gut": 0.85, "hepatology": 0.85, "immunity": 0.90,
    "neuron": 0.90, "gastroenterology": 0.88, "diabetes": 0.80, "oncogene": 0.75,
    "elife": 0.85,
}
JOURNAL_SUBSTR = [
    ("new england journal of medicine", 1.00),
    ("science translational medicine", 0.90), ("science advances", 0.80),
    ("science signaling", 0.78),
    ("nature reviews", 0.92), ("nature medicine", 0.95), ("nature genetics", 0.92),
    ("nature biotechnology", 0.92), ("nature methods", 0.90), ("nature immunology", 0.90),
    ("nature neuroscience", 0.90), ("nature communications", 0.78),
    ("lancet", 0.95),
    ("proceedings of the national academy", 0.88), ("nucleic acids research", 0.88),
    ("journal of clinical oncology", 0.90), ("journal of experimental medicine", 0.88),
    ("cancer cell", 0.90), ("molecular cell", 0.90), ("cell metabolism", 0.90),
    ("cell reports", 0.78),
    ("genome biology", 0.85), ("genome research", 0.85),
    ("plos medicine", 0.88), ("plos biology", 0.85), ("plos genetics", 0.78),
    ("plos pathogens", 0.78), ("plos computational biology", 0.75), ("plos one", 0.62),
    ("journal of the american medical association", 0.92), ("jama", 0.90), ("bmj", 0.85),
    ("bmc medicine", 0.80), ("bmc ", 0.60),
    ("frontiers in immunology", 0.70), ("frontiers in ", 0.62),
    ("scientific reports", 0.62),
    ("international journal of molecular sciences", 0.60),
    ("journal of biological chemistry", 0.72),
    ("peerj", 0.60), ("sensors", 0.55), ("cureus", 0.50),
]


class MultiCriteriaReranker:
    """用交叉编码器（bge-reranker-base）做相关性打分，再结合时效性与权威性排序。

    与双塔向量检索不同，交叉编码器把 (query, passage) 拼在一起过 Transformer，能建模词级
    交互，相关性判别更准——但代价是每个候选都要过一次模型，所以只对融合后的小候选池重排。

    三个准则：
      · relevance 交叉编码器 logit 过 sigmoid → [0,1]
      · recency   从 pub_year 线性衰减，current_year→1，往前 recency_span 年→0；缺年份记 0.5
      · authority 按期刊查权威性映射，未知期刊记 default_authority

    --------------------------------------------------------------------------
    mode="tiebreak"（**2026-08-12 起的默认**）与 mode="weighted"（旧行为）
    --------------------------------------------------------------------------
    旧行为 `final = 0.60·rel + 0.25·rec + 0.15·auth` 在 golden 检索评测上被实测为
    **净伤害检索**：T1 主指标 R@10 0.417 / MRR 0.231，比它自己的向量腿（0.490 / 0.349）还差，
    净毁掉 11 条命中，被毁的 26 条里 13 条是 ≤2015、被救的 15 条一条 ≤2015 都没有。

    机理不是归一化 bug（池内 min-max 归一后更差，MRR 0.207）。真因是**饱和型相关性分 +
    线性年份分相加**这个结构：交叉编码器认可的候选 rel 全挤在 1.0 附近，池内前 20 的
    rel std 只剩 ~0.08，而它们发表年份铺得很开、recency std ~0.18（2.2 倍）——
    于是 0.25 的权重下 recency 的**排序影响力几乎等于 rel（44.0% vs 47.7%）**。

    改法：recency 不再当加法项，改当**同分裁决**——先按 ε 把 rel 分档，同档内才比 recency。
    实测（golden T1，237 条评测集，已用真跑证伪过离线反推）：

        方案                  ≤2015  2016-19  2020-22  2023+   R@10    MRR
        旧 0.60/0.25/0.15     0.300   0.406    0.538   0.452   0.425   0.241
        tiebreak ε=0.02       0.525   0.469    0.538   0.476   0.503   0.315
                            +22.5pp  +6.3pp    持平  +2.4pp
                              ↑ **没有一格更差，三格严格更好**（2020-2022 打平）

    ⚠ 措辞（别说成「recency 无价值」）：**收益来自把 recency 从加法项移除**；
      同分裁决是在此基础上**以可控的小代价保留 recency 的原意**，买的是这个 benchmark
      测不到的临床时效价值（「同等相关时新指南优于旧指南」）。纯相关性的 MRR 更高（0.357），
      但它完全放弃了时效偏好。详见 `任务10/检索评测改进/README.md`。
    """

    # ε 默认 0.02 的理由（敏感度扫描见报告，ε 在 0.005~0.05 是一片平台不是刀刃）：
    #   · ε=0.005 支配 ε=0.01（R@10 与四个年份桶都相同、MRR 更高），所以 0.01 没有当默认的理由
    #   · ε=0.02 是**唯一在 2023+ 上也严格更好**的配置（0.476 > 0.452；ε=0.01 那格只是持平），
    #     而 2023+ 正是 recency 本该发挥作用的地方
    DEFAULT_TIEBREAK_EPS = 0.02

    def __init__(self, model_name: str = RERANKER_MODEL,
                 criteria_weights: Optional[Dict[str, float]] = None,
                 current_year: Optional[int] = None, recency_span: int = 20,
                 default_authority: float = 0.50,
                 mode: str = "tiebreak", tiebreak_eps: Optional[float] = None,
                 device: str = "cuda", max_length: int = 512, batch_size: int = 32,
                 verbose: bool = False):
        if mode not in ("tiebreak", "weighted"):
            raise ValueError(f"未知重排模式：{mode}（可选 tiebreak/weighted）")
        self.mode = mode
        self.eps = float(tiebreak_eps if tiebreak_eps is not None else self.DEFAULT_TIEBREAK_EPS)
        if self.eps <= 0:
            raise ValueError(f"tiebreak_eps 必须为正，收到 {self.eps}")
        self.w = criteria_weights or {"relevance": 0.6, "recency": 0.25, "authority": 0.15}
        s = sum(self.w.values())
        if abs(s - 1.0) > 1e-6:                 # 允许传非归一权重，内部归一
            self.w = {k: v / s for k, v in self.w.items()}
        from datetime import datetime
        self.current_year = current_year or datetime.now().year
        self.recency_span = max(1, recency_span)
        self.default_authority = default_authority
        self.max_length = max_length
        self.batch_size = batch_size
        self.verbose = verbose
        # 设备选择只有一个来源：向量化_建库.pick_device（cuda → mps → cpu）。
        # 本模块本来就 import 它取 BGEEmbedder，不引入新依赖。
        self.device = _jk.pick_device(device)
        print(f"[重排] 加载 {model_name} 到 {self.device} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

    # ---- 相关性：交叉编码器 ----
    @torch.inference_mode()
    def _relevance(self, query_text: str, passages: List[str]) -> List[float]:
        scores = []
        for i in range(0, len(passages), self.batch_size):
            batch = [[query_text, p or ""] for p in passages[i:i + self.batch_size]]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores

    # ---- 时效性：按年份线性衰减 ----
    def _recency(self, year) -> float:
        if year is None or not isinstance(year, (int, float)) or year <= 0:
            return 0.5
        s = 1.0 - (self.current_year - year) / self.recency_span
        return min(1.0, max(0.0, s))

    # ---- 权威性：按期刊查表 ----
    def _authority(self, journal) -> float:
        if not journal:
            return self.default_authority
        j = str(journal).strip().lower()
        if j in JOURNAL_EXACT:
            return JOURNAL_EXACT[j]
        for needle, w in JOURNAL_SUBSTR:
            if needle in j:
                return w
        return self.default_authority

    def sort_key(self, c: Candidate) -> tuple:
        """**排序的唯一来源**——降序比较（`reverse=True`），逐级如下。

        `mode="tiebreak"`：
            ① rel 分档   round(rel/ε)      —— 相关性仍是第一位，ε 内视为同档
            ② recency                      —— 同档内才比时效（recency 的原意就在这一级）
            ③ fused_score                  —— 收尾①：稳定，且**与上游候选顺序无关**
            ④ pmid 字符串 → chunk_id       —— 收尾②：把顺序钉成全序，chunk_id 唯一
        `mode="weighted"`：把①②换成加权总分，③④不变。

        ⚠ **为什么③④必须显式写出来**（2026-08-12 定的规矩）：
          同分裁决**按构造就会制造大量并列**。此前排序靠 Python 稳定排序的副产品收尾——
          即"并列时保持输入顺序"，而输入顺序是**融合序**。那是个从没被设计过的第三级，
          却会真正决定名次。后果是检索结果**悄悄依赖上游融合顺序**：
          P0 加 landmark 并行路、P2b 改融合权重，都会让它无声地变化。
          加权和几乎不并列，所以这个隐式依赖在旧行为下从来没露过头。
          写死成显式键 = 零成本，但把隐式依赖变成显式决定。
        ⚠ 用 chunk_id 收尾而不是只用 pmid：`pmid` 在本语料**并不唯一**（勘误记录与原文共用，
          见 docs/工程笔记.md 三·7），只比到 pmid 仍可能残留并列；chunk_id 100% 唯一，保证全序。
        """
        md = c.metadata or {}
        tail = (c.fused_score, str(md.get("pmid") or ""), c.chunk_id)
        if self.mode == "tiebreak":
            return (round((c.rel_score or 0.0) / self.eps), c.recency_score or 0.0) + tail
        return (c.rerank_score if c.rerank_score is not None else -1e9,) + tail

    def rerank(self, query_text: str, candidates: List[Candidate],
               top_k: Optional[int] = None) -> List[Candidate]:
        """对候选做多准则重排。query_text 用查询理解层的 core_text（英文、无 BGE 前缀）。

        副作用：给每个候选填 rel/recency/authority/rerank_score 与 **rerank_key**。
        `rerank_key` 是排序的唯一来源——`candidate_rows()` 也用它，避免两处排序法则不一致
        （2026-08-12 踩过：两边排序输入顺序不同，并列处名次就分道扬镳）。
        """
        if not candidates:
            return []
        t0 = time.time()
        rel = self._relevance(query_text, [c.text for c in candidates])
        for c, r in zip(candidates, rel):
            c.rel_score = r
            c.recency_score = self._recency((c.metadata or {}).get("pub_year"))
            c.authority_score = self._authority((c.metadata or {}).get("journal"))
            # 加权总分：weighted 模式用来排序；tiebreak 模式下只作展示与对照，不参与排序
            c.rerank_score = (self.w["relevance"] * c.rel_score
                              + self.w["recency"] * c.recency_score
                              + self.w["authority"] * c.authority_score)
        for c in candidates:
            c.rerank_key = self.sort_key(c)
        ranked = sorted(candidates, key=lambda c: c.rerank_key, reverse=True)
        if self.verbose:
            print(f"    [重排] {len(candidates)} 候选过交叉编码器（mode={self.mode}"
                  + (f", ε={self.eps}" if self.mode == "tiebreak" else "")
                  + f"），用时 {time.time()-t0:.2f}s")
        return ranked[:top_k] if top_k else ranked


# ============================================================================
# 二之二、候选池导出（调试检索层用）
# ============================================================================
def candidate_rows(out: dict) -> List[Dict[str, Any]]:
    """把一次 `search()` 的**完整候选池**摊平成逐行记录（不只是 top_k）。

    每行带三个名次，缺一个都不够用来定位问题：
      `fused_rank`   融合之后、重排之前的名次（向量+BM25 的合力）
      `final_rank`   重排之后的名次（rel/recency/authority 加权的结果）
      `in_top_k`     最终有没有进上下文

    有了这三个才分得清"检索没找到"与"找到了但被重排压下去"——
    只看最终答案，这两种表现完全一样。
    """
    pool = out.get("candidates") or []
    fused_rank = {id(c): i + 1 for i, c in enumerate(pool)}
    # ⚠ 必须用 `rerank_key` 排，不能自己再拼一套排序法则——`rerank()` 用的就是它。
    #   两处各排各的，在**打分并列**时会给出不同名次（稳定排序保持各自的输入顺序），
    #   于是"离线复算的名次"和"运行时的名次"对不上。2026-08-12 被证伪抓到过，
    #   当时 13/237 不一致；加权和几乎不并列所以一直没露头，换成同分裁决立刻暴露。
    if out.get("reranked"):
        if pool and pool[0].rerank_key is not None:
            ranked = sorted(pool, key=lambda c: c.rerank_key, reverse=True)
        else:   # 兼容：外部注入的重排器没填 rerank_key
            ranked = sorted(pool, key=lambda c: (c.rerank_score if c.rerank_score is not None else -1),
                            reverse=True)
    else:
        ranked = list(pool)
    top_ids = {id(c) for c in (out.get("results") or [])}

    rows = []
    for i, c in enumerate(ranked):
        md = c.metadata or {}
        rows.append({
            "final_rank": i + 1,
            "fused_rank": fused_rank.get(id(c)),
            "in_top_k": id(c) in top_ids,
            "pmcid": md.get("pmcid"), "pmid": md.get("pmid"),
            "pub_year": md.get("pub_year"), "journal": md.get("journal"),
            "section": md.get("section"),
            "source_title": (md.get("source_title") or "")[:160],
            "sources": "+".join(sorted(c.sources or [])),
            "cos_sim": c.cos_sim, "vec_rank": c.vec_rank,
            "bm25_score": c.bm25_score, "bm25_rank": c.bm25_rank,
            "fused_score": c.fused_score,
            "rel_score": c.rel_score, "recency_score": c.recency_score,
            "authority_score": c.authority_score, "rerank_score": c.rerank_score,
            "chunk_id": c.chunk_id,
            # P0：主库正文块是 None，landmark 条目是 "landmark"。
            # 验收要按它分开看——「召回了相关文献」与「召回了那篇 landmark」不是一回事。
            "source_type": md.get("source_type"),
            "trial_name": md.get("trial_name"),
            "text_head": (c.text or "")[:200].replace("\n", " "),
        })
    return rows


def dump_candidates(out: dict, path: str) -> str:
    """把候选池写成 `.jsonl` 或 `.csv`（按扩展名选）。返回实际写入的路径。

    ⚠ **导出里看不到的不等于库里没有。** 候选池本身是被层层截断的：

        向量 top_k_vector(50) + BM25 top_k_keyword(50) → 融合 → 截到 rerank_pool(50) → 重排

    所以融合第 51 名之后的文献**根本不在池子里**，导出中当然也没有。
    找不到目标文献时，先把 `rerank_pool` 与两个 `top_k_*` 调大再看一次，
    再下结论是"检索没召回"还是"召回了被压下去"。这两个截断值会写进导出的 meta。
    """
    rows = candidate_rows(out)
    eq = out.get("enhanced")
    meta = {
        "_type": "meta",
        "query": out.get("query"),
        "retrieval_query": getattr(eq, "core_text", None),
        "translated_from": getattr(eq, "translated_from", None),
        "translate_method": getattr(eq, "translate_method", None),
        "filters": getattr(eq, "filters", None),
        "fusion_strategy": out.get("fusion_strategy"),
        "reranked": out.get("reranked"),
        "limits": out.get("limits"),
        "n_candidates": len(rows),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "候选池被 top_k_vector / top_k_keyword / rerank_pool 三处截断；"
                "融合名次在 rerank_pool 之后的文献不会出现在这里，"
                "找不到目标文献时先调大这三个值再下结论",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if path.lower().endswith(".csv"):
        import csv
        cols = list(rows[0].keys()) if rows else []
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            # CSV 没有放 meta 的地方，写成同名 .meta.json，别把它塞进表头骗人
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def find_in_candidates(out: dict, wanted: Sequence[str], show_top: int = 5) -> None:
    """在候选池里找指定的 PMID / PMCID，打印它落在第几位、以及压在它上面的是什么。

    P0 灌 landmark 数据之后要反复用这个：验收锚在"目标文献进没进 top-k、排第几"，
    而不是最终答案——后者受生成侧方差影响（见 docs/工程笔记.md 三·1）。
    """
    rows = candidate_rows(out)
    keys = {str(w).strip().upper().replace("PMC", "") for w in wanted if str(w).strip()}

    def _hit(r: Dict[str, Any]) -> bool:
        pm = str(r.get("pmid") or "").upper()
        pc = str(r.get("pmcid") or "").upper().replace("PMC", "")
        return pm in keys or pc in keys

    found = [r for r in rows if _hit(r)]
    lim = out.get("limits") or {}
    print(f"\n候选池 {len(rows)} 条"
          f"（向量 {lim.get('top_k_vector')} + BM25 {lim.get('top_k_keyword')}"
          f" → 融合 → 截到 {lim.get('rerank_pool')}）｜最终 top_k = {lim.get('top_k')}")
    if not found:
        print(f"  ✗ 目标 {sorted(keys)} **不在候选池里**。")
        print("    这不代表库里没有——先把 --rerank-pool / --top-k-vector / --top-k-keyword")
        print("    调大再看一次，才分得清「没召回」与「召回了被压下去」。")
        return
    print(f"  找到 {len(found)} 条目标：")
    for r in found:
        print(f"    final#{r['final_rank']:<4} fused#{r['fused_rank']:<4} "
              f"{'进top_k' if r['in_top_k'] else '未进top_k':<9} "
              f"rel={_f(r['rel_score'])} rec={_f(r['recency_score'])} "
              f"auth={_f(r['authority_score'])} → rerank={_f(r['rerank_score'])}  "
              f"{str(r['pmcid'])} {str(r['source_title'])[:52]}")
    print(f"\n  作为对照，前 {show_top} 名：")
    for r in rows[:show_top]:
        print(f"    final#{r['final_rank']:<4} fused#{r['fused_rank']:<4} "
              f"rel={_f(r['rel_score'])} rec={_f(r['recency_score'])} "
              f"auth={_f(r['authority_score'])} → rerank={_f(r['rerank_score'])}  "
              f"{str(r['pmcid'])} {str(r['source_title'])[:52]}")


def _f(v: Any) -> str:
    return "  —  " if v is None else f"{float(v):.3f}"


# ============================================================================
# 三、完整检索流水线
# ============================================================================
class RetrievalPipeline:
    """把查询理解 → 多路检索 → 多准则重排串成一条 search()。

    加载成本（首次）：Chroma 4M 集合约 15.8GB / 数分钟（HNSW 载入），BGE-base(~0.4GB)
    与 bge-reranker-base(~1.1GB) 上 GPU。之后每次 search() 只有单查询开销。
    """

    def __init__(self, bm25_dir: str,
                 chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION,
                 model_key: str = "bge-base", reranker_model: str = RERANKER_MODEL,
                 corpus_meta_path: str = CORPUS_META,
                 translate: str = "dict",
                 rrf_k: int = 60, default_vector_weight: float = 0.7,
                 criteria_weights: Optional[Dict[str, float]] = None,
                 recency_span: int = 20,
                 rerank_mode: str = "tiebreak",
                 tiebreak_eps: Optional[float] = None,
                 verbose: bool = False,
                 load_reranker: bool = True,
                 use_landmark: bool = True,
                 landmark_path: str = LANDMARK_PATH,
                 landmark_collection: str = LANDMARK_COLLECTION,
                 landmark_quota: int = LANDMARK_QUOTA):
        self.verbose = verbose
        print("[流水线] 初始化查询理解处理器 ...", flush=True)
        self.processor = MedicalQueryProcessor(
            corpus_meta_path=corpus_meta_path, translate=translate, verbose=verbose)

        print(f"[流水线] 打开 Chroma 集合 {collection_name} @ {chroma_path} ...", flush=True)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)

        print(f"[流水线] 加载嵌入模型 {model_key} ...", flush=True)
        self.embedder = BGEEmbedder(model_key)

        print(f"[流水线] mmap 载入 BM25 索引 @ {bm25_dir} ...", flush=True)
        self.bm25 = bm25s.BM25.load(bm25_dir, mmap=True)
        self.bm25_doc_ids = pq.read_table(
            os.path.join(bm25_dir, "doc_ids.parquet")).column("chunk_id").to_pylist()
        with open(os.path.join(bm25_dir, "index_meta.json"), encoding="utf-8") as f:
            self.bm25_meta = json.load(f)

        self.retriever = MultiPathRetriever(
            self.collection, self.embedder, self.bm25, self.bm25_doc_ids,
            section_filter=SectionPostFilter(corpus_meta_path),
            rrf_k=rrf_k, default_vector_weight=default_vector_weight, verbose=verbose)

        self.reranker = None
        if load_reranker:
            self.reranker = MultiCriteriaReranker(
                reranker_model, criteria_weights=criteria_weights,
                recency_span=recency_span, mode=rerank_mode, tiebreak_eps=tiebreak_eps,
                verbose=verbose)

        # ---- P0：landmark 独立 collection（缺了就静静关掉这一路，不影响主链路）----
        self.landmark_quota = int(landmark_quota)
        self.landmark_col = None
        self.landmark_detail = "未启用"
        if use_landmark:
            try:
                lc = chromadb.PersistentClient(path=landmark_path)
                self.landmark_col = lc.get_collection(landmark_collection)
                self.landmark_detail = (f"{landmark_collection} @ {landmark_path}"
                                        f"（{self.landmark_col.count()} 条，保底 {self.landmark_quota}）")
                print(f"[流水线] landmark 集合：{self.landmark_detail}", flush=True)
            except Exception as e:                    # noqa: BLE001
                self.landmark_col = None
                self.landmark_detail = f"不可用（{type(e).__name__}: {e}）——这一路已关闭"
                print(f"[流水线] landmark 集合{self.landmark_detail}", flush=True)

        # ⚠ 单位是「块」不是「篇」：bm25_doc_ids 装的是 chunk_id。
        # 3,998,000 块实际只覆盖 2,274,167 篇（平均每篇 1.76 块），印成「篇」会让人
        # 把入库规模读大 1.76 倍——这正是 /qa/stats 要用 documents_note 去防的那个误读。
        print(f"[流水线] 就绪：Chroma {self.collection.count():,} 向量 | "
              f"BM25 {len(self.bm25_doc_ids):,} 块\n", flush=True)

    # ------------------------------------------------------------------
    # P0：landmark 路
    # ------------------------------------------------------------------
    def _landmark_search(self, eq, n: int) -> List[Candidate]:
        """在 landmark collection 上做一次向量检索，产出带 `source_type=landmark` 的候选。

        ⚠ **必须用同一个 `self.embedder`**：landmark 集合建库时用的就是它
        （`向量化_建库.py::BGEEmbedder`，CLS pooling + L2 归一化）。换成别的包装器，
        两个 collection 的余弦分就不在同一个尺度上，而下面要把它们放进同一个池子里排序。

        ⚠ `eq.vector_queries` **已含 BGE 指令前缀**，所以走 `_encode` 而不是 `embed_query`
        （后者会二次加前缀）——与 `_vector_search` 里的处理一致。
        """
        if self.landmark_col is None:
            return []
        qs = list(getattr(eq, "vector_queries", None) or [eq.core_text])
        vecs = self.embedder._encode(qs, batch_size=len(qs))
        res = self.landmark_col.query(
            query_embeddings=[v.tolist() for v in vecs],
            n_results=min(max(1, n), self.landmark_col.count()),
            include=["documents", "metadatas", "distances"])

        best: Dict[str, Candidate] = {}
        for ids, docs, metas, dists in zip(res["ids"], res["documents"],
                                           res["metadatas"], res["distances"]):
            for rank, (cid, doc, md, dist) in enumerate(zip(ids, docs, metas, dists), 1):
                cos = 1.0 - float(dist)
                c = best.get(cid)
                if c is None:
                    md = dict(md or {})
                    md.setdefault("source_type", "landmark")
                    # 让下游（重排的 recency/authority、上下文组装、引用）拿到熟悉的字段名。
                    # ⚠ 用 `or` 不能用 `setdefault`：建库时 pmcid 这个键**存在但是空串**
                    #   （NEJM/JAMA 原文不在 PMC，本来就没有 PMCID），setdefault 不会覆盖空串，
                    #   结果出处会渲染成一个空的 PMCID——引用链断在这里，而且不报错。
                    md["pmcid"] = md.get("pmcid") or f"PMID:{md.get('pmid', '')}"
                    md.setdefault("section", "landmark-abstract")
                    md["source_title"] = md.get("source_title") or md.get("title", "")
                    c = best[cid] = Candidate(chunk_id=cid, text=doc or "", metadata=md)
                    c.sources.add("landmark")
                    c.fusion_strategy = "landmark"
                if c.cos_sim is None or cos > c.cos_sim:
                    c.cos_sim = cos
                    c.vec_rank = rank
        out = sorted(best.values(), key=lambda c: c.cos_sim or 0.0, reverse=True)

        # ⚠ fused_score 必须落在**主库那一路的同一个尺度上**，否则它会凭数值大小
        # 压过所有主库候选（RRF 分只有 0.016 量级，余弦是 0.8 量级）。
        # 这里按「虚拟名次」给一个 RRF 等价分：**它不是竞争出来的分，只是让排序有意义**。
        # 真实信号在 `cos_sim` 里，dump 出来能看到。
        for rank, c in enumerate(out, 1):
            c.fused_score = 1.0 / (self.retriever.rrf_k + rank)
        return out

    def search(self, raw_query: str, top_k: int = 10,
               top_k_vector: int = 50, top_k_keyword: int = 50,
               fusion_strategy: str = "rrf", rerank: bool = True,
               rerank_pool: int = 50, dump: Optional[str] = None,
               use_landmark: Optional[bool] = None,
               landmark_quota: Optional[int] = None) -> dict:
        """一次完整检索。

        `rerank_pool` 为什么默认 50（golden 235 条实测，不是拍的）：
        池子提大反而变差——T1 主指标 R@10 随池子 50 → **0.450**、100 → 0.444、
        200 → 0.417，**单调下降**。机理是池子越大，重排把「不相关但新且权威」的
        文献捞上来的机会也越多；早期观察到的「融合第 58 名重排后进前 10」现象是真的
        （约 4.6% 属此类），但那是个案收益，**净效果为负**。

        ⚠ **诊断用途必须开到 200+**，两个用途别合并成一个默认值：要查一篇文献是
        「根本没进候选池」还是「进了池但被压下去」，50 的池子分不开这两种情况，
        只有 `dump` 出来的 `fused_rank` / `final_rank` 才分得开。

        Args:
            dump: 传路径则把**完整候选池**（不只 top_k）连同重排各分项导出，
                  `.jsonl` 或 `.csv`。查"目标文献落在第几位、被什么压下去了"用它——
                  只看最终答案或 top_k 是查不出来的。详见 `dump_candidates`。

        Returns dict: {query, enhanced(EnhancedQuery), fusion_strategy, reranked(bool),
                       candidates(融合后候选池), results(最终 top_k), limits(各层截断参数)}
        """
        eq = self.processor.process_query(raw_query)
        pool = self.retriever.retrieve(
            eq, top_k_vector=top_k_vector, top_k_keyword=top_k_keyword,
            fusion_strategy=fusion_strategy,
            final_k=rerank_pool if rerank else top_k)

        # ---- P0 landmark 路：融合之后、重排之前并入候选 ----
        # ⚠ 关掉时必须**一步都不做**（不编码、不查库、不动 pool），
        #   否则 P2a 那套「旧模式逐位复现旧数字」的回归断言会失效。
        want_lm = self.landmark_col is not None if use_landmark is None else bool(use_landmark)
        want_lm = want_lm and self.landmark_col is not None
        quota = self.landmark_quota if landmark_quota is None else int(landmark_quota)
        lm_cands: List[Candidate] = []
        if want_lm and quota > 0:
            lm_cands = self._landmark_search(eq, n=max(quota * 2, 4))
            pool = pool + lm_cands            # 一起进重排，让交叉编码器给它们真实的 rel 分

        if rerank and self.reranker is not None and pool:
            results = self.reranker.rerank(eq.core_text, pool, top_k=top_k)
        else:
            results = pool[:top_k]

        # ---- 保底：landmark 被重排挤出去时，**有条件地**补回来 ----
        # P0 原话是「保底 2 条」。**无条件保底是错的**（2026-08-15 在 golden 上实测抓到）：
        # 那 10 篇是心内科试验，而 golden 里绝大多数 query 与它们毫无关系——
        # 实测无关 query 上 landmark 的交叉编码器 rel 只有 **0.000~0.08**，
        # 而被它挤掉的主库候选 rel 是 **0.72~0.97**。无条件补回去等于
        # **在每一个上下文里塞 2 条 rel≈0.001 的无关文献**，白占 2/10 的证据位。
        #
        # 保底真正要修的不是"不相关也要给位置"，而是这个具体机理：
        # **同分裁决按 ε 分档后，档内比 recency——2014~2021 的 landmark 会输给 2023 的综述**
        # （P2a 已量到「老文献系统性吃亏」）。所以判据要贴着那个机理写：
        #
        #     只有当 landmark 的 rel **不低于它将要挤掉的那条**（容差 = 重排的 ε）时才补。
        #
        # 即：它本来就该在里面、只是被 recency 挤下去了 → 补回来；
        #     它本来就不相关 → 不补。**这条规则自校准，不用手调阈值。**
        lm_promoted = 0
        lm_rejected = 0
        if lm_cands and results:
            in_res = [c for c in results if (c.metadata or {}).get("source_type") == "landmark"]
            need = min(quota, len(lm_cands)) - len(in_res)
            if need > 0:
                have = {c.chunk_id for c in results}
                pending = [c for c in lm_cands if c.chunk_id not in have]
                mains = [c for c in results
                         if (c.metadata or {}).get("source_type") != "landmark"]
                floor_rel = min((c.rel_score for c in mains if c.rel_score is not None),
                                default=None)
                tol = getattr(self.reranker, "eps", 0.02) or 0.02
                if floor_rel is None:            # 没重排就没有 rel，退回无条件（仅调试路径）
                    missing = pending[:need]
                else:
                    # ⚠ 「被拒」只算**因不够相关**被挡下的，不算因配额满了没轮上的——
                    #   两者混在一起，这个数就说明不了「规则有没有在起作用」。
                    qualified = [c for c in pending
                                 if (c.rel_score or 0.0) >= floor_rel - tol]
                    missing = qualified[:need]
                    lm_rejected = len(pending) - len(qualified)
                if missing:
                    keep = [c for c in results
                            if (c.metadata or {}).get("source_type") != "landmark"]
                    keep = keep[:max(0, top_k - len(in_res) - len(missing))]
                    results = (in_res + missing + keep)[:top_k]
                    # 补回来的按**原本的排序键**放回原位，避免"保底进来的一定排最前"这种假象。
                    # ⚠ 方向必须与 `MultiCriteriaReranker.rerank` 一致：`reverse=True`（降序）。
                    #   写成升序会把整个结果倒过来，而且不报任何错。
                    if all(c.rerank_key is not None for c in results):
                        results.sort(key=lambda c: c.rerank_key, reverse=True)
                    else:                             # rerank=False 时没有 rerank_key
                        results.sort(key=lambda c: c.fused_score, reverse=True)
                    lm_promoted = len(missing)

        out = {"query": raw_query, "enhanced": eq, "fusion_strategy": fusion_strategy,
               "reranked": bool(rerank and self.reranker is not None),
               "candidates": pool, "results": results,
               "landmark": {
                   "enabled": bool(want_lm), "quota": quota if want_lm else 0,
                   "retrieved": len(lm_cands),
                   "in_results": sum(1 for c in results
                                     if (c.metadata or {}).get("source_type") == "landmark"),
                   "promoted": lm_promoted,
                   #: 够格进池、但 rel 低于「将被挤掉的那条」而**没有**保底进去的条数。
                   #: 这个数不该是 0——它是 0 说明保底又变回无条件了。
                   "rejected": lm_rejected,
                   "trials": [ (c.metadata or {}).get("trial_name", c.chunk_id)
                               for c in results
                               if (c.metadata or {}).get("source_type") == "landmark"],
               },
               "limits": {"top_k": top_k, "top_k_vector": top_k_vector,
                          "top_k_keyword": top_k_keyword, "rerank_pool": rerank_pool,
                          "pool_size": len(pool)}}
        if dump:
            dump_candidates(out, dump)
        return out


# ============================================================================
# CLI 演示
# ============================================================================
def _print_result(out: dict, top_n: int = 8):
    eq = out["enhanced"]
    print("=" * 96)
    print(f"查询：{out['query']}")
    print(f"检索主体：{eq.core_text}" + (f"   （中译英自：{eq.translated_from}）" if eq.translated_from else ""))
    if eq.filters:
        print(f"过滤：{json.dumps(eq.filters, ensure_ascii=False)}"
              + (f"  后置：{json.dumps(eq.post_filters, ensure_ascii=False)}" if eq.post_filters else ""))
    print(f"融合策略：{out['fusion_strategy']} | 候选池 {len(out['candidates'])} | "
          f"重排：{'是' if out['reranked'] else '否'}")
    print("-" * 96)
    for r, c in enumerate(out["results"][:top_n], 1):
        head, snip = c.brief()
        extra = ""
        if c.rerank_score is not None:
            extra = (f"  rerank={c.rerank_score:.3f} "
                     f"(rel={c.rel_score:.2f} rec={c.recency_score:.2f} auth={c.authority_score:.2f})")
        else:
            extra = f"  fused={c.fused_score:.4f}"
        print(f"[{r}] {head}{extra}")
        print(f"     {snip}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", default=os.path.join(ROOT, "data", "bm25_index_500k"))
    ap.add_argument("--chroma-path", default=CHROMA_PATH)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--query", default="heart attack prevention with aspirin, recent studies")
    ap.add_argument("--fusion", default="rrf", choices=["simple", "rrf", "weighted"])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--translate", default="dict", choices=["off", "dict", "llm"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    # ---- 调试检索层：导出完整候选池 + 定位目标文献 ----
    ap.add_argument("--dump-candidates", default="", metavar="PATH",
                    help="把完整候选池（不只 top_k）连同重排各分项导出，.jsonl 或 .csv")
    ap.add_argument("--find", default="", metavar="ID[,ID]",
                    help="在候选池里找这些 PMID/PMCID，打印落在第几位、被什么分数压下去")
    ap.add_argument("--top-k-vector", type=int, default=50)
    ap.add_argument("--top-k-keyword", type=int, default=50)
    ap.add_argument("--rerank-pool", type=int, default=50,
                    help="融合后进入重排的候选数。⚠ 生产默认 50 是实测最优（提到 100/200 反而更差）；"
                         "但**诊断时必须开到 200+**，否则「不在候选池」与「进了池被压下去」分不开")
    ap.add_argument("--rerank-mode", default="tiebreak", choices=["tiebreak", "weighted"],
                    help="tiebreak=recency 当同分裁决（默认，实测更优）；weighted=旧的加权总分")
    ap.add_argument("--tiebreak-eps", type=float, default=None,
                    help=f"同分裁决的 rel 分档宽度，默认 {MultiCriteriaReranker.DEFAULT_TIEBREAK_EPS}")
    args = ap.parse_args()

    pipe = RetrievalPipeline(
        bm25_dir=args.bm25, chroma_path=args.chroma_path, collection_name=args.collection,
        translate=args.translate, verbose=args.verbose, load_reranker=not args.no_rerank,
        rerank_mode=args.rerank_mode, tiebreak_eps=args.tiebreak_eps)
    out = pipe.search(args.query, top_k=args.top_k, fusion_strategy=args.fusion,
                      rerank=not args.no_rerank,
                      top_k_vector=args.top_k_vector, top_k_keyword=args.top_k_keyword,
                      rerank_pool=args.rerank_pool,
                      dump=args.dump_candidates or None)
    _print_result(out, top_n=args.top_k)
    if args.dump_candidates:
        print(f"\n候选池已导出：{args.dump_candidates}"
              + (f"（meta 另存 {args.dump_candidates}.meta.json）"
                 if args.dump_candidates.lower().endswith(".csv") else ""))
    if args.find:
        find_in_candidates(out, [x for x in args.find.split(",") if x.strip()])


if __name__ == "__main__":
    main()
