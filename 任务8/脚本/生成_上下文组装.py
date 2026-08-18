# -*- coding: utf-8 -*-
"""第七阶段（一）· 上下文组装器

把阶段六检索流水线吐出的候选（`Candidate`）整理成一段**可直接塞进提示词**的证据上下文：

    检索结果 ─▶ 统一成 DocumentChunk ─▶ Jaccard 近重去重 ─▶ 按相关性排序
             ─▶ 相关性×来源多样性 贪心选取 ─▶ 按 token 预算拼装（在完整句/段处截断）
             ─▶ 附带引用清单与统计元数据

设计要点（取舍理由见各方法 docstring）：
  1. **token 预算用精确分词器**：与 qwen3:8b 推理端同一套 BPE（从 Ollama GGUF 重建，见
     `生成_分词器.py`）。超预算的后果不对称——Ollama 超 num_ctx 会静默丢掉最前面的内容。
  2. **去重按 3-gram shingle 的 Jaccard**：PubMed 里同一段落经常被多篇（勘误、预印本、
     综述引用）重复收录，且阶段三切块本身带 overlap，不去重会让"多条证据"其实是同一句话，
     把假的一致性喂给模型。
  3. **多样性惩罚而非硬性轮转**：同一 doc 已入选 n 条时，其后续块的有效分乘 decay**n，
     并设同源硬上限。既避免"10 条证据全来自同一篇文章"的伪多源，也不会为了凑多样性把
     明显更相关的证据挤掉。
  4. **截断必须落在完整句/段**：半句话结尾的证据会让模型接着"补完"——那正是幻觉的来源。

用法：
    from 生成_上下文组装 import ContextAssembler, DocumentChunk     # 中文名需按路径导入
    asm = ContextAssembler(max_context_tokens=4000)
    ctx = asm.assemble_context(out["results"], query=q)   # out 来自 RetrievalPipeline.search
    ctx["context_text"] / ctx["metadata"] / ctx["selected_chunks"]

CLI 演示（离线，用阶段三的 1000 条样例块，不加载 4M 库）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_上下文组装.py --demo
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
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(mod_name: str, filename: str):
    """中文文件名模块按路径导入（与阶段五、六一致的做法）。"""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tk = _load_by_path("shengcheng_fenciqi", "生成_分词器.py")
TokenCounter = _tk.TokenCounter


# ============================================================================
# 一、文档块数据类
# ============================================================================
@dataclass
class DocumentChunk:
    """送进上下文的最小单位。

    与检索层的 `Candidate` 刻意解耦：生成层只关心「正文 / 元数据 / 相关性 / 来源 / 主键」，
    不关心它是向量路还是 BM25 路召回的、余弦多少。检索侧的原始信号会原样存进
    `metadata["_retrieval"]`，需要时可查，但不进提示词。
    """
    text: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str
    chunk_id: str

    # ---- 便捷读取（都是属性/方法，不是字段，保持数据类字段与任务书一致）----
    @property
    def journal(self) -> str:
        return str((self.metadata or {}).get("journal") or "").strip()

    @property
    def year(self) -> Optional[int]:
        y = (self.metadata or {}).get("pub_year")
        try:
            y = int(y)
            return y if y > 0 else None
        except (TypeError, ValueError):
            return None

    @property
    def section(self) -> str:
        return str((self.metadata or {}).get("section") or "").strip()

    @property
    def title(self) -> str:
        return str((self.metadata or {}).get("source_title") or "").strip()

    @property
    def source_type(self) -> str:
        """来源类别。主库正文块是空串，P0 的 landmark 条目是 `landmark`。

        组装器只用它做一件事：**不要把检索层特意放进来的东西又丢掉**。
        landmark 条目在检索侧已经过了一道有条件保底（rel 不低于被它挤掉的那条才补），
        到这里再被去重或相关性门槛刷掉，整条 P0 链就在最后一步无声失效。
        """
        return str((self.metadata or {}).get("source_type") or "").strip()

    def citation(self, marker: str) -> Dict[str, Any]:
        """出处记录：最终答案的参考文献列表与可溯源性都建立在这上面。"""
        m = self.metadata or {}
        return {
            "marker": marker, "chunk_id": self.chunk_id, "source": self.source,
            "pmcid": m.get("pmcid") or "", "pmid": m.get("pmid") or "",
            "title": self.title, "journal": self.journal, "pub_year": self.year,
            "section": self.section, "relevance_score": round(float(self.relevance_score), 6),
        }

    # ---- 从检索层对象转换 ----
    @classmethod
    def from_candidate(cls, c: Any, score_field: str = "auto") -> "DocumentChunk":
        """把阶段六的 `Candidate`（或等价的 dict）转成 DocumentChunk。

        相关性取值优先级（score_field="auto"）：rerank_score → rel_score → fused_score →
        cos_sim → 0.0。因为重排总分才是流水线的最终排序依据；只做检索不重排时退回融合分。
        """
        if isinstance(c, DocumentChunk):
            return c
        get = (lambda k, d=None: c.get(k, d)) if isinstance(c, dict) else (lambda k, d=None: getattr(c, k, d))
        meta = dict(get("metadata") or {})
        if score_field == "auto":
            score = next((v for v in (get("rerank_score"), get("rel_score"),
                                      get("fused_score"), get("cos_sim")) if v is not None), 0.0)
        else:
            score = get(score_field)
            score = 0.0 if score is None else score
        chunk_id = str(get("chunk_id") or meta.get("chunk_id") or "")
        # 来源 = 文献级主键：pmcid 100% 完整且唯一（pmid 会因勘误记录重复，见阶段三结论）
        source = str(meta.get("pmcid") or meta.get("doc_id") or chunk_id.split("#")[0] or "unknown")
        srcs = get("sources")
        meta["_retrieval"] = {
            "sources": sorted(srcs) if isinstance(srcs, (set, list, tuple)) else srcs,
            "cos_sim": get("cos_sim"), "bm25_score": get("bm25_score"),
            "fused_score": get("fused_score"), "rerank_score": get("rerank_score"),
            "rel_score": get("rel_score"), "recency_score": get("recency_score"),
            "authority_score": get("authority_score"),
        }
        return cls(text=str(get("text") or ""), metadata=meta,
                   relevance_score=float(score), source=source, chunk_id=chunk_id)


# ============================================================================
# 二、上下文组装器
# ============================================================================
_WORD = re.compile(r"[a-z0-9]+")
# 句末判定：句号/问号/叹号 + 空白。要求后面是空白，天然排除 "0.5"、"Fig.1" 这类小数与紧连编号
_SENT_END = re.compile(r"[.!?](?=\s)|[。！？]")
# 常见缩写：句点后接空格但并非句末，切在这里会把 "et al." 之类劈开。
# 注意要连内部的点一起匹配（"e.g" 而不是 "g"），否则 e.g. 的最后一个词元只是单字母 g。
_ABBREV = ("e.g", "i.e", "et al", "al", "vs", "cf", "fig", "figs", "eq", "ref", "refs",
           "approx", "vol", "no", "ca", "dr", "prof", "sd", "se", "ci", "resp")


def _is_abbrev_dot(text: str, pos: int) -> bool:
    """text[pos] 是句点时，判断它是不是缩写点（而非句末）。"""
    head = text[:pos].lower()
    for a in _ABBREV:
        # 要求缩写前面是非字母数字，避免 "casino." 被 "no" 命中
        if head.endswith(a) and (len(head) == len(a) or not head[-len(a) - 1].isalnum()):
            return True
    # 点分缩写的最后一截（e.g. / i.e. / U.S.）：单字母且前一个字符就是点。
    # 只认"前面是点"这一种，"...vitamin D." 这类真句末才不会被误判成缩写。
    return len(head) >= 2 and head[-1].isalpha() and head[-2] == "."


class ContextAssembler:
    """把检索结果组装成带出处、受 token 预算约束、来源多样的证据上下文。

    参数（默认值均可调；与生成阶段的 num_ctx 需配套，见 `生成_提示词模板.py` 的预算建议）：
      max_context_tokens  证据部分的 token 预算（不含提示词模板与答案）
      max_chunk_tokens    单块上限，防一条超长块吃掉整个预算
      similarity_threshold Jaccard ≥ 该值判为重复（默认 0.80）
      shingle_size        Jaccard 的词 n-gram 粒度（默认 3，见 `jaccard_similarity`）
      max_per_source      同一文献最多入选几块（硬上限）
      diversity_decay     同一文献第 n 块的有效分衰减系数 decay**n（软惩罚）
      min_relevance       相关性低于此值直接丢弃
      boundary_tail_ratio 截断时在末尾多大比例内回退找完整句/段（任务书口径：后 10%）
    """

    def __init__(self,
                 tokenizer: Any = "auto",
                 # 默认 2800：由 `生成_提示词模板.py::plan_budget` 在 num_ctx=8192 下算出的
                 # 证据预算上限是 2912（四段链最坏情形，见该函数），这里再留一点余量。
                 # 把 num_ctx 提到 16384 时可放宽到 ~11000，代价是 KV cache 多占约 1.2GB 显存。
                 max_context_tokens: int = 2800,
                 max_chunk_tokens: int = 600,
                 min_fragment_tokens: int = 80,
                 similarity_threshold: float = 0.80,
                 shingle_size: int = 3,
                 max_per_source: int = 3,
                 diversity_decay: float = 0.75,
                 min_relevance: float = 0.0,
                 min_chunk_chars: int = 20,
                 boundary_tail_ratio: float = 0.10,
                 truncate_marker: str = " […truncated]",
                 block_separator: str = "\n\n",
                 protected_source_types: Optional[Sequence[str]] = ("landmark",),
                 max_protected: int = 2,
                 verbose: bool = False):
        # ---- 加载 tokenizer ----
        # 按 duck typing 而不是 isinstance 判断：中文文件名要按路径导入，同一个
        # 生成_分词器.py 被不同调用方加载会得到两个不相等的 TokenCounter 类。
        if hasattr(tokenizer, "count") and hasattr(tokenizer, "truncate_to_tokens"):
            self.tokenizer = tokenizer
        elif tokenizer is None:
            self.tokenizer = TokenCounter(mode="heuristic", verbose=verbose)
        else:
            self.tokenizer = TokenCounter(mode=str(tokenizer), verbose=verbose)

        self.max_context_tokens = int(max_context_tokens)
        self.max_chunk_tokens = int(max_chunk_tokens)
        self.min_fragment_tokens = int(min_fragment_tokens)
        self.similarity_threshold = float(similarity_threshold)
        self.shingle_size = max(1, int(shingle_size))
        self.max_per_source = int(max_per_source)
        self.diversity_decay = float(diversity_decay)
        self.min_relevance = float(min_relevance)
        self.min_chunk_chars = int(min_chunk_chars)
        self.boundary_tail_ratio = float(boundary_tail_ratio)
        self.truncate_marker = truncate_marker
        self.block_separator = block_separator
        #: 受保护的来源类别（P0 landmark）。**只保护"不被丢掉"，不保证"排最前"**——
        #: 见 `assemble_context` 里那段说明。`max_protected` 是上限，防止一次塞太多把
        #: token 预算吃光；默认 2，与检索侧的 `landmark_quota` 对齐。
        self.protected_source_types = set(protected_source_types or ())
        self.max_protected = int(max_protected)
        self.verbose = verbose

    def _is_protected(self, c: "DocumentChunk") -> bool:
        return bool(self.protected_source_types) and c.source_type in self.protected_source_types

    # ---------------- 估算 token 数量 ----------------
    def estimate_tokens(self, text: str) -> int:
        """文本的 token 数。默认走与 qwen3:8b 同一套 BPE，此时是精确值而非估算
        （`self.tokenizer.exact` 为 True）；降级到字符启发式时才是估算，且刻意偏高。"""
        return self.tokenizer.count(text or "")

    # ---------------- 转换文档格式 ----------------
    def to_document_chunks(self, retrieved_docs: Sequence[Any]
                           ) -> Tuple[List[DocumentChunk], List[str]]:
        """接受 Candidate / dict / DocumentChunk 的混合列表，统一成 DocumentChunk。

        顺手丢掉过短的退化块：语料里确实存在正文只有 "xx" 的占位块（阶段三样例 1000 条里
        有 7 条，来自 PLoS Synopsis 一类条目），它们不可能支撑任何结论，却会白占一个引用位。
        Returns: (chunks, 被丢弃的过短块 id)
        """
        out, too_short = [], []
        for d in retrieved_docs or []:
            dc = DocumentChunk.from_candidate(d)
            if len(dc.text.strip()) < self.min_chunk_chars:
                too_short.append(dc.chunk_id)
                continue
            out.append(dc)
        return out, too_short

    # ---------------- 文本相似性 ----------------
    def _shingles(self, text: str) -> Set[str]:
        """归一化 → 词 n-gram 集合。大小写/标点/多空格差异不应影响重复判定。"""
        words = _WORD.findall((text or "").lower())
        n = self.shingle_size
        if len(words) < n:
            return set(words)
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    def jaccard_similarity(self, a: str, b: str) -> float:
        """两段文本的 Jaccard 相似度 |A∩B| / |A∪B|。

        用 3-gram（词序敏感）而不是词袋：同一学科的两段不同正文，词袋 Jaccard 常有 0.3~0.5
        的"虚高"，而只有真正复述同一段文字才会在 3-gram 上高度重合，阈值因此好定得多。
        """
        sa, sb = self._shingles(a), self._shingles(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        return inter / (len(sa) + len(sb) - inter)

    def deduplicate(self, chunks: List[DocumentChunk]
                    ) -> Tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        """按相关性从高到低保留代表块，丢弃与已保留块 Jaccard ≥ 阈值的近重复。

        先按相关性排序再线性扫描，保证"被留下的那条是这一簇里最相关的"。
        规模：候选池一般 ≤50，O(n²) 的两两比较可以接受（50 条约 1.2k 次集合运算）。
        """
        # ⚠ 受保护来源（P0 landmark）排在最前扫描：Jaccard 去重是"先到者留下"，
        #   谁先被扫到谁就是代表块。landmark 的相关性分未必最高，若按纯相关性排序，
        #   它会被一篇正文里大段引用了同一试验的综述当成"近重复"丢掉——
        #   而那恰恰是我们最不想丢的那一条（原文 vs 转述）。
        ordered = sorted(chunks, key=lambda c: (not self._is_protected(c),
                                                -c.relevance_score, c.chunk_id))
        kept: List[DocumentChunk] = []
        kept_shingles: List[Set[str]] = []
        seen_exact: Dict[str, str] = {}
        dropped: List[Dict[str, Any]] = []
        for c in ordered:
            norm = " ".join(_WORD.findall(c.text.lower()))
            if norm in seen_exact:                     # 完全相同正文的快速通道
                dropped.append({"chunk_id": c.chunk_id, "duplicate_of": seen_exact[norm],
                                "similarity": 1.0, "kind": "exact"})
                continue
            sh = self._shingles(c.text)
            hit, sim = None, 0.0
            for k, ksh in zip(kept, kept_shingles):
                inter = len(sh & ksh)
                s = 0.0 if not sh or not ksh else inter / (len(sh) + len(ksh) - inter)
                if s > sim:
                    hit, sim = k, s
            if hit is not None and sim >= self.similarity_threshold:
                dropped.append({"chunk_id": c.chunk_id, "duplicate_of": hit.chunk_id,
                                "similarity": round(sim, 4), "kind": "near"})
                continue
            kept.append(c)
            kept_shingles.append(sh)
            seen_exact[norm] = c.chunk_id
        return kept, dropped

    # ---------------- 选取：相关性优先 + 来源多样性 ----------------
    def _effective_score(self, c: DocumentChunk, used: Counter) -> float:
        """有效优先级 = 相关性 × decay^(该来源已入选块数)。

        软惩罚的原因：同一篇文章连着 3 块常常确实最相关（比如 Results 连续段落），硬轮转会
        把明显更差的证据换进来；而衰减能让"第 4 块同源"自然让位给别的文献的次优块。
        """
        return float(c.relevance_score) * (self.diversity_decay ** used[c.source])

    # ---------------- 单块格式化 ----------------
    def format_block(self, chunk: DocumentChunk, marker: str, body: Optional[str] = None) -> str:
        """一条证据的最终形态：一行出处头 + 正文。

        出处头只放模型判断证据质量与写参考文献需要的字段（来源、期刊、年份、章节、标题），
        **不放检索分数**——把 0.83 这类数字给模型看，它会当成"可信度"去解释。
        """
        head = f"[{marker}] {chunk.source or '?'}"
        if chunk.journal:
            head += f" · {chunk.journal}"
        head += f" ({chunk.year})" if chunk.year else " (year n/a)"
        if chunk.section:
            head += f" · {chunk.section}"
        if chunk.title:
            t = chunk.title if len(chunk.title) <= 110 else chunk.title[:107] + "..."
            head += f' · "{t}"'
        return f"{head}\n{(body if body is not None else chunk.text).strip()}"

    # ---------------- 在完整句/段处截断 ----------------
    def truncate_at_boundary(self, text: str, max_tokens: int) -> Tuple[str, bool]:
        """把文本截到 ≤ max_tokens，并保证结尾是**完整段落或完整句子**。

        做法：先按 token 硬截，再在结果的末 `boundary_tail_ratio`（默认 10%）里回退找边界——
        优先段落分隔（\\n\\n），其次句末标点（跳过 e.g./et al. 这类缩写点）。末 10% 里找不到
        边界就保留硬截结果（宁可少一句，也不放一个断句进提示词）。
        Returns: (文本, 是否被截断)
        """
        cut, truncated = self.tokenizer.truncate_to_tokens(text or "", max_tokens)
        if not truncated:
            return (text or "").strip(), False
        tail_start = int(len(cut) * (1.0 - self.boundary_tail_ratio))
        para = cut.rfind("\n\n", tail_start)
        if para > 0:
            return cut[:para].strip(), True
        best = -1
        for m in _SENT_END.finditer(cut):
            if m.start() < tail_start:
                continue
            if cut[m.start()] == "." and _is_abbrev_dot(cut, m.start()):
                continue                                   # e.g. / et al. / Fig. 等缩写点不算句末
            best = m.start() + 1
        if best > 0:
            return cut[:best].strip(), True
        return cut.strip(), True

    def _fit_block(self, chunk: DocumentChunk, marker: str, budget: int
                   ) -> Tuple[Optional[str], bool, int]:
        """把一条证据装进 budget 个 token；装不下就截断，截到最小片段还装不下则放弃。

        Returns: (block 文本或 None, 是否截断, 实际 token 数)
        整块（含出处头与截断标记）实际 token 数必须 ≤ budget——用测量-回退循环保证，
        而不是"估算头部开销"了事。
        """
        whole = self.format_block(chunk, marker)
        n = self.estimate_tokens(whole)
        if n <= budget and self.estimate_tokens(chunk.text) <= self.max_chunk_tokens:
            return whole, False, n
        overhead = n - self.estimate_tokens(chunk.text)          # 出处头 + 换行的开销
        body_budget = min(self.max_chunk_tokens, budget - overhead - self._marker_tokens())
        if body_budget < self.min_fragment_tokens:
            return None, False, 0                                 # 放不下有意义的片段，跳过
        for _ in range(4):                                        # 截断标记本身也占 token，测量后回退
            body, was_cut = self.truncate_at_boundary(chunk.text, body_budget)
            if not body.strip():
                return None, False, 0
            block = self.format_block(chunk, marker, body + (self.truncate_marker if was_cut else ""))
            nb = self.estimate_tokens(block)
            if nb <= budget:
                return block, was_cut, nb
            body_budget -= max(8, nb - budget)
            if body_budget < self.min_fragment_tokens:
                return None, False, 0
        return None, False, 0

    def _marker_tokens(self) -> int:
        if not hasattr(self, "_mt"):
            self._mt = self.estimate_tokens(self.truncate_marker)
        return self._mt

    # ---------------- 来源分析 ----------------
    def _analyze_sources(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """入选证据的来源构成——用来回答"这些证据是不是其实来自同一篇文章"。"""
        by_source = Counter(c.source for c in chunks)
        years = [c.year for c in chunks if c.year]
        paths = Counter()
        for c in chunks:
            s = ((c.metadata or {}).get("_retrieval") or {}).get("sources")
            if isinstance(s, (list, tuple, set)):
                paths["+".join(sorted(s)) or "unknown"] += 1
        return {
            "unique_sources": len(by_source),
            "by_source": dict(by_source.most_common()),
            "max_from_one_source": max(by_source.values()) if by_source else 0,
            "by_journal": dict(Counter(c.journal or "unknown" for c in chunks).most_common()),
            "by_section": dict(Counter(c.section or "unknown" for c in chunks).most_common()),
            "year_range": [min(years), max(years)] if years else None,
            "by_retrieval_path": dict(paths) or None,
        }

    # ---------------- 主入口：组装上下文 ----------------
    def assemble_context(self, retrieved_docs: Sequence[Any],
                         query: Optional[str] = None,
                         max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """检索结果 → 证据上下文。

        流程：转换 → 去重 → 按相关性排序 → 相关性×多样性贪心选取（同步受 token 预算约束）
        → 拼装（超长块在完整句/段处截断）→ 统计元数据。

        贪心时"装不下就跳过继续看下一条"而不是直接停：预算剩余 300 token 时，第 4 条 500
        token 的证据放不下，但第 5 条 200 token 的仍可能有价值。
        """
        budget = int(max_tokens if max_tokens is not None else self.max_context_tokens)
        chunks, too_short = self.to_document_chunks(retrieved_docs)
        n_retrieved = len(retrieved_docs or [])

        # 1) 去重
        unique_chunks, dedup_dropped = self.deduplicate(chunks)
        # 2) 相关性门槛 + 排序（相关性降序；同分按 chunk_id 保证可复现）
        # ⚠ 受保护来源豁免相关性门槛：它在检索侧已经过了一道**更严**的判据
        #   （rel 不低于被它挤掉的那条主库候选），这里再用一个固定阈值卡一次是重复把关，
        #   而且是用一把更钝的尺子推翻一把更准的。
        pool = [c for c in unique_chunks
                if c.relevance_score >= self.min_relevance or self._is_protected(c)]
        pool.sort(key=lambda c: (-c.relevance_score, c.chunk_id))

        # 3) 贪心选取：每轮取有效分最高者，同源过多降权/封顶，再按剩余预算装
        used: Counter = Counter()
        selected: List[DocumentChunk] = []
        blocks: List[str] = []
        citations: List[Dict[str, Any]] = []
        truncated_ids: List[str] = []
        skipped_budget: List[str] = []
        skipped_cap: List[str] = []
        remaining = budget
        sep_tokens = self.estimate_tokens(self.block_separator)
        candidates = list(pool)
        n_protected = 0                      # 已入选的受保护块数（受 max_protected 约束）

        # 条件是 remaining > 0 而不是 > min_fragment_tokens：min_fragment_tokens 约束的是
        # "截断后的片段至少要多长"，不该顺带把"整块本来就很短、放得下"的证据也挡在外面。
        while candidates and remaining > 0:
            avail = [c for c in candidates if used[c.source] < self.max_per_source]
            if not avail:
                skipped_cap.extend(c.chunk_id for c in candidates)
                break
            # ⚠ 受保护来源**先装**（上限 max_protected），这是一个刻意的取舍：
            #   它们的相关性分未必最高，按纯贪心可能排到最后，等预算耗尽就被跳过——
            #   那样整条 P0 链会在最后一步无声失效（检索特意保底放进来的东西没进提示词）。
            #   代价是最多 max_protected 块的预算被优先占用（约 1000 / 2800 token）。
            #   两种失败里，"少一条高相关正文"比"关键试验原文没进上下文"轻。
            protected = [c for c in avail if self._is_protected(c)]
            if protected and n_protected < self.max_protected:
                best = max(protected, key=lambda c: c.relevance_score)
            else:
                # max 并列时取先出现者；candidates 已按相关性有序，所以结果稳定可复现
                best = max(avail, key=lambda c: self._effective_score(c, used))
            candidates.remove(best)
            budget_here = remaining - (sep_tokens if blocks else 0)
            marker = f"S{len(selected) + 1}"
            block, was_cut, n_tok = self._fit_block(best, marker, budget_here)
            if block is None:
                skipped_budget.append(best.chunk_id)
                continue
            blocks.append(block)
            selected.append(best)
            citations.append(best.citation(marker))
            used[best.source] += 1
            if self._is_protected(best):
                n_protected += 1
            remaining -= n_tok + (sep_tokens if len(blocks) > 1 else 0)
            if was_cut:
                truncated_ids.append(best.chunk_id)
        skipped_cap.extend(c.chunk_id for c in candidates if used[c.source] >= self.max_per_source)

        final_context = self.block_separator.join(blocks)

        # 4) 元数据
        context_metadata: Dict[str, Any] = {
            "total_chunks_retrieved": n_retrieved,
            "unique_chunks_after_dedup": len(unique_chunks),
            "chunks_selected": len(selected),
            # P0：检索侧特意保底放进来的 landmark，最后有没有真的进提示词。
            # 这个数与检索侧的 landmark.in_results 对不上，就说明组装这一步把它丢了。
            "protected_selected": n_protected,
            "protected_available": sum(1 for c in unique_chunks if self._is_protected(c)),
            "estimated_tokens": self.estimate_tokens(final_context),
            "chunk_sources": self._analyze_sources(selected),
            # ---- 以下为便于排查与写报告的补充信息 ----
            "query": query,
            "budget_tokens": budget,
            "token_counter": {"mode": self.tokenizer.mode, "exact": self.tokenizer.exact,
                              "model": self.tokenizer.info.get("model")},
            "dropped_by_dedup": dedup_dropped,
            "dropped_too_short": too_short,
            "dropped_below_min_relevance": len(unique_chunks) - len(pool),
            "skipped_by_budget": skipped_budget,
            "skipped_by_source_cap": skipped_cap,
            "truncated_chunks": truncated_ids,
            "citations": citations,
            "params": {"similarity_threshold": self.similarity_threshold,
                       "shingle_size": self.shingle_size,
                       "max_per_source": self.max_per_source,
                       "diversity_decay": self.diversity_decay,
                       "max_chunk_tokens": self.max_chunk_tokens,
                       "min_relevance": self.min_relevance},
        }
        if self.verbose:
            print(f"    [上下文] 检索 {n_retrieved} → 去重后 {len(unique_chunks)} → 入选 "
                  f"{len(selected)}（{context_metadata['chunk_sources']['unique_sources']} 篇文献）"
                  f" | {context_metadata['estimated_tokens']}/{budget} tokens"
                  f" | 截断 {len(truncated_ids)} 条")
        return {"context_text": final_context, "metadata": context_metadata,
                "selected_chunks": selected}

    # ---------------- 参考文献列表（供最终答案附录）----------------
    @staticmethod
    def render_reference_list(selected_chunks: List[DocumentChunk],
                              citations: Optional[List[Dict[str, Any]]] = None) -> str:
        """把入选证据渲染成参考文献列表，标号与上下文里的 [S#] 一一对应。"""
        cits = citations or [c.citation(f"S{i}") for i, c in enumerate(selected_chunks, 1)]
        lines = []
        for c in cits:
            bits = [f"[{c['marker']}]"]
            if c.get("title"):
                bits.append(c["title"])
            jr = c.get("journal") or ""
            if c.get("pub_year"):
                jr = f"{jr} ({c['pub_year']})" if jr else f"({c['pub_year']})"
            if jr:
                bits.append(jr + ".")
            ids = [x for x in (c.get("pmcid"), f"PMID {c['pmid']}" if c.get("pmid") else "") if x]
            if ids:
                bits.append(" / ".join(ids))
            lines.append(" ".join(bits))
        return "\n".join(lines)


# ============================================================================
# CLI 演示：离线用阶段三的样例块，无需加载 4M 向量库
# ============================================================================
SAMPLE_JSONL = os.path.join(ROOT, "任务3", "样例_文本块_1000.jsonl")


def load_sample_chunks(path: str = SAMPLE_JSONL, limit: int = 40,
                       keyword: Optional[str] = None) -> List[DocumentChunk]:
    """从阶段三样例 jsonl 造一批 DocumentChunk（相关性用递减的假分数，仅供演示/验证）。

    ⚠ 这个文件是**阶段三的交付物**，按交付包规矩不重复收进本包。只拿到本包时
    `--demo` / `--offline` 会因此跑不了——下面把这件事说清楚，而不是抛一个裸的
    FileNotFoundError 让人去猜。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到演示用的样例证据文件：{path}\n"
            f"它是阶段三（任务3）的交付物，本包按规矩不重复收录。三种解法：\n"
            f"  1) 从阶段三交付包里取 样例_文本块_1000.jsonl，放到上面那个路径；\n"
            f"  2) 调用 load_sample_chunks(path=...) 指到你自己的位置；\n"
            f"  3) 跳过离线演示，直接用真检索：生成_流水线_测试.py --live --bm25 <BM25索引目录>")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if keyword and keyword.lower() not in (d.get("text") or "").lower():
                continue
            rows.append(d)
            if len(rows) >= limit:
                break
    out = []
    for i, d in enumerate(rows):
        meta = {k: d.get(k) for k in ("doc_id", "pmcid", "pmid", "journal", "pub_year",
                                      "section", "source_title", "chunk_index")}
        try:
            meta["pub_year"] = int(meta["pub_year"])
        except (TypeError, ValueError):
            pass
        out.append(DocumentChunk(text=d["text"], metadata=meta,
                                 relevance_score=round(0.95 - 0.01 * i, 4),
                                 source=str(meta.get("pmcid") or meta.get("doc_id")),
                                 chunk_id=d["chunk_id"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用阶段三样例块跑一次组装")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--keyword", default=None)
    ap.add_argument("--show", type=int, default=1600, help="打印上下文前 N 字符")
    args = ap.parse_args()

    asm = ContextAssembler(max_context_tokens=args.budget, verbose=True)
    docs = load_sample_chunks(limit=args.limit, keyword=args.keyword)
    print(f"载入样例块 {len(docs)} 条 | 分词器 {asm.tokenizer.mode}(精确={asm.tokenizer.exact})\n")
    ctx = asm.assemble_context(docs, query="(demo)")
    md = ctx["metadata"]
    print(json.dumps({k: v for k, v in md.items()
                      if k in ("total_chunks_retrieved", "unique_chunks_after_dedup",
                               "chunks_selected", "estimated_tokens", "budget_tokens",
                               "chunk_sources", "truncated_chunks")},
                     ensure_ascii=False, indent=2))
    print("\n--- context_text（前 %d 字符）---" % args.show)
    print(ctx["context_text"][:args.show])
    print("\n--- 参考文献 ---")
    print(ContextAssembler.render_reference_list(ctx["selected_chunks"], md["citations"]))


if __name__ == "__main__":
    main()
