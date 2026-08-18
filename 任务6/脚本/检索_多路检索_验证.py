# -*- coding: utf-8 -*-
"""第六阶段 · 多路检索 + 多准则重排 端到端验证

在 4M Chroma（向量）+ 已构建 BM25 索引（关键词，验证阶段用 50 万子集）上跑通整条链路，
并对每个断言做【可计算】的核验：每条 PASS/FAIL 都由真实数据算出、汇入总 ok 标志，绝不
硬编码结论。过滤类断言特意构造成非空洞——必须确有 BM25 候选被过滤掉才算通过。

覆盖：
  A. 多路检索：向量/关键词各自返回、BM25 独立召回（贡献了向量 top-k 没有的 id）
  B. 三种融合：simple/rrf/weighted 各自的打分公式独立重算并核对，排序单调递减
  C. 过滤下推到 BM25：pub_year 数值过滤、section 后置过滤都对关键词来源候选生效（非空洞）
  D. 多准则重排：relevance/recency/authority ∈[0,1]、权重和为1、总分=加权和、
     时效性随年份单调、权威性已知>未知，以及**排序契约**——
     按 sort_key 单调 / rel 分档单调 / 同档内 recency 单调 / **全序无并列** /
     weighted 模式仍按总分单调
     ⚠ 2026-08-12（P2a）起 recency 改为同分裁决，`rerank_score` **不再是排序键**，
       旧断言「按 rerank_score 单调递减」已按新契约重写，别改回去

用法：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_多路检索_验证.py --bm25 E:\\rag\\data\\bm25_index_500k
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import importlib.util
import json
import io
import re
import sys
import time
from statistics import median

import pyarrow.parquet as pq

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORT_PATH = os.path.join(ROOT, "report_data", "多路检索验证报告.txt")

# ⚠ 报告先攒在内存里，跑完再决定要不要落盘。**不能在这里 open(..., "w")**：
# 那样文件在第一条断言跑起来之前就被清空了，任何「跑完再判断该不该覆盖」的判据都来不及生效。
_REPORT = io.StringIO()
_STDOUT = sys.stdout          # 真终端。收尾时「写了/没写」这句只给人看，不进报告正文


class _Tee:
    def __init__(self, *s): self.streams = s
    def write(self, x):
        for st in self.streams:
            try: st.write(x)
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass


sys.stdout = _Tee(sys.stdout, _REPORT)


# ---------------------------------------------------------------------------
# 防降级：弱运行不许覆盖强运行的报告
# ---------------------------------------------------------------------------
_TOTAL_RE = re.compile(r"验证结论：(\d+)\s*/\s*(\d+)\s*项通过")
#: 报告里那行「就绪：Chroma N 向量 | BM25 M 块」——M 就是这一轮真正查的语料规模。
#: ⚠ 必须同时认「块」与「篇」：2026-08-18 把印法从「篇」改成「块」（单位本来就错，
#: BM25 的条目是 chunk 不是文献），而**冻结报告仍是旧写法**，它们不会重跑。
#: 只认新写法的话，比对旧报告时读出 0，判据会退化成「永远拒绝写入」。
_CORPUS_RE = re.compile(r"BM25\s+([\d,]+)\s*[篇块]")


def _refuse_downgrade(path, new_total, new_corpus, force=False):
    """已有报告更强时，拒绝覆盖。返回 True 表示「别写」。

    现象：这个脚本的冻结报告是用**4M BM25 索引**（3,998,000 块）跑出来的，
    而 docstring 里写的用法是 `--bm25 ...bm25_index_500k`（500,248 篇）。
    照 docstring 跑一遍，就把 4M 那份**静默覆盖**成 500k 那份了。

    根因：**两轮都是 25/25**——项数判据对这个差别完全看不见。
    这正是「强弱不是一个标量」的第四种形态：变的不是数量，是**证据的种类**
    （在多大的真实语料上验过）。

    决定：两条判据并列，缺一不可 ——

      1. 新报告项数 >= 旧报告项数；
      2. 新报告的 BM25 语料规模 >= 旧报告的规模。

    第 2 条就是「按证据种类比」在这个脚本上的具体形态。要更新报告就用不小于旧报告的
    索引跑，或者显式 `--force-report`。

    依据数字：500k 子集 500,248 块 vs 全量 3,998,000 块，两者都跑出 25/25。
    """
    if force or not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return False
    m = _TOTAL_RE.search(text)
    if not m:
        return False
    old_total = int(m.group(2))
    mc = _CORPUS_RE.search(text)
    old_corpus = int(mc.group(1).replace(",", "")) if mc else 0

    if new_total >= old_total and new_corpus >= old_corpus:
        return False
    print("\n" + "=" * 70)
    if new_corpus < old_corpus:
        print(f"⚠ 拒绝覆盖：已有报告是在 {old_corpus:,} 块的语料上验的，本轮只有 {new_corpus:,} 块。")
        print(f"  两轮项数分别是 {old_total} 与 {new_total} —— **项数在这条判据里不作数**：")
        print(f"  变的不是验了多少条，是在多大的真实语料上验的。")
        print(f"  要更新报告：用不小于旧报告的索引跑（如 --bm25 <4M 索引目录>）。")
    else:
        print(f"⚠ 拒绝覆盖：已有报告 {old_total} 项，本轮只有 {new_total} 项 —— 这是降级。")
    print(f"  只想看这轮结果：--force-report。")
    print("=" * 70)
    return True

_spec = importlib.util.spec_from_file_location("dlretr", os.path.join(_HERE, "检索_多路检索.py"))
_dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dl)
RetrievalPipeline = _dl.RetrievalPipeline
SectionPostFilter = _dl.SectionPostFilter
match_where = _dl.match_where


# ----------------------------------------------------------------------------
# 计算式核验器：每条结论都是布尔变量，汇入总 ok
# ----------------------------------------------------------------------------
CHECKS = []


def check(name, passed, detail=""):
    passed = bool(passed)
    CHECKS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return passed


def is_sorted_desc(xs, eps=1e-9):
    return all(xs[i] >= xs[i + 1] - eps for i in range(len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", default=os.path.join(ROOT, "data", "bm25_index_500k"))
    ap.add_argument("--chroma-path", default=_dl.CHROMA_PATH)
    ap.add_argument("--collection", default=_dl.COLLECTION)
    ap.add_argument("--force-report", action="store_true",
                    help="允许用更弱的一轮覆盖已有报告（默认拒绝：项数更少、或语料规模更小）")
    args = ap.parse_args()

    print("=" * 96)
    print("多路检索 + 多准则重排 · 端到端验证")
    print("=" * 96)
    t_start = time.time()
    pipe = RetrievalPipeline(bm25_dir=args.bm25, chroma_path=args.chroma_path,
                             collection_name=args.collection, verbose=True)
    R = pipe.retriever
    RR = pipe.reranker
    col = pipe.collection
    print(f"[加载完成] 用时 {time.time()-t_start:.0f}s\n")

    # ========================================================================
    # A. 多路检索：两路各自返回 + BM25 独立召回
    # ========================================================================
    print("=" * 96)
    print("A. 多路检索 —— 向量/关键词各自返回，BM25 提供独立召回")
    print("=" * 96)
    q = "RCT evidence for pembrolizumab in NSCLC"
    eq = pipe.processor.process_query(q)
    vec_lists = R._vector_search(eq, 50, push_where=False)
    kw_list = R._keyword_search(eq, top_k=50)
    n_vec = sum(len(l) for _, l in vec_lists)
    check("向量检索返回非空", n_vec > 0, f"{len(vec_lists)} 个变体，共 {n_vec} 条命中")
    check("BM25 关键词检索返回非空", len(kw_list) > 0, f"BM25 top50 命中 {len(kw_list)} 条")

    vec_ids = {cid for _, l in vec_lists for cid, _, _ in l}
    kw_ids = {cid for cid, _, _ in kw_list}
    kw_only = kw_ids - vec_ids
    check("BM25 贡献了向量 top-k 之外的候选（独立召回）", len(kw_only) > 0,
          f"关键词独有 {len(kw_only)} / {len(kw_ids)} 条，向量∩关键词 {len(vec_ids & kw_ids)} 条")

    # ========================================================================
    # B. 三种融合：独立重算打分公式，核对排序
    # ========================================================================
    print("\n" + "=" * 96)
    print("B. 融合策略 —— simple / rrf / weighted 打分公式独立重算 + 排序单调")
    print("=" * 96)

    # -- rrf：独立按 Σ w/(k+rank) 重算，与融合器输出逐条比对 --
    fused_rrf = R._fuse(vec_lists, kw_list, "rrf")
    exp_rrf = {}
    for w, l in vec_lists:
        for cid, rank, _ in l:
            exp_rrf[cid] = exp_rrf.get(cid, 0.0) + w / (R.rrf_k + rank)
    for cid, rank, _ in kw_list:
        exp_rrf[cid] = exp_rrf.get(cid, 0.0) + R.keyword_weight / (R.rrf_k + rank)
    max_err = max(abs(fused_rrf[cid].fused_score - exp_rrf[cid]) for cid in fused_rrf)
    check("RRF 分数 = Σ w/(k+rank)（逐候选独立重算一致）", max_err < 1e-9,
          f"最大误差 {max_err:.2e}，候选 {len(fused_rrf)}")
    rrf_sorted = sorted(fused_rrf.values(), key=lambda c: c.fused_score, reverse=True)
    check("RRF 排序按分数单调递减", is_sorted_desc([c.fused_score for c in rrf_sorted]))

    # -- weighted：独立 min-max 归一 + 加权重算 --
    fused_w = R._fuse(vec_lists, kw_list, "weighted")
    cos = {cid: c.cos_sim for cid, c in fused_w.items() if c.cos_sim is not None}
    bm = {cid: c.bm25_score for cid, c in fused_w.items() if c.bm25_score is not None}
    ncos, nbm = R._minmax(cos), R._minmax(bm)
    vw = R.default_vector_weight
    werr = max(abs(fused_w[cid].fused_score - (vw * ncos.get(cid, 0.0) + (1 - vw) * nbm.get(cid, 0.0)))
               for cid in fused_w)
    check("weighted 分数 = vw·norm(cos)+(1-vw)·norm(bm25)（独立重算一致）", werr < 1e-9,
          f"vw={vw}，最大误差 {werr:.2e}")

    # -- simple：命中两路者必排在只命中一路者之前 --
    fused_s = R._fuse(vec_lists, kw_list, "simple")
    s_sorted = sorted(fused_s.values(), key=lambda c: c.fused_score, reverse=True)
    two_src = [i for i, c in enumerate(s_sorted) if len(c.sources) == 2]
    one_src = [i for i, c in enumerate(s_sorted) if len(c.sources) == 1]
    both_before_single = (not two_src or not one_src or max(two_src) < min(one_src))
    check("simple 去重合并：命中两路者全部排在只命中一路者之前", both_before_single,
          f"两路命中 {len(two_src)} 条，单路 {len(one_src)} 条")

    # -- 三策略覆盖同一候选并集（融合不丢候选）--
    same_set = set(fused_rrf) == set(fused_w) == set(fused_s)
    check("三种策略作用于同一候选并集（策略只改定序不改集合）", same_set,
          f"并集大小 {len(fused_rrf)}")

    # ========================================================================
    # C. 过滤下推到 BM25（非空洞）：数值过滤 + section 后置过滤
    # ========================================================================
    print("\n" + "=" * 96)
    print("C. 过滤对关键词来源候选生效 —— 构造成必有 BM25 候选被过滤才算通过")
    print("=" * 96)

    # C1. pub_year 数值过滤：阈值取 BM25 命中年份中位数 → 约一半 BM25 候选应被剔除
    kw_ids_list = [cid for cid, _, _ in kw_list]
    kg = col.get(ids=kw_ids_list, include=["metadatas"])
    kw_year = {cid: (m or {}).get("pub_year") for cid, m in zip(kg["ids"], kg["metadatas"])}
    years = [y for y in kw_year.values() if isinstance(y, (int, float)) and y > 0]
    thr = int(median(years)) if years else 2015
    eqf = pipe.processor.process_query(q)
    eqf.filters = {"pub_year": {"$gte": thr}}      # 直接注入数值过滤，测 match_where 对 BM25 候选的作用
    eqf.post_filters = {}
    below = {cid for cid, y in kw_year.items() if isinstance(y, (int, float)) and y < thr}
    res_f = R.retrieve(eqf, top_k_vector=50, top_k_keyword=50, fusion_strategy="rrf")
    final_ids = {c.chunk_id for c in res_f}
    leaked = below & final_ids
    all_ok_year = all((c.metadata.get("pub_year") or 0) >= thr for c in res_f)
    check(f"过滤 pub_year>=%d 下 BM25 命中里 <阈值 的候选被剔除（非空洞）" % thr,
          len(below) > 0 and len(leaked) == 0,
          f"BM25 命中 <{thr} 有 {len(below)} 条，最终泄漏 {len(leaked)} 条")
    check("过滤后全部最终候选 pub_year>=阈值（含关键词来源）", all_ok_year,
          f"最终 {len(res_f)} 条，最低年份 "
          f"{min((c.metadata.get('pub_year') or 0) for c in res_f) if res_f else 'NA'}")

    # C2. section 后置过滤（methods：7726 种写法，走后置过滤而非 $in 下推）
    sf = SectionPostFilter(_dl.CORPUS_META)
    methods_variants = sf.c2r.get("methods", set())
    q_sec = "gene expression analysis, in the methods section"
    eqs = pipe.processor.process_query(q_sec)
    is_post = eqs.post_filters.get("section_canon") == "methods"
    check("methods 章节走后置过滤（section_canon=methods，未下推 $in）", is_post,
          f"filters={json.dumps(eqs.filters, ensure_ascii=False)} post={json.dumps(eqs.post_filters, ensure_ascii=False)}")
    kw_sec = R._keyword_search(eqs, top_k=50)
    kw_sec_ids = [cid for cid, _, _ in kw_sec]
    sg = col.get(ids=kw_sec_ids, include=["metadatas"])
    kw_sec_meta = {cid: (m or {}) for cid, m in zip(sg["ids"], sg["metadatas"])}
    kw_non_methods = {cid for cid, m in kw_sec_meta.items() if m.get("section") not in methods_variants}
    res_sec = R.retrieve(eqs, top_k_vector=50, top_k_keyword=50, fusion_strategy="rrf")
    sec_final_ids = {c.chunk_id for c in res_sec}
    leaked_sec = kw_non_methods & sec_final_ids
    all_methods = all((c.metadata.get("section") in methods_variants) for c in res_sec)
    check("section 后置过滤把非-methods 的 BM25 候选剔除（非空洞）",
          len(kw_non_methods) > 0 and len(leaked_sec) == 0,
          f"BM25 命中里非-methods {len(kw_non_methods)} 条，最终泄漏 {len(leaked_sec)} 条")
    check("过滤后全部最终候选 section∈methods 写法表（复用同一 canonical_to_raw）", all_methods,
          f"最终 {len(res_sec)} 条")

    # ========================================================================
    # D. 多准则重排
    # ========================================================================
    print("\n" + "=" * 96)
    print("D. 多准则重排 —— relevance × recency × authority")
    print("=" * 96)
    check("准则权重之和为 1", abs(sum(RR.w.values()) - 1.0) < 1e-9,
          f"权重 {RR.w}")
    # 时效性单调 + 值域
    r_new, r_mid, r_old = RR._recency(RR.current_year), RR._recency(RR.current_year - 10), RR._recency(RR.current_year - 100)
    check("时效性随年份单调递减且∈[0,1]",
          1.0 >= r_new > r_mid > r_old >= 0.0,
          f"今年={r_new:.2f} 前10年={r_mid:.2f} 前100年={r_old:.2f}")
    # 权威性 已知>未知 且值域
    a_top = RR._authority("Nature")
    a_mid = RR._authority("PLoS ONE")
    a_unk = RR._authority("Journal of Nowhere Unlisted XYZ")
    check("权威性 已知顶刊>开放获取巨刊>未知刊 且∈[0,1]",
          1.0 >= a_top >= a_mid > a_unk >= 0.0 and a_unk == RR.default_authority,
          f"Nature={a_top:.2f} PLoS ONE={a_mid:.2f} 未知={a_unk:.2f}")

    # 端到端重排：总分=加权和、按总分单调、各准则∈[0,1]
    out = pipe.search("heart attack prevention with aspirin, recent studies",
                      top_k=10, fusion_strategy="rrf", rerank=True, rerank_pool=50)
    results = out["results"]
    check("重排返回非空", len(results) > 0, f"{len(results)} 条")
    rng_ok = all(0.0 <= c.rel_score <= 1.0 and 0.0 <= c.recency_score <= 1.0
                 and 0.0 <= c.authority_score <= 1.0 for c in results)
    check("三准则分数均∈[0,1]", rng_ok)
    recompute_err = max(abs(c.rerank_score - (RR.w["relevance"] * c.rel_score
                        + RR.w["recency"] * c.recency_score
                        + RR.w["authority"] * c.authority_score)) for c in results)
    check("重排总分 = Σ 准则权重×准则分（逐条独立重算一致）", recompute_err < 1e-9,
          f"最大误差 {recompute_err:.2e}")

    # ------------------------------------------------------------------
    # 排序契约 —— 2026-08-12（P2a）之后重写
    #
    # 旧断言是「最终结果按 rerank_score 单调递减」。P2a 把 recency 从加法项改成
    # **同分裁决**之后，`rerank_score` **不再是排序键**（它仍被算出来，供展示、
    # 供 weighted 模式使用、供报告里的对照分析用），所以那条断言必然 FAIL——
    # 这是设计变更，不是回归。
    #
    # 这里换成验**真正的契约**，而且比原来更严：
    #   ① 结果按 sort_key 单调递减（排序键是唯一来源）
    #   ② 该模式的语义（tiebreak：rel 分档单调 + 同档内 recency 单调）
    #   ③ **全序**：排序键两两不等，没有残余并列
    #      —— 这条最关键。并列一旦存在，名次就由输入顺序（融合序）决定，
    #         检索结果会悄悄依赖上游，P0/P2b 动融合时会无声漂移。
    #   ④ weighted 模式**仍然**按总分单调（旧行为没被改坏，可随时切回做对照）
    # ------------------------------------------------------------------
    keys = [RR.sort_key(c) for c in results]
    check(f"最终结果按 sort_key 单调递减（mode={RR.mode}）",
          all(keys[i] >= keys[i + 1] for i in range(len(keys) - 1)),
          f"排序键 {len(keys[0])} 级")

    if RR.mode == "tiebreak":
        buckets = [round(c.rel_score / RR.eps) for c in results]
        check("① rel 分档单调不增（相关性仍是第一位）", is_sorted_desc(buckets),
              f"分档 {buckets}")
        # ⚠ 必须在**全池**上验，不能只看 top10：top10 的 rel 分档往往两两不同
        #   （实测 [35,34,32,31,27,24,21,20,19,17]），一个同档相邻对都没有——
        #   那样断言会**空洞通过**，什么也没检验到。项目铁律：PASS 必须由真实数据算出，
        #   且不能是空洞的（见模块 docstring 里过滤类断言"非空洞"的同款要求）。
        #   所以这里排全池，并把"同档相邻对数 > 0"本身也纳入通过条件。
        ranked_pool = sorted(out["candidates"], key=RR.sort_key, reverse=True)
        pb = [round(c.rel_score / RR.eps) for c in ranked_pool]
        same_bucket_ok, checked = True, 0
        for i in range(len(ranked_pool) - 1):
            if pb[i] == pb[i + 1]:
                checked += 1
                if ranked_pool[i].recency_score < ranked_pool[i + 1].recency_score - 1e-9:
                    same_bucket_ok = False
        check("② 同档内 recency 单调不增（同等相关时新的优先，全池非空洞检验）",
              same_bucket_ok and checked > 0,
              f"全池 {len(ranked_pool)} 条，同档相邻对 {checked} 处"
              + ("　⚠ 0 处 → 本条未构成检验，判为不通过" if checked == 0 else ""))

    pool_keys = [RR.sort_key(c) for c in out["candidates"]]
    check("③ 排序键在全池上构成全序（无残余并列 → 名次不依赖上游融合顺序）",
          len(set(pool_keys)) == len(pool_keys),
          f"{len(pool_keys)} 条候选，唯一键 {len(set(pool_keys))} 个")

    _saved_mode = RR.mode
    try:
        RR.mode = "weighted"
        w_sorted = sorted(out["candidates"], key=RR.sort_key, reverse=True)
        check("④ weighted 模式仍按总分单调递减（旧行为可随时切回做对照）",
              is_sorted_desc([c.rerank_score for c in w_sorted]),
              f"全池 {len(w_sorted)} 条")
    finally:
        RR.mode = _saved_mode

    # 重排确实改变了顺序（相对纯融合序）
    pool = out["candidates"]
    fusion_order = [c.chunk_id for c in sorted(pool, key=lambda c: c.fused_score, reverse=True)][:10]
    rerank_order = [c.chunk_id for c in results]
    reordered = fusion_order != rerank_order
    check("重排相对纯融合序发生了重排（顺序被改变）", reordered,
          f"top10 顺序 {'不同' if reordered else '相同'}")

    # ========================================================================
    # 汇总（结论由 ok 变量计算，不硬编码）
    # ========================================================================
    ok = all(p for _, p, _ in CHECKS)
    n_pass = sum(1 for _, p, _ in CHECKS if p)
    print("\n" + "=" * 96)
    print(f"验证结论：{n_pass}/{len(CHECKS)} 项通过  →  {'全部通过' if ok else '存在失败项'}")
    print("=" * 96)
    if not ok:
        for name, p, detail in CHECKS:
            if not p:
                print(f"  失败：{name} — {detail}")
    print(f"[总用时 {time.time()-t_start:.0f}s]")

    # 汇总打完了才决定落不落盘。下面这几句只走真终端、不进报告正文。
    body = _REPORT.getvalue()
    sys.stdout = _STDOUT
    mc = _CORPUS_RE.search(body)
    new_corpus = int(mc.group(1).replace(",", "")) if mc else 0
    if _refuse_downgrade(_REPORT_PATH, len(CHECKS), new_corpus, force=args.force_report):
        print(f"  本轮 {n_pass}/{len(CHECKS)} 项（语料 {new_corpus:,} 块）的完整输出见上方终端回滚，未落盘。")
    else:
        with open(_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"报告已写入 {_REPORT_PATH}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
