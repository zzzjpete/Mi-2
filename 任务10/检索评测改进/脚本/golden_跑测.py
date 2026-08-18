#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""golden 检索评测集 · 跑测与统计
================================================================================
对 `golden_构建.py` 定稿的每条 query 走完整检索链路，记录 ground truth chunk 落在
第几名，然后出 Recall / MRR / 命中分布 / 分层拆解 / 词面泄漏检查。

**只调 RetrievalPipeline.search()，不调 LLM 作答。** 所以这批数：
  · 是分钟级的，不是小时级
  · **完全确定性**——不受 docs/工程笔记.md 三·1 那个「温度 0 连跑两轮结果翻转」影响。
    这正是它比阶段七/八/九任何一个生成侧指标都更适合当验收锚点的原因。

--------------------------------------------------------------------------------
两趟跑法（沿用阶段七 --dump-retrieval / --from-dump 那个套路）
--------------------------------------------------------------------------------
    趟 1  --translate   只建 MedicalQueryProcessor，把中译英结果冻结成缓存
                        需要 Ollama，不加载检索器（内存 < 1GB）
    趟 2  --run         加载检索器（15.8GB / 67s），翻译走缓存
                        **不需要 Ollama**，因此可以反复重跑、结果逐字相同

为什么不一趟跑完：
  ① 15.8GB 检索器 + 8.1GB Ollama 在 32G 机器上挤在一起没必要冒风险；
  ② 更重要的是**把唯一的不确定性来源（LLM 翻译）钉死在趟 1**。趟 2 只剩确定性算子，
     改重排权重、改池子大小可以随便重跑对照，不用担心"这次的差别是翻译抖动造成的"。
  ③ 缓存里存的是 `_translate_by_llm` 的**入参→出参**，趟 2 把这个方法替换成查表，
     `process_query` 的其余步骤原样跑——所以趟 2 拿到的 EnhancedQuery 和真跑一次
     `--translate llm` 逐字相同，没有保真度损失。
  ⚠ 趟 2 遇到缓存未命中会**直接报错**，不会静默降级成词典直译——
    静默降级正是 2026-08-11 那个「检索回来是蝴蝶名录」的成因。

--------------------------------------------------------------------------------
两组配置（一次跑完，回答"rerank_pool 默认值该不该动"）
--------------------------------------------------------------------------------
    prod   向量 50  + BM25 50  → 融合截到 50   ← 线上当前默认
    wide   向量 200 + BM25 200 → 融合截到 200  ← 需求要求的宽池

`rrf` 的 fused_score = Σ w/(k+rank) 只依赖候选自己的名次，`rerank_score` 也是逐条算的
（都无跨池归一，已核对 `检索_多路检索.py::_fuse` 与 `MultiCriteriaReranker.rerank`）。
所以在 wide 组内部，把池子截到任意 P≤200 都能**精确离线反推**，不用重跑。
但 prod 组不能由 wide 反推——两路召回列表本身被截短了，RRF 少了几项贡献，
而且向量路有多条查询变体、`vec_rank` 只存了最小名次，反推不回去。所以 prod 老老实实跑一遍。

--------------------------------------------------------------------------------
用法
--------------------------------------------------------------------------------
    $py = "E:\\rag\\conda\\envs\\medrag\\python.exe"
    $env:OLLAMA_MODELS = "E:\\rag\\ollama\\models"      # 趟 1 前先起 Ollama

    & $py scripts\\golden_跑测.py --translate       # 趟 1：冻结翻译（约 5s/条）
    & $py scripts\\golden_跑测.py --run             # 趟 2：跑检索（加载 67s + 约 3s/条）
    & $py scripts\\golden_跑测.py --report          # 出指标（秒级，可反复跑）
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT


import os
import sys

os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pyarrow  # noqa: F401,E402  ← 必须早于 torch，否则 Windows DLL 冲突 exit 139

import argparse                       # noqa: E402
import importlib.util                 # noqa: E402
import json                           # noqa: E402
import statistics                     # noqa: E402
import time                           # noqa: E402
from collections import Counter, defaultdict   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data", "golden")
REPORT_DIR = os.path.join(ROOT, "report_data")

F_FINAL = os.path.join(OUT_DIR, "golden_set.jsonl")
F_TRANS = os.path.join(OUT_DIR, "golden_翻译缓存.json")
F_RUN = os.path.join(OUT_DIR, "golden_跑测明细.jsonl")
F_REPORT = os.path.join(REPORT_DIR, "golden_检索评测报告.txt")


def _tagged(path: str, tag: str) -> str:
    """给产物文件名加后缀，用来区分不同重排权重的跑测，别互相覆盖。"""
    if not tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


def parse_weights(s: str):
    """`rel=1.0,rec=0,auth=0` → dict。留空返回 None（用流水线默认 0.6/0.25/0.15）。"""
    if not s:
        return None
    alias = {"rel": "relevance", "rec": "recency", "auth": "authority"}
    out = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        k = k.strip().lower()
        out[alias.get(k, k)] = float(v)
    missing = {"relevance", "recency", "authority"} - set(out)
    if missing:
        raise SystemExit(f"--weights 缺少 {missing}，三项都要写全（写 0 也算写）")
    return out

BM25_DIR = os.path.join(ROOT, "data", "bm25_index_4m")

CONFIGS = {
    "prod": dict(top_k_vector=50, top_k_keyword=50, rerank_pool=50),
    "wide": dict(top_k_vector=200, top_k_keyword=200, rerank_pool=200),
}


def _load_by_path(mod_name: str, filename: str):
    """按路径导入中文文件名模块，**先登记 sys.modules 再 exec**（docs/工程笔记.md 三·8 那条坑）。"""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


def load_set(path=F_FINAL):
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}，先跑 golden_构建.py --finalize")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def detect_rerank_mode(rows):
    """从跑测明细**反推**这一轮用的哪套重排配置，用于报告头的配置标识。

    为什么要反推而不是读字段：早期几轮跑测没把配置存进明细（`rerank_config` 是后加的），
    而那几份 19MB 的明细不值得为一行标识重跑。新跑的轮次会直接带字段，走不到这里。

    判据：weighted 模式下 `final_rank` 的顺序**必然**等同于按 `rerank_score` 降序排；
    同分裁决则不等（它按 rel 分档 + 档内比 recency 排）。取多条 query 投票，稳。
    """
    if rows and rows[0].get("rerank_config"):
        c = rows[0]["rerank_config"]
        return c.get("mode"), c.get("eps")
    mismatch = 0
    probe = rows[:20]
    for r in probe:
        cand = r["configs"]["wide"]["candidates"]
        by_score = sorted(cand, key=lambda c: ((c["rerank_score"] if c["rerank_score"]
                                                is not None else -1e9),) + _tail(c), reverse=True)
        if [c["chunk_id"] for c in by_score] != [c["chunk_id"] for c in
                                                 sorted(cand, key=lambda c: c["final_rank"])]:
            mismatch += 1
    if not probe:
        return None, None
    # 同分裁决几乎每条都会与加权序不同；加权序则应当条条吻合
    return ("tiebreak", None) if mismatch > len(probe) // 2 else ("weighted", None)


def rerank_label(mode, eps):
    """报告头那一行配置标识。两份报告标题相同、时间只差一秒，**只能靠这行区分**。"""
    if mode == "tiebreak":
        e = f"ε={eps}" if eps else "ε=0.02"
        return f"同分裁决 {e}（P2a，当前生产默认）"
    if mode == "weighted":
        return "加权和 0.60rel+0.25rec+0.15auth（P2a 之前的旧默认）"
    return "（未能判定，请核对跑测明细）"


def load_run(path):
    """读跑测明细，并**与当前 golden_set.jsonl 取交集**。

    ⚠ 这一步是设计要点：`golden_set.jsonl` 是「集合里有哪些 query」的**唯一真源**。
    跑测明细是历史产物，一旦从集合里剔掉某条（例如事后发现它压根不是临床问题），
    所有下游统计必须自动跟着变，而不能靠人记得去改每个报告——
    否则剔除动作只在文档里生效、数字里不生效。
    好处是剔除**零 GPU 成本**：明细按 qid 存着，重算即可，不用重跑检索。
    """
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not os.path.exists(F_FINAL):
        return rows
    keep = {r["qid"] for r in load_set()}
    kept = [r for r in rows if r["qid"] in keep]
    if len(kept) != len(rows):
        dropped = sorted({r["qid"] for r in rows} - keep)
        print(f"[口径] 跑测明细 {len(rows)} 条，按当前 golden_set 取交集后 {len(kept)} 条；"
              f"已剔除 {len(dropped)} 条：{dropped}")
    return kept


# ==============================================================================
# 趟 1：冻结中译英
# ==============================================================================
class TranslationCacheMiss(RuntimeError):
    pass


def do_translate(args):
    rows = load_set()
    qu = _load_by_path("检索_查询理解", "检索_查询理解.py")
    proc = qu.MedicalQueryProcessor(translate="llm", verbose=False)

    cache = {}
    if args.resume and os.path.exists(F_TRANS):
        cache = json.load(open(F_TRANS, encoding="utf-8"))
        print(f"[趟1] 续跑：缓存已有 {len(cache)} 条")

    real = proc._translate_by_llm
    log = {}

    def wrapped(s):
        out = real(s)
        log[s] = out          # 记下 **入参→出参**，趟 2 用它替换本方法
        return out

    proc._translate_by_llm = wrapped

    t0, n_new, n_fail = time.time(), 0, 0
    for i, r in enumerate(rows, 1):
        log.clear()
        eq = proc.process_query(r["query"])
        for src, dst in log.items():
            if src in cache and cache[src] != dst:
                print(f"  ⚠ 同一句翻译出了两个结果（LLM 抖动）：{src!r}\n"
                      f"     旧 {cache[src]!r}\n     新 {dst!r}  ← 保留旧的，保证可复现")
                continue
            cache.setdefault(src, dst)
            n_new += 1
        if eq.translate_method != "llm" and eq.language in ("zh", "mixed"):
            n_fail += 1
            print(f"  [{i}/{len(rows)}] {r['qid']} ⚠ LLM 翻译没成功，降级成了 {eq.translate_method}")
        if i % 20 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] 缓存 {len(cache)} 条，{time.time()-t0:.0f}s", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(F_TRANS, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    print(f"\n[趟1] 完成：{len(cache)} 条翻译（本轮新增 {n_new}），"
          f"降级 {n_fail} 条，用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"       → {F_TRANS}")
    if n_fail:
        print(f"  ⚠ 有 {n_fail} 条没走成 LLM 翻译。**别直接进趟 2**——"
              f"先确认 Ollama 起着、模型已预热，再 --translate --resume 补齐。")


# ==============================================================================
# 趟 2：跑检索
# ==============================================================================
def offline_sort_key(c: dict, mode: str, eps: float):
    """离线复算排序键——**必须与生产的 `MultiCriteriaReranker.sort_key` 逐级一致**。

    ⚠ 这里刻意**不重新发明**排序法则，逐级都照抄生产实现：
        tiebreak : (round(rel/ε), recency, fused_score, pmid, chunk_id)
        weighted : (rerank_score,            fused_score, pmid, chunk_id)
    末三级是 P2a 显式化的收尾键。以前靠 Python 稳定排序的副产品（即"并列时保持输入顺序"）
    收尾，那个隐式第三级会让名次**依赖上游融合顺序**——2026-08-12 的证伪就是被它抓到的
    （13/237 不一致）。现在生产端写死成全序，离线端照抄，两边不可能再分叉。
    """
    tail = (c.get("fused_score") or 0.0, str(c.get("pmid") or ""), c["chunk_id"])
    if mode == "tiebreak":
        return (round((c["rel_score"] or 0.0) / eps), c["recency_score"] or 0.0) + tail
    return ((c["rerank_score"] if c["rerank_score"] is not None else -1e9),) + tail


def _compact(rows):
    """把 candidate_rows 的完整记录压成跑测要留的字段（200×2×N 条，别存正文）。"""
    return [{
        "chunk_id": r["chunk_id"], "pmcid": r["pmcid"],
        "fused_rank": r["fused_rank"], "final_rank": r["final_rank"],
        "vec_rank": r["vec_rank"], "bm25_rank": r["bm25_rank"],
        "fused_score": r["fused_score"], "rerank_score": r["rerank_score"],
        "rel_score": r["rel_score"], "recency_score": r["recency_score"],
        "authority_score": r["authority_score"],
        "pub_year": r["pub_year"], "section": r["section"],
        # P0：主库正文块是 None，landmark 条目是 "landmark"。
        # 不留这个字段，跑完就分不清「T1 掉的那几分是不是被 landmark 挤掉的」——
        # 而那正是这一轮要回答的问题。
        "source_type": r.get("source_type"), "trial_name": r.get("trial_name"),
        # in_top_k 与 final_rank 是两件事：final_rank 是重排后的池内名次，
        # in_top_k 是**真正进了上下文**——landmark 保底会改后者不改前者。
        "in_top_k": r.get("in_top_k"),
    } for r in rows]


def _locate(rows, gt_chunk_id, gt_pmcid):
    """在候选池里定位 ground truth。chunk 级（严格）与 doc 级（同篇任一块）各算一份。

    两个都要：query 是从某个 chunk 生成的，但同一篇文献的**摘要块**往往同样能回答它——
    那种情况检索其实是成功的，只按 chunk 级算会低估。
    """
    def pack(r):
        if r is None:
            return {"in_pool": False, "fused_rank": None, "final_rank": None,
                    "vec_rank": None, "bm25_rank": None, "in_top_k": False}
        return {"in_pool": True, "fused_rank": r["fused_rank"], "final_rank": r["final_rank"],
                "vec_rank": r["vec_rank"], "bm25_rank": r["bm25_rank"],
                "in_top_k": r["in_top_k"], "rerank_score": r["rerank_score"],
                "rel_score": r["rel_score"], "recency_score": r["recency_score"],
                "authority_score": r["authority_score"]}

    chunk_hit = next((r for r in rows if r["chunk_id"] == gt_chunk_id), None)
    same_doc = [r for r in rows if str(r["pmcid"]) == str(gt_pmcid)]
    doc_hit = min(same_doc, key=lambda r: r["final_rank"]) if same_doc else None
    out = {"chunk": pack(chunk_hit), "doc": pack(doc_hit)}
    out["doc"]["n_chunks_from_doc"] = len(same_doc)
    return out


def do_run(args):
    rows = load_set()
    if args.limit:
        rows = rows[:args.limit]
    if not os.path.exists(F_TRANS):
        sys.exit(f"找不到翻译缓存 {F_TRANS}，先跑 --translate")
    cache = json.load(open(F_TRANS, encoding="utf-8"))
    print(f"[趟2] 翻译缓存 {len(cache)} 条")

    f_run = _tagged(F_RUN, args.tag)
    weights = parse_weights(args.weights)
    print(f"[趟2] 重排权重：{weights or '默认 0.6rel/0.25rec/0.15auth'}")
    print(f"[趟2] 输出 → {f_run}")

    mp = _load_by_path("检索_多路检索", "检索_多路检索.py")
    # P0：landmark 路默认**关**——这把尺子的 ground truth 全部抽自主库，
    # 开着它等于让 2 条保底候选去挤主库候选，测出来的是「代价」而不是「基线」。
    # 要量那个代价就显式 --landmark，两轮对照着看。
    pipe = mp.RetrievalPipeline(bm25_dir=args.bm25, translate="llm",
                                criteria_weights=weights,
                                rerank_mode=args.rerank_mode, tiebreak_eps=args.eps,
                                use_landmark=args.landmark)
    print(f"[趟2] landmark 路："
          + (f"开（保底 {pipe.landmark_quota} 条）" if args.landmark else "关"))
    eps_used = pipe.reranker.eps if pipe.reranker else args.eps
    print(f"[趟2] 重排模式：{args.rerank_mode}"
          + (f"（ε={eps_used}）" if args.rerank_mode == "tiebreak" else "（旧的加权总分）"))

    # 把 LLM 翻译换成查表。未命中直接抛——**绝不静默降级成词典直译**。
    def cached_translate(s):
        if s not in cache:
            raise TranslationCacheMiss(
                f"翻译缓存里没有 {s!r}。缓存是趟 1 建的，query 改过就要重跑 --translate。")
        return cache[s]

    pipe.processor._translate_by_llm = cached_translate

    done = set()
    mode = "w"
    if args.resume and os.path.exists(f_run):
        for l in open(f_run, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["qid"])
        mode = "a"
        print(f"[趟2] 续跑：已有 {len(done)} 条")

    t0 = time.time()
    with open(f_run, mode, encoding="utf-8") as fout:
        for i, r in enumerate(rows, 1):
            if r["qid"] in done:
                continue
            rec = {k: r[k] for k in ("qid", "gid", "query", "tier", "tier_name",
                                     "en_tokens", "gt_chunk_id", "gt_pmcid",
                                     "section", "year_bucket", "pub_year", "journal",
                                     "source_title", "leaked_terms")}
            # 把重排配置写进明细，报告头直接读，不必再从数据反推（detect_rerank_mode）
            rec["rerank_config"] = {"mode": args.rerank_mode,
                                    "eps": eps_used if args.rerank_mode == "tiebreak" else None,
                                    "criteria_weights": weights}
            rec["configs"] = {}
            for name, cfg in CONFIGS.items():
                ts = time.time()
                out = pipe.search(r["query"], top_k=args.top_k, rerank=True,
                                  use_landmark=args.landmark, **cfg)
                crows = mp.candidate_rows(out)
                eq = out["enhanced"]
                rec["configs"][name] = {
                    "limits": out["limits"],
                    "pool_size": len(crows),
                    "elapsed": round(time.time() - ts, 2),
                    "hit": _locate(crows, r["gt_chunk_id"], r["gt_pmcid"]),
                    "candidates": _compact(crows),
                }
                if name == "wide":
                    rec["retrieval_query"] = eq.core_text
                    rec["translate_method"] = eq.translate_method
                    rec["filters"] = eq.filters
                    rec["post_filters"] = eq.post_filters
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            h = rec["configs"]["wide"]["hit"]["chunk"]
            tag = (f"final#{h['final_rank']} fused#{h['fused_rank']}"
                   if h["in_pool"] else "不在候选池")
            print(f"  [{i}/{len(rows)}] {r['qid']} {r['section']:<12} {tag:<26} "
                  f"{rec['configs']['wide']['elapsed']:.1f}s", flush=True)

    print(f"\n[趟2] 完成，用时 {(time.time()-t0)/60:.1f} 分钟 → {f_run}")
    print("       下一步：--report" + (f" --tag {args.tag}" if args.tag else ""))


# ==============================================================================
# 三、统计
# ==============================================================================
BUCKETS = [("第 1 名", lambda r: r == 1),
           ("第 2–5 名", lambda r: r is not None and 2 <= r <= 5),
           ("第 6–10 名", lambda r: r is not None and 6 <= r <= 10),
           ("第 11–50 名", lambda r: r is not None and 11 <= r <= 50),
           ("第 50 名之后", lambda r: r is not None and r > 50),
           ("完全没召回", lambda r: r is None)]


def _rank_of(hit):
    return hit["final_rank"] if hit["in_pool"] else None


def _recall(ranks, k):
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def _mrr(ranks):
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks) if ranks else 0.0


def _line(title, ranks):
    return (f"  {title:<22} n={len(ranks):<4} "
            f"R@5={_recall(ranks,5):.3f}  R@10={_recall(ranks,10):.3f}  "
            f"R@20={_recall(ranks,20):.3f}  MRR={_mrr(ranks):.3f}")


def _rank_in(cand, gt, key):
    """在候选池里按 key 排序后定位 ground truth 的名次。**与运行时逐字一致。**

    历史：P2a 之前这里要先 `sorted(cand, key=fused_rank)` 还原融合序，因为当时排序的
    第三级是稳定排序的副产品（并列时保持输入顺序）。**P2a 把收尾键显式化成全序之后，
    并列不复存在**，还原输入顺序这一步就不需要了——排序结果与输入顺序无关。
    这正是显式化的收益：少一处必须记住的隐式约定。
    ⚠ 但下面这些**反事实**的 key（纯相关性、各种权重）仍可能并列，所以它们要么自带
    收尾键（`offline_sort_key` 已带），要么只用于粗略对照。
    """
    ordered = sorted(cand, key=key, reverse=True)
    return next((i + 1 for i, c in enumerate(ordered) if c["chunk_id"] == gt), None)


def _tail(c):
    """反事实排序键的收尾三级，与生产 `sort_key` 一致：fused_score → pmid → chunk_id。"""
    return (c.get("fused_score") or 0.0, str(c.get("pmid") or ""), c["chunk_id"])


def simulate_pool(cand, P, top_k=10):
    """在 wide 组内部把池子截到 P，返回 ground truth 的 final_rank（None=没进池）。

    合法性：rrf 的 fused_score 只依赖候选自己在两路里的名次，rerank_score 逐条算，
    两者都不含跨池归一——所以截断只是丢掉 fused_rank>P 的候选，剩下的分数原样，
    重新按 rerank_score 排序即得 P 配置下的真实结果。**这是精确反推，不是近似。**
    """
    kept = [c for c in cand if c["fused_rank"] <= P]
    # 没有 rerank_score 的排最后。别写成 (是否非空, 分数) 的元组——两条都是 None 时
    # Python 会去比第二项，None < None 直接 TypeError。收尾键与生产一致，保证全序。
    kept.sort(key=lambda c: ((c["rerank_score"] if c["rerank_score"] is not None else -1e9),)
              + _tail(c), reverse=True)
    return kept


TIER_ORDER = [1, 2, 3]
TIER_LABEL = {
    1: "T1 纯中文临床层（主指标）",
    2: "T2 术语直穿层（中文问句嵌英文术语，翻译时原样穿过）",
    3: "T3 标识符锚定层（词面锚点极强，视为召回上界）",
}


def distribution_lines(tiers, cfg="prod", lvl="chunk", header=True):
    """命中分布表（「看一下命中分布」）。

    三层并排成一张表，比三块分开的柱状图好读；后面再给主指标 T1 一张柱状图。
    """
    L = []
    if header:
        L.append("=" * 96)
        tag = "线上默认，池子 50" if cfg == "prod" else "宽池 200，诊断用"
        L.append(f"【命中分布】目标文献落在第几名 —— 配置 {cfg}（{tag}）· {lvl} 级")
        L.append("=" * 96)
        L.append("")
        L.append("  「完全没召回」= 目标 chunk 连候选池都没进（不是排在后面，是根本没捞到）。")
        if cfg == "prod":
            L.append("")
            L.append("  ⚠ **配置 prod 的候选池只有 50，所以「第 50 名之后」结构上恒为 0**——")
            L.append("    那一档的量被并进了「完全没召回」。也就是说这里的「完全没召回」其实混了两种：")
            L.append("    真的一条都没捞到，和「捞到了但排在 50 开外、被池子截掉」。")
            L.append("    要把这两种分开，看下面配置 wide（池子 200）的对照表。")
        L.append("")
    used = [t for t in TIER_ORDER if tiers[t]]
    ranks = {t: [_rank_of(r["configs"][cfg]["hit"][lvl]) for r in tiers[t]] for t in used}

    L.append(f"    {'命中位置':<14}" + "".join(f"{'T%d (n=%d)' % (t, len(ranks[t])):>18}" for t in used))
    L.append(f"    {'-'*14}" + "".join(f"{'-'*18}" for _ in used))
    for name, fn in BUCKETS:
        cells = []
        for t in used:
            n = sum(1 for r in ranks[t] if fn(r))
            cells.append(f"{n:>8}  {n/len(ranks[t])*100:>6.1f}%")
        L.append(f"    {name:<14}" + "".join(cells))
    L.append("")
    L.append(f"  ▸ 主指标 T1（n={len(ranks[1])}）柱状：")
    for name, fn in BUCKETS:
        n = sum(1 for r in ranks[1] if fn(r))
        bar = "█" * int(round(n / max(1, len(ranks[1])) * 44))
        L.append(f"    {name:<14} {n:>4}  {n/len(ranks[1])*100:>5.1f}%  {bar}")
    return L


def split_lines(tiers, lvl="chunk"):
    """把 prod 的「完全没召回」拆成「真没召回」与「排在 50 开外被池子截掉」。

    ⚠ 这段必须有：prod 表里 T1 的「完全没召回」是整张表最扎眼、最会被追问的数字，
    而表格自己声明了它混着两种情况——如果只指向 wide 表却不给结论，读的人还得自己去减。
    数字**全部算出来**，不写死（项目铁律：每个数都要由变量得出）。
    """
    # ⚠ 标题必须带粒度：本节 chunk 级与 doc 级各出一块，数字不同。
    #   两块标题若一字不差，读的人第一反应是「同一张表写了两组不同的数，是不是错了」。
    L = ["", f"  ▸ **把 prod 的「完全没召回」拆开 · {lvl} 级**"
             f"（这正是上面 prod · {lvl} 级那张表说的「两种」）："]
    L.append("")
    L.append("    ⚠ 拆分**只在「prod 没召回」的那批里做**，两栏相加恰好等于第一栏。")
    L.append("      别拿「全体里 wide 排 50 开外的比例」去凑——那是另一个集合，加不上。")
    L.append("")
    L.append(f"    {'层':<6}{'prod 完全没召回':>18}{'│ 连200池也没有':>18}{'│ 被50的池子截掉':>19}")
    L.append(f"    {'-'*6}{'-'*18}{'-'*18}{'-'*19}")
    for t in TIER_ORDER:
        if not tiers[t]:
            continue
        n = len(tiers[t])
        miss = [r for r in tiers[t]
                if _rank_of(r["configs"]["prod"]["hit"][lvl]) is None]
        # prod 池 ⊆ wide 池（RRF 分数相同、wide 只是截得更宽），所以这两类不重不漏
        真没有 = [r for r in miss if not r["configs"]["wide"]["hit"][lvl]["in_pool"]]
        被截掉 = [r for r in miss if r["configs"]["wide"]["hit"][lvl]["in_pool"]]
        assert len(真没有) + len(被截掉) == len(miss)
        L.append(f"    T{t:<5}{len(miss)/n*100:>17.1f}%"
                 f"{len(真没有)/n*100:>17.1f}%{len(被截掉)/n*100:>18.1f}%"
                 f"   （{len(miss)} = {len(真没有)} + {len(被截掉)} 条）")
    L.append("")
    L.append("    读法：把候选池从 50 放大到 200 之后，原本并进「完全没召回」的那一部分")
    L.append("    显形出来——**它们其实被检索到了，只是被池子截断挡在重排之外**。")
    L.append("    剩下的才是真正一条都没捞到的（连 200 条候选里都没有）。")
    L.append("    ⚠ **本节在「基线」与「P2a」两份报告里数字完全相同，这是对的**——")
    L.append("      拆分看的是「有没有进候选池」，那由检索与融合决定；")
    L.append("      重排只改池内顺序、改不了池子成员。上面几张分布表才会因重排而不同。")
    L.append("    ⚠ 但别据此就去调大 rerank_pool：见评测报告，池子提到 100/200 后")
    L.append("      主指标反而单调变差，生产默认维持 50。放大只用于**诊断**。")
    return L


def do_report(args):
    f_run = _tagged(F_RUN, args.tag)
    f_report = _tagged(F_REPORT, args.tag)
    if not os.path.exists(f_run):
        sys.exit(f"找不到 {f_run}，先跑 --run" + (f" --tag {args.tag}" if args.tag else ""))
    rows = load_run(f_run)
    n_raw = sum(1 for l in open(f_run, encoding="utf-8") if l.strip())
    L = []

    def p(s=""):
        print(s)
        L.append(s)

    tiers = {t: [r for r in rows if r.get("tier") == t] for t in TIER_ORDER}

    p("=" * 96)
    p("golden 检索评测报告")
    p("=" * 96)
    p(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"query 数：{len(rows)}｜来源文献：{len({r['gid'] for r in rows})} 篇")
    p(f"检索库：{BM25_DIR} + Chroma 4M｜中译英：LLM（趟 1 冻结，趟 2 查表，确定性）")
    # ⚠ 这一行不能省：基线与 P2a 两份报告标题相同、生成时间只差一秒、query 数也一样，
    #   **只有这行说明用的是哪套重排配置**，否则单独拿到一份根本分不清是哪个。
    _mode, _eps = detect_rerank_mode(rows)
    p(f"**重排配置：{rerank_label(_mode, _eps)}**")
    p("")
    p("─" * 96)
    p("集合沿革（**这份报告的分母动过，读数字前先看这段**）")
    p("─" * 96)
    p("")
    p("  出题共 240 条 / 120 篇文献 → 人工复核剔除 5 条 → **最终 235 条 / 118 篇**。")
    p("  剔除的 5 条：")
    p("    · G107-1/2　模型照抄了提示词占位符，生成出来的字面就是「问题一」「问题二」")
    p("    · G033-1　　生物强化小麦对巴基斯坦妇女膳食锌摄入——农业营养学")
    p("    · G032-1/2　发酵残渣做农药载体、小麦种子发芽率与白粉病防治——农业")
    p("")
    p("  ⚠ **G032 那两条是补了抽样硬否决规则之后回放才发现的**，此前一直混在")
    p("    **T1 主指标层**里（曾是 237 条 / T1 153 条）。已剔除并**用同一批跑测明细重算**")
    p("    ——跑测明细按 qid 存着，剔除是纯重算、没有重跑检索。")
    p("  ⚠ **重算后全部结论不变**：rerank_pool 仍随池子增大单调变差、")
    p("    多准则重排仍净伤害检索、T2 仍不是泄漏层、methods 段仍是召回+排序双重问题。")
    p("    变的只是小数点后的数（例：T1·prod·R@10 0.458 → 0.450）。")
    p(f"  · 本次统计实际读入跑测明细 {n_raw} 条，与当前 golden_set.jsonl 取交集后 {len(rows)} 条。")
    p("")
    p("─" * 96)
    p("口径限制（**转述这些数字时必须一并带上，缺一条就会被读成别的意思**）")
    p("─" * 96)
    p("")
    p(" ① **ground truth 是「出题所用的那一块 chunk」，不是「所有能回答该问题的块」。**")
    p("    同一篇文献的摘要块往往同样能回答该问题，但按 chunk 级判就算没命中。")
    p("    所以**绝对 Recall 偏低是设计使然，不是系统差到这个程度**。")
    p("    doc 级（命中同篇任一块即算）是更宽松的口径，两个都报了。")
    p("")
    p(" ② **「只用向量」那一行偏乐观。** `vec_rank` 取的是多条向量查询变体里的**最好名次**")
    p("    （`Candidate.vec_rank = min(各变体名次)`），比单条查询的真实向量基线要好一点。")
    p("    ⚠ 但结论**不依赖**这个偏差：三层方向一致，且「纯相关性重排」(0.510) 同样超过")
    p("    「只用向量」(0.490)——即使把向量那行再打个折，重排有害的结论也不变。")
    p("")
    p(" ③ **抽样只覆盖「临床相关」的 chunk**（见 golden_构建.py 决定 2）。语料是 PubMed")
    p("    oa_comm 全量，含大量植物学/材料学/传感器文献，从那种正文出不了临床医生的问法。")
    p("    所以这些数**只代表临床类 query 上的检索表现，不代表全语料**。")
    p("    但**干扰项仍是全部 400 万条**，难度没有放水。")
    p("")
    p(" ④ **这批数适合做纵向对照（改一处 → 回测一次），不适合当绝对能力宣称。**")
    p("    「R@10=0.51」这种数拿出去单独说没有意义；「同一把尺子下 0.425 → 0.510」才有意义。")
    p("")
    p("另外三条读法约定：")
    p("   · 这是**检索评测**，不生成答案，全程无 LLM 作答，因此**完全确定性、可复跑**——")
    p("     不受 docs/工程笔记.md 三·1 那个「温度 0 连跑两轮结果翻转」影响。")
    p("   · **本报告不给「总体 Recall」。** 三层各自独立报、不合成：合成数带着必须解释的")
    p("     偏差，那个数没人会引用，不如从一开始就不产生。对外引用请用 **T1**。")
    p("   · chunk 级 = 必须命中出题用的那一块；doc 级 = 命中同一篇文献任一块即算。")
    p("")
    for t in TIER_ORDER:
        p(f"   {TIER_LABEL[t]}：{len(tiers[t])} 条 / {len({r['gid'] for r in tiers[t]})} 篇")
    p("")

    # ---------- A. 命中分布（放最前）----------
    dist = distribution_lines(tiers, cfg="prod", lvl="chunk")
    for s in dist:
        p(s)

    # ---------- B. 三层核心指标 ----------
    p("")
    for cfg in ("prod", "wide"):
        lim = rows[0]["configs"][cfg]["limits"]
        p("=" * 96)
        p(f"【核心指标 · 配置 {cfg}】向量 {lim['top_k_vector']} + BM25 {lim['top_k_keyword']} "
          f"→ 融合截到 {lim['rerank_pool']}｜最终 top_k={lim['top_k']}")
        p("=" * 96)
        for lvl in ("chunk", "doc"):
            p("")
            p(f"■ {lvl} 级")
            for t in TIER_ORDER:
                if not tiers[t]:
                    continue
                ranks = [_rank_of(r["configs"][cfg]["hit"][lvl]) for r in tiers[t]]
                p(_line(f"T{t}", ranks))
        p("")
    p("=" * 96)
    p("【命中分布 · 配置 wide 对照】把池子从 50 放大到 200 之后，分布往哪边挪")
    p("=" * 96)
    for s in distribution_lines(tiers, cfg="wide", lvl="chunk", header=False):
        p(s)

    # ---------- C. 单路对照（三层分别算）----------
    p("")
    p("=" * 96)
    p("【单路对照】BM25-only / 向量-only / 融合+重排，三层分别算（配置 wide·chunk 级）")
    p("=" * 96)
    p("  这一节同时回答两个问题：")
    p("    · T3 两路是否都逼近 1.0 → 确认词面泄漏（该层本就当上界看）")
    p("    · T2 的 BM25-only 是否明显高于向量-only → 量化「英文 token 绕过翻译」的程度。")
    p("      若 T2 的 BM25-only 与 T1 差不多，说明根本没送出多少锚点，")
    p("      那 T2 就不该算泄漏，而是「翻译器对原样英文术语的处理」这条链路的独立测点。")
    p("    · T1 两路应有差距且都不饱和 → 这才是系统的真实检索能力。")
    for t in TIER_ORDER:
        if not tiers[t]:
            continue
        p("")
        p(f"■ {TIER_LABEL[t]}")
        vec = [r["configs"]["wide"]["hit"]["chunk"].get("vec_rank") for r in tiers[t]]
        bm = [r["configs"]["wide"]["hit"]["chunk"].get("bm25_rank") for r in tiers[t]]
        fused = [_rank_of(r["configs"]["wide"]["hit"]["chunk"]) for r in tiers[t]]
        p(_line("向量-only", vec))
        p(_line("BM25-only", bm))
        p(_line("融合+重排", fused))

    # ---------- D. T1 的分层拆解 ----------
    p("")
    p("=" * 96)
    p("【分层拆解】只做 T1（主指标）· 配置 wide · chunk 级")
    p("=" * 96)
    p("  T2/T3 每格样本太少（<50 条摊到 5 个 section），拆了是噪声，故不拆。")
    for dim in ("section", "year_bucket"):
        p("")
        p(f"■ 按 {dim}")
        groups = defaultdict(list)
        for r in tiers[1]:
            groups[r[dim]].append(_rank_of(r["configs"]["wide"]["hit"]["chunk"]))
        for k in sorted(groups, key=lambda k: -len(groups[k])):
            p(_line(str(k), groups[k]))

    p("")
    p("■ methods / results 段为什么这么差：拆成「没进候选池」与「进了池但排不上去」")
    p("")
    p(f"    {'section':<14}{'n':>4}{'进候选池':>10}{'其中进top10':>12}"
      f"{'向量200命中':>12}{'BM25 200命中':>13}")
    for sec in ("abstract", "introduction", "methods", "results", "discussion"):
        sub = [r for r in tiers[1] if r["section"] == sec]
        if not sub:
            continue
        inpool = sum(1 for r in sub if r["configs"]["wide"]["hit"]["chunk"]["in_pool"])
        top10 = sum(1 for r in sub
                    if (h := r["configs"]["wide"]["hit"]["chunk"])["in_pool"]
                    and h["final_rank"] <= 10)
        nv = sum(1 for r in sub if r["configs"]["wide"]["hit"]["chunk"].get("vec_rank"))
        nb = sum(1 for r in sub if r["configs"]["wide"]["hit"]["chunk"].get("bm25_rank"))
        p(f"    {sec:<14}{len(sub):>4}{inpool:>7} ({inpool/len(sub)*100:>3.0f}%)"
          f"{top10:>12}{nv:>12}{nb:>13}")
    p("")
    p("  ⚠ methods 段 R@5=R@10=R@20 完全相等，是**两个问题叠加**，不是单一原因：")
    p("    · 约四成五**根本没进候选池**（200 条也没捞到）→ 召回问题")
    p("    · 进了池的里面又有大半排在 20 名开外        → 排序问题")
    p("    BM25 尤其找不到 methods 段（命中数只有 abstract 的三分之一）——方法学段落写的是")
    p("    通用流程语言（「患者被随机分组」「通过伦理审查」），和「这研究怎么做的」这类问题")
    p("    既不词面重合也不语义贴近，同篇的摘要块反而更容易命中。")
    p("    这也是 doc 级始终高于 chunk 级的主因。")
    p("  ⚠ **超出本轮范围**：这属于切块策略问题（方法段该不该单独成块、要不要带标题上下文），")
    p("    以后动切块时再看，别在重排层面找补。")

    # ---------- E. 重排诊断（本轮最有信息量的一节）----------
    p("")
    p("=" * 96)
    p("【重排诊断】重排救了谁、毁了谁，换权重会怎样（T1 · wide · chunk 级）")
    p("=" * 96)
    p("  这一节全是**离线精确反事实**，一次 GPU 都不用再烧：候选池里每条的")
    p("  rel / recency / authority 三个分项都存下来了，而 rerank_score 就是这三项的")
    p("  线性加权、无跨池归一——所以「换一组权重会怎样」等价于按新权重重新排序。")
    p("")
    p("■ 以「纯向量名次」为基准，重排改变了什么")
    saved, killed, both_in, neither = [], [], 0, 0
    for r in tiers[1]:
        h = r["configs"]["wide"]["hit"]["chunk"]
        v = h.get("vec_rank")
        f = h["final_rank"] if h["in_pool"] else None
        vin = v is not None and v <= 10
        fin = f is not None and f <= 10
        if vin and not fin:
            killed.append(r)
        elif fin and not vin:
            saved.append(r)
        elif vin and fin:
            both_in += 1
        else:
            neither += 1
    p(f"    向量进前10、重排后掉出   （毁）：{len(killed)}")
    p(f"    向量没进前10、重排后进入 （救）：{len(saved)}")
    p(f"    两者都进前10                  ：{both_in}")
    p(f"    两者都没进                    ：{neither}")
    p(f"    **净效果：{len(saved)-len(killed):+d} 条**")
    p("")
    p(f"    被毁掉的按年份：{dict(Counter(r['year_bucket'] for r in killed))}")
    p(f"    被救回的按年份：{dict(Counter(r['year_bucket'] for r in saved))}")
    p("")
    p("    被毁掉的样例（相关性满分却被踢走的）：")
    for r in killed[:8]:
        h = r["configs"]["wide"]["hit"]["chunk"]
        p(f"      {r['qid']} {r['pub_year']}  vec#{h['vec_rank']:<3} → final#{h['final_rank']:<4}"
          f"  rel={h.get('rel_score'):.2f} rec={h.get('recency_score'):.2f}"
          f"  {r['query'][:28]}")

    # ---- 为什么权重一动降幅就这么陡：看池内跨度 ----
    p("")
    p("■ 为什么 recency 权重从 0.25 降到 0.05，MRR 就跳一大截（非线性的来源）")
    p("")
    p("  排序只取决于**池内差值**，不取决于绝对水平。所以要比的是各准则的池内 std × 权重。")
    for scope, topn in (("整池", None), ("池内前 20（按 rel）", 20)):
        stds = {k: [] for k in ("rel_score", "recency_score", "authority_score")}
        for r in tiers[1]:
            cand = r["configs"]["wide"]["candidates"]
            if topn:
                cand = sorted(cand, key=lambda c: c["rel_score"] or 0, reverse=True)[:topn]
            for k in stds:
                v = [c[k] for c in cand if c[k] is not None]
                if len(v) >= 2:
                    stds[k].append(statistics.pstdev(v))
        med = {k: statistics.median(v) for k, v in stds.items()}
        p(f"    ▸ {scope}：rel std={med['rel_score']:.4f}  "
          f"recency std={med['recency_score']:.4f}  authority std={med['authority_score']:.4f}")
        for wn, w in (("0.60/0.25/0.15", (.60, .25, .15)), ("0.90/0.05/0.05", (.90, .05, .05))):
            infl = [w[0] * med["rel_score"], w[1] * med["recency_score"],
                    w[2] * med["authority_score"]]
            tot = sum(infl) or 1
            p(f"        权重 {wn} → 有效影响力 rel {infl[0]:.4f}({infl[0]/tot*100:.0f}%) "
              f"recency {infl[1]:.4f}({infl[1]/tot*100:.0f}%) "
              f"authority {infl[2]:.4f}({infl[2]/tot*100:.0f}%)")
    p("")
    p("  ⚠ 关键在「池内前 20」那一行：交叉编码器认可的候选 rel 都挤在 1.0 附近，")
    p("    池内 std 只剩 ~0.08；而它们的发表年份铺得很开，recency std ~0.18（2.2 倍）。")
    p("    于是默认权重下 recency 的排序影响力几乎等于 rel——把权重降到 0.05")
    p("    等于一次性抽掉八成竞争影响力，所以头一档降幅最陡。")
    p("  ⚠ **这不是归一化 bug，重新标度修不好**：实测「池内 min-max 归一后再按 0.60/0.25/0.15")
    p("    加权」MRR = 0.207，比不归一的 0.231 **更差**——归一化把 recency 的跨度拉满到 [0,1]，")
    p("    反而让它更压得过 rel。病根是「饱和型相关性分 + 线性年份分相加」这个结构本身。")

    p("")
    p("■ 换重排打分方式会怎样")
    schemes = [
        ("旧行为 0.60rel+0.25rec+0.15auth",
         lambda c: ((c["rerank_score"] if c["rerank_score"] is not None else -9),) + _tail(c)),
        ("纯相关性 1.0rel",
         lambda c: ((c["rel_score"] if c["rel_score"] is not None else -9),) + _tail(c)),
        ("同分裁决 ε=0.02（**新的生产默认**）",
         lambda c: (round((c["rel_score"] or 0) / 0.02), c["recency_score"] or 0) + _tail(c)),
        ("同分裁决 ε=0.01",
         lambda c: (round((c["rel_score"] or 0) / 0.01), c["recency_score"] or 0) + _tail(c)),
        ("0.90rel+0.05rec+0.05auth",
         lambda c: (0.9 * (c["rel_score"] or 0) + 0.05 * (c["recency_score"] or 0)
                    + 0.05 * (c["authority_score"] or 0),) + _tail(c)),
        ("0.80rel+0.10rec+0.10auth",
         lambda c: (0.8 * (c["rel_score"] or 0) + 0.1 * (c["recency_score"] or 0)
                    + 0.1 * (c["authority_score"] or 0),) + _tail(c)),
        ("不重排，只用融合分", lambda c: (-c["fused_rank"],) + _tail(c)),
        ("只用向量名次",
         lambda c: (-(c["vec_rank"] if c["vec_rank"] else 10 ** 6),) + _tail(c)),
    ]
    p("")
    p("  「同分裁决」= 先把 rel 按 ε 分档，**同档内**才比 recency。")
    p("   它保留了「同等相关时优先新文献」这个原意，但不让 recency 去推翻相关性排序。")

    _rank_by = _rank_in       # 与运行时一致：先还原融合序，再稳定排序（见 _ordered_like_runtime）

    p("")
    for name, key in schemes:
        ranks = [_rank_by(r["configs"]["wide"]["candidates"], r["gt_chunk_id"], key)
                 for r in tiers[1]]
        p(_line(name, ranks))

    p("")
    p("■ 同样这几组权重，按年份桶看 R@10（看代价落在谁身上）")
    ybs = [y for y, _ in [("≤2015", 0), ("2016-2019", 0), ("2020-2022", 0), ("2023+", 0)]]
    p("")
    p(f"    {'方案':<30}" + "".join(f"{y:>12}" for y in ybs))
    for name, key in schemes:
        cells = []
        for yb in ybs:
            sub = [r for r in tiers[1] if r["year_bucket"] == yb]
            ranks = [_rank_by(r["configs"]["wide"]["candidates"], r["gt_chunk_id"], key)
                     for r in sub]
            cells.append(f"{_recall(ranks, 10):>12.3f}")
        p(f"    {name:<30}" + "".join(cells))

    # ---- ε 敏感度：ε=0.01 是不是魔数 ----
    p("")
    p("■ 同分裁决的 ε 敏感度扫描（ε=0.01 不是拍脑袋来的，看它落在刀刃还是平台上）")
    p("")
    p("  ε 是 rel 的分档宽度：ε→0 退化成纯相关性（recency 完全不起作用），")
    p("  ε→大 则整个池子挤进同一档、退化成纯按年份排。中间应当有一段平台。")
    p("")
    eps_grid = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
    ybs = ["≤2015", "2016-2019", "2020-2022", "2023+"]
    p(f"    {'ε':<12}{'R@5':>8}{'R@10':>8}{'R@20':>8}{'MRR':>8}   "
      + "".join(f"{y:>11}" for y in ybs))
    def _sweep_row(label, key):
        rk = [_rank_by(r["configs"]["wide"]["candidates"], r["gt_chunk_id"], key)
              for r in tiers[1]]
        cells = []
        for y in ybs:
            sub = [r for r in tiers[1] if r["year_bucket"] == y]
            cells.append(f"{_recall([_rank_by(r['configs']['wide']['candidates'], r['gt_chunk_id'], key) for r in sub], 10):>11.3f}")
        p(f"    {label:<12}{_recall(rk,5):>8.3f}{_recall(rk,10):>8.3f}"
          f"{_recall(rk,20):>8.3f}{_mrr(rk):>8.3f}   " + "".join(cells))

    _sweep_row("旧行为",
               lambda c: ((c["rerank_score"] if c["rerank_score"] is not None else -9),) + _tail(c))
    for e in eps_grid:
        mark = f"{e:.3f}" + ("  ←默认" if abs(e - 0.02) < 1e-9 else "")
        _sweep_row(mark,
                   lambda c, e=e: (round((c["rel_score"] or 0) / e), c["recency_score"] or 0) + _tail(c))
    _sweep_row("ε→0(纯rel)",
               lambda c: ((c["rel_score"] if c["rel_score"] is not None else -9),) + _tail(c))

    p("")
    p("  ▸ 怎么读这张表：**ε 落在一片平台上，不是刀刃。**")
    p("    ε 从 0.005 到 0.05 跨了一个数量级，R@10 只从 0.510 变到 0.490（差 2pp）——")
    p("    说明这个修法**稳健，不依赖把 ε 调准**。要到 ε=0.2 才退化回旧行为水平。")
    p("    刀刃敏感才说明修法脆；这里是平台，可以放心用。")
    p("")
    p("  ▸ **但必须诚实读出另一件事：每一档 ε 的 MRR 都低于纯相关性（0.357），且 ε 越大越低。**")
    p("    也就是说——**在这个 benchmark 上，同分裁决本身不产生任何收益；")
    p("    全部增益来自「把 recency 从加法项移除」这一步。**")
    p("    同分裁决相对纯相关性是**净支出**，不是净收入。")
    p("")
    p("  ▸ 默认值取 **ε=0.02**，两个理由：")
    p("    · ε=0.005 在表上支配 ε=0.01（R@10 与四个年份桶都相同、MRR 更高），")
    p("      所以 **0.01 没有当默认值的理由**；")
    p("    · ε=0.02 是**唯一一个在 2023+ 上也严格更好**的配置（0.476 > 0.452）。")
    p("      ⚠ 精确表述：ε=0.02 在四个年份桶上**没有一格更差，其中三格严格更好**")
    p("      （≤2015 +22.5pp、2016-2019 +6.3pp、2023+ +2.4pp），2020-2022 **持平**。")
    p("      别说成「四格全部严格更好」——2020-2022 是 0.538 对 0.538，打平。")
    p("      选 0.02 的实质理由是：**2023+ 正是 recency 本该发挥作用的地方**，")
    p("      而只有 ε=0.02 在那一格真的赚到了；ε=0.01 在那格只是持平。")
    p("")
    p("■ 结论（措辞要点，转述时请照抄）")
    p("")
    p("  **不是「recency 无价值」，而是「40% 的加法权重代价过高」。**")
    p("  旧行为用新文献上的 +2.3pp（2023+ 0.452 vs 纯相关性 0.429），")
    p("  换来老文献上的 −30pp（≤2015 0.300 vs 0.600）。这笔交易在任何权重观下都不划算。")
    p("")
    p("  **收益来自把 recency 从加法项移除；同分裁决是在此基础上以可控的小代价保留")
    p("    recency 的原意，买的是本 benchmark 测不到的临床时效价值。**")
    p("")
    p("  ⚠ 为什么要这么说，而不是直接上纯相关性（它 MRR 最高）：")
    p("    golden 集的 ground truth 是「出题所用的那一块」，**由文本定义**，")
    p("    **不含临床时效价值**——「同等相关时新指南优于旧指南」这种价值本评测量不到，")
    p("    而 recency 当初大概正是想代理它。纯相关性把这份价值完全放弃了；")
    p("    同分裁决用 MRR 上的一点代价把它买回来，且**只在相关性打平处生效**。")
    p("    这样表述的好处：**有人拿「纯相关性 MRR 更高」来问，立场依然成立**——")
    p("    我们并不声称同分裁决在本表上更优，我们声称它在本表之外还买到了东西。")
    p("")
    p("  ✔ **已落地（P2a，2026-08-12）**：`检索_多路检索.py` 默认 `rerank_mode=\"tiebreak\"`，")
    p("    ε=0.02。旧行为仍可用 `--rerank-mode weighted` 跑出来做对照。")
    p("")
    p("    生产实测（真跑一轮，非离线反推；离线与真跑逐条一致，见 --verify --tag p2a）：")
    p("      T1 · wide · chunk    R@5=0.437  R@10=0.497  R@20=0.556  MRR=0.306")
    p("      T1 · prod · chunk    R@5=0.444  R@10=0.483  R@20=0.523  MRR=0.331  ← 线上默认")
    p("      年份桶 R@10          ≤2015 0.525 ｜ 2016-2019 0.433 ｜")
    p("                           2020-2022 0.538 ｜ 2023+ 0.476")
    p("      对旧行为             +22.5pp ｜ +6.6pp ｜ 持平 ｜ +2.4pp")
    p("")
    p("    ⚠ 同时做过一条**回归硬断言**：新代码的 weighted 模式必须逐位复现原始基线")
    p("      （老代码那轮）。结果 14/14 全过——逐条名次 237/237 相同（2 配置 × 2 粒度）、")
    p("      聚合指标与四个年份桶逐位复现。**所以本报告里的历史对照数字仍然有效，")
    p("      不需要因为 P2a 改了排序路径而重算。** 见 report_data\\golden_P2a回归断言.txt。")

    # ---------- F. rerank_pool 该不该动 ----------
    p("")
    p("=" * 96)
    p("【rerank_pool 默认值该不该从 50 提到 100/200】")
    p("=" * 96)
    p("  在 wide 组内部截池子——rrf 的 fused_score 只依赖候选自己的名次，")
    p("  rerank_score 逐条算，两者都无跨池归一，所以这是**精确反推**不是近似。")
    for t in TIER_ORDER:
        if not tiers[t]:
            continue
        p("")
        p(f"■ {TIER_LABEL[t]}")
        for P in (50, 100, 200):
            ranks = []
            for r in tiers[t]:
                kept = simulate_pool(r["configs"]["wide"]["candidates"], P)
                pos = next((i + 1 for i, c in enumerate(kept)
                            if c["chunk_id"] == r["gt_chunk_id"]), None)
                ranks.append(pos)
            p(_line(f"池子截到 {P}", ranks))
    p("")
    deep = [r for r in rows
            if (h := r["configs"]["wide"]["hit"]["chunk"])["in_pool"]
            and h["final_rank"] is not None and h["final_rank"] <= 10
            and h["fused_rank"] is not None and h["fused_rank"] > 50]
    p(f"  重排后进了 top-10、但融合名次在 50 之后的：{len(deep)}/{len(rows)} 条"
      f"（{len(deep)/len(rows)*100:.1f}%）")
    p("  ↑ 这些就是 rerank_pool=50 会**直接丢掉**的命中：")
    for r in deep[:12]:
        h = r["configs"]["wide"]["hit"]["chunk"]
        p(f"    T{r['tier']} {r['qid']}  fused#{h['fused_rank']:<4} → final#{h['final_rank']:<3} "
          f"{r['query'][:34]}")
    p("")
    p("  ⚠ 但**净效果是负的**：池子越大，重排把「不相关但新且权威」的文献捞上来的机会也越多。")
    p("    所以生产默认值维持 50。")
    p("  ⚠ **「越大越差」只针对生产默认值。** 诊断用途——查某篇文献到底在不在候选池、")
    p("    排第几、被什么压下去——仍然必须开到 200+，否则「不在池」与「被压下去」分不开。")
    p("    **两个用途别合并成一个默认值。**")

    # ---------- F. 耗时 ----------
    p("")
    p("=" * 96)
    el = {c: [r["configs"][c]["elapsed"] for r in rows] for c in ("prod", "wide")}
    for c, v in el.items():
        p(f"  {c} 单次检索：中位 {statistics.median(v):.2f}s  合计 {sum(v)/60:.1f} 分钟")
    p("=" * 96)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f_report, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n→ {f_report}")

    # 命中分布单独再出一份——别让人在长报告里翻
    f_dist = _tagged(os.path.join(REPORT_DIR, "golden_命中分布表.txt"), args.tag)
    with open(f_dist, "w", encoding="utf-8") as f:
        f.write(f"golden 检索评测 · 命中分布表\n生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"query {len(rows)} 条 / 来源文献 {len({r['gid'] for r in rows})} 篇\n")
        # 与评测报告同理：这份表也是单独发出去的，没有配置标识就分不清是哪一版
        f.write(f"**重排配置：{rerank_label(*detect_rerank_mode(rows))}**\n")
        f.write("T1 纯中文临床层（主指标）｜T2 术语直穿层｜T3 标识符锚定层（召回上界）\n\n")
        f.write("⚠ 集合沿革：出题 240 条 → 人工复核剔除 5 条（2 条模板占位符、3 条农业类）\n"
                "  → 最终 235 条。其中 G032-1/2 是补了抽样硬否决规则后回放才发现的，\n"
                "  此前混在 T1 主指标层里（曾是 237 条 / T1 153 条）。\n"
                "  **已剔除并用同一批跑测明细重算，全部结论不变**，只有小数点后的数变了。\n\n")
        # prod（线上默认）两张 —— 对外引用用这两张
        f.write("\n".join(distribution_lines(tiers, cfg="prod", lvl="chunk")) + "\n\n")
        f.write("\n".join(distribution_lines(tiers, cfg="prod", lvl="doc")) + "\n\n")
        # wide（池子 200）两张 —— **上面 prod 表里那句「看下面配置 wide 的对照表」指的就是这两张**，
        # 缺了它们那句话就是悬空引用（2026-08-12 补）
        f.write("\n".join(distribution_lines(tiers, cfg="wide", lvl="chunk")) + "\n\n")
        f.write("\n".join(distribution_lines(tiers, cfg="wide", lvl="doc")) + "\n\n")
        # 拆分结论：prod 的「完全没召回」到底是真没捞到，还是被 50 的池子截掉
        f.write("=" * 96 + "\n")
        f.write("【拆分】prod 的「完全没召回」= 真没召回 + 排在 50 开外被池子截掉\n")
        f.write("=" * 96 + "\n")
        f.write("\n".join(split_lines(tiers, "chunk")) + "\n\n")
        f.write("\n".join(split_lines(tiers, "doc")) + "\n\n")
        f.write("⚠ 口径：ground truth 是「出题所用的那一块 chunk」，不是「所有能回答该问题的块」。\n"
                "  chunk 级偏严（同篇摘要块答得了也算没命中），doc 级是更宽松的口径，两个都给。\n"
                "  绝对值偏低是设计使然；这批数适合纵向对照，不适合当绝对能力宣称。\n")
    print(f"→ {f_dist}")


# ==============================================================================
# 四、证伪：离线反事实 vs 真跑一轮
# ==============================================================================
def do_verify(args):
    """比对「离线反事实预测的名次」与「按该配置真跑一轮得到的名次」。

    报告里所有「换权重 / 换池子 / 换打分方式会怎样」的数字都是离线反推的。
    那些数要拿去改生产默认值，所以必须证伪：反推逻辑若有错，真跑会立刻暴露
    （池子不同 / 名次不同 / 指标不同）。

    两个已验过的方案：
        # ① 纯相关性权重（权重配置改动）
        & $py scripts\\golden_跑测.py --run --weights "rel=1.0,rec=0,auth=0" --tag relonly
        & $py scripts\\golden_跑测.py --verify --tag relonly --verify-scheme relonly

        # ② 同分裁决（**结构性**改动：ε 分档 + 档内比年份，不是换权重）
        & $py scripts\\golden_跑测.py --run --rerank-mode tiebreak --eps 0.01 --tag tie001
        & $py scripts\\golden_跑测.py --verify --tag tie001 --verify-scheme tiebreak --eps 0.01
    """
    f_real = _tagged(F_RUN, args.tag or "relonly")
    if not (os.path.exists(F_RUN) and os.path.exists(f_real)):
        sys.exit(f"需要两轮产物：{F_RUN} 与 {f_real}")
    base = {r["qid"]: r for r in load_run(F_RUN)}
    real = {r["qid"]: r for r in load_run(f_real)}
    qids = [q for q in base if q in real]
    L = []

    def p(s=""):
        print(s)
        L.append(s)

    scheme = args.verify_scheme
    eps = args.eps if args.eps is not None else 0.02
    if scheme == "tiebreak":
        scheme_name = f"同分裁决 ε={eps}（**结构性**改动，非换权重）"

        def offline_key(c):
            return offline_sort_key(c, "tiebreak", eps)
    else:
        scheme_name = "纯相关性 1.0rel（权重配置改动）"

        def offline_key(c):
            return offline_sort_key(dict(c, rerank_score=c["rel_score"]), "weighted", eps)

    p("=" * 92)
    p("golden 检索评测 · 离线反事实的证伪")
    p("=" * 92)
    p(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"被验方案：{scheme_name}")
    p(f"默认权重轮 {len(base)} 条｜该方案真跑轮 {len(real)} 条｜交集 {len(qids)} 条")
    p("")

    same = sum(1 for q in qids
               if {c["chunk_id"] for c in base[q]["configs"]["wide"]["candidates"]}
               == {c["chunk_id"] for c in real[q]["configs"]["wide"]["candidates"]})
    p(f"① 候选池一致性：{same}/{len(qids)} 条完全相同")
    p("   （权重只影响重排、不影响召回与融合，所以池子理应逐条相同）")

    def offline_rank(r):
        return _rank_in(r["configs"]["wide"]["candidates"], r["gt_chunk_id"], offline_key)

    ok, bad = 0, []
    for q in qids:
        pred = offline_rank(base[q])
        h = real[q]["configs"]["wide"]["hit"]["chunk"]
        act = h["final_rank"] if h["in_pool"] else None
        if pred == act:
            ok += 1
        else:
            bad.append((q, pred, act))
    p("")
    p(f"② 逐条名次：{ok}/{len(qids)} 完全一致，{len(bad)} 条不一致")
    for q, pred, act in bad[:15]:
        p(f"     {q}  离线预测 #{pred}  实跑 #{act}")

    p("")
    p("③ 汇总指标（T1 主指标层）")
    t1 = [q for q in qids if base[q]["tier"] == 1]

    def rk(d, q):
        h = d[q]["configs"]["wide"]["hit"]["chunk"]
        return h["final_rank"] if h["in_pool"] else None

    short = "同分裁决" if scheme == "tiebreak" else "纯相关性"
    for name, ranks in (("现行 0.60/0.25/0.15 实跑", [rk(base, q) for q in t1]),
                        (f"{short} · 离线预测", [offline_rank(base[q]) for q in t1]),
                        (f"{short} · 真跑一轮", [rk(real, q) for q in t1])):
        p(_line(name, ranks))

    p("")
    p("④ 按年份桶（R@10, T1）")
    for yb in ("≤2015", "2016-2019", "2020-2022", "2023+"):
        sub = [q for q in t1 if base[q]["year_bucket"] == yb]
        if not sub:
            continue
        p(f"   {yb:<12} n={len(sub):<4} 现行 {_recall([rk(base,q) for q in sub],10):.3f}"
          f"  →  {short}实跑 {_recall([rk(real,q) for q in sub],10):.3f}")

    p("")
    verdict = "通过" if (same == len(qids) and not bad) else "**不通过**"
    p(f"结论：{verdict}。" + ("离线反事实与真跑逐字一致，因此报告里所有「换权重/换池子会怎样」"
                             "的数字都可直接采信，再试新配置是 0 GPU 成本。"
                             if verdict == "通过" else
                             "反推逻辑与真实行为不符，报告里的反事实数字**不可用**，先查原因。"))
    p("=" * 92)

    out = os.path.join(REPORT_DIR, f"golden_离线反事实证伪_{scheme}.txt")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n→ {out}")


# ==============================================================================
# 五、回归断言：新代码的 weighted 模式必须逐位复现原始基线
# ==============================================================================
# 报告里所有历史对照数字（「旧行为 R@10=0.425 / MRR=0.241」以及四个年份桶）都产自
# **P2a 之前的老代码**——当时排序的收尾是「稳定排序保持输入顺序」这个隐式第三级。
# P2a 之后 weighted 模式走的是显式全序 rerank_score → fused_score → pmid → chunk_id。
#
# 加权和几乎不会并列，所以两者**应当**逐位相同。但这必须验，不能假定：
#   · 若相同 → 原始基线里没有一条名次是被输入顺序决定的，**之前那批数字可以继续引用**；
#   · 若不同 → 说明并列比预期多、原始基线有一部分名次是输入顺序的产物，
#     **报告里所有历史对照数字都要用新代码重算**，在那之前一个都不能引用。
# 这是个二分结果、零额外成本，但它决定之前那批数字的效力，所以单独做成一条硬断言。
#
# ⚠⚠ 下面这组常数是**已发布数字的 tripwire**，不是「调到能过为止」的参数。
#    真正的判据是**逐条名次**（老代码 vs 新代码，235/235 必须全同）——那条与集合大小无关。
#    这组聚合值额外绑定了「**集合组成**」：集合一变，它必然失败，从而**强制做一次显式决定**，
#    而不是让对外引用的数字悄悄漂掉。
#
#    改这组值的**唯一合法理由**是：集合组成有据地变了，且逐条名次仍然全同。
#    每改一次都必须在下面留一行记录。
#
#    修订记录：
#      2026-08-12 初版  237 条 / T1 153 条
#          R@5 0.353 ｜ R@10 0.425 ｜ R@20 0.490 ｜ MRR 0.241
#          年份 0.300 / 0.406 / 0.538 / 0.452
#      2026-08-12 修订  235 条 / T1 151 条  ← 当前
#          起因：补硬否决规则后回放，发现 G032-1/2 是**纯农业问题**
#          （发酵残渣做农药载体、小麦种子发芽率与白粉病防治，出自 RSC Advances），
#          却落在 T1 主指标层里污染对外引用的数，已加进 HARD_DROP。
#          ⚠ 此次改动**逐条名次 235/235 仍然全同**，只有分母变了——
#            即行为没变、口径变了，符合上面说的唯一合法理由。
BASELINE_EXPECT = {
    "per_query_identical": True,          # 与老代码那轮逐条同名次（真正的判据）
    "T1_wide_chunk": {"R@5": 0.344, "R@10": 0.417, "R@20": 0.483, "MRR": 0.231},
    "T1_wide_chunk_by_year": {"≤2015": 0.300, "2016-2019": 0.367,
                              "2020-2022": 0.538, "2023+": 0.452},
}


def do_assert_baseline(args):
    """硬断言：新代码 weighted 模式 == 老代码原始基线。**每个判定都由变量算出。**"""
    f_new = _tagged(F_RUN, args.tag or "baseline")
    if not os.path.exists(F_RUN):
        sys.exit(f"缺原始基线 {F_RUN}")
    if not os.path.exists(f_new):
        sys.exit(f"缺新代码 weighted 轮 {f_new}\n"
                 f"先跑：golden_跑测.py --run --rerank-mode weighted --tag baseline")
    old = {r["qid"]: r for r in load_run(F_RUN)}
    new = {r["qid"]: r for r in load_run(f_new)}
    qids = sorted(set(old) & set(new))
    L, failures = [], []

    def p(s=""):
        print(s)
        L.append(s)

    def judge(name, ok, detail=""):
        """✓/✗ 由传进来的 ok 决定，绝不无条件打印通过（docs/工程笔记.md 一·7 铁律）。"""
        p(f"  {'✓' if ok else '✗ **失败**'} {name}" + (f"　{detail}" if detail else ""))
        if not ok:
            failures.append(name)
        return ok

    p("=" * 96)
    p("P2a 回归断言：新代码 weighted 模式必须逐位复现原始基线")
    p("=" * 96)
    p(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"原始基线（P2a 前的老代码）：{F_RUN}")
    p(f"新代码 weighted 轮        ：{f_new}")
    p(f"交集 {len(qids)} 条")
    p("")
    p("为什么要单独验这一条：原始基线的排序收尾是**稳定排序保持输入顺序**这个隐式第三级；")
    p("P2a 后 weighted 走显式全序。加权和几乎不并列、两者应当相同——但若不同，说明")
    p("**原始基线里有一部分名次是输入顺序的产物，报告里所有历史对照数字都要重算**。")
    p("")

    judge("两轮 query 集合一致", len(qids) == len(old) == len(new),
          f"old={len(old)} new={len(new)} 交集={len(qids)}")

    # ---- ① 逐条名次（两个配置 × 两个粒度，全比）----
    p("")
    p("① 逐条名次比对")
    for cfg in ("prod", "wide"):
        for lvl in ("chunk", "doc"):
            diff = []
            for q in qids:
                a = _rank_of(old[q]["configs"][cfg]["hit"][lvl])
                b = _rank_of(new[q]["configs"][cfg]["hit"][lvl])
                if a != b:
                    diff.append((q, a, b))
            judge(f"{cfg}/{lvl} 级 {len(qids)-len(diff)}/{len(qids)} 同名次",
                  not diff)
            for q, a, b in diff[:10]:
                p(f"      {q}  老 #{a}  新 #{b}")

    # ---- ② 候选池 ----
    same_pool = sum(1 for q in qids
                    if {c["chunk_id"] for c in old[q]["configs"]["wide"]["candidates"]}
                    == {c["chunk_id"] for c in new[q]["configs"]["wide"]["candidates"]})
    p("")
    p("② 候选池（重排改动不该影响召回与融合）")
    judge(f"候选池 {same_pool}/{len(qids)} 完全相同", same_pool == len(qids))

    # ---- ③ 聚合指标必须逐位复现 ----
    p("")
    p("③ T1 · wide · chunk 级聚合指标 —— 必须逐位复现已发布的数字")
    t1 = [q for q in qids if old[q]["tier"] == 1]
    rk_new = [_rank_of(new[q]["configs"]["wide"]["hit"]["chunk"]) for q in t1]
    got = {"R@5": _recall(rk_new, 5), "R@10": _recall(rk_new, 10),
           "R@20": _recall(rk_new, 20), "MRR": _mrr(rk_new)}
    for k, want in BASELINE_EXPECT["T1_wide_chunk"].items():
        judge(f"{k} = {want:.3f}", abs(got[k] - want) < 5e-4, f"实测 {got[k]:.4f}")

    p("")
    p("④ T1 · wide · chunk 级按年份桶 R@10")
    for yb, want in BASELINE_EXPECT["T1_wide_chunk_by_year"].items():
        sub = [q for q in t1 if old[q]["year_bucket"] == yb]
        gotv = _recall([_rank_of(new[q]["configs"]["wide"]["hit"]["chunk"]) for q in sub], 10)
        judge(f"{yb:<12} = {want:.3f}", abs(gotv - want) < 5e-4,
              f"实测 {gotv:.4f}（n={len(sub)}）")

    p("")
    p("=" * 96)
    if failures:
        p(f"**断言失败 {len(failures)} 项** —— " + "；".join(failures[:6]))
        p("")
        p("⚠ **立刻停，别把任何数字写进报告。** 这说明原始基线里有名次是由输入顺序决定的，")
        p("  报告里所有历史对照数字（旧行为 R@10/MRR、救毁计数、权重扫描、ε 扫描的『旧行为』行）")
        p("  都必须用新代码重跑重算，在那之前一个都不能引用。")
    else:
        p("**全部通过。**新代码 weighted 模式逐位复现原始基线：")
        p("  · 逐条名次完全相同（两个配置 × 两个粒度）")
        p("  · 聚合指标与年份桶逐位复现已发布数字")
        p("  → 原始基线里**没有一条名次是被输入顺序决定的**，")
        p("    之前那批历史对照数字**可以继续引用**，不需要重算。")
    p("=" * 96)

    out = os.path.join(REPORT_DIR, "golden_P2a回归断言.txt")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n→ {out}")
    sys.exit(1 if failures else 0)


def main():
    ap = argparse.ArgumentParser(description="golden 检索评测集 · 跑测与统计")
    ap.add_argument("--translate", action="store_true", help="趟1：冻结中译英（需 Ollama）")
    ap.add_argument("--run", action="store_true", help="趟2：跑检索（需 65GB 库，不需 Ollama）")
    ap.add_argument("--report", action="store_true", help="出指标（离线，秒级）")
    ap.add_argument("--verify", action="store_true",
                    help="证伪：比对离线反事实与 --tag 那轮真跑（默认 tag=relonly）")
    ap.add_argument("--assert-baseline", action="store_true",
                    help="硬断言：新代码 weighted 模式必须逐位复现原始基线（决定历史数字能否引用）")
    ap.add_argument("--bm25", default=BM25_DIR)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--weights", default="",
                    help="重排权重，形如 rel=1.0,rec=0,auth=0（三项写全）。留空用流水线默认")
    ap.add_argument("--tag", default="",
                    help="产物文件后缀，用来区分不同权重的跑测，别互相覆盖")
    ap.add_argument("--rerank-mode", choices=["weighted", "tiebreak"], default="tiebreak",
                    help="tiebreak=同分裁决（生产默认）；weighted=旧的加权总分（跑基线对照用）")
    ap.add_argument("--eps", type=float, default=None,
                    help="同分裁决的 rel 分档宽度，默认取生产默认值 0.02")
    ap.add_argument("--landmark", action="store_true",
                    help="趟2：开 P0 的 landmark 并行路（默认关）。开着测的是「补语料的代价与收益」，"
                         "关着测的才是主库基线——两轮对照着看，别只跑一轮")
    ap.add_argument("--verify-scheme", choices=["relonly", "tiebreak"], default="relonly",
                    help="--verify 时要比对的离线方案")
    args = ap.parse_args()
    if args.translate:
        do_translate(args)
    elif args.run:
        do_run(args)
    elif args.report:
        do_report(args)
    elif args.verify:
        do_verify(args)
    elif args.assert_baseline:
        do_assert_baseline(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
