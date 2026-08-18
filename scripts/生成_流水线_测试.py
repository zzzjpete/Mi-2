# -*- coding: utf-8 -*-
"""第七阶段（二）· 生成流水线的完整流程测试

三种跑法，用途不同，不要混着看结论：

  --offline   用阶段三的 1000 条样例块当证据（PLoS Biology 2003–2005，基础生物学）。
              **测的是链路本身**：四段是否都跑通、JSON 是否解析得出、引用是否落地、
              无证据时是否老实拒答。不测答案质量——这批语料回答不了临床问题。
  --live      走阶段六真检索（400 万向量 + BM25）。测试题就是阶段一压测裸模型时留下的
              四类错题，RAG 就是来修它们的。首次加载约 15.8GB RSS + 数分钟。
  --ablation  同一道题跑三种链路配置（1/2/4 次模型调用），把"四段链值不值那 3~4 倍耗时"
              这个阶段七之一留下的未决问题量化掉。

产物（都落在 report_data\\，是后续写报告的证据链，只读不改）：
    生成_流水线测试_<mode>.jsonl   每次生成一条：查询/答案/全部指标/引用来源
    生成_流水线测试报告_<mode>.txt  人读版：答案全文 + 分阶段耗时 + 出处清单 + 汇总判定

用法：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_流水线_测试.py --offline
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_流水线_测试.py --ablation
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_流水线_测试.py --live \\
        --bm25 E:\\rag\\data\\bm25_index_4m
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
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "report_data")


def _load_by_path(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pipe_mod = _load_by_path("shengcheng_liushuixian", "生成_流水线.py")
_asm_mod = _pipe_mod._asm_mod
MedicalGenerationPipeline = _pipe_mod.MedicalGenerationPipeline


# ============================================================================
# 一、测试查询集
# ============================================================================
#: 离线题：**必须能被 2003–2005 PLoS Biology 样例语料覆盖**，否则测的就不是链路而是拒答。
#: keyword 决定从样例块里挑哪些当证据（模拟检索命中）。
OFFLINE_QUERIES: List[Dict[str, Any]] = [
    {"id": "off-1", "category": "证据充分·基因关联",
     "query": "What candidate genes have been associated with type 2 diabetes, "
              "and how strong is the evidence?",
     "keyword": "diabetes", "expect": "应给出带 [S#] 的结论，并说明关联强度的不确定性"},
    {"id": "off-2", "category": "证据充分·机制",
     "query": "How do bacteria control chemotaxis, and what makes the network robust?",
     "keyword": "chemotaxis", "expect": "应基于证据描述机制，不得引入语料外的通路名"},
    {"id": "off-3", "category": "证据充分·方法",
     "query": "Can gene expression data be used to predict patient survival? "
              "What methods were evaluated?",
     "keyword": "survival", "expect": "应点明方法名与其局限"},
    {"id": "off-4", "category": "证据不足·必须拒答",
     "query": "What is the recommended starting dose of pembrolizumab for "
              "non-small cell lung cancer in 2024 guidelines?",
     "keyword": "protein", "expect": "语料里没有该药与该指南，应明确说证据不足、不得编造剂量"},
]

#: 在线题：阶段一压测裸 qwen3:8b 时留下的四类错，是阶段七的验收用例。
#: why 一栏写的是"裸模型当时错在哪"，评估时对着看。
LIVE_QUERIES: List[Dict[str, Any]] = [
    {"id": "live-1", "category": "① 编造参考文献",
     "query": "What are the key published studies on CRISPR-Cas9 off-target effects, "
              "and what did they find?",
     "why": "裸模型当时给出的标题/期刊/卷页全是编的",
     "expect": "参考文献必须来自检索证据，[S#] 与文末列表一一对应，不得出现编造编号"},
    {"id": "live-2", "category": "② 罕见病药名张冠李戴",
     "query": "Which enzyme replacement therapies are used for Fabry disease, "
              "and what evidence supports them?",
     "why": "裸模型把法布里病的品牌名安到了别的药上",
     "expect": "药名须能在证据里逐一核到；核不到就说证据不足"},
    {"id": "live-3", "category": "③ 药物分类错误",
     "query": "How are disease-modifying therapies for multiple sclerosis classified, "
              "and what are examples of each class?",
     "why": "裸模型把单抗 / S1P 调节剂 / 干扰素三类混为一谈",
     "expect": "分类须有证据支持，证据不足的类别应明说而不是硬分"},
    {"id": "live-4", "category": "④ 最新药物时间线自相矛盾",
     "query": "What is the current evidence for anti-amyloid monoclonal antibodies "
              "in Alzheimer disease, and when were the key trials reported?",
     "why": "裸模型给出的年份前后自相矛盾",
     "expect": "年份须来自证据元数据；证据年份跨度要在'证据强度与一致性'里说明"},
]


# ============================================================================
# 二、显示与统计
# ============================================================================
def print_result(case: Dict[str, Any], res: Dict[str, Any], show_answer: bool = True):
    gm = res["generation_metrics"]
    cm = res["context_metadata"]
    cc = res.get("citation_check") or {}
    bar = "=" * 96

    print(f"\n{bar}\n[{case['id']}] {case['category']}\n问题：{case['query']}")
    if case.get("why"):
        print(f"裸模型当时的错：{case['why']}")
    print(f"期望：{case['expect']}\n{'-' * 96}")

    if show_answer:
        print(res["answer"])
        print("-" * 96)

    print(f"耗时 {gm['total_time_seconds']}s | 模型调用 {gm['llm_calls']} 次 | "
          f"答案 {gm['answer_chars']} 字 | 上下文 {gm['context_tokens']} tok | "
          f"in/out {gm['total_prompt_tokens']}/{gm['total_output_tokens']} tok")
    print("分阶段耗时：" + "  ".join(f"{k}={v}s" for k, v in gm["stage_times"].items()))
    ok_str = "  ".join(f"{k}={'✓' if v else ('—' if v is None else '✗')}"
                       for k, v in gm["stage_success"].items())
    print("阶段成功：  " + ok_str)
    print(f"证据：检索 {cm['total_chunks_retrieved']} → 去重后 {cm['unique_chunks_after_dedup']} "
          f"→ 入选 {cm['chunks_selected']}（{cm['chunk_sources']['unique_sources']} 篇文献）")
    print(f"引用：用了 {cc.get('used', [])} / 可用 {cc.get('available', [])}"
          + (f" | ⚠ 编造 {cc['fabricated']}" if cc.get("fabricated") else ""))

    if res["sources"]:
        print("出处清单：")
        for s in res["sources"]:
            used = "·" if s["marker"] in (cc.get("used") or []) else " "
            print(f"  {used}[{s['marker']}] {s['pmcid']}  {s['journal']} ({s['pub_year']}) "
                  f"{s['section']}\n      {s['title'][:78]}")


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总判定。每一条都由记录里的数据算出——不写死任何结论。"""
    n = len(records)
    if not n:
        return {"cases": 0}
    ok_chain = sum(1 for r in records
                   if all(v for v in r["generation_metrics"]["stage_success"].values()
                          if v is not None))
    json_stages = [(r["intermediate_results"].get("evidence_evaluation"),
                    r["intermediate_results"].get("review_feedback")) for r in records]
    json_total = sum(1 for a, b in json_stages for x in (a, b) if x is not None)
    json_ok = sum(1 for a, b in json_stages for x in (a, b) if x is not None and x.get("json_ok"))
    fab = [r["query"] for r in records if (r.get("citation_check") or {}).get("fabricated")]
    no_cite = [r["query"] for r in records
               if not (r.get("citation_check") or {}).get("has_citation")]
    times = [r["generation_metrics"]["total_time_seconds"] for r in records]
    chars = [r["generation_metrics"]["answer_chars"] for r in records]
    calls = [r["generation_metrics"]["llm_calls"] for r in records]
    return {
        "cases": n,
        "all_stages_ok": ok_chain,
        "json_stage_total": json_total,
        "json_stage_ok": json_ok,
        "fabricated_citation_cases": len(fab),
        "fabricated_queries": fab,
        "no_citation_cases": len(no_cite),
        "no_citation_queries": no_cite,
        "time_seconds": {"min": round(min(times), 2), "max": round(max(times), 2),
                         "mean": round(sum(times) / n, 2)},
        "answer_chars": {"min": min(chars), "max": max(chars),
                         "mean": round(sum(chars) / n, 1)},
        "llm_calls": {"min": min(calls), "max": max(calls), "total": sum(calls)},
    }


def print_summary(s: Dict[str, Any]) -> int:
    """打印汇总并返回未通过项数。每个 ✓/✗ 都由 s 里的数值比对得出。"""
    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    checks = [
        (f"全部用例四段链无失败阶段（{s['all_stages_ok']}/{s['cases']}）",
         s["all_stages_ok"] == s["cases"]),
        (f"结构化阶段 JSON 全部解析成功（{s['json_stage_ok']}/{s['json_stage_total']}）",
         s["json_stage_total"] == 0 or s["json_stage_ok"] == s["json_stage_total"]),
        (f"无编造出处编号（{s['fabricated_citation_cases']} 例）",
         s["fabricated_citation_cases"] == 0),
        (f"每份答案都带真实引用（缺引用 {s['no_citation_cases']} 例）",
         s["no_citation_cases"] == 0),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    t, c, k = s["time_seconds"], s["answer_chars"], s["llm_calls"]
    print(f"\n  耗时     min {t['min']}s / 均 {t['mean']}s / max {t['max']}s")
    print(f"  答案长度 min {c['min']} / 均 {c['mean']} / max {c['max']} 字")
    print(f"  模型调用 共 {k['total']} 次（单题 {k['min']}~{k['max']} 次）")
    failed = sum(1 for _, ok in checks if not ok)
    print(f"\n  {len(checks) - failed}/{len(checks)} 项通过")
    return failed


# ============================================================================
# 三、日志
# ============================================================================
def _dump_stage(stage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """结构化阶段的日志形态：解析成功只留结果，失败则连原始输出与错因一起留下。"""
    if stage is None:
        return None
    out: Dict[str, Any] = {"parsed": stage.get("parsed"), "json_ok": stage.get("json_ok")}
    if not stage.get("json_ok"):
        out["json_error"] = stage.get("json_error", "")
        out["raw_text"] = stage.get("raw_text", "")
    if stage.get("filter"):
        out["filter"] = {k: v for k, v in stage["filter"].items() if k != "context_text"}
    return out


def log_records(records: List[Dict[str, Any]], cases: List[Dict[str, Any]],
                mode: str, summary: Dict[str, Any]) -> Dict[str, str]:
    """写 jsonl（机器读，含全部指标与来源）+ txt（人读，含答案全文）。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    jsonl_path = os.path.join(REPORT_DIR, f"生成_流水线测试_{mode}.jsonl")
    txt_path = os.path.join(REPORT_DIR, f"生成_流水线测试报告_{mode}.txt")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for case, r in zip(cases, records):
            gm = r["generation_metrics"]
            f.write(json.dumps({
                "ts": r["timestamp"], "mode": mode,
                "case_id": case["id"], "category": case["category"],
                "query": r["query"], "expect": case.get("expect", ""),
                "answer": r["answer"],
                "metrics": {                       # 任务书要求的关键指标：耗时/答案长度/阶段
                    "total_time_seconds": gm["total_time_seconds"],
                    "answer_chars": gm["answer_chars"],
                    "llm_calls": gm["llm_calls"],
                    "stage_times": gm["stage_times"],
                    "stage_success": gm["stage_success"],
                    "token_counts": gm["token_counts"],
                    "context_tokens": gm["context_tokens"],
                    "chain": gm["chain"],
                },
                "citation_check": r.get("citation_check"),
                "sources": r["sources"],
                "context_metadata": {k: v for k, v in r["context_metadata"].items()
                                     if k != "citations"},
                # 解析失败时**必须把原始输出留下**——只记 parsed 的话，日志里看到的是
                # 一个空列表，完全看不出模型到底吐了什么、卡在哪个畸形上（这一条是被
                # `"relevance":3"` 那次坑出来的：查因时只能重新跑一遍模型才拿得到原文）。
                "intermediate": {
                    "evidence_evaluation": _dump_stage(
                        r["intermediate_results"]["evidence_evaluation"]),
                    "draft_answer": r["intermediate_results"]["draft_answer"],
                    "review_feedback": _dump_stage(
                        r["intermediate_results"]["review_feedback"]),
                },
            }, ensure_ascii=False) + "\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"生成流水线测试报告 · mode={mode}\n生成时间：{stamp}\n")
        f.write(f"用例 {len(records)} 个\n")
        for case, r in zip(cases, records):
            gm = r["generation_metrics"]
            cc = r.get("citation_check") or {}
            f.write("\n" + "=" * 96 + "\n")
            f.write(f"[{case['id']}] {case['category']}\n问题：{r['query']}\n")
            if case.get("why"):
                f.write(f"裸模型当时的错：{case['why']}\n")
            f.write(f"期望：{case.get('expect', '')}\n" + "-" * 96 + "\n")
            f.write(r["answer"] + "\n" + "-" * 96 + "\n")
            f.write(f"耗时 {gm['total_time_seconds']}s | 调用 {gm['llm_calls']} 次 | "
                    f"答案 {gm['answer_chars']} 字 | 上下文 {gm['context_tokens']} tok\n")
            f.write("分阶段耗时：" + json.dumps(gm["stage_times"], ensure_ascii=False) + "\n")
            f.write("阶段成功：  " + json.dumps(gm["stage_success"], ensure_ascii=False) + "\n")
            f.write(f"引用：用了 {cc.get('used')} / 可用 {cc.get('available')}"
                    f" / 编造 {cc.get('fabricated')}\n")
            f.write("出处清单：\n")
            for s in r["sources"]:
                f.write(f"  [{s['marker']}] {s['pmcid']} | {s['journal']} ({s['pub_year']}) "
                        f"| {s['section']} | {s['title']}\n      {s['url']}\n")
        f.write("\n" + "=" * 96 + "\n汇总\n" + "=" * 96 + "\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return {"jsonl": jsonl_path, "txt": txt_path}


# ============================================================================
# 四、跑法
# ============================================================================
def run_offline(args) -> int:
    pipe = MedicalGenerationPipeline(max_context_tokens=args.budget, model_name=args.model,
                                     verbose=True)
    records, cases = [], []
    for case in OFFLINE_QUERIES:
        docs = _asm_mod.load_sample_chunks(limit=args.limit, keyword=case["keyword"])
        print(f"\n[{case['id']}] 样例证据 {len(docs)} 条（keyword={case['keyword']!r}）")
        res = pipe.generate(case["query"], retrieved_docs=docs,
                            evaluate=not args.no_evaluate, review=not args.no_review)
        records.append(res)
        cases.append(case)
        print_result(case, res)
    s = summarize(records)
    failed = print_summary(s)
    paths = log_records(records, cases, "offline", s)
    print(f"\n日志：{paths['jsonl']}\n      {paths['txt']}")
    return 1 if failed else 0


def run_live(args) -> int:
    print("加载阶段六检索流水线（约 15.8GB RSS，Chroma HNSW 载入）……")
    t0 = time.time()
    mp = _load_by_path("jiansuo_duolu", "检索_多路检索.py")
    retr = mp.RetrievalPipeline(bm25_dir=args.bm25)
    print(f"检索器就绪，用时 {time.time() - t0:.1f}s")

    pipe = MedicalGenerationPipeline(retriever=retr, max_context_tokens=args.budget,
                                     model_name=args.model, verbose=True)
    records, cases = [], []
    for case in LIVE_QUERIES:
        res = pipe.generate(case["query"], top_k=args.top_k,
                            evaluate=not args.no_evaluate, review=not args.no_review)
        records.append(res)
        cases.append(case)
        print_result(case, res)
    s = summarize(records)
    failed = print_summary(s)
    paths = log_records(records, cases, "live", s)
    print(f"\n日志：{paths['jsonl']}\n      {paths['txt']}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# 两趟跑法：检索与生成分开，避免两个大内存进程共存
# ---------------------------------------------------------------------------
# 检索器约 15.8 GB RSS，Ollama 的 llama-server 常驻约 8.1 GB，32 G 机器同时开会换页。
# 但这两件事本来就不必同时发生——`generate(retrieved_docs=...)` 早就支持离线回放。
# 于是拆成：① --dump-retrieval 只加载检索器、把候选序列化到磁盘，全程不碰 Ollama；
#           ② --from-dump 只加载 Ollama、从磁盘读候选生成，全程不碰检索器。
# 副产品：检索结果固化成文件后，重跑生成/换链路配置不必再等 30s 加载，评测迭代快得多。

#: `Candidate` 需要保留的字段。`DocumentChunk.from_candidate` 对 dict 走 `.get`、
#: 对对象走 `getattr`，所以回放时直接喂 dict 即可，不必还原成 Candidate 类。
CAND_FIELDS = ("chunk_id", "text", "metadata", "cos_sim", "vec_rank", "bm25_score",
               "bm25_rank", "fused_score", "fusion_strategy", "rel_score",
               "recency_score", "authority_score", "rerank_score")


def candidate_to_dict(c: Any) -> Dict[str, Any]:
    d = {k: getattr(c, k, None) for k in CAND_FIELDS}
    s = getattr(c, "sources", None)
    d["sources"] = sorted(s) if isinstance(s, (set, list, tuple)) else s
    return d


def run_dump(args) -> int:
    """第一趟：只跑检索，把候选写到磁盘。**不加载 Ollama。**"""
    print("加载阶段六检索流水线（约 15.8GB RSS）……本趟不会连 Ollama")
    t0 = time.time()
    mp = _load_by_path("jiansuo_duolu", "检索_多路检索.py")
    retr = mp.RetrievalPipeline(bm25_dir=args.bm25)
    load_s = time.time() - t0
    print(f"检索器就绪，用时 {load_s:.1f}s")

    out_path = args.dump_retrieval
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bm25_dir": args.bm25, "top_k": args.top_k,
        "retriever_load_seconds": round(load_s, 1), "queries": [],
    }
    for case in LIVE_QUERIES:
        t1 = time.time()
        out = retr.search(case["query"], top_k=args.top_k)
        el = time.time() - t1
        cands = [candidate_to_dict(c) for c in out["results"]]
        payload["queries"].append({
            "case": {k: case[k] for k in ("id", "category", "query", "why", "expect")},
            "seconds": round(el, 3), "n_results": len(cands),
            "fusion_strategy": out.get("fusion_strategy"),
            "reranked": bool(out.get("reranked")),
            "candidates": cands,
        })
        srcs = {(c.get("metadata") or {}).get("pmcid") for c in cands}
        yrs = [(c.get("metadata") or {}).get("pub_year") for c in cands]
        yrs = sorted(y for y in yrs if y)
        print(f"  [{case['id']}] {el:.2f}s  命中 {len(cands)} 条 / {len(srcs)} 篇文献"
              f"  年份 {yrs[0] if yrs else '-'}~{yrs[-1] if yrs else '-'}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    size_mb = os.path.getsize(out_path) / 1024 ** 2
    total = sum(q["n_results"] for q in payload["queries"])
    print(f"\n已写入 {out_path}（{size_mb:.1f} MB，{len(payload['queries'])} 个查询 / "
          f"{total} 条候选）")
    print("下一步（新开一个进程，检索器随本进程退出而释放）：")
    print(f"  & $py scripts\\生成_流水线_测试.py --from-dump {out_path}")
    return 0 if total else 1


def run_from_dump(args) -> int:
    """第二趟：从磁盘读候选跑生成。**不加载检索器。**"""
    with open(args.from_dump, encoding="utf-8") as f:
        payload = json.load(f)
    print(f"读入检索快照 {args.from_dump}")
    print(f"  生成于 {payload['created']} | top_k={payload['top_k']} | "
          f"{len(payload['queries'])} 个查询")

    pipe = MedicalGenerationPipeline(max_context_tokens=args.budget, model_name=args.model,
                                     verbose=True)
    records, cases = [], []
    for q in payload["queries"]:
        case = q["case"]
        res = pipe.generate(case["query"], retrieved_docs=q["candidates"],
                            evaluate=not args.no_evaluate, review=not args.no_review)
        # 检索耗时来自快照，补进指标里，这样总耗时与一趟跑法可比
        res["generation_metrics"]["stage_times"]["retrieval"] = q["seconds"]
        res["generation_metrics"]["retrieval_from_snapshot"] = True
        records.append(res)
        cases.append(case)
        print_result(case, res)

    s = summarize(records)
    s["retrieval_snapshot"] = {"created": payload["created"], "top_k": payload["top_k"],
                              "retriever_load_seconds": payload.get("retriever_load_seconds"),
                              "search_seconds": [q["seconds"] for q in payload["queries"]]}
    failed = print_summary(s)
    paths = log_records(records, cases, "live", s)
    print(f"\n日志：{paths['jsonl']}\n      {paths['txt']}")
    return 1 if failed else 0


#: 消融配置：名字 → (evaluate, review, 预期模型调用次数)
ABLATION = [
    ("直接作答（对照基线）", False, False, 1),
    ("评估+作答", True, False, 2),
    ("完整四段链", True, True, 4),
]


def run_ablation(args) -> int:
    """同题三种链路配置对照——回答"四段链值不值 3~4 倍耗时"。

    ⚠ 这里只量化**耗时、答案长度、引用落地、审查发现的问题数**这些可自动统计的量。
    答案是否更忠于证据仍需人工标注判定；本脚本给的是做那件事之前的成本侧事实。
    """
    pipe = MedicalGenerationPipeline(max_context_tokens=args.budget, model_name=args.model,
                                     verbose=True)
    rows: List[Dict[str, Any]] = []
    records, cases = [], []
    for case in OFFLINE_QUERIES:
        docs = _asm_mod.load_sample_chunks(limit=args.limit, keyword=case["keyword"])
        for name, ev, rv, expect_calls in ABLATION:
            print(f"\n[{case['id']}] {name}")
            res = pipe.generate(case["query"], retrieved_docs=docs, evaluate=ev, review=rv)
            gm = res["generation_metrics"]
            cc = res.get("citation_check") or {}
            rf = (res["intermediate_results"].get("review_feedback") or {}).get("parsed") or {}
            rows.append({
                "case": case["id"], "config": name,
                "llm_calls": gm["llm_calls"], "calls_as_expected": gm["llm_calls"] == expect_calls,
                "seconds": gm["total_time_seconds"], "chars": gm["answer_chars"],
                "citations_used": len(cc.get("used") or []),
                "fabricated": len(cc.get("fabricated") or []),
                "review_issues": len(rf.get("issues") or []) if rf else None,
                "review_verdict": rf.get("verdict") if rf else None,
            })
            records.append(res)
            cases.append(dict(case, id=f"{case['id']}·{name}", category=name))

    print("\n" + "=" * 96)
    print("消融对照（同一批证据，同一道题，只改链路配置）")
    print("=" * 96)
    hdr = f"{'用例':<8}{'配置':<20}{'调用':>5}{'耗时s':>8}{'字数':>7}{'引用':>5}{'编造':>5}{'审查问题':>9}"
    print(hdr + "\n" + "-" * 96)
    for r in rows:
        print(f"{r['case']:<8}{r['config']:<20}{r['llm_calls']:>5}{r['seconds']:>8.1f}"
              f"{r['chars']:>7}{r['citations_used']:>5}{r['fabricated']:>5}"
              f"{str(r['review_issues'] if r['review_issues'] is not None else '—'):>9}")

    by_cfg: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    print("-" * 96)
    base = None
    for name, _, _, _ in ABLATION:
        g = by_cfg.get(name, [])
        if not g:
            continue
        mt = sum(x["seconds"] for x in g) / len(g)
        mc = sum(x["chars"] for x in g) / len(g)
        base = base or mt
        print(f"{name:<20} 均耗时 {mt:>6.1f}s（基线的 {mt / base:.2f}×） 均字数 {mc:>6.0f} "
              f"编造出处 {sum(x['fabricated'] for x in g)} 例")

    calls_ok = sum(1 for r in rows if r["calls_as_expected"])
    fab_total = sum(r["fabricated"] for r in rows)
    print("\n" + "=" * 96)
    checks = [(f"各配置模型调用次数与预期一致（{calls_ok}/{len(rows)}）", calls_ok == len(rows)),
              (f"所有配置均无编造出处（{fab_total} 例）", fab_total == 0)]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = sum(1 for _, ok in checks if not ok)

    s = summarize(records)
    s["ablation_rows"] = rows
    paths = log_records(records, cases, "ablation", s)
    print(f"\n日志：{paths['jsonl']}\n      {paths['txt']}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="用样例块测链路（不需向量库）")
    ap.add_argument("--live", action="store_true", help="走真检索测四类验收错题（一趟跑完，吃内存）")
    ap.add_argument("--ablation", action="store_true", help="同题三种链路配置对照")
    ap.add_argument("--dump-retrieval", metavar="PATH", default=None,
                    help="两趟跑法第一趟：只跑检索，把候选写到 PATH（不连 Ollama）")
    ap.add_argument("--from-dump", metavar="PATH", default=None,
                    help="两趟跑法第二趟：从 PATH 读候选跑生成（不加载检索器）")
    ap.add_argument("--bm25", default=os.path.join(ROOT, "data", "bm25_index_4m"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=30, help="离线模式每题取多少样例块当证据")
    ap.add_argument("--budget", type=int, default=2400, help="证据上下文 token 预算")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--no-evaluate", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args()

    if args.dump_retrieval:
        return run_dump(args)
    if args.from_dump:
        return run_from_dump(args)
    if args.offline:
        return run_offline(args)
    if args.ablation:
        return run_ablation(args)
    if args.live:
        return run_live(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
