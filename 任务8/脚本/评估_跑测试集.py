# -*- coding: utf-8 -*-
"""第八阶段 · 用上周的测试 query 再跑一遍：验证评估器 / 缓存 / 批量处理

上周（阶段七）的测试题就是阶段一压测裸模型时留下的**四类错题**，检索结果已经固化在
`report_data\\检索快照_live.json` 里。本脚本拿同一批题，把阶段八的三件新东西全部过一遍：

    ① 批量处理  —— 串行 vs 并行各跑一趟（**每趟前清空缓存**，否则第二趟白捡便宜），
                    回答"单卡跑本地模型，多线程到底有没有用"这个问题——用实测，不用直觉。
    ② 缓存      —— 第三趟不清缓存，看命中率、省下多少秒、以及**命中的答案是否逐字相同**。
    ③ 答案评估  —— 四维指标全跑一遍；顺带用同一把尺子去量阶段七已经存下来的**裸模型答案**
                    （复用 `生成_对比评测.jsonl`，不额外调模型），得到 RAG vs 裸模型的对照。

⚠ 口径（报告里必须一起写）：本次没有人工标准答案，`reference_kind="evidence"`，
   ①②两维量的是**答案对检索证据的覆盖/贴合**，是"忠于证据"的代理指标，**不是正确性**。
   人工标注评测集仍是阶段七、八共同的遗留项。

用法（需 Ollama 在跑；不需要 65GB 向量库，检索结果读快照）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\评估_跑测试集.py
    ... --workers 4          # 并行趟用几个线程（默认按 CPU 与工作类型推荐）
    ... --skip-serial        # 跳过串行趟（省约 3 分钟，但就没有加速比可比了）
    ... --refs <gold.jsonl>  # 有人工标准答案时传进来，评估口径自动升级为 gold
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
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "report_data")
SNAPSHOT = os.path.join(REPORT_DIR, "检索快照_live.json")
COMPARE_JSONL = os.path.join(REPORT_DIR, "生成_对比评测.jsonl")
OUT_JSONL = os.path.join(REPORT_DIR, "评估_测试集.jsonl")
OUT_TXT = os.path.join(REPORT_DIR, "评估_测试集报告.txt")
OUT_JSON = os.path.join(REPORT_DIR, "评估_测试集汇总.json")     # 机器读，供转 Word 用
CACHE_PATH = os.path.join(REPORT_DIR, "评估_生成缓存.json")


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pipe_mod = _load("shengcheng_liushuixian", "生成_流水线.py")
ev = _load("pinggu_pinggu", "评估_答案评估器.py")
gc_ = _load("shengcheng_huancun", "生成_缓存.py")
bp = _load("shengcheng_piliang", "生成_批量处理.py")

MedicalGenerationPipeline = _pipe_mod.MedicalGenerationPipeline
LLMGenerator = _pipe_mod.LLMGenerator
ContextAssembler = _pipe_mod.ContextAssembler


# ============================================================================
# 一、跑一趟
# ============================================================================
class Runner:
    """按线程持有各自的流水线：`ContextAssembler` 不是为并发设计的，一个线程一个最省心。

    **生成器和缓存是共享的**——生成器只是 HTTP 客户端（无状态），缓存内部有锁。
    共享它们才有意义：缓存要跨线程命中，才叫缓存。
    """

    def __init__(self, generator: Any, budget: int, verbose: bool = False):
        self.generator = generator
        self.budget = budget
        self.verbose = verbose
        self._local = threading.local()

    def pipeline(self) -> Any:
        p = getattr(self._local, "pipe", None)
        if p is None:
            p = MedicalGenerationPipeline(
                assembler=ContextAssembler(max_context_tokens=self.budget, verbose=False),
                generator=self.generator, verbose=self.verbose)
            self._local.pipe = p
        return p

    def generate(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipeline().generate(item["case"]["query"],
                                        retrieved_docs=item["candidates"])


def run_pass(name: str, items: List[Dict[str, Any]], runner: Runner, workers: int,
             pipeline_cache: Optional[Any] = None,
             cache_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """跑一趟四道题。pipeline_cache 非 None 时先查缓存（键 = 查询 + 检索证据）。"""
    print(f"\n{'=' * 96}\n{name}（并发 {workers}）\n{'=' * 96}")
    hits = {"n": 0}
    lock = threading.Lock()

    def work(item: Dict[str, Any]) -> Dict[str, Any]:
        q = item["case"]["query"]
        if pipeline_cache is not None:
            key = gc_.make_pipeline_key(pipeline_cache, q, item["candidates"],
                                        **(cache_params or {}))
            cached = pipeline_cache.get(key)
            if cached is not None:
                with lock:
                    hits["n"] += 1
                out = dict(cached)
                out["from_cache"] = True
                return out
            res = runner.generate(item)
            res["from_cache"] = False
            pipeline_cache.set(key, res, temperature=(cache_params or {}).get("temperature"),
                               meta={"elapsed": res["generation_metrics"]["total_time_seconds"]})
            return res
        res = runner.generate(item)
        res["from_cache"] = False
        return res

    def progress(done: int, total: int, r: Dict[str, Any]):
        v = r.get("value") or {}
        gm = (v.get("generation_metrics") or {})
        tag = "缓存命中" if v.get("from_cache") else f"{gm.get('llm_calls', '?')} 次调用"
        print(f"  {done}/{total} {'ok' if r['ok'] else 'FAIL'}  {r['seconds']:>6.2f}s  {tag}"
              + ("" if r["ok"] else f"  · {r['error'][:90]}"))

    proc = bp.ParallelBatchProcessor(max_workers=workers, kind="llm")
    out = proc.run(items, work, progress=progress)
    st = dict(proc.stats)
    st["cache_hits"] = hits["n"]
    st["name"] = name
    print(f"  → 墙钟 {st['wall_seconds']}s | 成功 {st['ok']}/{st['items']} | "
          f"缓存命中 {hits['n']}")
    return {"stats": st, "results": out}


# ============================================================================
# 二、评估
# ============================================================================
def evidence_blob(item: Dict[str, Any]) -> str:
    return "\n".join((c.get("text") or "") for c in item["candidates"])


def evaluate_answers(pairs: List[Tuple[str, str, str, str]], workers: int
                     ) -> List[Dict[str, Any]]:
    """并行评估（纯 CPU 工作，这里的并行是实打实有效的）。

    pairs: [(case_id, 答案, 参照证据, 人工标准答案或"")]
    """
    e = ev.AnswerEvaluator()
    proc = bp.ParallelBatchProcessor(max_workers=workers, kind="cpu")
    out = proc.run(pairs, lambda p: e.evaluate(p[1], reference=(p[3] or None),
                                               evidence=p[2], case_id=p[0]))
    return [r["value"] if r["ok"] else {"case_id": pairs[i][0], "error": r["error"]}
            for i, r in enumerate(out)]


def load_bare_answers() -> Dict[str, str]:
    """阶段七已经跑好的裸模型答案，直接复用，不再花模型时间。"""
    out: Dict[str, str] = {}
    if not os.path.exists(COMPARE_JSONL):
        return out
    with open(COMPARE_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = (r.get("case") or {}).get("id")
            ans = (r.get("bare") or {}).get("answer")
            if cid and ans:
                out[cid] = ans
    return out


def load_refs(path: Optional[str]) -> Dict[str, str]:
    """人工标准答案（jsonl，每行 {"case_id": …, "reference": …}）。没有就返回空。"""
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("case_id") and r.get("reference"):
                out[r["case_id"]] = r["reference"]
    return out


# ============================================================================
# 三、主流程
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=SNAPSHOT)
    ap.add_argument("--workers", type=int, default=None, help="并行趟的线程数")
    ap.add_argument("--skip-serial", action="store_true", help="跳过串行趟（就没有加速比了）")
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--num-ctx", type=int, default=12288)
    ap.add_argument("--ttl", type=float, default=7 * 24 * 3600, help="缓存 TTL（秒）")
    ap.add_argument("--cache-temperature", type=float, default=0.3,
                    help="流水线级缓存的温度门限。四段链最高段是 0.3，设 0.0 则整条链都不缓存")
    ap.add_argument("--refs", default=None, help="人工标准答案 jsonl（有则评估口径升级为 gold）")
    args = ap.parse_args()

    if not os.path.exists(args.snapshot):
        print(f"找不到检索快照 {args.snapshot}\n"
              f"请先跑：生成_流水线_测试.py --dump-retrieval {args.snapshot} --bm25 <BM25索引目录>")
        return 2

    snap = json.load(open(args.snapshot, encoding="utf-8"))
    items = snap["queries"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    workers = args.workers or bp.recommended_workers("llm", len(items))
    eval_workers = bp.recommended_workers("cpu", len(items) * 2)

    print("=" * 96)
    print("阶段八 · 用上周的测试 query 再跑一遍（评估器 / 缓存 / 批量处理）")
    print("=" * 96)
    print(f"检索快照：{args.snapshot}（生成于 {snap['created']}，{len(items)} 道题，"
          f"top_k={snap['top_k']}）")
    print(f"CPU 逻辑核 {bp.CPU_COUNT}｜生成并发 {workers}（kind=llm）｜评估并发 {eval_workers}（kind=cpu）")
    print(f"ROUGE 后端：{ev.ROUGE_BACKEND}")
    for it in items:
        print(f"  · [{it['case']['id']}] {it['case']['category']}｜{it['case']['query'][:66]}…")

    # ---- 缓存：阶段级（温度 0 的 ①③）+ 流水线级（整条结果）----
    stage_cache = gc_.GenerationCache(namespace="stage", ttl_seconds=args.ttl,
                                      max_temperature=0.0, max_entries=200)
    pipe_cache = gc_.GenerationCache(namespace="pipeline", ttl_seconds=args.ttl,
                                     max_temperature=args.cache_temperature,
                                     max_entries=200, path=CACHE_PATH)
    cache_params = {"model": args.model, "num_ctx": args.num_ctx, "budget": args.budget,
                    "temperature": 0.3, "chain": "evaluate+review"}
    print(f"\n缓存配置：TTL {args.ttl / 3600:.1f} h｜阶段级门限 0.0（只缓存 ①③）｜"
          f"流水线级门限 {args.cache_temperature}")
    if args.cache_temperature > 0:
        print("  ⚠ 流水线级门限 > 0 意味着**把一次抽样的结果固定下来**：②答案生成段温度 0.3，"
              "\n    本来每次都会不同。评测复跑要稳定时这是想要的，面向用户要多样性时应设 0.0。")

    gen = LLMGenerator(model_name=args.model, num_ctx=args.num_ctx, verbose=False)
    cached_gen = gc_.CachedLLMGenerator(gen, stage_cache)
    runner = Runner(cached_gen, budget=args.budget, verbose=False)
    print(f"Ollama 连接：{gen.connection['ok']}｜模型 {gen.model_name}")

    passes: List[Dict[str, Any]] = []

    # ---- 趟 1：串行（冷）----
    if not args.skip_serial:
        stage_cache.clear()
        p1 = run_pass("趟 1 · 串行（缓存已清空，冷启动）", items, runner, workers=1)
        passes.append(p1)
    else:
        p1 = None

    # ---- 趟 2：并行（冷）----
    stage_cache.clear()                     # 关键：不清的话这一趟白捡上一趟的缓存，加速比就假了
    p2 = run_pass(f"趟 2 · 并行 {workers} 线程（缓存已清空，冷启动）", items, runner,
                  workers=workers)
    passes.append(p2)

    # ---- 趟 3：并行（热，走流水线级缓存）----
    #  先把趟 2 的结果灌进流水线缓存，再跑一遍看命中
    for it, r in zip(items, p2["results"]):
        if r["ok"]:
            key = gc_.make_pipeline_key(pipe_cache, it["case"]["query"], it["candidates"],
                                        **cache_params)
            pipe_cache.set(key, r["value"], temperature=cache_params["temperature"],
                           meta={"elapsed": r["value"]["generation_metrics"]["total_time_seconds"]})
    p3 = run_pass("趟 3 · 并行（缓存已预热，走流水线级缓存）", items, runner, workers=workers,
                  pipeline_cache=pipe_cache, cache_params=cache_params)
    passes.append(p3)
    pipe_cache.save()

    # ---- 评估 ----
    print(f"\n{'=' * 96}\n答案评估（四维指标）\n{'=' * 96}")
    refs = load_refs(args.refs)
    if refs:
        print(f"载入人工标准答案 {len(refs)} 条 → 评估口径 = gold")
    else:
        print("未提供人工标准答案 → 参照 = 该题检索到的证据原文（reference_kind=evidence）")
        print("  口径：①②量的是「答案对证据的覆盖」，不是正确性。")

    rag_answers = [(it["case"]["id"], r["value"]["answer"], evidence_blob(it),
                    refs.get(it["case"]["id"], ""))
                   for it, r in zip(items, p2["results"]) if r["ok"]]
    t0 = time.time()
    rag_evals = evaluate_answers(rag_answers, eval_workers)
    eval_seconds = time.time() - t0

    bare = load_bare_answers()
    bare_answers = [(it["case"]["id"], bare[it["case"]["id"]], evidence_blob(it),
                     refs.get(it["case"]["id"], ""))
                    for it in items if it["case"]["id"] in bare]
    bare_evals = evaluate_answers(bare_answers, eval_workers) if bare_answers else []

    for (cid, ans, _, _), r in zip(rag_answers, rag_evals):
        print(f"\n[{cid}]  {len(ans)} 字")
        print(ev.format_evaluation(r))
    print(f"\n评估 {len(rag_evals) + len(bare_evals)} 份答案用时 {eval_seconds:.2f}s"
          f"（并发 {eval_workers}）")

    rag_agg = ev.AnswerEvaluator.aggregate(rag_evals) if rag_evals else {}
    bare_agg = ev.AnswerEvaluator.aggregate(bare_evals) if bare_evals else {}

    # ---- 汇总判定：每一条都由上面算出的数据比对得出 ----
    report = build_report(stamp, snap, items, passes, p1, p2, p3, rag_evals, bare_evals,
                          rag_agg, bare_agg, stage_cache, pipe_cache, cached_gen,
                          workers, eval_workers, eval_seconds, args, refs)
    print(report["text"])

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for it, r, e in zip(items, p2["results"], rag_evals):
            f.write(json.dumps({
                "ts": stamp, "case_id": it["case"]["id"], "category": it["case"]["category"],
                "query": it["case"]["query"],
                "answer": r["value"]["answer"] if r["ok"] else "",
                "metrics": (r["value"]["generation_metrics"] if r["ok"] else {}),
                "evaluation": e,
                "sources": (r["value"]["sources"] if r["ok"] else []),
            }, ensure_ascii=False) + "\n")
        for (cid, ans, _, _), e in zip(bare_answers, bare_evals):
            f.write(json.dumps({"ts": stamp, "case_id": cid, "side": "bare",
                                "answer": ans, "evaluation": e}, ensure_ascii=False) + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report["summary"], f, ensure_ascii=False, indent=2)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report["text"] + "\n")
        f.write("\n" + "=" * 96 + "\n答案全文（趟 2 的结果）\n" + "=" * 96 + "\n")
        for (cid, ans, _, _), e in zip(rag_answers, rag_evals):
            f.write(f"\n{'-' * 96}\n[{cid}]\n{'-' * 96}\n{ans}\n")
            f.write(ev.format_evaluation(e) + "\n")
    print(f"\n日志：{OUT_JSONL}\n      {OUT_TXT}\n      {OUT_JSON}\n缓存：{CACHE_PATH}")
    return 0 if report["failed"] == 0 else 1


def build_report(stamp, snap, items, passes, p1, p2, p3, rag_evals, bare_evals,
                 rag_agg, bare_agg, stage_cache, pipe_cache, cached_gen,
                 workers, eval_workers, eval_seconds, args, refs) -> Dict[str, Any]:
    """所有结论都从上面的实测数据算出。没有一条是写死的。"""
    L: List[str] = []

    def w(s: str = ""):
        L.append(s)

    w("\n" + "=" * 96)
    w("汇总")
    w("=" * 96)
    w(f"时间：{stamp}｜检索快照生成于 {snap['created']}｜{len(items)} 道题")

    # —— ① 批量处理 ——
    w("\n【一】批量处理：串行 vs 并行（两趟都是冷缓存，起跑线相同）")
    w(f"  {'趟':<34}{'并发':>5}{'墙钟 s':>10}{'成功':>6}{'缓存命中':>9}")
    for p in passes:
        s = p["stats"]
        w(f"  {s['name']:<34}{s['workers']:>5}{s['wall_seconds']:>10.2f}"
          f"{s['ok']:>4}/{s['items']:<2}{s['cache_hits']:>9}")
    speedup = None
    if p1 is not None and p2["stats"]["wall_seconds"] > 0:
        speedup = p1["stats"]["wall_seconds"] / p2["stats"]["wall_seconds"]
        w(f"  并行相对串行：{speedup:.2f}×"
          f"（{p1['stats']['wall_seconds']:.1f}s → {p2['stats']['wall_seconds']:.1f}s）")
        w("  ⚠ 本机是单卡 + 单个 Ollama 进程，多个请求争同一份权重与同一块 GPU；"
          "\n    这个倍数量的是**本机实际**能拿到多少并行收益，不是线程池的理论上限。")
    else:
        w("  （跳过了串行趟，无加速比可比）")

    # —— ② 缓存 ——
    w("\n【二】缓存：冷 vs 热（同一批查询、同一批证据）")
    cold, hot = p2["stats"], p3["stats"]
    w(f"  冷（趟 2）墙钟 {cold['wall_seconds']:.2f}s，缓存命中 {cold['cache_hits']}")
    w(f"  热（趟 3）墙钟 {hot['wall_seconds']:.2f}s，缓存命中 {hot['cache_hits']}/{hot['items']}")
    ratio = (cold["wall_seconds"] / hot["wall_seconds"]) if hot["wall_seconds"] > 0 else None
    # 全命中时热趟只剩哈希与字典查找，比值动辄五位数——那个数字没有意义，
    # 有意义的是「替下来多少模型时间」，所以这里报省下的秒数，比值只给量级。
    saved = pipe_cache.stats["seconds_saved"]
    w(f"  缓存替下来的模型时间：{saved:.1f}s（四题合计，即冷跑时这些答案真花掉的时间）")
    if ratio:
        w(f"  热趟墙钟不到冷趟的 1/{min(int(ratio), 10 ** 6):,}"
          f"（命中后只剩哈希与字典查找，不再调用模型）")
    identical = [r3["value"]["answer"] == r2["value"]["answer"]
                 for r2, r3 in zip(p2["results"], p3["results"]) if r2["ok"] and r3["ok"]]
    w(f"  命中返回的答案与首次逐字相同：{sum(identical)}/{len(identical)}")
    w(f"  阶段级缓存（只缓存温度 0 的 ①③）：{json.dumps(cached_gen.info(), ensure_ascii=False)}")
    w(f"  流水线级缓存：{json.dumps(pipe_cache.info(), ensure_ascii=False)}")

    # —— ③ 评估 ——
    kind = rag_evals[0].get("reference_kind") if rag_evals else "none"
    w(f"\n【三】答案评估（{len(rag_evals)} 份 RAG 答案，参照类型 = {kind}）")
    w(f"  {'用例':<10}{'ROUGE-L f':>11}{'关键信息召回':>14}{'幻觉风险':>10}"
      f"{'可读性':>8}{'均句长':>8}{'中英混排':>9}")
    for e in rag_evals:
        rl = e["rouge"]["rouge-l"]["f"] if e.get("rouge") else float("nan")
        ki = e["key_info"]["recall"] if e.get("key_info") else None
        w(f"  {e['case_id']:<10}{rl:>11.4f}{(ki if ki is not None else 0):>14.4f}"
          f"{e['hallucination']['risk_score']:>10.4f}"
          f"{e['readability']['readability_score']:>8.4f}"
          f"{e['readability']['avg_sentence_length']:>8.1f}"
          f"{e['readability']['mixed_language_ratio']:>9.2f}")
    w(f"  均值：{json.dumps(rag_agg, ensure_ascii=False)}")

    # —— ④ RAG vs 裸模型 ——
    if bare_evals:
        w(f"\n【四】同一把尺子量裸模型（复用阶段七已存的答案，未额外调模型）")
        w(f"  {'':<8}{'ROUGE-L f':>11}{'关键信息召回':>14}{'幻觉风险':>10}"
          f"{'无出处信号':>11}{'可读性':>8}")
        for label, agg in (("RAG", rag_agg), ("裸模型", bare_agg)):
            w(f"  {label:<8}{agg['rouge_l_f']:>11.4f}{agg['key_info_recall']:>14.4f}"
              f"{agg['hallucination_risk']:>10.4f}"
              f"{agg['hallucination_signals_unmitigated']:>11}"
              f"{agg['readability_score']:>8.4f}")
        w("  ⚠ 这里的 ROUGE / 召回是对**同一批检索证据**算的：RAG 看得见这些证据、裸模型看不见，")
        w("    所以 RAG 更高是意料之中——它量的是「有没有贴着证据写」，不是「谁答得更对」。")
        w("    真正需要人工标注才能回答的是后者。")

    # —— 判定 ——
    checks: List[Tuple[str, bool]] = []
    checks.append((f"并行趟四道题全部成功（{p2['stats']['ok']}/{p2['stats']['items']}）",
                   p2["stats"]["ok"] == p2["stats"]["items"]))
    checks.append((f"批量输出顺序与输入一致（逐条核对 index）",
                   all(r["index"] == i for i, r in enumerate(p2["results"]))))
    checks.append((f"热缓存趟全部命中（{hot['cache_hits']}/{hot['items']}）",
                   hot["cache_hits"] == hot["items"]))
    checks.append(("缓存命中返回的答案与首次逐字相同",
                   bool(identical) and all(identical)))
    checks.append((f"缓存命中后墙钟显著下降（{cold['wall_seconds']:.1f}s → "
                   f"{hot['wall_seconds']:.2f}s）",
                   hot["wall_seconds"] < cold["wall_seconds"] / 10))
    checks.append((f"阶段级缓存只收下了温度 0 的调用"
                   f"（写入 {stage_cache.stats['sets']}，因高温跳过 "
                   f"{cached_gen.cache_stats['skipped']}）",
                   stage_cache.stats["sets"] > 0 and cached_gen.cache_stats["skipped"] > 0))
    checks.append((f"每份答案都拿到四个维度的分数（{len(rag_evals)} 份）",
                   bool(rag_evals) and all(
                       e.get("rouge") and e.get("key_info")
                       and "risk_score" in e["hallucination"]
                       and "readability_score" in e["readability"] for e in rag_evals)))
    if bare_evals:
        checks.append((f"RAG 对证据的贴合度高于裸模型"
                       f"（ROUGE-L {rag_agg['rouge_l_f']:.4f} vs "
                       f"{bare_agg['rouge_l_f']:.4f}）",
                       rag_agg["rouge_l_f"] > bare_agg["rouge_l_f"]))
        checks.append((f"RAG 的关键信息召回高于裸模型"
                       f"（{rag_agg['key_info_recall']:.4f} vs "
                       f"{bare_agg['key_info_recall']:.4f}）",
                       rag_agg["key_info_recall"] > bare_agg["key_info_recall"]))

    w("\n" + "=" * 96)
    for label, ok in checks:
        w(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = sum(1 for _, ok in checks if not ok)
    w(f"\n  {len(checks) - failed}/{len(checks)} 项通过")
    w("\n  ⚠ 口径提醒：本轮没有人工标准答案，ROUGE 与关键信息召回的参照是**检索到的证据原文**，")
    w("    量的是「答案有没有贴着证据写」；幻觉信号是**措辞层面**的启发式；")
    w("    可读性分是约定的经验带。三者都不能单独当作「答得对不对」的判据。")

    # 机器读的汇总：转 Word 与后续统计都读它，不去正则解析上面这段人读文本
    summary = {
        "timestamp": stamp,
        "snapshot_created": snap["created"],
        "cases": len(items),
        "case_ids": [it["case"]["id"] for it in items],
        "environment": {"cpu_count": bp.CPU_COUNT, "gen_workers": workers,
                        "eval_workers": eval_workers, "rouge_backend": ev.ROUGE_BACKEND,
                        "model": args.model, "num_ctx": args.num_ctx,
                        "budget": args.budget},
        "passes": [p["stats"] for p in passes],
        "serial_vs_parallel_speedup": (round(speedup, 2) if speedup else None),
        "cache": {
            "ttl_seconds": args.ttl,
            "stage_max_temperature": 0.0,
            "pipeline_max_temperature": args.cache_temperature,
            "cold_wall_seconds": cold["wall_seconds"],
            "hot_wall_seconds": hot["wall_seconds"],
            "hot_hits": hot["cache_hits"],
            "model_seconds_saved": round(saved, 1),
            "speedup": (round(ratio, 1) if ratio else None),
            "answers_identical": f"{sum(identical)}/{len(identical)}",
            "stage_cache": cached_gen.info(),
            "pipeline_cache": pipe_cache.info(),
        },
        "evaluation": {
            "reference_kind": kind,
            "has_gold_refs": bool(refs),
            "eval_seconds": round(eval_seconds, 2),
            "per_case": [{
                "case_id": e["case_id"],
                "rouge_l_f": (e["rouge"]["rouge-l"]["f"] if e.get("rouge") else None),
                "rouge_1_f": (e["rouge"]["rouge-1"]["f"] if e.get("rouge") else None),
                "rouge_2_f": (e["rouge"]["rouge-2"]["f"] if e.get("rouge") else None),
                "key_info_recall": (e["key_info"]["recall"] if e.get("key_info") else None),
                "key_info_per_category": (
                    {c: {"overlap": v["overlap"], "gt": v["gt_matches"],
                         "recall": v["recall"]}
                     for c, v in e["key_info"]["per_category"].items()}
                    if e.get("key_info") else {}),
                "hallucination_risk": e["hallucination"]["risk_score"],
                "hallucination_level": e["hallucination"]["risk_level"],
                "signals_total": e["hallucination"]["signals_total"],
                "signals_unmitigated": e["hallucination"]["signals_unmitigated"],
                "by_signal": e["hallucination"]["by_signal"],
                "readability_score": e["readability"]["readability_score"],
                "avg_sentence_length": e["readability"]["avg_sentence_length"],
                "sentences": e["readability"]["sentences"],
                "mixed_language_ratio": e["readability"]["mixed_language_ratio"],
                "chars": e["readability"]["chars"],
            } for e in rag_evals],
            "rag_aggregate": rag_agg,
            "bare_aggregate": bare_agg,
            "bare_cases": len(bare_evals),
        },
        "checks": [{"label": lb, "passed": ok} for lb, ok in checks],
        "failed": failed,
    }
    return {"text": "\n".join(L), "failed": failed, "summary": summary}


if __name__ == "__main__":
    sys.exit(main())
