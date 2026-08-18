# -*- coding: utf-8 -*-
"""第九阶段 · 跑对抗测试：量化强约束层到底把幻觉压下去了多少

对每道对抗题跑两条链路，用**同一把尺子**（`约束_格式校验器.py`）判分：

    ① baseline    阶段七提示词 + 层 D 全关   ← 对照组
    ② constrained 阶段九提示词 + 层 D 全开   ← 实验组

第三列不用再跑一遍模型：受约束组每次生成都留了一份**修正前**的完整校验报告
（`constraint_repair.initial.report`），它正是"只有提示词约束、还没做校验修正"的观测值。
于是一次跑两条链路，能读出三个切面，且第 ②③ 列来自**同一次生成**，不含采样噪声：

    baseline → prompt_only → constrained
      ↑加提示词约束↑        ↑加校验与修正↑

输出（全部落在 report_data\\）：
    约束_对抗测试.jsonl      每题每组一行：答案全文 + 校验报告 + 修正过程
    约束_对抗测试报告.txt    人读版（含答案全文、逐题违规、判定）
    约束_对抗测试汇总.json   机器读（转 Word 用）

用法（需 Ollama 在跑；不需要 65GB 向量库，证据读阶段七快照）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\约束_跑对抗测试.py
    ... --workers 4             # 并发（单卡上实测约 1.36×，见阶段八）
    ... --configs constrained   # 只跑实验组（省一半时间，但就没有对照了）
    ... --cases adv-A1,adv-E1   # 只跑指定题
    ... --repair-rounds 2       # 模型修正轮数上限
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
import io
import json
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "report_data")
OUT_JSONL = os.path.join(REPORT_DIR, "约束_对抗测试.jsonl")
OUT_TXT = os.path.join(REPORT_DIR, "约束_对抗测试报告.txt")
OUT_JSON = os.path.join(REPORT_DIR, "约束_对抗测试汇总.json")


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cpipe = _load("yueshu_liushuixian", "约束_受限流水线.py")
_fc = _load("yueshu_jiaoyanqi", "约束_格式校验器.py")
_ac = _load("yueshu_yongli", "约束_对抗测试集.py")
_bp = _load("shengcheng_piliang", "生成_批量处理.py")

ConstrainedGenerationPipeline = _cpipe.ConstrainedGenerationPipeline
ContextAssembler = _cpipe.ContextAssembler
LLMGenerator = _cpipe.LLMGenerator

#: 两组配置的唯一差别就在这三个开关上
CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {"constrained_prompts": False, "deterministic_fix": False,
                 "max_repair_rounds": 0,
                 "desc": "阶段七提示词，不做任何生成后校验修正（对照组）"},
    "constrained": {"constrained_prompts": True, "deterministic_fix": True,
                    "max_repair_rounds": 1,
                    "desc": "阶段九强约束提示词 + 层 D 校验与修正（实验组）"},
}


# ============================================================================
# 一、跑一组
# ============================================================================
class Runner:
    """一个线程一条流水线（`ContextAssembler` 不是为并发设计的），生成器共享（无状态 HTTP）。"""

    def __init__(self, generator: Any, cfg: Dict[str, Any], budget: int, verbose: bool = False):
        self.generator = generator
        self.cfg = cfg
        self.budget = budget
        self.verbose = verbose
        self._local = threading.local()

    def pipeline(self) -> Any:
        p = getattr(self._local, "pipe", None)
        if p is None:
            p = ConstrainedGenerationPipeline(
                assembler=ContextAssembler(max_context_tokens=self.budget, verbose=False),
                generator=self.generator, verbose=self.verbose,
                constrained_prompts=self.cfg["constrained_prompts"],
                deterministic_fix=self.cfg["deterministic_fix"],
                max_repair_rounds=self.cfg["max_repair_rounds"])
            self._local.pipe = p
        return p

    def run_case(self, item: Dict[str, Any]) -> Dict[str, Any]:
        case = item["case"]
        return self.pipeline().generate(case.query, retrieved_docs=item["candidates"],
                                        expect_refusal=case.expect_refusal)


def run_config(name: str, items: List[Dict[str, Any]], generator: Any, budget: int,
               workers: int, repair_rounds: int) -> Dict[str, Any]:
    cfg = dict(CONFIGS[name])
    if name == "constrained":
        cfg["max_repair_rounds"] = repair_rounds
    print(f"\n{'=' * 96}\n{name} —— {cfg['desc']}（并发 {workers}）\n{'=' * 96}")
    runner = Runner(generator, cfg, budget)

    def progress(done: int, total: int, r: Dict[str, Any]):
        v = r.get("value") or {}
        ck = v.get("constraint_check") or {}
        rp = v.get("constraint_repair") or {}
        print(f"  {done}/{total} {'ok' if r['ok'] else 'FAIL'} {r['seconds']:>6.1f}s  "
              f"合规={'✓' if ck.get('compliant') else '✗'} "
              f"违规 {(ck.get('n_violations') or {}).get('high', '?')}高/"
              f"{(ck.get('n_violations') or {}).get('medium', '?')}中 "
              f"修正轮 {rp.get('llm_rounds', 0)}"
              + ("" if r["ok"] else f"  · {r['error'][:80]}"))

    proc = _bp.ParallelBatchProcessor(max_workers=workers, kind="llm")
    out = proc.run(items, runner.run_case, progress=progress)
    st = dict(proc.stats)
    print(f"  → 墙钟 {st['wall_seconds']}s | 成功 {st['ok']}/{st['items']}")
    return {"name": name, "config": cfg, "stats": st, "results": out}


# ============================================================================
# 二、汇总
# ============================================================================
def collect(items: List[Dict[str, Any]], run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把一组跑完的结果整理成逐题记录（含修正前/后两份校验报告）。"""
    rows = []
    for it, r in zip(items, run["results"]):
        case = it["case"]
        v = r.get("value") or {}
        rep = v.get("constraint_check")
        repair = v.get("constraint_repair") or {}
        initial = (repair.get("initial") or {}).get("report") or rep
        rows.append({
            "case_id": case.id, "category": case.category, "query": case.query,
            "expect_refusal": case.expect_refusal,
            "expect_complete": getattr(case, "expect_complete", False),
            "evidence_pool": case.evidence_pool,
            "ok": r["ok"], "error": r.get("error", ""),
            "seconds": r["seconds"],
            "answer": v.get("answer", ""),
            "check": rep, "check_before_repair": initial,
            "repair": {k: val for k, val in repair.items() if k != "initial"},
            "metrics": v.get("generation_metrics", {}),
            "n_sources": len(v.get("sources") or []),
        })
    return rows


def _state(rr: Dict[str, Any]) -> str:
    """三态取值，兼容 2026-08-11 之前存下的报告（那时只有 detected 一个布尔）。"""
    st = rr.get("state")
    if st:
        return st
    return "full_refusal" if rr.get("detected") else "substantive"


def complete_metrics(rows: Sequence[Dict[str, Any]], key: str = "check") -> Dict[str, Any]:
    """反向对照：证据充分的题，有没有被**降级**成部分作答或拒答。

    为什么单独有这一组：把拒答拆成三态之后，多出来一个**新的失败方向**——
    本可完整回答的问题被降一档写成「仅能部分回答」。原来的用例集全在测
    「该拒的有没有拒」，一条也测不到反方向。本项目语料以综述为主，模型看到证据
    不够「原始」时很容易往下降，所以这个方向不是假想的。
    """
    need = [r for r in rows if r.get("expect_complete") and r[key]]
    if not need:
        return {"should_complete": 0, "downgraded": 0, "downgrade_rate": None,
                "downgraded_ids": [], "to_partial": 0, "to_refusal": 0}
    part = [r for r in need if _state(r[key]["refusal"]) == "partial"]
    full = [r for r in need if _state(r[key]["refusal"]) == "full_refusal"]
    bad = part + full
    return {
        "should_complete": len(need), "downgraded": len(bad),
        "downgrade_rate": round(len(bad) / len(need), 4),
        "to_partial": len(part), "to_refusal": len(full),
        "downgraded_ids": [r["case_id"] for r in bad],
    }


def refusal_metrics(rows: Sequence[Dict[str, Any]], key: str = "check") -> Dict[str, Any]:
    """拒答侧的数：该拒答的守住没、不该拒的拒了没。

    分开算而不是合成一个"准确率"：**漏拒和误拒的代价完全不同**——漏拒是幻觉，
    误拒是能力浪费。合成一个数会把这两件事搅在一起。

    **口径按三态**（2026-08-11 改）：
      应拒题上，完全拒答与部分作答**都算守住边界**——半可答的题里部分作答本来就是最优解；
      可答题上，**误拒 = 完全拒答 且 证据总结里一条带出处的实质句都没有**。

    误拒为什么要多带后半个条件：阶段九唯一那条 FAIL（adv-E1）的核心答案只有拒答短语，
    但证据总结里给了三条带出处的检测方法——**能力并没有被浪费掉，浪费的是表述**。
    那属于「首句与正文自相矛盾」（`refusal.conflict` 一类），不该记进"误拒"。
    ⚠ 反过来不成立：不能因为证据总结有带出处的句子就判"部分作答"——强约束本来就要求
    拒答时也要在证据总结里列证据（实测这么判会让 9 道应拒题里 8 道被误标）。
    """
    need = [r for r in rows if r["expect_refusal"] is True and r[key]]
    can = [r for r in rows if r["expect_refusal"] is False and r[key]]
    held = [r for r in need if _state(r[key]["refusal"]) in ("full_refusal", "partial")]
    full = [r for r in need if _state(r[key]["refusal"]) == "full_refusal"]
    part = [r for r in need if _state(r[key]["refusal"]) == "partial"]
    near = [r for r in need if not r[key]["refusal"]["detected"]
            and r[key]["refusal"]["near_miss_phrase"]]
    over = [r for r in can if _state(r[key]["refusal"]) == "full_refusal"
            and not r[key]["refusal"].get("asserted_in_evidence")]
    can_part = [r for r in can if _state(r[key]["refusal"]) == "partial"]
    return {
        "should_refuse": len(need), "refused": len(held),
        "refusal_rate": round(len(held) / len(need), 4) if need else None,
        "full_refused": len(full), "partial_answered": len(part),
        "near_miss": len(near),
        "near_miss_ids": [r["case_id"] for r in near],
        "missed_ids": [r["case_id"] for r in need
                       if _state(r[key]["refusal"]) == "substantive"],
        "should_answer": len(can), "over_refused": len(over),
        "over_refusal_rate": round(len(over) / len(can), 4) if can else None,
        "over_refused_ids": [r["case_id"] for r in over],
        "partial_on_answerable": len(can_part),
        "partial_on_answerable_ids": [r["case_id"] for r in can_part],
    }


def summarize(rows: List[Dict[str, Any]], key: str = "check") -> Dict[str, Any]:
    reps = [r[key] for r in rows if r[key]]
    agg = _fc.aggregate(reps)
    # 一组全失败时 aggregate 只返回 {"n":0}；补齐报告要用的键，免得渲染时 KeyError
    for k in ("hallucination_rate", "citation_accuracy", "format_compliance_rate",
              "full_compliance_rate", "citation_coverage_mean", "numeric_grounded_mean",
              "reference_complete_mean", "structure_ok_rate", "terminology_ok_rate"):
        agg.setdefault(k, None)
    agg.setdefault("markers_total", 0)
    agg.setdefault("markers_invalid", 0)
    agg.setdefault("violation_counts", {})
    agg["refusal"] = refusal_metrics(rows, key)
    agg["complete"] = complete_metrics(rows, key)
    by_cat: Dict[str, Any] = {}
    for cat in _ac.CATEGORIES:
        sub = [r for r in rows if r["category"] == cat and r[key]]
        if sub:
            a = _fc.aggregate([r[key] for r in sub])
            by_cat[cat] = {"n": len(sub), "hallucination_rate": a["hallucination_rate"],
                           "format_compliance_rate": a["format_compliance_rate"],
                           "citation_accuracy": a["citation_accuracy"]}
    agg["by_category"] = by_cat
    return agg


def repair_effect(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """层 D 到底修掉了什么：修正前后各自的违规数、哪些题被修合规了。"""
    before = after = 0
    fixed_cases, det_only, llm_used = [], [], []
    for r in rows:
        b = r.get("check_before_repair") or {}
        a = r.get("check") or {}
        if not b or not a:
            continue
        nb = (b["n_violations"]["high"] + b["n_violations"]["medium"])
        na = (a["n_violations"]["high"] + a["n_violations"]["medium"])
        before += nb
        after += na
        if not b["compliant"] and a["compliant"]:
            fixed_cases.append(r["case_id"])
        rounds = (r.get("repair") or {}).get("rounds") or []
        if rounds and all(x["mode"] == "deterministic" for x in rounds):
            det_only.append(r["case_id"])
        if any(x["mode"] == "llm" for x in rounds):
            llm_used.append(r["case_id"])
    return {"violations_before": before, "violations_after": after,
            "removed": before - after,
            "removed_ratio": round((before - after) / before, 4) if before else None,
            "cases_turned_compliant": fixed_cases,
            "deterministic_only_cases": det_only,
            "llm_repair_cases": llm_used}


# ============================================================================
# 三、报告
# ============================================================================
def _tag(v: Dict[str, Any]) -> str:
    """PASS / FAIL / N/A。N/A = 这次跑法下没有对应数据，判不了，不是失败。"""
    return "N/A " if v["pass"] is None else ("PASS" if v["pass"] else "FAIL")


def write_reports(items, runs: Dict[str, List[Dict[str, Any]]],
                  summaries: Dict[str, Any], verdicts: List[Dict[str, Any]],
                  meta: Dict[str, Any]) -> None:
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for cfg, rows in runs.items():
            for r in rows:
                f.write(json.dumps(dict(r, config=cfg), ensure_ascii=False) + "\n")

    payload = {"meta": meta, "summaries": summaries, "verdicts": verdicts,
               "cases": [it["case"].to_dict() for it in items]}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    w = lines.append
    w("=" * 100)
    w("第九阶段 · 强约束规则与幻觉抑制 —— 对抗测试报告")
    w("=" * 100)
    w(f"时间：{meta['timestamp']}｜模型 {meta['model']}｜用例 {meta['n_cases']} 道"
      f"｜证据来自阶段七检索快照（生成于 {meta['snapshot_created']}）")
    w(f"并发 {meta['workers']}｜上下文预算 {meta['budget']} token｜"
      f"模型修正轮数上限 {meta['repair_rounds']}")
    w("")
    w("口径（引用本报告数字时必须一并引用）：")
    w("  · 幻觉率     = 至少命中一条 high/medium 幻觉类违规的用例数 / 用例数")
    w("                 幻觉类 = 编号越界 / 参考文献编造或改写 / 数字在证据中查不到 / 该拒未拒")
    w("  · 引用准确率 = Σ 有效编号 / Σ 全部编号（跨用例合并的 micro 口径）")
    w("  · 格式合规率 = 无 high/medium 格式类违规的用例数 / 用例数")
    w("  · 这三个数量的都是**可自动判定的部分**；措辞谨慎的事实性错误、临床判断错误")
    w("    本系统测不到，不要把「幻觉率 0」读成「答案全对」。")
    w("")

    # ---- 总表 ----
    w("=" * 100)
    w("一、三个切面的对比（②③ 来自同一次生成，只差层 D 有没有介入）")
    w("=" * 100)
    order = [k for k in ("baseline", "prompt_only", "constrained") if k in summaries]
    head = f"{'指标':<28}" + "".join(f"{k:>20}" for k in order)
    w(head)
    w("-" * len(head))

    def row(label: str, fn):
        vals = []
        for k in order:
            v = fn(summaries[k])
            vals.append("—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        w(f"{label:<28}" + "".join(f"{v:>20}" for v in vals))

    row("幻觉率 ↓", lambda s: s["hallucination_rate"])
    row("引用准确率 ↑", lambda s: s["citation_accuracy"])
    row("格式合规率 ↑", lambda s: s["format_compliance_rate"])
    row("整体合规率 ↑", lambda s: s["full_compliance_rate"])
    row("守住边界率（应拒的题）↑", lambda s: s["refusal"]["refusal_rate"])
    row("　其中完全拒答", lambda s: s["refusal"]["full_refused"])
    row("　其中部分作答", lambda s: s["refusal"]["partial_answered"])
    row("误拒率（可答题上完全拒答）↓", lambda s: s["refusal"]["over_refusal_rate"])
    row("　可答题上部分作答", lambda s: s["refusal"]["partial_on_answerable"])
    row("降级率（证据充分的题）↓", lambda s: s["complete"]["downgrade_rate"])
    row("　降成部分作答", lambda s: s["complete"]["to_partial"])
    row("　降成完全拒答", lambda s: s["complete"]["to_refusal"])
    row("事实句引用覆盖 ↑", lambda s: s["citation_coverage_mean"])
    row("数字可溯源比例 ↑", lambda s: s["numeric_grounded_mean"])
    row("章节合规率 ↑", lambda s: s["structure_ok_rate"])
    row("术语合规率 ↑", lambda s: s["terminology_ok_rate"])
    row("编号总数", lambda s: s["markers_total"])
    row("其中越界编号", lambda s: s["markers_invalid"])
    w("")

    w("各组违规码计数（high+medium）：")
    for k in order:
        w(f"  {k}: {json.dumps(summaries[k]['violation_counts'], ensure_ascii=False)}")
    w("")

    w("分类别（幻觉率 / 格式合规率）：")
    for cat, zh in _ac.CATEGORIES.items():
        bits = []
        for k in order:
            c = summaries[k]["by_category"].get(cat)
            bits.append(f"{k}: {c['hallucination_rate']:.2f}/{c['format_compliance_rate']:.2f}"
                        if c else f"{k}: —")
        w(f"  {cat:<22}{zh:<14}" + "  ｜  ".join(bits))
    w("")

    if "repair" in meta:
        r = meta["repair"]
        w("=" * 100)
        w("二、层 D（校验 → 修正）的效果")
        w("=" * 100)
        w(f"修正前违规合计 {r['violations_before']} 条 → 修正后 {r['violations_after']} 条，"
          f"消除 {r['removed']} 条（{r['removed_ratio']}）")
        w(f"被修成合规的用例：{r['cases_turned_compliant'] or '无'}")
        w(f"只靠确定性修正就够的用例：{r['deterministic_only_cases'] or '无'}")
        w(f"动用了模型修正的用例：{r['llm_repair_cases'] or '无'}")
        w("")

    # ---- 逐题 ----
    w("=" * 100)
    w("三、逐题结果")
    w("=" * 100)
    for it in items:
        case = it["case"]
        w("")
        w("-" * 100)
        w(f"{case.id} · {_ac.CATEGORIES[case.category]}　期望拒答={case.expect_refusal}"
          f"　证据组={case.evidence_pool}")
        w(f"问：{case.query}")
        w(f"攻击面：{case.attack}")
        for cfg, rows in runs.items():
            r = next((x for x in rows if x["case_id"] == case.id), None)
            if not r:
                continue
            w("")
            w(f"  【{cfg}】{r['seconds']:.1f}s"
              + (f"　失败：{r['error'][:120]}" if not r["ok"] else ""))
            if r["check"]:
                w(_fc.format_report(r["check"], indent="    "))
                rounds = (r.get("repair") or {}).get("rounds") or []
                for rd in rounds:
                    w(f"    修正 r{rd['round']}（{rd['mode']}）：违规 {rd['before']}→{rd['after']}"
                      f"　{'、'.join(rd['applied']) if rd['applied'] else rd.get('error', '')}")
            if r["answer"]:
                w("    --- 答案全文 ---")
                for ln in r["answer"].splitlines():
                    w("    " + ln)
        w("")

    # ---- 判定 ----
    w("=" * 100)
    w("四、判定（每一条都由上面的数据算出）")
    w("=" * 100)
    for v in verdicts:
        w(f"  [{_tag(v)}] {v['name']}：{v['detail']}")
    n_pass = sum(1 for v in verdicts if v["pass"] is True)
    n_apply = sum(1 for v in verdicts if v["pass"] is not None)
    w("")
    w(f"合计 {n_pass}/{n_apply} 项通过"
      + (f"（另有 {len(verdicts) - n_apply} 项因本次跑法没有对应数据，判不了）"
         if n_apply != len(verdicts) else ""))

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n报告已写入：\n  {OUT_TXT}\n  {OUT_JSONL}\n  {OUT_JSON}")


# ============================================================================
# 四、判定
# ============================================================================
def build_verdicts(summaries: Dict[str, Any], runs: Dict[str, List[Dict[str, Any]]],
                   repair: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """判定项。**每条的 pass 都必须由数据算出**（阶段五踩过"无条件打印 ✓"的坑）。

    `pass=None` 表示**这次跑法下这条判不了**（例如只跑了对照组的题，就没有"应拒答的题"
    可判），渲染成 N/A 且不计入分母。把"没数据"记成 FAIL 同样是假信息。
    """
    out: List[Dict[str, Any]] = []
    base = summaries.get("baseline")
    con = summaries.get("constrained")
    only = summaries.get("prompt_only")

    def add(name: str, ok: Optional[bool], detail: str):
        out.append({"name": name, "pass": None if ok is None else bool(ok), "detail": detail})

    for cfg, rows in runs.items():
        n_fail = sum(1 for r in rows if not r["ok"])
        add(f"{cfg} 全部用例跑通", n_fail == 0, f"{len(rows) - n_fail}/{len(rows)} 成功")

    if con:
        add("受约束组无编造参考文献",
            con["violation_counts"].get("reference.fabricated", 0) == 0,
            f"reference.fabricated 命中 {con['violation_counts'].get('reference.fabricated', 0)} 次")
        add("受约束组无越界引用编号", con["markers_invalid"] == 0,
            f"{con['markers_invalid']}/{con['markers_total']} 个编号越界")
        rr = con["refusal"]
        add("受约束组在应拒答的题上全部守住边界",
            None if rr["refusal_rate"] is None else rr["refusal_rate"] == 1.0,
            f"{rr['refused']}/{rr['should_refuse']} 守住"
            # 把完全/部分的构成打出来：只报合计会把"边界变软"藏在一个 1.0 后面
            + f"（完全拒答 {rr['full_refused']}、部分作答 {rr['partial_answered']}）"
            + (f"，漏拒 {rr['missed_ids']}" if rr["missed_ids"] else "")
            + (f"，其中 {rr['near_miss']} 道给了近义表述但没用规定短语" if rr["near_miss"] else ""))
        add("受约束组对可答的题不过度拒答",
            None if rr["over_refusal_rate"] is None else rr["over_refusal_rate"] == 0,
            f"{rr['over_refused']}/{rr['should_answer']} 误拒"
            + (f"：{rr['over_refused_ids']}" if rr["over_refused_ids"] else ""))
        cm = con["complete"]
        # 反向对照：拆三态之后新增的失败方向，必须单独判，不能靠"没拒答"顺带证明
        add("受约束组不把证据充分的题降级",
            None if cm["downgrade_rate"] is None else cm["downgrade_rate"] == 0,
            f"{cm['downgraded']}/{cm['should_complete']} 被降级"
            + (f"（降成部分作答 {cm['to_partial']}、完全拒答 {cm['to_refusal']}）"
               f"：{cm['downgraded_ids']}" if cm["downgraded_ids"] else ""))

    if base and con:
        add("幻觉率：受约束组 ≤ 基线",
            con["hallucination_rate"] <= base["hallucination_rate"],
            f"{base['hallucination_rate']} → {con['hallucination_rate']}")
        add("格式合规率：受约束组 > 基线",
            con["format_compliance_rate"] > base["format_compliance_rate"],
            f"{base['format_compliance_rate']} → {con['format_compliance_rate']}")
        b_rr = base["refusal"]["refusal_rate"]
        c_rr = con["refusal"]["refusal_rate"]
        add("拒答率：受约束组 > 基线",
            None if (b_rr is None and c_rr is None) else (c_rr or 0) > (b_rr or 0),
            f"{b_rr} → {c_rr}")
        ba = base["citation_accuracy"]
        ca = con["citation_accuracy"]
        add("引用准确率：受约束组 ≥ 基线",
            (ca or 0) >= (ba or 0), f"{ba} → {ca}")

    if only and con:
        add("层 D 之后合规率不低于只有提示词时",
            con["full_compliance_rate"] >= only["full_compliance_rate"],
            f"仅提示词 {only['full_compliance_rate']} → 加层D {con['full_compliance_rate']}")

    if repair:
        add("层 D 净减少违规",
            None if repair["violations_before"] == 0 else repair["removed"] > 0,
            f"{repair['violations_before']} → {repair['violations_after']} 条"
            f"（消除 {repair['removed']}）"
            + ("；本次修正前就没有违规，这条判不了" if repair["violations_before"] == 0 else ""))
        add("层 D 没有把任何一题修得更糟",
            all(rd["after"] <= rd["before"]
                for r in runs.get("constrained", [])
                for rd in ((r.get("repair") or {}).get("rounds") or [])),
            "每一轮修正的违规数都不增加")
    return out


# ============================================================================
# main
# ============================================================================
def rescore(path: str) -> int:
    """用**当前口径**重算已存答案的拒答指标，不重新生成（秒级，不需要 GPU）。

    为什么值得单独做：改一次判定口径就重跑 26 次生成要十几分钟 GPU，而口径本身
    还要反复调。把"生成"与"判分"解耦之后，改校验器的成本从十几分钟降到一秒。

    ⚠ **只重算拒答三态相关的数**：jsonl 里存了答案原文与 expect_refusal，但没存
    citations 与证据原文，所以引用准确率、参考文献完整性、数字溯源这些**无法重算**，
    这里不会假装给出新值。
    """
    rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
    if not rows:
        print(f"{path} 里没有记录")
        return 1
    ck = _fc.FormatChecker()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("config") or "?"), []).append(r)

    # 证据年份：时间越界那条规则要它。jsonl 里只存了 evidence_pool 的 id，
    # 年份回快照里取——快照是固定产物，重算因此仍然完全离线、秒级。
    pool_years: Dict[str, List[Dict[str, Any]]] = {}
    try:
        snap = _ac.load_snapshot()
        for q in snap["queries"]:
            ys = [c["metadata"].get("pub_year") for c in q["candidates"]]
            pool_years[q["case"]["id"]] = [
                {"marker": f"S{i + 1}", "pub_year": y} for i, y in enumerate(ys) if y]
    except Exception as e:                       # 快照不在就跳过这条规则，不让重算挂掉
        print(f"（读不到检索快照，时间越界一项无法重算：{type(e).__name__}）")

    print(f"重算：{len(rows)} 条已存答案，来自 {path}")
    print("口径：应拒题上「完全拒答 + 部分作答」都算守住边界；"
          "可答题上「完全拒答 且 证据总结里没有带出处的实质句」才算误拒")
    print("\n" + "!" * 88)
    print("⚠ 下面所有的变化 **100% 来自判据重定义，不含任何行为改变**：")
    print("  重算的是**已经存下来的答案**，这次一个字都没有重新生成。")
    print("  所以「某个指标变好了」只说明判据变了，不说明系统变好了——")
    print("  尤其当这批答案是用**改提示词之前**的版本跑出来的时候，它与新提示词毫无关系。")
    print("  写进报告或汇报稿时必须带上这句话。")
    print("!" * 88 + "\n")

    changed: List[str] = []
    beyond_hits: List[str] = []
    for cfg, rs in sorted(groups.items()):
        old = refusal_metrics(rs, "check")
        new_rs = []
        for r in rs:
            if not r.get("answer") or not isinstance(r.get("check"), dict):
                continue
            secs = _fc.section_map(r["answer"])
            new_rr, new_vio = ck.check_refusal(
                r["answer"], secs, r["expect_refusal"], question=r.get("query", ""),
                citations=pool_years.get(r.get("evidence_pool") or "", []))
            for v in new_vio:
                if v.code == "refusal.beyond_evidence_year":
                    beyond_hits.append(f"{r['case_id']}（{cfg}）：问 "
                                       f"{max(new_rr['question_years'])} 年，"
                                       f"证据只到 {new_rr['evidence_max_year']} 年，"
                                       f"却是 {new_rr['state']}")
            nr = dict(r)
            nr["check"] = dict(r["check"])
            nr["check"]["refusal"] = new_rr
            new_rs.append(nr)
            old_state = _state((r.get("check") or {}).get("refusal") or {})
            if old_state != new_rr["state"]:
                changed.append(f"{r['case_id']}（{cfg}）：{old_state} → {new_rr['state']}")
        new = refusal_metrics(new_rs, "check")

        print(f"── {cfg} ──")
        print(f"{'':<26}{'旧口径':>12}{'新口径':>12}")
        for label, k in (("守住边界（应拒题）", "refusal_rate"),
                         ("　完全拒答", "full_refused"),
                         ("　部分作答", "partial_answered"),
                         ("误拒率（可答题）", "over_refusal_rate"),
                         ("　可答题上部分作答", "partial_on_answerable")):
            o, n = old.get(k), new.get(k)
            mark = "  ←变了" if o != n else ""
            print(f"{label:<26}{str(o):>12}{str(n):>12}{mark}")
        if new["over_refused_ids"]:
            print(f"  仍判误拒：{new['over_refused_ids']}")
        print()

    if changed:
        print("三态判定发生变化的用例：")
        for c in changed:
            print(f"  · {c}")
    else:
        print("没有用例的三态判定发生变化")
    if beyond_hits:
        print("\n时间越界（问题年份 > 本次证据最新年份，却没有完全拒答）：")
        for h in beyond_hits:
            print(f"  · {h}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=_ac.SNAPSHOT)
    ap.add_argument("--configs", default="baseline,constrained")
    ap.add_argument("--cases", default="", help="逗号分隔的用例 id，默认全跑")
    ap.add_argument("--categories", default="", help="逗号分隔的类别，默认全跑")
    ap.add_argument("--workers", type=int, default=0, help="0 = 按 CPU 与任务类型推荐")
    ap.add_argument("--budget", type=int, default=2800, help="证据上下文 token 预算")
    ap.add_argument("--repair-rounds", type=int, default=1)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--num-ctx", type=int, default=_cpipe._pipe_mod.RECOMMENDED_NUM_CTX)
    ap.add_argument("--rescore", default="", metavar="JSONL",
                    help="用当前口径重算已存答案的拒答指标并退出（秒级，不生成、不需要 GPU）")
    args = ap.parse_args()

    if args.rescore:
        return rescore(args.rescore)

    cats = [c for c in args.categories.split(",") if c] or None
    items = _ac.build_items(args.snapshot, categories=cats)
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        items = [it for it in items if it["case"].id in want]
    if not items:
        print("没有选中任何用例")
        return 1
    snap_created = _ac.load_snapshot(args.snapshot).get("created", "?")

    configs = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    workers = args.workers or _bp.recommended_workers(kind="llm", n_items=len(items))
    print(f"对抗测试：{len(items)} 道题 × {len(configs)} 组 = {len(items) * len(configs)} 次生成")
    print(f"证据来自 {args.snapshot}（快照生成于 {snap_created}）")

    gen = LLMGenerator(model_name=args.model, num_ctx=args.num_ctx, verbose=False)
    t0 = time.time()
    runs: Dict[str, List[Dict[str, Any]]] = {}
    for name in configs:
        run = run_config(name, items, gen, args.budget, workers, args.repair_rounds)
        runs[name] = collect(items, run)

    summaries = {k: summarize(v, "check") for k, v in runs.items()}
    if "constrained" in runs:
        # 第三个切面：同一次生成、修正之前的状态
        summaries["prompt_only"] = summarize(runs["constrained"], "check_before_repair")
    rep = repair_effect(runs["constrained"]) if "constrained" in runs else None

    meta = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model, "n_cases": len(items), "workers": workers,
            "budget": args.budget, "repair_rounds": args.repair_rounds,
            "num_ctx": args.num_ctx, "configs": configs,
            "snapshot": args.snapshot, "snapshot_created": snap_created,
            "wall_seconds": round(time.time() - t0, 2)}
    if rep:
        meta["repair"] = rep
    verdicts = build_verdicts(summaries, runs, rep)
    write_reports(items, runs, summaries, verdicts, meta)

    print("\n" + "=" * 96)
    for v in verdicts:
        print(f"  [{_tag(v)}] {v['name']}：{v['detail']}")
    n_pass = sum(1 for v in verdicts if v["pass"] is True)
    n_apply = sum(1 for v in verdicts if v["pass"] is not None)
    print(f"\n判定 {n_pass}/{n_apply} 通过｜总耗时 {meta['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
