# -*- coding: utf-8 -*-
"""第七阶段（二）· 接入前后对比评测：RAG vs 裸 qwen3:8b

阶段一压测裸模型时留下四道题（编造参考文献 / 罕见病药名张冠李戴 / 药物分类错误 /
最新药物时间线自相矛盾），整个 RAG 系统就是为了修它们而建的。这个脚本把两边并排放：

    同一道题 ─┬─▶ 裸 qwen3:8b（无检索、无证据，仅医学助手 system prompt）
              └─▶ RAG 四段链（阶段六检索 → 上下文组装 → 四段提示词）

**公平性上做了两件事**，否则这个对比没有说服力：
  1. 裸模型用的是项目里日常问答那套 system prompt（`多轮问答.py` 的同款，明确要求"不得编造
     参考文献"），而不是随手写一个差的——不能靠削弱对照组来赢。
  2. 两边同模型、同 `num_ctx`、同 `think=False`；裸模型温度取 0.3，与 RAG 的答案生成段一致。

**能自动判定的**（下面 A/B/C 三组，结论全部由数据算出）：
  A 出处可溯源性 —— RAG 的每个 [S#] 都能给出 PMCID 与 PMC 链接；裸模型给出的"参考文献"
    有多少条带可解析标识符（PMID/DOI/PMC），以及这些标识符能否在检索语料里核到。
  B 年份可核性 —— 答案里出现的年份，有多少能在**同一道题检索到的真实证据原文**中找到。
  C 术语可核性 —— 从证据原文自动构建术语表（药名后缀 -mab/-nib/-stat… + 试验名式样），
    看两边答案提到的术语有多少在证据里出现过。

**不能自动判定的**（必须人读，脚本只负责并排呈现）：答得全不全、临床上对不对、
语气是否恰当。所以产物里始终包含两边的答案全文。

⚠ 一个必须说清的口径：**"核不到" ≠ "错"**。证据池只有该题检索回来的 top_k 条，
裸模型说的东西可能是对的、只是不在这几条里。B/C 两组量的是**"这套系统能不能替你验证它"**，
不是"它说的是不是真的"。这正是 RAG 的卖点：可溯源，而不是全知。

用法（需 Ollama 在跑；检索快照由 生成_流水线_测试.py --dump-retrieval 产出）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_对比评测.py \\
        --snapshot E:\\rag\\report_data\\检索快照_live.json

    # 复用已跑好的 RAG 结果（省约 3 分钟），只补跑裸模型：
    ... --rag-jsonl E:\\rag\\report_data\\生成_流水线测试_live.jsonl
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
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

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
MedicalGenerationPipeline = _pipe_mod.MedicalGenerationPipeline
LLMGenerator = _pipe_mod.LLMGenerator


# ============================================================================
# 一、裸模型的 system prompt（与项目日常问答同款，不削弱对照组）
# ============================================================================
BARE_SYSTEM = (
    "你是一名严谨的医学知识助手，面向有医学背景的用户。回答要求："
    "1) 先给结论再解释；2) 分点作答，避免空话；3) 涉及药物使用通用名；"
    "4) 证据不足或存在争议时明确指出，不要臆断；5) **不得编造参考文献或数据**；"
    "6) 如果引用了具体研究，请给出可核对的出处（作者/期刊/年份/PMID）；"
    "7) 回答语言与提问语言保持一致。"
)


# ============================================================================
# 二、自动判定
# ============================================================================
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_PMID = re.compile(r"\bPMID[:\s]*(\d{6,9})\b", re.I)
_PMC = re.compile(r"\bPMC\d{5,9}\b", re.I)
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
#: 一行看起来像参考文献：编号开头 + 含年份，或含 "et al."
_REFLINE = re.compile(r"^\s*(?:\[\d+\]|\(\d+\)|\d+[.)])\s+.{10,}$", re.M)
_ETAL = re.compile(r"\bet\s+al\.?", re.I)
#: 药名/试验名式样：常见药物后缀 + 全大写试验名（CLARITY-AD、EMERGE 等）
_DRUGLIKE = re.compile(r"\b[A-Za-z][A-Za-z\-]{3,}(?:mab|nib|stat|zumab|ximab|umab|ase|"
                       r"parin|sartan|prazole|tinib|ciclib|degib|afil|vir)\b", re.I)
_TRIALLIKE = re.compile(r"\b(?:[A-Z]{3,}(?:-[A-Z0-9]+)?|[A-Z][a-z]+-[A-Z]{2,})\b")
#: 试验名式样里会混进的普通缩写，逐个排掉（否则 FDA/RCT 会被当成试验名）
_TRIAL_STOP = {"FDA", "EMA", "RCT", "MS", "AD", "FD", "ERT", "DMT", "DMTS", "CNS", "MRI",
               "PET", "CSF", "USA", "US", "EU", "WHO", "NHS", "AAN", "III", "II", "IV",
               "DNA", "RNA", "PCR", "ELISA", "HR", "OR", "CI", "SD", "SE", "NSCLC",
               "CRISPR", "CAS", "PMID", "PMC", "DOI", "JAMA", "NEJM", "BMJ", "S1P",
               "ARIA", "APOE", "CDR", "SB", "ADAS", "MMSE", "EML", "AGAL", "GLA"}


def _terms(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """从一段文本里抽出 (年份, 药名式样, 试验名式样)，全部小写归一。"""
    years = set(_YEAR.findall(text))
    drugs = {m.group(0).lower() for m in _DRUGLIKE.finditer(text)}
    trials = {m.group(0) for m in _TRIALLIKE.finditer(text)} - _TRIAL_STOP
    trials = {t for t in trials if len(t) >= 4}
    return years, drugs, {t.lower() for t in trials}


def check_answer(answer: str, evidence_blob: str, evidence_meta: Dict[str, Any],
                 is_rag: bool) -> Dict[str, Any]:
    """把一份答案对着该题的真实证据核一遍。所有计数都由文本算出。"""
    ev = evidence_blob.lower()
    years, drugs, trials = _terms(answer)

    def grounded(items: Set[str]) -> Tuple[int, List[str]]:
        miss = sorted(x for x in items if x not in ev)
        return len(items) - len(miss), miss

    y_ok, y_miss = grounded(years)
    d_ok, d_miss = grounded(drugs)
    t_ok, t_miss = grounded(trials)

    # A 出处可溯源性
    pmids = set(_PMID.findall(answer))
    pmcs = {x.upper() for x in _PMC.findall(answer)}
    dois = set(_DOI.findall(answer))
    reflines = len(_REFLINE.findall(answer))
    etal = len(_ETAL.findall(answer))
    if is_rag:
        # RAG 的出处由代码生成：可溯源率 = 有 PMCID 的来源数 / 来源总数
        srcs = evidence_meta.get("sources") or []
        traceable = sum(1 for s in srcs if s.get("pmcid"))
        cite_total, cite_traceable = len(srcs), traceable
        cite_note = "出处由代码从检索元数据生成，模型无法自行拼造"
    else:
        # 裸模型：把它自己给出的标识符算作"声称的出处"，再看能否在语料里核到
        claimed = pmids | pmcs | dois
        corpus_ids = {str(s.get("pmid")) for s in (evidence_meta.get("sources") or [])} | \
                     {str(s.get("pmcid")).upper() for s in (evidence_meta.get("sources") or [])}
        cite_total = len(claimed) if claimed else reflines
        cite_traceable = len(claimed & corpus_ids)
        cite_note = "标识符能否在该题检索到的证据里核到"

    return {
        "chars": len(answer),
        "years": {"n": len(years), "grounded": y_ok, "missing": y_miss},
        "drugs": {"n": len(drugs), "grounded": d_ok, "missing": d_miss},
        "trials": {"n": len(trials), "grounded": t_ok, "missing": t_miss},
        "citations": {"total": cite_total, "traceable": cite_traceable, "note": cite_note,
                      "pmids": sorted(pmids), "pmcids": sorted(pmcs),
                      "dois": sorted(dois), "reference_lines": reflines, "et_al": etal},
    }


# ============================================================================
# 三、跑对比
# ============================================================================
def run(args) -> int:
    snap = json.load(open(args.snapshot, encoding="utf-8"))
    print(f"检索快照 {args.snapshot}｜生成于 {snap['created']}｜{len(snap['queries'])} 道题")

    rag_by_query: Dict[str, Dict[str, Any]] = {}
    if args.rag_jsonl and os.path.exists(args.rag_jsonl):
        for line in open(args.rag_jsonl, encoding="utf-8"):
            r = json.loads(line)
            rag_by_query[r["query"]] = r
        print(f"复用已有 RAG 结果 {args.rag_jsonl}（{len(rag_by_query)} 条）")

    gen = LLMGenerator(model_name=args.model, num_ctx=args.num_ctx, verbose=False)
    pipe = None
    if len(rag_by_query) < len(snap["queries"]):
        pipe = MedicalGenerationPipeline(generator=gen, max_context_tokens=args.budget,
                                         verbose=True)

    rows: List[Dict[str, Any]] = []
    for q in snap["queries"]:
        case, query = q["case"], q["case"]["query"]
        blob = "\n".join((c.get("text") or "") for c in q["candidates"])

        # ---- RAG 侧 ----
        if query in rag_by_query:
            r = rag_by_query[query]
            rag_answer = r["answer"]
            rag_seconds = r["metrics"]["total_time_seconds"]
            rag_meta = {"sources": r["sources"]}
        else:
            res = pipe.generate(query, retrieved_docs=q["candidates"])
            rag_answer = res["answer"]
            rag_seconds = res["generation_metrics"]["total_time_seconds"]
            rag_meta = {"sources": res["sources"]}

        # ---- 裸模型侧 ----
        print(f"\n[{case['id']}] 裸模型作答中……")
        t0 = time.time()
        br = gen.generate(query, system_prompt=BARE_SYSTEM, temperature=0.3,
                          max_tokens=args.bare_max_tokens)
        bare_answer, bare_seconds = br["text"], round(time.time() - t0, 2)
        print(f"  {bare_seconds}s | {br['eval_count']} tok"
              + ("（被 num_predict 截断）" if br["truncated"] else ""))

        rag_chk = check_answer(rag_answer, blob, rag_meta, is_rag=True)
        bare_chk = check_answer(bare_answer, blob, {"sources": rag_meta["sources"]}, is_rag=False)
        rows.append({"case": case, "rag": {"answer": rag_answer, "seconds": rag_seconds,
                                           "check": rag_chk, "sources": rag_meta["sources"]},
                     "bare": {"answer": bare_answer, "seconds": bare_seconds,
                              "check": bare_chk, "truncated": br["truncated"]}})

    return report(rows, args)


def _fmt(chk: Dict[str, Any]) -> str:
    y, d, t = chk["years"], chk["drugs"], chk["trials"]
    c = chk["citations"]
    return (f"{chk['chars']:>5} 字 | 年份 {y['grounded']}/{y['n']} 可核 | "
            f"药名 {d['grounded']}/{d['n']} | 试验名 {t['grounded']}/{t['n']} | "
            f"出处 {c['traceable']}/{c['total']} 可溯源")


def report(rows: List[Dict[str, Any]], args) -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 100)
    print("接入前后对比（每题两行：RAG / 裸模型）")
    print("=" * 100)
    for r in rows:
        print(f"\n[{r['case']['id']}] {r['case']['category']}")
        print(f"  裸模型当时的错：{r['case'].get('why', '')}")
        print(f"  RAG  {_fmt(r['rag']['check'])}  {r['rag']['seconds']}s")
        print(f"  裸   {_fmt(r['bare']['check'])}  {r['bare']['seconds']}s")
        bc = r["bare"]["check"]["citations"]
        if bc["pmids"] or bc["dois"] or bc["pmcids"]:
            print(f"       裸模型声称的标识符：PMID{bc['pmids']} PMC{bc['pmcids']} DOI{bc['dois']}")
        elif bc["reference_lines"]:
            print(f"       裸模型给了 {bc['reference_lines']} 行参考文献样式的内容，但无任何可解析标识符")

    # ---- 汇总（结论全部由上面的计数算出）----
    n = len(rows)
    agg = {}
    for side in ("rag", "bare"):
        yg = sum(r[side]["check"]["years"]["grounded"] for r in rows)
        yn = sum(r[side]["check"]["years"]["n"] for r in rows)
        dg = sum(r[side]["check"]["drugs"]["grounded"] for r in rows)
        dn = sum(r[side]["check"]["drugs"]["n"] for r in rows)
        ct = sum(r[side]["check"]["citations"]["traceable"] for r in rows)
        cn = sum(r[side]["check"]["citations"]["total"] for r in rows)
        agg[side] = {"years": (yg, yn), "drugs": (dg, dn), "cites": (ct, cn),
                     "chars": sum(r[side]["check"]["chars"] for r in rows) / n,
                     "seconds": sum(r[side]["seconds"] for r in rows) / n}

    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    hdr = f"{'':<8}{'年份可核':>14}{'药名可核':>14}{'出处可溯源':>16}{'均字数':>9}{'均耗时':>9}"
    print(hdr + "\n" + "-" * 100)
    for side, label in (("rag", "RAG"), ("bare", "裸模型")):
        a = agg[side]
        print(f"{label:<8}{a['years'][0]}/{a['years'][1]:<12}"
              f"{a['drugs'][0]}/{a['drugs'][1]:<12}"
              f"{a['cites'][0]}/{a['cites'][1]:<14}"
              f"{a['chars']:>9.0f}{a['seconds']:>9.1f}s")

    def rate(t):
        return (t[0] / t[1]) if t[1] else 0.0

    checks = [
        ("RAG 的出处 100% 可溯源到 PMCID",
         rate(agg["rag"]["cites"]) == 1.0 and agg["rag"]["cites"][1] > 0),
        ("裸模型的出处可溯源率低于 RAG",
         rate(agg["bare"]["cites"]) < rate(agg["rag"]["cites"])),
        ("RAG 的年份可核率高于裸模型",
         rate(agg["rag"]["years"]) > rate(agg["bare"]["years"])),
        ("RAG 的药名可核率不低于裸模型",
         rate(agg["rag"]["drugs"]) >= rate(agg["bare"]["drugs"])),
    ]
    print()
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = sum(1 for _, ok in checks if not ok)
    print(f"\n  {len(checks) - failed}/{len(checks)} 项成立")
    print("\n  ⚠ 口径提醒：「可核」= 能在该题检索回的 top_k 条证据原文里找到。"
          "\n    核不到 ≠ 一定错——证据池就那么大。这组数量的是「系统能不能替你验证它」，"
          "\n    不是「它说的是不是真的」。答得全不全、临床上对不对，仍须人读下面的答案全文。")

    # ---- 落盘 ----
    jsonl_path = os.path.join(REPORT_DIR, "生成_对比评测.jsonl")
    txt_path = os.path.join(REPORT_DIR, "生成_对比评测报告.txt")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"ts": stamp, **r}, ensure_ascii=False) + "\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"接入前后对比评测（RAG vs 裸 qwen3:8b）\n生成时间：{stamp}\n用例 {n} 个\n")
        f.write(f"\n裸模型 system prompt（与项目日常问答同款，明确要求不得编造参考文献）：\n"
                f"{BARE_SYSTEM}\n")
        for r in rows:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"[{r['case']['id']}] {r['case']['category']}\n问题：{r['case']['query']}\n")
            f.write(f"裸模型当时的错：{r['case'].get('why', '')}\n")
            f.write(f"期望：{r['case'].get('expect', '')}\n")
            f.write("\n" + "-" * 46 + " RAG " + "-" * 46 + "\n")
            f.write(r["rag"]["answer"] + "\n")
            f.write(f"\n[指标] {_fmt(r['rag']['check'])}  {r['rag']['seconds']}s\n")
            f.write("\n" + "-" * 44 + " 裸模型 " + "-" * 44 + "\n")
            f.write(r["bare"]["answer"] + "\n")
            f.write(f"\n[指标] {_fmt(r['bare']['check'])}  {r['bare']['seconds']}s\n")
            bc = r["bare"]["check"]["citations"]
            f.write(f"[裸模型声称的标识符] PMID={bc['pmids']} PMC={bc['pmcids']} DOI={bc['dois']} "
                    f"参考文献行数={bc['reference_lines']} et al.={bc['et_al']}\n")
            f.write(f"[裸模型答案里核不到的年份] {r['bare']['check']['years']['missing']}\n")
            f.write(f"[裸模型答案里核不到的药名] {r['bare']['check']['drugs']['missing']}\n")
        f.write("\n" + "=" * 100 + "\n汇总\n" + "=" * 100 + "\n")
        f.write(json.dumps({k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                for kk, vv in v.items()} for k, v in agg.items()},
                           ensure_ascii=False, indent=2) + "\n")
        f.write("\n口径：「可核」= 能在该题检索回的 top_k 条证据原文里找到。核不到 ≠ 一定错，"
                "证据池就那么大。这组数量的是「系统能不能替你验证它」，不是「它说的是不是真的」。\n")
    print(f"\n日志：{jsonl_path}\n      {txt_path}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(REPORT_DIR, "检索快照_live.json"))
    ap.add_argument("--rag-jsonl", default=os.path.join(REPORT_DIR, "生成_流水线测试_live.jsonl"),
                    help="复用已跑好的 RAG 结果；缺失则现场跑")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--num-ctx", type=int, default=12288)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--bare-max-tokens", type=int, default=1500,
                    help="与 RAG 答案生成段的 max_tokens 一致，避免因长度上限不同产生偏差")
    args = ap.parse_args()
    if not os.path.exists(args.snapshot):
        print(f"找不到检索快照 {args.snapshot}\n"
              f"请先跑：生成_流水线_测试.py --dump-retrieval {args.snapshot} --bm25 <BM25索引目录>")
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
