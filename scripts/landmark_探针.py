# -*- coding: utf-8 -*-
"""P0 验收探针 —— landmark 那 10 篇在**对应问题**上进没进上下文、进的那一块里有没有那个数

## 为什么单独有这个脚本

golden 那把尺子测的是**主库**的检索质量：它的 ground truth 全部抽自 `merged_4m.parquet`，
所以 landmark 条目在 golden 上**永远是负担、不可能是收益**——它只会挤掉主库候选。
拿 golden 的 T1 掉了多少去判断 P0 成不成功，方向就反了。

P0 的收益必须在**它自己的题面上**量：问 HFpEF 用什么 SGLT2 抑制剂，
EMPEROR-Preserved / DELIVER 有没有进上下文。这就是本脚本。

## 验收判据（用户 2026-08-15 定的口径）

不是「有没有召回相关文献」，而是**两层**：

    ① 召回层：landmark 条目进没进最终上下文（`source_type=landmark` 出现在 results 里）
    ② 数字层：**进的那一块原文里，主要终点的 HR / 95%CI / 入组人数在不在**

只看①会得出一个成功的假结论——库里本来就有一堆"沾边"的次要分析与综述，
召回率上去了，模型照样可能从转述里凑一个错数字出来。

用法（要 65GB 主库 + 15.8GB 内存，约 70 秒加载 + 每题 3 秒）：
    ... landmark_探针.py                 # 开关各跑一遍，出对照表
    ... landmark_探针.py --off           # 只跑关掉 landmark 的一轮（看基线）
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import re
import sys
import time
from typing import Any, Dict, List

import pyarrow  # noqa: F401  必须早于 torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPORT = os.path.join(ROOT, "report_data", "landmark_探针报告.txt")
BM25_DIR = os.path.join(ROOT, "data", "bm25_index_4m")

#: 每条 = (题面, 期望命中的 trial_name)。题面是**英文**——探针要量的是检索与保底，
#: 不是中译英；中文那一层已经在 golden 与服务层各自量过了，混进来会分不清是谁的锅。
PROBES: List[Dict[str, Any]] = [
    {"q": "What is the evidence for SGLT2 inhibitors in heart failure with preserved "
          "ejection fraction? Give the primary outcome hazard ratio.",
     "want": ["EMPEROR-Preserved", "DELIVER"]},
    {"q": "Dapagliflozin in heart failure with mildly reduced or preserved ejection "
          "fraction: primary endpoint result",
     "want": ["DELIVER"]},
    {"q": "Does sacubitril/valsartan reduce hospitalization in HFpEF?",
     "want": ["PARAGON-HF"]},
    {"q": "Spironolactone in heart failure with preserved ejection fraction: "
          "did it meet its primary endpoint?",
     "want": ["TOPCAT 主结果", "TOPCAT 区域差异"]},
    {"q": "SGLT2 inhibitor outcomes in heart failure with reduced ejection fraction",
     "want": ["DAPA-HF", "EMPEROR-Reduced"]},
    {"q": "Should healthy older adults take aspirin for primary prevention of "
          "cardiovascular disease?",
     "want": ["ASPREE", "USPSTF 2022 阿司匹林一级预防"]},
    {"q": "Aspirin for primary prevention in patients with diabetes: benefit versus "
          "bleeding risk",
     "want": ["ASCEND"]},
    {"q": "USPSTF recommendation on aspirin use to prevent cardiovascular disease 2022",
     "want": ["USPSTF 2022 阿司匹林一级预防"]},
]

#: 数字层的判据。RCT 与指南分开——**指南没有 HR 是结构使然，不是缺陷**。
_RE_RATIO = re.compile(r"(hazard ratio|risk ratio|rate ratio|odds ratio|relative risk|"
                       r"\bHR\b|\bRR\b|\bOR\b)", re.I)
_RE_CI = re.compile(r"95%\s*(confidence interval|CI)", re.I)
_RE_N = re.compile(r"[\d,]{3,9}\s*(patients|participants|adults|persons)", re.I)
_RE_GRADE = re.compile(r"\b[A-D]\s+recommendation|grade\s+[A-D]\b", re.I)
_RE_AGE = re.compile(r"\b\d{2}\s*(to|-|–)\s*\d{2}\s*years|aged\s*\d{2}", re.I)


def numbers_present(text: str, level: str) -> Dict[str, bool]:
    if level == "guideline":
        return {"推荐等级": bool(_RE_GRADE.search(text)), "年龄分层": bool(_RE_AGE.search(text))}
    return {"效应量": bool(_RE_RATIO.search(text)), "95%CI": bool(_RE_CI.search(text)),
            "入组人数": bool(_RE_N.search(text))}


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(pipe, use_landmark: bool, top_k: int = 10) -> List[Dict[str, Any]]:
    out_rows = []
    for p in PROBES:
        t0 = time.time()
        out = pipe.search(p["q"], top_k=top_k, rerank=True, use_landmark=use_landmark)
        lm = [c for c in out["results"]
              if (c.metadata or {}).get("source_type") == "landmark"]
        hit_names = [(c.metadata or {}).get("trial_name", "?") for c in lm]
        want_hit = [w for w in p["want"] if w in hit_names]
        nums = {}
        for c in lm:
            md = c.metadata or {}
            nums[md.get("trial_name", "?")] = numbers_present(
                c.text or "", md.get("evidence_level", "RCT"))
        out_rows.append({
            "q": p["q"], "want": p["want"], "landmark_in_context": hit_names,
            "want_hit": want_hit, "numbers": nums,
            "landmark_meta": out.get("landmark", {}),
            "elapsed": round(time.time() - t0, 2),
        })
    return out_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", default=BM25_DIR)
    ap.add_argument("--off", action="store_true", help="只跑关掉 landmark 的那一轮")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    mp = _load("mp", os.path.join(ROOT, "scripts", "检索_多路检索.py"))
    pipe = mp.RetrievalPipeline(bm25_dir=args.bm25, translate="dict")

    lines: List[str] = []

    def p(s: str = ""):
        print(s)
        lines.append(s)

    p("=" * 100)
    p("P0 验收探针：landmark 在自己的题面上进没进上下文 + 进的那一块里有没有那个数")
    p("=" * 100)
    p(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　题目 {len(PROBES)} 条（英文，绕开中译英那一层）")
    p(f"landmark 集合：{pipe.landmark_detail}")
    p("")
    p("⚠ 这把尺子只量 P0 的**收益**。P0 的**代价**（挤掉主库候选）在 golden 上量，")
    p("  两把尺子各测一半，缺一边就会得出片面结论。")
    p("")

    rows_on = run(pipe, use_landmark=True, top_k=args.top_k)
    rows_off = None if args.off else run(pipe, use_landmark=False, top_k=args.top_k)

    p("① 召回层：landmark 有没有进上下文")
    p("-" * 100)
    p(f"{'题面':<52}{'期望':<28}{'开':<6}{'关':<6}命中")
    n_hit = 0
    for i, r in enumerate(rows_on):
        off_n = len(rows_off[i]["landmark_in_context"]) if rows_off else 0
        ok = bool(r["want_hit"])
        n_hit += int(ok)
        p(f"{r['q'][:50]:<52}{'/'.join(w[:12] for w in r['want']):<28}"
          f"{len(r['landmark_in_context']):<6}{off_n:<6}"
          f"{'✓ ' + '/'.join(w[:14] for w in r['want_hit']) if ok else '✗'}")
    p("-" * 100)
    p(f"**{n_hit}/{len(rows_on)} 条题面上，期望的 landmark 进了上下文**")
    if rows_off is not None:
        tot_off = sum(len(r["landmark_in_context"]) for r in rows_off)
        p(f"对照：关掉 landmark 路时进上下文的 landmark 条目 {tot_off} 条"
          f"（应当恒为 0，否则说明开关没生效）")

    p("")
    p("② 数字层：进上下文的那一块里，要害数字在不在")
    p("-" * 100)
    total = ok_all = 0
    for r in rows_on:
        for name, chk in r["numbers"].items():
            total += 1
            good = all(chk.values())
            ok_all += int(good)
            p(f"  {name:<30}{'✓' if good else '✗'}  "
              + "　".join(f"{k} {'✓' if v else '✗'}" for k, v in chk.items()))
    p("-" * 100)
    if total:
        p(f"**{ok_all}/{total} 次进上下文的 landmark，原文里带着该类型的要害信息**")
    else:
        p("没有任何 landmark 进上下文——数字层无从谈起（先看①）")

    p("")
    p("③ 保底机制的动作（每题）")
    p("-" * 100)
    p(f"{'题面':<52}{'池内':<8}{'进上下文':<10}{'保底补入':<10}{'因不够相关被拒'}")
    for r in rows_on:
        m = r["landmark_meta"]
        p(f"{r['q'][:50]:<52}{m.get('retrieved', 0):<8}{m.get('in_results', 0):<10}"
          f"{m.get('promoted', 0):<10}{m.get('rejected', 0)}")
    p("")
    p("⚠ 「因不够相关被拒」恒为 0 说明保底又变回了无条件——那会往每个上下文里塞无关文献。")
    p("  实测无关 query 上 landmark 的 rel 只有 0.000~0.08，而被它挤掉的主库候选是 0.72~0.97。")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(REPORT.replace(".txt", ".json"), "w", encoding="utf-8") as f:
        json.dump({"on": rows_on, "off": rows_off}, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {REPORT}")
    return 0 if n_hit == len(rows_on) else 1


if __name__ == "__main__":
    sys.exit(main())
