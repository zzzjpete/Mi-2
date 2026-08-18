# -*- coding: utf-8 -*-
"""P0 · landmark 开 / 关 两轮的对照 —— 代价落在哪、换回了什么

⚠ **这把尺子只量代价，不量收益。** golden 的 ground truth 全部抽自 `merged_4m.parquet`
（主库建库输入表），所以 landmark 条目在它上面**只可能挤占位置、不可能加分**。
拿 T1 掉了多少去判断 P0 成不成功，方向是反的。收益要用 `landmark_探针.py` 量。

两把尺子各测一半，缺一边就会得出片面结论：
    代价（本脚本）  golden 235 条，看 R@k / MRR / GT 名次位移
    收益（探针）    8 道 landmark 自己题面的题，看进没进上下文 + 那一块里有没有那个数

用法（纯离线，秒级；两轮跑测明细必须都已存在）：
    ... landmark_对照.py
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import importlib.util
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("g", os.path.join(ROOT, "scripts", "golden_跑测.py"))
g = importlib.util.module_from_spec(spec)
sys.modules["g"] = g
spec.loader.exec_module(g)

OFF = os.path.join(ROOT, "data", "golden", "golden_跑测明细_p2a.jsonl")      # tiebreak, landmark 关（已发布基线）
ON = os.path.join(ROOT, "data", "golden", "golden_跑测明细_lm3.jsonl")        # tiebreak, landmark 开

off = {r["qid"]: r for r in g.load_run(OFF)}
on = {r["qid"]: r for r in g.load_run(ON)}
qids = sorted(set(off) & set(on))
print(f"交集 {len(qids)} 条（off={len(off)} on={len(on)}）\n")


def tier_rows(d, tier):
    return [d[q] for q in qids if d[q]["tier"] == tier]


def metrics(rows, cfg, lvl):
    ranks = [g._rank_of(r["configs"][cfg]["hit"][lvl]) for r in rows]
    return (g._recall(ranks, 5), g._recall(ranks, 10), g._recall(ranks, 20), g._mrr(ranks))


print("① final_rank 口径（与历史数字同尺，可直接对照）")
print(f"{'层/配置/粒度':<26}{'R@5':>16}{'R@10':>16}{'R@20':>16}{'MRR':>16}")
for tier in (1, 2, 3):
    for cfg in ("prod", "wide"):
        for lvl in ("chunk", "doc"):
            a = metrics(tier_rows(off, tier), cfg, lvl)
            b = metrics(tier_rows(on, tier), cfg, lvl)
            if lvl == "doc" and cfg == "wide":
                continue
            name = f"T{tier} · {cfg} · {lvl}"
            cells = "".join(f"{x:.3f}→{y:.3f}({y-x:+.3f})".rjust(16) for x, y in zip(a, b))
            print(f"{name:<26}{cells}")
print()

# ---- in_top_k 口径：真正进上下文（保底会改这个，不改 final_rank）----
print("② in_top_k 口径（真正进上下文的比例；保底只影响这个）")


def topk_rate(rows, cfg, lvl):
    n = sum(1 for r in rows if r["configs"][cfg]["hit"][lvl].get("in_top_k"))
    return n / len(rows) if rows else 0.0


for tier in (1, 2, 3):
    for cfg in ("prod",):
        a_c, b_c = topk_rate(tier_rows(off, tier), cfg, "chunk"), topk_rate(tier_rows(on, tier), cfg, "chunk")
        a_d, b_d = topk_rate(tier_rows(off, tier), cfg, "doc"), topk_rate(tier_rows(on, tier), cfg, "doc")
        print(f"  T{tier} · {cfg}   chunk {a_c:.3f}→{b_c:.3f}({b_c-a_c:+.3f})   "
              f"doc {a_d:.3f}→{b_d:.3f}({b_d-a_d:+.3f})")
print()

# ---- landmark 到底动了多少次 ----
print("③ landmark 在 235 条上的实际动作（prod 配置）")
n_pool = n_ctx = n_q_with_ctx = 0
per_trial = {}
for q in qids:
    cs = on[q]["configs"]["prod"]["candidates"]
    lm = [c for c in cs if c.get("source_type") == "landmark"]
    inctx = [c for c in lm if c.get("in_top_k")]
    n_pool += len(lm)
    n_ctx += len(inctx)
    n_q_with_ctx += int(bool(inctx))
    for c in inctx:
        per_trial[c.get("trial_name")] = per_trial.get(c.get("trial_name"), 0) + 1
print(f"  进候选池 {n_pool} 条次（{len(qids)} 条 query，每题取 4 条）")
print(f"  进上下文 {n_ctx} 条次，涉及 {n_q_with_ctx} 条 query（占 {n_q_with_ctx/len(qids):.1%}）")
for k, v in sorted(per_trial.items(), key=lambda x: -x[1]):
    print(f"     {k:<28}{v} 次")

# ---- 被 landmark 挤掉的那些：原本在第几名 ----
print()
print("④ 代价落在哪：off 时 GT 在 top10、on 时掉出去的题")
lost = []
for q in qids:
    for cfg in ("prod",):
        ha = off[q]["configs"][cfg]["hit"]["chunk"]
        hb = on[q]["configs"][cfg]["hit"]["chunk"]
        ra, rb = g._rank_of(ha), g._rank_of(hb)
        if ra is not None and ra <= 10 and (rb is None or rb > 10):
            lost.append((q, off[q]["tier"], ra, rb))
gained = []
for q in qids:
    ha = off[q]["configs"]["prod"]["hit"]["chunk"]
    hb = on[q]["configs"]["prod"]["hit"]["chunk"]
    ra, rb = g._rank_of(ha), g._rank_of(hb)
    if (ra is None or ra > 10) and rb is not None and rb <= 10:
        gained.append((q, off[q]["tier"], ra, rb))
print(f"  掉出 top10：{len(lost)} 条　进入 top10：{len(gained)} 条　净 {len(gained)-len(lost):+d}")
for q, t, ra, rb in lost[:8]:
    print(f"     T{t} {q}  #{ra} → {rb}")

