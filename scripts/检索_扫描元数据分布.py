# -*- coding: utf-8 -*-
"""
第五阶段 · 检索系统（一）· 扫描语料元数据分布

为什么需要它：查询理解要「提取过滤条件」，但过滤条件必须落到语料的真实取值上，
否则生成的 where 子句看着漂亮、实际一条都命中不了。两个具体问题：

  1. section 取值很脏：'Results' / 'RESULTS' / '3. Results' / 'Results and Discussion'
     混在一起。用户说 "in the methods" 时，得展开成库里所有等价写法的 $in 列表。
  2. pub_year 覆盖范围未知：用户说「近五年」，若语料根本没有那几年的文献，
     过滤后会返回空。得知道真实上下界，好在结果里给出提示。

输入：data/vectors/subset_4000000_s42.parquet（建库用的同一份子集，元数据与库内一致）
输出：data/dict/corpus_meta.json

用法：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_扫描元数据分布.py
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import pyarrow.parquet as pq

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SUBSET = os.path.join(ROOT, "data", "vectors", "subset_4000000_s42.parquet")
OUT = os.path.join(ROOT, "data", "dict", "corpus_meta.json")

# 规范章节 -> 判定关键词（对小写化、去编号后的章节名做匹配）
SECTION_RULES = [
    ("abstract",     [r"^abstract$", r"^summary$"]),
    ("introduction", [r"^introduction$", r"^background$", r"^intro$"]),
    ("methods",      [r"method", r"^materials?$", r"material.*method", r"experimental",
                      r"^study design$", r"^patients and methods$", r"^statistical analysis$"]),
    ("results",      [r"^results?$", r"^findings$", r"result.*discussion"]),
    ("discussion",   [r"^discussions?$", r"result.*discussion", r"^interpretation$"]),
    ("conclusion",   [r"^conclusions?$", r"concluding remark", r"^summary and conclusion"]),
]
# 非正文章节（参考文献、致谢、利益冲突等）——检索时通常该排除
NONBODY_RULES = [
    r"^references?$", r"acknowledg", r"competing interest", r"conflict of interest",
    r"authors?.{0,3} contribution", r"^funding$", r"supporting information",
    r"supplementary", r"^abbreviations?$", r"^ethics", r"consent", r"^disclosure",
    r"data availability", r"^copyright", r"^footnotes?$", r"additional file",
]


def canon_section(raw):
    """把库里的原始 section 值映射到规范章节名；认不出返回 None。"""
    s = raw.strip().lower()
    s = re.sub(r"^[\divxlc]+[.．、)\s]+", "", s)      # 去 "3. " / "IV) " 之类编号
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    for pat in NONBODY_RULES:
        if re.search(pat, s):
            return "_nonbody"
    for canon, pats in SECTION_RULES:
        for pat in pats:
            if re.search(pat, s):
                return canon
    return None


def scan(args):
    t0 = time.time()
    pf = pq.ParquetFile(args.subset)
    total = pf.metadata.num_rows
    print(f"[扫描] {args.subset}  共 {total:,} 行", flush=True)

    sec = Counter()
    yr = Counter()
    jrn = Counter()
    done = 0
    for b in pf.iter_batches(batch_size=131072, columns=["section", "pub_year", "journal"]):
        d = b.to_pydict()
        sec.update(d["section"])
        yr.update(d["pub_year"])
        jrn.update(d["journal"])
        done += b.num_rows
        if done % 1_000_000 < 131072:
            print(f"  ... {done:,}/{total:,}", flush=True)

    # 章节：原始值 -> 规范名，再倒排成 规范名 -> [原始值...]（供 Chroma $in 使用）
    canon_map = {}
    canon_count = Counter()
    unmapped = Counter()
    for raw, n in sec.items():
        c = canon_section(str(raw))
        if c is None:
            unmapped[raw] += n
            continue
        canon_map.setdefault(c, []).append(str(raw))
        canon_count[c] += n
    # 每个规范名下按出现频次排序，只保留覆盖 99% 的原始值（避免 $in 列表爆炸）
    trimmed = {}
    for c, raws in canon_map.items():
        raws_sorted = sorted(raws, key=lambda r: -sec[r])
        cum, keep, tot = 0, [], canon_count[c]
        for r in raws_sorted:
            keep.append(r)
            cum += sec[r]
            if cum >= 0.99 * tot:
                break
        trimmed[c] = keep

    years = {int(k): v for k, v in yr.items() if isinstance(k, (int, float)) and k}
    ys = sorted(years)
    cum, cutoff_p95 = 0, None
    tot_y = sum(years.values())
    for y in ys:                                   # 覆盖 95% 文献的年份下界
        cum += years[y]
        if cutoff_p95 is None and cum >= 0.05 * tot_y:
            cutoff_p95 = y

    out = {
        "meta": {
            "source": os.path.basename(args.subset),
            "rows": total,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scan_seconds": round(time.time() - t0, 1),
        },
        "pub_year": {
            "min": ys[0] if ys else None,
            "max": ys[-1] if ys else None,
            "p05": cutoff_p95,
            "histogram": {str(y): years[y] for y in ys},
            "top_years": [[str(y), n] for y, n in yr.most_common(15)],
        },
        "section": {
            "distinct_raw_values": len(sec),
            "canonical_counts": dict(canon_count),
            "canonical_to_raw": trimmed,
            "unmapped_top": [[str(k), v] for k, v in unmapped.most_common(25)],
            "unmapped_total": sum(unmapped.values()),
        },
        "journal": {
            "distinct": len(jrn),
            "top": [[str(k), v] for k, v in jrn.most_common(50)],
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n[完成] 用时 {out['meta']['scan_seconds']}s -> {args.out}")
    print(f"\n年份：{out['pub_year']['min']} – {out['pub_year']['max']}，"
          f"95% 文献在 {cutoff_p95} 年之后")
    print("  近年分布：", {str(y): years.get(y, 0) for y in range(2015, (ys[-1] or 2015) + 1)})
    print(f"\n章节：原始取值 {len(sec)} 种 -> 规范 6 类")
    for c, n in canon_count.most_common():
        print(f"  {c:14s} {n:9,}  （{len(trimmed[c])} 种写法，如 {trimmed[c][:3]}）")
    print(f"  未映射合计 {sum(unmapped.values()):,}，top: {[k for k, _ in unmapped.most_common(8)]}")
    print(f"\n期刊：{len(jrn)} 种，top5 {[k for k, _ in jrn.most_common(5)]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    scan(args)


if __name__ == "__main__":
    main()
