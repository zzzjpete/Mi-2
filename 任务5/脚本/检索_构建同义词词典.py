# -*- coding: utf-8 -*-
"""
第五阶段 · 检索系统（一）· 从 MeSH 构建医学同义词词典

任务书里的 MEDICAL_SYNONYMS 只是个示例，并注明「实际应用中这个词典应该更全面，
可以从 UMLS、MeSH 等医学标准术语库构建」。本脚本就干这件事。

选 MeSH 而非 UMLS 的理由：
  * MeSH 是 NLM 官方叙词表，免费直接下载；UMLS 要注册 UTS 账号走 license 审批（数天）。
  * 本项目语料就是 PubMed，MeSH 与 PubMed 同源（PubMed 文献本来就用 MeSH 标引），最对口。

输入：data/mesh/desc2026.xml（DescriptorRecordSet，298MB）
输出：data/dict/mesh_synonyms.json
  {
    "meta": {...统计...},
    "descriptors": { "D009203": {"pref": "...", "type": "disease", "tree": [...], "terms": [...]} },
    "index":       { "归一化词面": ["D009203", ...] }     # 词面 -> 主题词，用于 gazetteer 匹配
  }

降噪策略（不做的话查询会被扩散成噪声）：
  1. 只保留 6 类树号（C 疾病 / D 药物化学 / E 诊疗技术 / A 解剖 / B 生物 / G 生理过程），
     丢掉 N 卫生服务、I 社会学、L 情报学、V 出版类型、Z 地理等——这些是噪声大户。
  2. 丢掉 IsPermutedTermYN="Y" 的机器轮排词。
  3. 倒装词面（"Infarction, Myocardial"）还原成自然语序（"myocardial infarction"）。
  4. 过于通用的单词入口词（control/time/water/...）整条丢弃。

用法：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_构建同义词词典.py
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

from lxml import etree

try:
    sys.stdout.reconfigure(encoding="utf-8")   # 控制台是 GBK，统一 UTF-8
except Exception:
    pass

MESH_XML = os.path.join(ROOT, "data", "mesh", "desc2026.xml")
OUT_JSON = os.path.join(ROOT, "data", "dict", "mesh_synonyms.json")

# MeSH 树号首字母 -> 实体类型。未列出的分支整条丢弃。
TREE_TYPE = {
    "C": "disease",     # Diseases
    "D": "drug",        # Chemicals and Drugs
    "E": "procedure",   # Analytical, Diagnostic and Therapeutic Techniques
    "A": "anatomy",     # Anatomy
    "B": "organism",    # Organisms
    "G": "process",     # Phenomena and Processes
}
# 一个主题词常挂多个树号，按此优先级定型（疾病/药物对检索最有用，排前面）
TYPE_PRIORITY = ["disease", "drug", "procedure", "process", "anatomy", "organism"]

# 过于通用的词面：即便挂在医学树下也不进词典，否则 "control"、"time" 这类词会污染扩展
GENERIC_STOPWORDS = {
    "time", "water", "control", "controls", "cells", "cell", "tissue", "tissues",
    "growth", "development", "structure", "function", "activity", "process",
    "state", "states", "system", "systems", "group", "groups", "factor", "factors",
    "level", "levels", "rate", "rates", "change", "changes", "effect", "effects",
    "response", "responses", "role", "roles", "type", "types", "form", "forms",
    "value", "values", "size", "shape", "color", "light", "heat", "cold", "air",
    "food", "foods", "male", "female", "adult", "adults", "child", "children",
    "human", "humans", "animal", "animals", "male", "aged", "young", "old",
    "history", "science", "research", "study", "reports", "report", "methods",
    "attention", "memory", "movement", "behavior", "learning", "emotions",
    "character", "personality", "family", "friends", "life", "death", "health",
    "disease", "diseases", "syndrome", "infection", "injury", "pain", "fever",
    "weight", "pressure", "flow", "volume", "area", "surface", "body", "head",
    "back", "face", "hand", "foot", "leg", "arm", "skin", "hair", "eye", "ear",
    "mouth", "nose", "neck", "chest", "gases", "ions", "acids", "salts", "oils",
    "products", "materials", "equipment", "software", "internet", "telephone",
}

# 允许的词面字符集（小写化后）。排除带书名号/括号引注/奇怪符号的化学全名
TERM_RE = re.compile(r"^[a-z0-9][a-z0-9 \-'/+.]*$")


def _norm(s):
    """词面归一化：小写 + 压空白。"""
    return re.sub(r"\s+", " ", s).strip().lower()


def _deinvert(s):
    """MeSH 倒装词面还原：'Infarction, Myocardial' -> 'myocardial infarction'。

    多级倒装同理：'Antibodies, Monoclonal, Humanized' -> 'humanized monoclonal antibodies'。
    """
    if ", " not in s:
        return s
    parts = [p.strip() for p in s.split(",")]
    if any(not p for p in parts):
        return s
    return " ".join(reversed(parts))


def _acceptable(term):
    """词面是否收入词典。"""
    if len(term) < 3 or len(term) > 80:
        return False
    if term in GENERIC_STOPWORDS:
        return False
    if not TERM_RE.match(term):
        return False
    if term.endswith("-") or term.endswith("."):
        return False
    # 纯数字/编号类
    if re.fullmatch(r"[0-9 \-.]+", term):
        return False
    return True


def pick_type(tree_numbers):
    """按树号定实体类型。F03（精神障碍）归入 disease。"""
    cands = set()
    for tn in tree_numbers:
        if tn.startswith("F03"):
            cands.add("disease")
            continue
        t = TREE_TYPE.get(tn[:1])
        if t:
            cands.add(t)
    for t in TYPE_PRIORITY:
        if t in cands:
            return t
    return None


def build(args):
    t0 = time.time()
    print(f"[MeSH] 解析 {args.xml} ...", flush=True)

    descriptors = {}
    index = {}
    stat = Counter()

    ctx = etree.iterparse(args.xml, events=("end",), tag="DescriptorRecord",
                          load_dtd=False, no_network=True, resolve_entities=False)
    n_rec = 0
    for _, rec in ctx:
        n_rec += 1
        try:
            ui = rec.findtext("DescriptorUI")
            pref = rec.findtext("DescriptorName/String")
            if not ui or not pref:
                continue
            trees = [e.text for e in rec.findall("TreeNumberList/TreeNumber") if e.text]
            etype = pick_type(trees)
            if etype is None:
                stat["丢弃_树号不在保留分支"] += 1
                continue

            # 收词面：主题词名 + 全部概念下的非轮排 Term
            raw_terms = [pref]
            for term in rec.findall("ConceptList/Concept/TermList/Term"):
                if term.get("IsPermutedTermYN") == "Y":
                    stat["丢弃_轮排词"] += 1
                    continue
                s = term.findtext("String")
                if s:
                    raw_terms.append(s)

            terms, seen = [], set()
            for s in raw_terms:
                t = _norm(_deinvert(s))
                if t in seen:
                    continue
                seen.add(t)
                if not _acceptable(t):
                    stat["丢弃_词面不合格"] += 1
                    continue
                terms.append(t)

            if not terms:
                stat["丢弃_无可用词面"] += 1
                continue

            descriptors[ui] = {
                "pref": pref,
                "type": etype,
                "tree": trees[:3],
                "terms": terms,
            }
            stat[f"保留_{etype}"] += 1
            stat["词面总数"] += len(terms)
            for t in terms:
                index.setdefault(t, []).append(ui)
        finally:
            rec.clear()
            while rec.getprevious() is not None:
                del rec.getparent()[0]

    # 一个词面挂到多个主题词 = 该词歧义（如 "cold" 既是低温也是感冒）
    ambiguous = {t: uis for t, uis in index.items() if len(uis) > 1}
    stat["歧义词面"] = len(ambiguous)

    meta = {
        "source": os.path.basename(args.xml),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "descriptor_records_scanned": n_rec,
        "descriptors_kept": len(descriptors),
        "surface_forms_indexed": len(index),
        "by_type": {t: stat[f"保留_{t}"] for t in TYPE_PRIORITY},
        "ambiguous_surface_forms": len(ambiguous),
        "filters": {
            "kept_tree_branches": TREE_TYPE,
            "dropped_permuted_terms": stat["丢弃_轮排词"],
            "dropped_out_of_branch": stat["丢弃_树号不在保留分支"],
            "dropped_bad_surface": stat["丢弃_词面不合格"],
        },
        "build_seconds": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "descriptors": descriptors, "index": index},
                  f, ensure_ascii=False)

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"\n[完成] 用时 {meta['build_seconds']}s -> {args.out} ({size_mb:.1f} MB)")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    # 抽样自检：几个典型词查一下扩展效果
    print("\n[抽检] 典型医学词的 MeSH 同义词：")
    for probe in ["myocardial infarction", "metformin", "type 2 diabetes mellitus",
                  "hypertension", "crispr-cas systems", "tumor microenvironment"]:
        uis = index.get(probe)
        if not uis:
            print(f"  {probe:32s} -> （未收录）")
            continue
        d = descriptors[uis[0]]
        syns = [t for t in d["terms"] if t != probe][:6]
        print(f"  {probe:32s} -> [{d['type']}] {syns}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=MESH_XML)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
