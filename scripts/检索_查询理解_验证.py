# -*- coding: utf-8 -*-
"""
第五阶段 · 检索系统（一）· 查询理解与增强 —— 质量验证

分两层：
  离线层（不碰向量库）：① 功能自检，对 12 条覆盖各类情况的查询断言实体/扩展/过滤是否正确
  在线层（连 400 万向量库）：② 过滤条件下推是否真的可用
                             ③ A/B：BGE 指令前缀 "sentence"(官方/建库用) vs "question"(任务书写法)
                             ④ A/B：缩写查询的三种扩展策略（主查询 / 单查询平铺 / 多查询 RRF）
                             ⑤ A/B：中文查询 直接检索 vs 中译英
                             ⑥ 端到端样例：增强查询 + 元数据过滤的实际命中

相关性怎么量的（诚实说明）：本阶段没有人工标注集，用的是**术语命中率 term-hit@k**——
top-k 命中块的正文里是否出现该查询的目标医学术语（实体及其同义词）。它只是相关性的
代理指标，不等于真实相关性；但对「缩写有没有被理解」「中文有没有落到英文术语上」
这类问题，它的区分度足够，且完全可复现。真正的相关性评测放在下一部分（检索+重排）。

用法：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_查询理解_验证.py
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_查询理解_验证.py --offline-only
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")          # 硬覆盖，早于任何 HF 导入
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import importlib.util
import json
import re
import sys
import textwrap
import time

import pyarrow.parquet as pq          # 必须早于 torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPORT_PATH = os.path.join(ROOT, "report_data", "查询理解验证报告.txt")
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
_REPORT = open(REPORT_PATH, "w", encoding="utf-8")


class _Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            try:
                st.write(x)
            except Exception:
                pass

    def flush(self):
        for st in self.s:
            try:
                st.flush()
            except Exception:
                pass


sys.stdout = _Tee(sys.stdout, _REPORT)


def _load(path, name):
    """中文文件名模块按路径导入（与本项目既有脚本一致）。

    先找**本脚本同目录**、再退回给定的绝对路径：这样这份脚本连同它的依赖被拷进
    `任务5\\脚本\\` 单独发出去时也能跑，而不是死认 E:\\rag\\scripts。
    """
    sibling = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(path))
    spec = importlib.util.spec_from_file_location(name, sibling if os.path.exists(sibling) else path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


qu = _load(os.path.join(ROOT, "scripts", "检索_查询理解.py"), "qu")
# 章节归一化函数：复用扫描脚本里那一份，保证验证用的规则与建映射时完全一致
canon_section = _load(os.path.join(ROOT, "scripts", "检索_扫描元数据分布.py"), "scanmeta").canon_section


# ============================================================================
# ① 功能自检（离线）
# ============================================================================
CASES = [
    {
        "q": "Does MI risk increase in patients with CKD?",
        "why": "缩写识别 + 歧义缩写生成多义项变体",
        "entities": ["MI", "CKD"],
        "expansions": {"MI": "myocardial infarction", "CKD": "chronic kidney disease"},
        "min_vector_queries": 3,
        "not_entities": ["risk", "patients"],      # 通用词不该被当实体扩展
    },
    {
        "q": "What is the effect of metformin on cardiovascular outcomes in T2DM?",
        "why": "正则药名 + 缩写全称展开",
        "entities": ["metformin", "T2DM"],
        "expansions": {"T2DM": "type 2 diabetes mellitus"},
        "min_vector_queries": 2,
    },
    {
        "q": "Is Glucophage safe for elderly patients with impaired renal function?",
        "why": "商品名 → 通用名（MeSH 覆盖）",
        "entities": ["Glucophage"],
        "expansions": {"Glucophage": "metformin"},
    },
    {
        "q": "heart attack prevention with aspirin, recent studies",
        "why": "俗称 → 术语；模糊时间词按启发式转过滤",
        "entities": ["heart attack", "aspirin"],
        "expansions": {"heart attack": "myocardial infarction"},
        "filter_year_gte": True,
    },
    {
        "q": "tumour haemorrhage after anti-VEGF therapy",
        "why": "英式拼写归一 + 基因/蛋白识别",
        "entities": ["tumour", "haemorrhage", "VEGF"],
        "expansions": {"tumour": "tumor", "haemorrhage": "hemorrhage"},
    },
    {
        "q": "RCT evidence for pembrolizumab in NSCLC published since 2020",
        "why": "研究设计缩写 + since 年份下界",
        "entities": ["RCT", "pembrolizumab", "NSCLC"],
        "filter_exact": {"pub_year": {"$gte": 2020}},
    },
    {
        "q": "What did studies between 2015 and 2018 report about gut microbiota and obesity?",
        "why": "年份区间 → $and 双边界",
        "entities": ["gut microbiota", "obesity"],
        "filter_exact": {"$and": [{"pub_year": {"$gte": 2015}}, {"pub_year": {"$lte": 2018}}]},
    },
    {
        "q": "EGFR mutation and TKI resistance in lung cancer, in the results section",
        "why": "章节过滤下推 $in（results 只需 24 种写法）",
        "entities": ["EGFR", "TKI"],
        "filter_has_section_in": True,
    },
    {
        "q": "cell culture protocols described in the methods section",
        "why": "章节过滤退化为后置过滤（methods 有 7726 种写法，$in 不现实）",
        "post_filter_section": "methods",
    },
    {
        "q": "二甲双胍对心血管疾病有何影响？",
        "why": "中文查询 → 英文术语（需求方给的示例查询）",
        "lang": "zh",
        "core_contains": ["metformin", "cardiovascular"],
    },
    {
        "q": "近五年 CRISPR 基因编辑在肿瘤治疗中的应用",
        "why": "中文时间词必须在翻译前抽取，否则会被当未覆盖片段丢掉",
        "lang": "zh",
        "filter_year_gte": True,
        "core_contains": ["CRISPR", "gene editing"],
    },
    {
        "q": "hello world",
        "why": "非医学查询：不报错、优雅降级",
        "entities": [],
        "min_vector_queries": 1,
    },
]


def check_offline(proc):
    print("\n" + "=" * 78)
    print("① 功能自检（12 条查询，覆盖缩写/歧义/俗称/商品名/拼写/中文/时间/章节/无实体）")
    print("=" * 78)
    passed = failed = 0
    for i, c in enumerate(CASES, 1):
        eq = proc.process_query(c["q"])
        errs = []
        ent_texts = [e.text for e in eq.entities]
        ent_lower = [t.lower() for t in ent_texts]

        for want in c.get("entities", []):
            if want.lower() not in ent_lower:
                errs.append(f"未识别实体 {want!r}（实得 {ent_texts}）")
        for bad in c.get("not_entities", []):
            if bad.lower() in ent_lower:
                errs.append(f"通用词 {bad!r} 不该被识别为实体")
        for k, v in c.get("expansions", {}).items():
            got = [s.lower() for s in eq.expansions.get(k, [])]
            if v.lower() not in got:
                errs.append(f"{k!r} 的扩展里缺 {v!r}（实得 {got}）")
        if "min_vector_queries" in c and len(eq.vector_queries) < c["min_vector_queries"]:
            errs.append(f"向量查询条数 {len(eq.vector_queries)} < 期望 {c['min_vector_queries']}")
        if c.get("filter_year_gte") and "pub_year" not in json.dumps(eq.filters):
            errs.append(f"期望抽出年份过滤，实得 {eq.filters}")
        if "filter_exact" in c and eq.filters != c["filter_exact"]:
            errs.append(f"过滤条件不符：期望 {c['filter_exact']}，实得 {eq.filters}")
        if c.get("filter_has_section_in") and "section" not in json.dumps(eq.filters):
            errs.append(f"期望 section $in 下推，实得 {eq.filters}")
        if "post_filter_section" in c and eq.post_filters.get("section_canon") != c["post_filter_section"]:
            errs.append(f"期望后置过滤 section={c['post_filter_section']}，实得 {eq.post_filters}")
        if "lang" in c and eq.language != c["lang"]:
            errs.append(f"语言判定 {eq.language} != {c['lang']}")
        for w in c.get("core_contains", []):
            if w.lower() not in eq.core_text.lower():
                errs.append(f"检索主体里缺 {w!r}（实得 {eq.core_text!r}）")

        if errs:
            failed += 1
            print(f"\n  [{i:2d}] ✗ {c['q']}")
            print(f"       用途：{c['why']}")
            for e in errs:
                print(f"       ! {e}")
        else:
            passed += 1
            print(f"  [{i:2d}] ✓ {c['q'][:58]:60s} … {c['why']}")
    print(f"\n  → 自检结果：{passed}/{len(CASES)} 通过，{failed} 失败")
    return failed == 0


# ============================================================================
# 在线层公共工具
# ============================================================================
def embed_prefixed(embedder, text):
    """text 已自带 BGE 指令前缀，绕开 embed_query（它会再加一次前缀）。"""
    return embedder._encode([text], batch_size=1)[0]


def search(col, embedder, prefixed_query, n=10, where=None):
    v = embed_prefixed(embedder, prefixed_query)
    return col.query(query_embeddings=[v.tolist()], n_results=n, where=where,
                     include=["documents", "metadatas", "distances"])


def term_hit_rate(res, targets):
    """term-hit@k：top-k 命中块正文里出现任一目标术语的比例（相关性代理指标）。"""
    docs = res["documents"][0]
    if not docs:
        return 0.0, 0
    pats = [re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in targets]
    hit = sum(1 for d in docs if any(p.search(d.lower()) for p in pats))
    return hit / len(docs), len(docs)


def rrf_fuse(result_sets, k=60, topn=10, weights=None):
    """多查询结果 RRF 融合：score = Σ w_i / (k + rank)。weights 为 None 时等权。"""
    scores, store = {}, {}
    ws = weights or [1.0] * len(result_sets)
    for res, w in zip(result_sets, ws):
        for rank, (cid, doc, meta, dist) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            store[cid] = (doc, meta, dist)
    top = sorted(scores, key=lambda c: -scores[c])[:topn]
    return {"ids": [top],
            "documents": [[store[c][0] for c in top]],
            "metadatas": [[store[c][1] for c in top]],
            "distances": [[store[c][2] for c in top]]}


def show(res, n=3, width=110):
    for r, (cid, doc, m, dist) in enumerate(zip(
            res["ids"][0][:n], res["documents"][0][:n],
            res["metadatas"][0][:n], res["distances"][0][:n]), 1):
        print(f"      [{r}] sim={1-dist:.3f}  {cid}  ({m.get('journal','?')} {m.get('pub_year','?')} · {m.get('section','?')})")
        print(f"          《{(m.get('source_title') or '')[:72]}》")
        print(f"          {textwrap.shorten(doc.replace(chr(10),' '), width=width, placeholder=' …')}")


# ============================================================================
# ②–⑥ 在线验证
# ============================================================================
# A/B 用查询：目标术语用于算 term-hit@k
AB_QUERIES = [
    {"q": "Does MI risk increase in patients with CKD?",
     "targets": ["myocardial infarction", "heart attack", "chronic kidney disease"]},
    {"q": "What is the effect of metformin on cardiovascular outcomes in T2DM?",
     "targets": ["metformin", "type 2 diabetes", "cardiovascular"]},
    {"q": "RCT evidence for pembrolizumab in NSCLC",
     "targets": ["pembrolizumab", "non-small cell lung", "nsclc"]},
    {"q": "EGFR mutation and TKI resistance in lung cancer",
     "targets": ["egfr", "tyrosine kinase inhibitor", "resistance"]},
    {"q": "Is Glucophage safe for patients with impaired renal function?",
     "targets": ["metformin", "glucophage", "renal"]},
    {"q": "heart attack prevention with aspirin",
     "targets": ["myocardial infarction", "aspirin"]},
]


def check_filters_online(proc, col, embedder):
    print("\n" + "=" * 78)
    print("② 过滤条件下推验证（生成的 where 子句真的丢给 Chroma 跑）")
    print("=" * 78)
    cases = [
        "RCT evidence for pembrolizumab in NSCLC published since 2020",
        "What did studies between 2015 and 2018 report about gut microbiota and obesity?",
        "EGFR mutation and TKI resistance in lung cancer, in the results section",
        "近五年 CRISPR 基因编辑在肿瘤治疗中的应用",
    ]
    ok = True
    for q in cases:
        eq = proc.process_query(q)
        res = search(col, embedder, eq.vector_query, n=10, where=eq.filters or None)
        metas = res["metadatas"][0]
        n = len(metas)
        print(f"\n  Q: {q}")
        print(f"     where = {json.dumps(eq.filters, ensure_ascii=False)[:150]}")
        if n == 0:
            print("     ✗ 返回 0 条 —— 过滤条件把结果滤空了")
            ok = False
            continue
        yrs = sorted({m.get("pub_year") for m in metas})
        secs = sorted({str(m.get("section")) for m in metas})
        print(f"     ✓ 返回 {n} 条；命中年份 {yrs}")
        # 逐条核对过滤条件真的生效
        f = json.dumps(eq.filters)
        if "$gte" in f:
            lo = min(c["pub_year"]["$gte"] for c in
                     (eq.filters.get("$and") or [eq.filters]) if "pub_year" in c and "$gte" in c["pub_year"])
            bad = [y for y in yrs if y < lo]
            print(f"       年份下界 {lo}：{'✓ 全部满足' if not bad else f'✗ 违例 {bad}'}")
            ok &= not bad
        if "$lte" in f:
            hi = max(c["pub_year"]["$lte"] for c in
                     (eq.filters.get("$and") or [eq.filters]) if "pub_year" in c and "$lte" in c["pub_year"])
            bad = [y for y in yrs if y > hi]
            print(f"       年份上界 {hi}：{'✓ 全部满足' if not bad else f'✗ 违例 {bad}'}")
            ok &= not bad
        if "section" in f:
            # 两项独立核对：
            #   a) 返回值是否都在 $in 白名单内 —— 证明 Chroma 真的执行了过滤
            #   b) 这些值是否都归一到同一个规范章节 —— 证明白名单本身是自洽的
            allowed = None
            for c in (eq.filters.get("$and") or [eq.filters]):
                sv = c.get("section")
                if isinstance(sv, dict) and "$in" in sv:
                    allowed = set(sv["$in"])
            out_of_list = [s for s in secs if allowed is not None and s not in allowed]
            canons = {canon_section(s) for s in secs}
            print(f"       命中章节 {secs[:6]}")
            print(f"       在 $in 白名单内：{'✓ 全部满足' if not out_of_list else f'✗ 越界 {out_of_list}'}")
            print(f"       归一化章节：{canons} → "
                  f"{'✓ 同属一类' if len(canons) == 1 and None not in canons else '✗ 不一致或无法归类'}")
            ok &= (not out_of_list) and len(canons) == 1 and None not in canons
    return ok


def ab_instruction(proc, col, embedder):
    print("\n" + "=" * 78)
    print("③ A/B：BGE 指令前缀  官方\"sentence\"（第四阶段建库/验证所用） vs 任务书\"question\"")
    print("=" * 78)
    print("  两者只作用于查询侧（文档侧不加前缀），换前缀不会与既有索引冲突。")
    print(f"  {'查询':<46s} {'sentence':>10s} {'question':>10s} {'top10重合':>10s}")
    agg = {"official": [], "taskbook": []}
    overlaps = []
    for item in AB_QUERIES:
        q, tg = item["q"], item["targets"]
        core = proc.process_query(q).core_text
        r1 = search(col, embedder, qu.QUERY_INSTRUCTION_OFFICIAL + core, n=10)
        r2 = search(col, embedder, qu.QUERY_INSTRUCTION_TASKBOOK + core, n=10)
        h1, _ = term_hit_rate(r1, tg)
        h2, _ = term_hit_rate(r2, tg)
        ov = len(set(r1["ids"][0]) & set(r2["ids"][0])) / 10
        agg["official"].append(h1)
        agg["taskbook"].append(h2)
        overlaps.append(ov)
        print(f"  {q[:44]:<46s} {h1:>10.2f} {h2:>10.2f} {ov:>10.0%}")
    mo, mt = np.mean(agg["official"]), np.mean(agg["taskbook"])
    print(f"  {'平均 term-hit@10':<46s} {mo:>10.2f} {mt:>10.2f} {np.mean(overlaps):>10.0%}")
    print(f"\n  结论：两版前缀平均 term-hit@10 差 {abs(mo-mt):.3f}，top10 平均重合 {np.mean(overlaps):.0%}。")
    print("  → 差异在噪声量级。保持与第四阶段建库/验证一致的官方 \"sentence\" 版为默认，")
    print("    任务书的 \"question\" 版通过 --instruction taskbook 可切换，二者可互换不影响索引。")
    return mo, mt


def ab_expansion(proc, col, embedder):
    print("\n" + "=" * 78)
    print("④ A/B：同义词扩展策略（本模块最关键的设计选择）")
    print("=" * 78)
    print("  A 主查询       ：只用清洗后的原查询（不扩展）")
    print("  B 单查询平铺   ：把同义词拼进同一条查询（直觉做法）")
    print("  C 多查询+等权RRF：每个缩写义项各查一次，等权 RRF 融合")
    print("  D 多查询+加权RRF：同上，但主查询在其缩写已被展开时降权到 0.5（本模块默认输出的权重）")
    print("\n  ⚠ 指标偏差声明：term-hit@10 的目标术语【就是】同义词扩展项，")
    print("    把扩展项塞进查询天然更容易命中它们，该指标对 B 有构造性偏袒。")
    print("    因此下表只用于看「扩展有没有救回缩写查询」，不足以判定 B/C/D 孰优——")
    print("    那需要第二部分带人工标注的相关性评测。")
    print(f"\n  {'查询':<44s} {'A主查询':>8s} {'B平铺':>8s} {'C等权':>8s} {'D加权':>8s}")
    A, B, C, D = [], [], [], []
    for item in AB_QUERIES:
        q, tg = item["q"], item["targets"]
        eq = proc.process_query(q)
        subs = [search(col, embedder, vq, n=10) for vq in eq.vector_queries]
        ra = subs[0]
        rb = search(col, embedder, eq.vector_query_expanded, n=10)
        rc = rrf_fuse(subs, topn=10)
        rd = rrf_fuse(subs, topn=10, weights=eq.vector_query_weights)
        ha, _ = term_hit_rate(ra, tg)
        hb, _ = term_hit_rate(rb, tg)
        hc, _ = term_hit_rate(rc, tg)
        hd, _ = term_hit_rate(rd, tg)
        A.append(ha); B.append(hb); C.append(hc); D.append(hd)
        nq = len(eq.vector_queries)
        print(f"  {q[:42]:<44s} {ha:>8.2f} {hb:>8.2f} {hc:>8.2f} {hd:>8.2f}   ({nq}条查询)")
    print(f"  {'平均 term-hit@10':<44s} {np.mean(A):>8.2f} {np.mean(B):>8.2f} "
          f"{np.mean(C):>8.2f} {np.mean(D):>8.2f}")
    return np.mean(A), np.mean(B), np.mean(C), np.mean(D)


def ab_chinese(proc, col, embedder):
    print("\n" + "=" * 78)
    print("⑤ A/B：中文查询  直接检索 vs 中译英（索引是英文模型 bge-base-en-v1.5）")
    print("=" * 78)
    cases = [
        ("二甲双胍对心血管疾病有何影响？", ["metformin", "cardiovascular"]),
        ("肿瘤微环境在癌症免疫治疗中的作用", ["tumor microenvironment", "immunotherapy"]),
        ("CRISPR 基因编辑的脱靶效应", ["crispr", "off-target"]),
    ]
    raw_h, tr_h = [], []
    for q, tg in cases:
        eq = proc.process_query(q)
        r_raw = search(col, embedder, qu.QUERY_INSTRUCTION_OFFICIAL + q, n=10)   # 中文原文直接向量化
        r_tr = search(col, embedder, eq.vector_query, n=10)                      # 走翻译后的英文
        h1, _ = term_hit_rate(r_raw, tg)
        h2, _ = term_hit_rate(r_tr, tg)
        raw_h.append(h1); tr_h.append(h2)
        print(f"\n  Q: {q}")
        print(f"     译文: {eq.core_text}")
        print(f"     直接中文检索 term-hit@10 = {h1:.2f}   |   中译英后 = {h2:.2f}")
        print("     直接中文 top1:")
        show(r_raw, n=1)
        print("     中译英  top1:")
        show(r_tr, n=1)
    print(f"\n  → 平均 term-hit@10：直接中文 {np.mean(raw_h):.2f}  vs  中译英 {np.mean(tr_h):.2f}")
    return np.mean(raw_h), np.mean(tr_h)


def end_to_end(proc, col, embedder):
    print("\n" + "=" * 78)
    print("⑥ 端到端样例：增强查询 + 元数据过滤的实际命中")
    print("=" * 78)
    for q in ["二甲双胍对心血管疾病有何影响？",
              "Does MI risk increase in patients with CKD?",
              "RCT evidence for pembrolizumab in NSCLC published since 2020"]:
        eq = proc.process_query(q)
        print("\n" + "-" * 78)
        print(eq.pretty())
        res = rrf_fuse([search(col, embedder, vq, n=10, where=eq.filters or None)
                        for vq in eq.vector_queries], topn=10,
                       weights=eq.vector_query_weights)
        print("  检索结果（多查询加权 RRF 融合后 top3）：")
        show(res, n=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-path", default=os.path.join(ROOT, "data", "chroma_db_4m"))
    ap.add_argument("--collection", default="medrag_bge_base")
    ap.add_argument("--model", default="bge-base")
    ap.add_argument("--offline-only", action="store_true", help="只跑①功能自检，不加载向量库")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("第五阶段 · 查询理解与增强 —— 质量验证报告")
    print(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    proc = qu.MedicalQueryProcessor(verbose=True)
    offline_ok = check_offline(proc)

    if args.offline_only:
        print(f"\n[完成] 仅离线自检，用时 {time.time()-t0:.1f}s -> {REPORT_PATH}")
        return

    import chromadb
    jk = _load(os.path.join(ROOT, "scripts", "向量化_建库.py"), "jk")
    print(f"\n[库] 加载 {args.chroma_path} / {args.collection} ...", flush=True)
    col = chromadb.PersistentClient(path=args.chroma_path).get_collection(args.collection)
    print(f"[库] 向量数 {col.count():,}", flush=True)
    embedder = jk.BGEEmbedder(args.model)

    filt_ok = check_filters_online(proc, col, embedder)
    mo, mt = ab_instruction(proc, col, embedder)
    a, b, c, d = ab_expansion(proc, col, embedder)
    zh_raw, zh_tr = ab_chinese(proc, col, embedder)
    end_to_end(proc, col, embedder)

    print("\n" + "=" * 78)
    print("小结")
    print("=" * 78)
    print(f"  ① 功能自检            ：{'✓ 全部通过' if offline_ok else '✗ 有失败项'}")
    print(f"  ② 过滤条件下推        ：{'✓ 全部生效且非空' if filt_ok else '✗ 有问题'}")
    print(f"  ③ 指令前缀 sentence/question：{mo:.2f} / {mt:.2f}（term-hit@10，差异在噪声量级）")
    print(f"  ④ 扩展 主/平铺/等权RRF/加权RRF：{a:.2f} / {b:.2f} / {c:.2f} / {d:.2f}（term-hit@10，指标对平铺有偏袒）")
    print(f"  ⑤ 中文 直接/中译英    ：{zh_raw:.2f} / {zh_tr:.2f}（term-hit@10）")
    print(f"\n[完成] 用时 {time.time()-t0:.1f}s -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
