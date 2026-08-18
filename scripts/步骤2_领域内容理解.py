"""
步骤2_领域内容理解.py — 阶段二第二步：领域内容理解（对应任务第2点）

用 bge-m3 tokenizer 给全文算 token 长度 -> 按短/中/长分层抽样 -> 精读并量化：
  · 结构：是否遵循 IMRaD（背景-方法-结果-结论）
  · 术语：缩写密度（EGFR/PCI…）、同一概念的不同表述（同义/英式美式拼写）
  · （可选）高频内容词

产物：
  - 在 parsed_full.parquet 上新增列 token_len（供第三步复用），另存 parsed_tok.parquet
  - report_data\step2.json
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import json
import re
from collections import Counter
from pathlib import Path

import os
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from transformers import AutoTokenizer

DATA = Path(os.path.join(ROOT, "data", "pubmed"))
REPORT = Path(os.path.join(ROOT, "report_data"))
IN_PARQUET = DATA / "parsed_full.parquet"
OUT_PARQUET = DATA / "parsed_tok.parquet"

# 分块单元 = 标题 + 正文（摘要按第一步策略视为可选，不参与）
def doc_text(r):
    return (r["title"] or "") + "\n\n" + (r["body_text"] or "")

IMRAD_CANON = {
    "introduction": ["introduction"],
    "background": ["background"],
    "methods": ["method", "material"],
    "results": ["result"],
    "discussion": ["discussion"],
    "conclusions": ["conclusion"],
}

# 同一概念的不同表述（演示检索层面的同义/拼写问题）
VARIANT_PAIRS = [
    ("myocardial infarction", "heart attack"),
    ("neoplasm", "tumor"),
    ("hypertension", "high blood pressure"),
    ("tumor", "tumour"),          # 美式 vs 英式
    ("randomized", "randomised"),  # 美式 vs 英式
]

ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")
ACRONYM_STOP = {"THE", "AND", "FOR", "WITH", "THIS", "THAT", "FIG", "NOT",
                "ARE", "WAS", "ALL", "III", "II", "IV", "USA", "DNA", "RNA"}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
STOP = set("""the and for with this that from were was are have has had not
which can may our their study data using used between within results
methods introduction discussion conclusion these those they them been
also into than then when here there such more most other into each one two
three both figure table used also more will these show shown showed found
observed obtained analysis based different high low value values case cases
group groups control number time""".split())


def main():
    df = pd.read_parquet(IN_PARQUET)
    print("加载", len(df), "篇。加载 bge-m3 tokenizer…")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    # ---- token 长度（分批，避免一次性构建超长序列）----
    texts = df.apply(doc_text, axis=1).tolist()
    lens = []
    B = 128
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], add_special_tokens=True,
                  truncation=False, return_length=True)
        lens.extend(enc["length"])
        print(f"  tokenized {min(i+B,len(texts))}/{len(texts)}", end="\r")
    print()
    df["token_len"] = lens

    # ---- 分层：按 token 三分位分短/中/长 ----
    q33, q66 = df["token_len"].quantile([1/3, 2/3])
    q33, q66 = int(q33), int(q66)

    def band(x):
        return "short" if x <= q33 else ("medium" if x <= q66 else "long")
    df["band"] = df["token_len"].apply(band)

    # ---- 每档抽 6 篇，精读用 ----
    samples = {}
    for b in ["short", "medium", "long"]:
        sub = df[df["band"] == b].sample(n=6, random_state=11)
        rows = []
        for _, r in sub.iterrows():
            titles = [t for t in (r["sec_titles"] or "").split(" | ") if t]
            acr = ACRONYM_RE.findall(r["body_text"] or "")
            acr = [a for a in acr if a not in ACRONYM_STOP]
            rows.append({
                "pmcid": r["pmcid"], "journal": r["journal"],
                "year": int(r["pub_year"]) if pd.notna(r["pub_year"]) else None,
                "tokens": int(r["token_len"]), "n_sections": int(r["n_sections"]),
                "sec_titles": titles[:8],
                "uniq_acronyms": len(set(acr)),
            })
        samples[b] = rows

    # ---- IMRaD 覆盖率（正文章节标题里出现各标准段的比例）----
    sec_lower = df["sec_titles"].fillna("").str.lower()
    imrad_cov = {}
    for canon, keys in IMRAD_CANON.items():
        hit = sec_lower.apply(lambda s: any(k in t for t in s.split(" | ")
                                            for k in keys))
        imrad_cov[canon] = round(100 * hit.mean(), 1)
    # 同时含 方法+结果+讨论/结论 的“完整 IMRaD”
    def full_imrad(s):
        segs = s.split(" | ")
        has = lambda keys: any(k in t for t in segs for k in keys)
        return (has(["method", "material"]) and has(["result"])
                and (has(["discussion"]) or has(["conclusion"])))
    full_cov = round(100 * sec_lower.apply(full_imrad).mean(), 1)

    # ---- 缩写密度 ----
    per_doc_uacr, per_1k = [], []
    all_acr = Counter()
    for _, r in df.iterrows():
        acr = [a for a in ACRONYM_RE.findall(r["body_text"] or "")
               if a not in ACRONYM_STOP]
        u = len(set(acr))
        per_doc_uacr.append(u)
        if r["token_len"]:
            per_1k.append(1000 * len(acr) / r["token_len"])
        all_acr.update(acr)
    acr_summary = {
        "median_unique_per_doc": int(pd.Series(per_doc_uacr).median()),
        "mean_acronyms_per_1k_tokens": round(pd.Series(per_1k).mean(), 1),
        "top20": all_acr.most_common(20),
    }

    # ---- 同义/拼写变体（doc frequency）----
    variants = []
    for a, b in VARIANT_PAIRS:
        na = int(df["body_text"].str.contains(a, case=False, regex=False).sum())
        nb = int(df["body_text"].str.contains(b, case=False, regex=False).sum())
        variants.append({"a": a, "docs_a": na, "b": b, "docs_b": nb})

    # ---- （可选）高频内容词：取 400 篇样本，词频 ----
    samp_txt = " ".join(df["body_text"].sample(400, random_state=3).tolist()).lower()
    wc = Counter(w for w in WORD_RE.findall(samp_txt) if w not in STOP)
    top_terms = wc.most_common(25)

    summary = {
        "token_bands": {"q33": q33, "q66": q66,
                        "short_max": q33, "long_min": q66 + 1,
                        "counts": {b: int((df["band"] == b).sum())
                                   for b in ["short", "medium", "long"]}},
        "samples": samples,
        "imrad_coverage_pct": imrad_cov,
        "full_imrad_pct": full_cov,
        "acronyms": acr_summary,
        "variant_pairs": variants,
        "top_content_terms_sample400": top_terms,
        "token_len_glance": {
            "median": int(df["token_len"].median()),
            "p90": int(df["token_len"].quantile(0.90)),
            "p95": int(df["token_len"].quantile(0.95)),
            "p99": int(df["token_len"].quantile(0.99)),
            "max": int(df["token_len"].max()),
        },
    }

    df.to_parquet(OUT_PARQUET)
    REPORT.mkdir(exist_ok=True)
    (REPORT / "step2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== step2 摘要 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:1500])
    print("…")
    print("已存:", OUT_PARQUET, "和", REPORT / "step2.json")


if __name__ == "__main__":
    main()
