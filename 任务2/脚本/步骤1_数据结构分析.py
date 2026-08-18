"""
步骤1_数据结构分析.py — 阶段二第一步：数据加载 + 结构分析（全量解析 oa_comm baseline 样本包）

做三件事（对应任务第1点）：
  1) 字段完整性：逐字段缺失率（重点盯 abstract 是否 >1%）
  2) 基础质量：超短文本 / 编码错误
  3) 关键字段：journal / pub_date 能否做元数据过滤器，pmid 能否做溯源链接

产物：
  - E:\\rag\\data\\pubmed\\parsed_full.parquet   全量结构化表（供后续步骤复用）
  - E:\\rag\\report_data\\step1.json             报告用的统计结果
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import json
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd
from pandas.api import types as pdt
from lxml import etree

DATA_DIR = Path(os.path.join(ROOT, "data", "pubmed"))
STEM = "oa_comm_xml.PMC000xxxxxx.baseline.2026-06-18"
TARGZ = DATA_DIR / f"{STEM}.tar.gz"
OUT_PARQUET = DATA_DIR / "parsed_full.parquet"
REPORT_DIR = Path(os.path.join(ROOT, "report_data"))
REPORT_DIR.mkdir(exist_ok=True)

THIS_YEAR = 2026
IMRAD_KEYS = ("introduction", "background", "method", "material",
              "result", "discussion", "conclusion")


def text_of(el):
    return " ".join(el.itertext()).strip() if el is not None else ""


def first_abstract(root):
    """取主摘要：跳过 graphical/teaser 等非正文摘要，取第一个非空。"""
    for ab in root.findall(".//front//abstract"):
        atype = (ab.get("abstract-type") or "").lower()
        if atype in ("graphical", "teaser", "graphical-abstract"):
            continue
        t = text_of(ab)
        if t:
            return t
    return ""


def pub_year(root):
    """优先 ppub/epub/collection 的年份。"""
    best = None
    for pd_el in root.findall(".//front//pub-date"):
        y = pd_el.findtext("year")
        if not y:
            continue
        ptype = (pd_el.get("pub-type") or pd_el.get("date-type") or "").lower()
        try:
            yi = int(y)
        except ValueError:
            continue
        # ppub/collection 更代表"发表年"，优先；否则先记下
        if ptype in ("ppub", "collection", "pub"):
            return yi
        if best is None:
            best = yi
    return best


def article_id(root, idtype):
    for aid in root.iter("article-id"):
        if aid.get("pub-id-type") == idtype:
            return (aid.text or "").strip()
    return ""


def journal_title(root):
    el = root.find(".//journal-meta//journal-title")
    return text_of(el)


def sec_titles(root):
    body = root.find(".//body")
    if body is None:
        return []
    return [text_of(t) for t in body.findall(".//sec/title") if text_of(t)]


def parse_all():
    rows = []
    n_fail = 0
    with tarfile.open(TARGZ, "r:gz") as tar:
        for m in tar:
            if not m.name.endswith(".xml"):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            try:
                root = etree.parse(f).getroot()
            except Exception:
                n_fail += 1
                continue
            title = text_of(root.find(".//title-group/article-title"))
            abstract = first_abstract(root)
            body = text_of(root.find(".//body"))
            stitles = sec_titles(root)
            joined = " ".join([title, abstract, body])
            rows.append({
                "file": m.name.split("/")[-1],
                "pmcid": article_id(root, "pmc"),
                "pmid": article_id(root, "pmid"),
                "doi": article_id(root, "doi"),
                "journal": journal_title(root),
                "pub_year": pub_year(root),
                "title": title,
                "abstract": abstract,
                "body_text": body,
                "title_chars": len(title),
                "abstract_chars": len(abstract),
                "body_chars": len(body),
                "n_sections": len(stitles),
                "sec_titles": " | ".join(stitles[:40]),
                "n_fffd": joined.count("�"),
            })
    return pd.DataFrame(rows), n_fail


def miss_rate(df, col):
    """空缺=NaN 或 空/纯空白字符串。兼容 pandas 3.0 的 str dtype。"""
    s = df[col]
    if pdt.is_numeric_dtype(s):
        m = s.isna()
    else:
        t = s.astype("string").str.strip()
        m = t.isna() | (t == "")
    return int(m.sum()), round(100 * m.mean(), 2)


def main():
    print("解析中（全量 oa_comm baseline 样本包）……")
    df, n_fail = parse_all()
    n = len(df)
    print(f"成功解析 {n} 篇，解析失败 {n_fail} 篇")

    # ---- 1. 字段完整性 ----
    fields = ["pmcid", "pmid", "doi", "journal", "pub_year",
              "title", "abstract", "body_text"]
    completeness = {}
    for c in fields:
        cnt, rate = miss_rate(df, c)
        completeness[c] = {"missing": cnt, "missing_rate_pct": rate,
                           "present": n - cnt}

    # ---- 2. 基础质量 ----
    short_abstract = int((df["abstract_chars"].between(1, 100)).sum())
    empty_body = int((df["body_chars"] == 0).sum())
    short_body = int((df["body_chars"].between(1, 500)).sum())
    enc_err = int((df["n_fffd"] > 0).sum())

    # ---- 3. 关键字段：元数据可行性 ----
    journals = df.loc[df["journal"].str.len() > 0, "journal"]
    top_journals = journals.value_counts().head(10)
    years = df["pub_year"].dropna().astype(int)
    year_min = int(years.min()) if len(years) else None
    year_max = int(years.max()) if len(years) else None
    last5 = int((years >= THIS_YEAR - 4).sum()) if len(years) else 0
    year_hist = {int(k): int(v) for k, v in
                 sorted(years.value_counts().items())}
    pmid_cov = round(100 * (df["pmid"].str.len() > 0).mean(), 2)

    # IMRaD 预览（为后续第2/4步铺垫）
    def is_imrad(s):
        s = (s or "").lower()
        return any(k in s for k in IMRAD_KEYS)
    imrad_docs = int(df["sec_titles"].apply(
        lambda s: any(is_imrad(t) for t in s.split(" | "))).sum())
    has_sections = int((df["n_sections"] > 0).sum())

    summary = {
        "n_docs": n, "n_parse_fail": n_fail,
        "completeness": completeness,
        "quality": {
            "short_abstract_1_100": short_abstract,
            "empty_body": empty_body,
            "short_body_1_500": short_body,
            "encoding_error_docs": enc_err,
        },
        "metadata": {
            "journal_distinct": int(journals.nunique()),
            "journal_top10": {k: int(v) for k, v in top_journals.items()},
            "year_min": year_min, "year_max": year_max,
            "year_last5_count": last5,
            "year_hist": year_hist,
            "pmid_coverage_pct": pmid_cov,
        },
        "structure_preview": {
            "docs_with_sections": has_sections,
            "docs_with_imrad_titles": imrad_docs,
            "median_sections": int(df["n_sections"].median()),
        },
        "length_glance_chars": {
            "abstract_median": int(df["abstract_chars"].median()),
            "body_median": int(df["body_chars"].median()),
            "body_p95": int(df["body_chars"].quantile(0.95)),
            "body_max": int(df["body_chars"].max()),
        },
    }

    df.to_parquet(OUT_PARQUET)
    (REPORT_DIR / "step1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 结果摘要 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n全量结构化表已存: {OUT_PARQUET}")
    print(f"报告数据已存: {REPORT_DIR / 'step1.json'}")


if __name__ == "__main__":
    main()
