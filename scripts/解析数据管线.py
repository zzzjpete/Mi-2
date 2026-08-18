"""
解析数据管线.py — 验证 PubMed oa_comm 数据的「下载 → 解析」管线。

阶段一末：证明数据可下载、可解析、可结构化，为阶段二的数据尽调铺路。

策略：baseline 包内含几十万个小 .xml，不全量解压；用 tarfile 流式读取前 N 篇，
用 lxml 解析 JATS XML 提取结构化字段（PMCID/标题/摘要/正文长度）；
并用 pandas 读取 filelist.csv 元数据。证明数据可下载、可解析、可结构化。
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import sys
import tarfile
from pathlib import Path
import pandas as pd
from lxml import etree

DATA_DIR = Path(os.path.join(ROOT, "data", "pubmed"))
STEM = "oa_comm_xml.PMC000xxxxxx.baseline.2026-06-18"
TARGZ = DATA_DIR / f"{STEM}.tar.gz"
FILELIST = DATA_DIR / f"{STEM}.filelist.csv"
N = 5


def text_of(el):
    return " ".join(el.itertext()).strip() if el is not None else ""


def parse_filelist():
    print("=== [1] filelist 元数据 (pandas) ===")
    df = pd.read_csv(FILELIST)
    print("文献条目数:", len(df))
    print("列名:", list(df.columns))
    print(df.head(3).to_string())
    print()
    return df


def parse_articles(n=N):
    print(f"=== [2] 流式解析前 {n} 篇 JATS XML (lxml) ===")
    rows = []
    with tarfile.open(TARGZ, "r:gz") as tar:
        count = 0
        for m in tar:
            if not m.name.endswith(".xml"):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            try:
                root = etree.parse(f).getroot()
            except Exception as e:
                print("  解析失败", m.name, e)
                continue
            pmcid = ""
            for aid in root.iter("article-id"):
                if aid.get("pub-id-type") == "pmc":
                    pmcid = (aid.text or "").strip()
            title = text_of(root.find(".//title-group/article-title"))
            abstract = text_of(root.find(".//abstract"))
            body_chars = len(text_of(root.find(".//body")))
            rows.append({
                "file": m.name.split("/")[-1],
                "pmcid": pmcid,
                "title": title[:70],
                "abstract_chars": len(abstract),
                "body_chars": body_chars,
            })
            count += 1
            if count >= n:
                break
    df = pd.DataFrame(rows)
    print(df.to_string())
    print()
    # 演示「可结构化落盘」：存一份样本，供后续切块/向量化使用
    out = DATA_DIR / "sample_parsed.parquet"
    df.to_parquet(out)
    print("结构化样本已保存:", out)
    return df


if __name__ == "__main__":
    if not TARGZ.exists():
        print("找不到数据包:", TARGZ)
        sys.exit(1)
    if FILELIST.exists():
        parse_filelist()
    else:
        print("(未找到 filelist.csv，跳过元数据步骤)\n")
    parse_articles()
    print("\n解析管线验证完成：成功从 tar.gz 流式提取结构化文献字段。")
