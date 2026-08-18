"""
步骤4_分割策略分析.py — 阶段二第四步：文本分割策略（对应任务第4点）

思路：第二步知道 ~70% 文章有 IMRaD 章节结构，第三步知道整篇远超 512。
这一步验证「按章节切」够不够——把正文按 JATS 顶层 <sec> 切开，量每个章节的 token，
看章节本身是不是也常常 > 512（若是，则需在章节内再做递归切兜底）。
然后模拟「章节感知 + 递归兜底」策略，给出最终 chunk 规模，并与「无脑定长切」对比。

产物：
  - E:\\rag\\report_data\\section_hist.png
  - E:\\rag\\report_data\\step4.json
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
import tarfile
from lxml import etree
from transformers import AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(os.path.join(ROOT, "data", "pubmed"))
REPORT = Path(os.path.join(ROOT, "report_data"))
STEM = "oa_comm_xml.PMC000xxxxxx.baseline.2026-06-18"
TARGZ = DATA / f"{STEM}.tar.gz"
IMG = REPORT / "section_hist.png"

SIZE, OVERLAP = 512, 64


def text_of(el):
    return " ".join(el.itertext()).strip() if el is not None else ""


def top_sections(root):
    """返回顶层 <sec> 的 (title, text)；无 sec 的文章整篇正文作为一个单元。"""
    body = root.find(".//body")
    if body is None:
        return []
    secs = body.findall("sec")  # 顶层直接子 sec
    if not secs:
        t = text_of(body)
        return [("(no-section)", t)] if t else []
    out = []
    for s in secs:
        title = text_of(s.find("title"))
        out.append((title or "(untitled)", text_of(s)))
    return out


def chunks_for(L, size=SIZE, overlap=OVERLAP):
    if L <= size:
        return 1
    return max(1, math.ceil((L - overlap) / (size - overlap)))


def main():
    print("加载 tokenizer…")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    print("重解析正文、抽取顶层章节…")
    sec_texts, sec_doc, sec_title = [], [], []
    doc_secmeta = {}  # file -> list of section token lens (填充后)
    files_order = []
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
                continue
            fn = m.name.split("/")[-1]
            files_order.append(fn)
            secs = top_sections(root)
            doc_secmeta[fn] = []
            for title, txt in secs:
                sec_texts.append(txt)
                sec_doc.append(fn)
                sec_title.append(title)

    print(f"共 {len(sec_texts)} 个章节，来自 {len(files_order)} 篇。分词计长…")
    lens = []
    B = 256
    for i in range(0, len(sec_texts), B):
        enc = tok(sec_texts[i:i + B], add_special_tokens=True,
                  truncation=False, return_length=True)
        lens.extend(enc["length"])
        print(f"  {min(i+B,len(sec_texts))}/{len(sec_texts)}", end="\r")
    print()

    sdf = pd.DataFrame({"file": sec_doc, "title": sec_title, "tok": lens})

    # ---- 章节级分布 ----
    st = sdf["tok"]
    sec_dist = {
        "n_sections": int(len(sdf)),
        "median": int(st.median()), "mean": int(st.mean()),
        "p90": int(st.quantile(.90)), "p95": int(st.quantile(.95)),
        "max": int(st.max()),
        "pct_le_512": round(100 * (st <= 512).mean(), 1),
        "pct_le_1024": round(100 * (st <= 1024).mean(), 1),
        "pct_over_512": round(100 * (st > 512).mean(), 1),
    }

    # ---- 策略模拟：章节感知 + 递归兜底(512/64) ----
    sdf["chunks"] = st.apply(chunks_for)
    total_sec_aware = int(sdf["chunks"].sum())
    sec_needing_split = int((st > SIZE).sum())
    per_doc = sdf.groupby("file")["chunks"].sum()
    strategy = {
        "config": f"{SIZE}/{OVERLAP}",
        "total_chunks": total_sec_aware,
        "mean_chunks_per_doc": round(per_doc.mean(), 1),
        "sections_fit_whole_pct": round(100 * (st <= SIZE).mean(), 1),
        "sections_needing_recursive": sec_needing_split,
    }

    # 对比：无脑对整篇定长切（用第三步的 token_len）
    tdf = pd.read_parquet(DATA / "parsed_tok.parquet")
    naive_total = int(tdf["token_len"].apply(chunks_for).sum())

    summary = {
        "section_distribution": sec_dist,
        "strategy_sim": strategy,
        "naive_wholedoc_total_chunks": naive_total,
        "img": str(IMG),
    }
    REPORT.mkdir(exist_ok=True)
    (REPORT / "step4.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图：章节 token 分布 ----
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(st.clip(upper=sec_dist["p95"]), bins=60,
            color="#B279A2", edgecolor="white", linewidth=.3)
    ax.axvline(512, color="#E45756", ls="--", lw=1.4, label="512 target")
    ax.axvline(sec_dist["median"], color="#54A24B", ls="--", lw=1.4,
               label=f"median {sec_dist['median']}")
    ax.set_title("Token length per section (top-level <sec>)", fontsize=10)
    ax.set_xlabel("tokens (clipped at p95)")
    ax.set_ylabel("# sections")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG, dpi=150)

    print("图已存:", IMG)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
