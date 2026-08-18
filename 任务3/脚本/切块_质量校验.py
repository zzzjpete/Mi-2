"""
切块_质量校验.py — 对切块产物做统计 + 质量抽样校验（对应任务第 4、5 点）

规模可达 ~9200 万块，故采用**按包流式聚合**（一次只读一个 parquet 的必要列），
避免把全部 chunk 读进内存。产出：
  report_data/chunking_stats.json
  report_data/切块质量报告.md

统计：总块数 / 每篇块数 / token 分布 / 整篇 vs 拆分 / 每篇块数分档 / 章节分布
质量：超模型限制 / 超 chunk_size / 空标题率 / 超短块 / doc_id 唯一性 / 行数完整性
深度抽查（多块文献）：截断（句子边界）+ 同章节相邻块 token 重叠 ≈ overlap
用法：
  python 切块_质量校验.py                 # 全量校验 data/chunks
  python 切块_质量校验.py --sample-docs 500 --deep-files 2
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT_PATH as ROOT

import os
import json
import argparse
import random
from collections import Counter
from pathlib import Path

os.environ["HF_HOME"] = str(ROOT / "hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as pds
from transformers import AutoTokenizer

CHUNK_DIR = ROOT / "data" / "chunks"
REPORT = ROOT / "report_data"
CHUNK_SIZE = 512
OVERLAP = 64
MODEL_LIMIT = 8192

_tok = None


def get_tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _tok


def pctl(a, q):
    return int(np.percentile(a, q))


def token_overlap(a, b, maxk=200):
    tok = get_tok()
    ai = tok.encode(a, add_special_tokens=False)[-maxk:]
    bi = tok.encode(b, add_special_tokens=False)[:maxk]
    for k in range(min(len(ai), len(bi)), 0, -1):
        if ai[-k:] == bi[:k]:
            return k
    return 0


SENT_END = tuple(list(".?!;") + ['."', '.”', ')', ']'])


def global_pass(files):
    """按包流式聚合全局统计（内存受控）。"""
    tok_parts = []              # 各包 token_count（int32）
    pdt_parts = []             # 各包每篇 total_chunks（chunk_index==0 行）
    title_empty_parts = []      # 各包每篇 source_title 是否空（对齐 pdt）
    doc_ids = set()
    section_counter = Counter()
    n_chunk = 0

    for f in files:
        tt = pq.read_table(f, columns=["token_count"])
        tc = tt.column("token_count").to_numpy()
        tok_parts.append(tc)
        n_chunk += len(tc)

        t2 = pq.read_table(f, columns=["chunk_index", "total_chunks",
                                       "source_title", "doc_id"])
        mask0 = pc.equal(t2.column("chunk_index"), 0)
        tot = pc.filter(t2.column("total_chunks"), mask0).to_numpy()
        pdt_parts.append(tot)
        st = pc.filter(t2.column("source_title"), mask0)
        st_empty = pc.equal(pc.utf8_length(pc.utf8_trim_whitespace(st)), 0).to_numpy(
            zero_copy_only=False)
        title_empty_parts.append(st_empty)
        dids = pc.filter(t2.column("doc_id"), mask0).to_pylist()
        doc_ids.update(dids)

        sc = pq.read_table(f, columns=["section"]).column("section")
        vc = pc.value_counts(sc)
        for pair in vc:
            section_counter[pair["values"].as_py()] += pair["counts"].as_py()
        print(f"  聚合 {f.name} 完成", flush=True)

    tok_all = np.concatenate(tok_parts)
    per_doc_total = np.concatenate(pdt_parts)
    title_empty = np.concatenate(title_empty_parts)
    n_doc = len(per_doc_total)
    return dict(tok_all=tok_all, per_doc_total=per_doc_total,
                title_empty=title_empty, doc_ids=doc_ids,
                section_counter=section_counter, n_chunk=n_chunk, n_doc=n_doc)


def deep_sample(files, deep_files, sample_docs, rng):
    """从若干包里抽多块文献，查截断 + 同章节相邻块重叠。"""
    chosen = files[:deep_files] if deep_files <= len(files) else files
    ds = pds.dataset([str(f) for f in chosen], format="parquet")
    idx0 = ds.to_table(columns=["doc_id", "chunk_index", "total_chunks"],
                       filter=(pds.field("chunk_index") == 0) &
                              (pds.field("total_chunks") > 1)).to_pandas()
    pool = idx0["doc_id"].tolist()
    picks = rng.sample(pool, min(sample_docs, len(pool)))
    tbl = ds.to_table(
        filter=pds.field("doc_id").isin(picks),
        columns=["doc_id", "chunk_index", "total_chunks", "section",
                 "token_count", "text"]).to_pandas().sort_values(["doc_id", "chunk_index"])

    ends_sent = n_text = garbage = 0
    overlaps, cross = [], []
    previews = []
    for did, g in tbl.groupby("doc_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            t = g.loc[i, "text"] or ""
            n_text += 1
            if "�" in t:
                garbage += 1
            if t.rstrip().endswith(SENT_END):
                ends_sent += 1
            if i + 1 < len(g):
                ov = token_overlap(g.loc[i, "text"], g.loc[i + 1, "text"])
                (overlaps if g.loc[i, "section"] == g.loc[i + 1, "section"]
                 else cross).append(ov)
        if len(previews) < 3:
            previews.append({"doc_id": did,
                             "total_chunks": int(g["total_chunks"].iloc[0]),
                             "sections": g["section"].tolist(),
                             "token_counts": g["token_count"].tolist()})
    return dict(
        deep_files=[f.name for f in chosen],
        sampled_multi_chunk_docs=len(picks), sampled_chunks=n_text,
        garbage_char_chunks=garbage,
        ends_at_sentence_boundary_pct=round(100 * ends_sent / n_text, 1) if n_text else None,
        same_section_adjacent_pairs=len(overlaps),
        overlap_tokens_median=int(np.median(overlaps)) if overlaps else None,
        overlap_tokens_mean=round(float(np.mean(overlaps)), 1) if overlaps else None,
        overlap_present_ge32tok_pct=round(100 * np.mean([o >= 32 for o in overlaps]), 1) if overlaps else None,
        cross_section_adjacent_pairs=len(cross),
        cross_section_overlap_median=int(np.median(cross)) if cross else None,
    ), previews


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-docs", type=int, default=500)
    ap.add_argument("--deep-files", type=int, default=2)
    ap.add_argument("--glob", default="chunks_*.parquet")
    args = ap.parse_args()

    files = sorted(CHUNK_DIR.glob(args.glob))
    if not files:
        raise SystemExit(f"没有找到 {CHUNK_DIR / args.glob}")
    print(f"校验 {len(files)} 个 parquet")

    G = global_pass(files)
    tc = G["tok_all"]
    n_chunk, n_doc = G["n_chunk"], G["n_doc"]
    per_doc = G["per_doc_total"]
    whole = int((per_doc == 1).sum())

    tok_dist = {"min": int(tc.min()), "p5": pctl(tc, 5), "p25": pctl(tc, 25),
                "median": int(np.median(tc)), "mean": round(float(tc.mean()), 1),
                "p75": pctl(tc, 75), "p95": pctl(tc, 95), "p99": pctl(tc, 99),
                "max": int(tc.max())}

    def bucket(n):
        return ("1(整篇)" if n == 1 else "2-5" if n <= 5 else "6-20"
                if n <= 20 else "21-50" if n <= 50 else ">50")
    bnames, bcounts = np.unique([bucket(n) for n in per_doc], return_counts=True)
    dist_buckets = {k: int(v) for k, v in zip(bnames, bcounts)}
    top_sections = dict(G["section_counter"].most_common(15))

    present_title_chunks = int(per_doc[~G["title_empty"]].sum())
    stats = {
        "files": [f.name for f in files],
        "chunk_size": CHUNK_SIZE, "chunk_overlap": OVERLAP,
        "original_documents": n_doc, "total_chunks": n_chunk,
        "chunks_per_doc": round(n_chunk / n_doc, 2),
        "whole_doc_docs": whole, "split_docs": n_doc - whole,
        "whole_doc_pct": round(100 * whole / n_doc, 1),
        "token_count_distribution": tok_dist,
        "chunks_per_doc_buckets": dist_buckets,
        "top_sections": top_sections,
    }
    quality = {
        "duplicate_doc_ids": n_doc - len(G["doc_ids"]),
        "row_count_integrity_ok": bool(int(per_doc.sum()) == n_chunk),
        "chunks_over_model_limit_8192": int((tc > MODEL_LIMIT).sum()),
        "chunks_over_chunk_size_512": int((tc > CHUNK_SIZE).sum()),
        "empty_source_title_chunks": n_chunk - present_title_chunks,
        "source_title_present_pct": round(100 * present_title_chunks / n_chunk, 3),
        "tiny_chunks_lt10tok": int((tc < 10).sum()),
        "tiny_chunks_pct": round(100 * float((tc < 10).mean()), 2),
    }

    rng = random.Random(42)
    deep, previews = deep_sample(files, args.deep_files, args.sample_docs, rng)

    result = {"stats": stats, "quality": quality, "deep_sample": deep,
              "previews": previews}
    REPORT.mkdir(exist_ok=True)
    (REPORT / "chunking_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result)
    print(json.dumps({"stats": stats, "quality": quality, "deep_sample": deep},
                     ensure_ascii=False, indent=1))
    print(f"\n已写：{REPORT / 'chunking_stats.json'}  &  {REPORT / '切块质量报告.md'}")


def write_markdown(r):
    s, q, d = r["stats"], r["quality"], r["deep_sample"]
    td = s["token_count_distribution"]
    m = []
    m.append("# 切块质量与统计报告（全量 oa_comm）\n")
    m.append(f"> 数据源：PubMed oa_comm baseline 2026-06-18（PMC000–010）| "
             f"chunk_size={s['chunk_size']} / overlap={s['chunk_overlap']} | 分词器 bge-m3\n")
    m.append("## 一、规模统计\n")
    m.append("| 指标 | 值 |")
    m.append("|---|---|")
    m.append(f"| 原始文献数 | {s['original_documents']:,} |")
    m.append(f"| 文本块总数 | {s['total_chunks']:,} |")
    m.append(f"| 平均每篇块数 | {s['chunks_per_doc']} |")
    m.append(f"| 整篇不分割文献 | {s['whole_doc_docs']:,}（{s['whole_doc_pct']}%）|")
    m.append(f"| 需拆分文献 | {s['split_docs']:,} |\n")
    m.append("## 二、块 token 长度分布\n")
    m.append("| min | p5 | p25 | 中位 | 均值 | p75 | p95 | p99 | max |")
    m.append("|---|---|---|---|---|---|---|---|---|")
    m.append(f"| {td['min']} | {td['p5']} | {td['p25']} | {td['median']} | {td['mean']} "
             f"| {td['p75']} | {td['p95']} | {td['p99']} | {td['max']} |\n")
    m.append("**每篇块数分档**：" +
             " · ".join(f"{k}: {v:,}" for k, v in s["chunks_per_doc_buckets"].items()) + "\n")
    m.append("**Top 章节**：" +
             " · ".join(f"{k}({v:,})" for k, v in list(s["top_sections"].items())[:10]) + "\n")
    m.append("## 三、质量校验\n")
    m.append("| 检查项 | 结果 |")
    m.append("|---|---|")
    m.append(f"| doc_id 重复数 | {q['duplicate_doc_ids']} |")
    m.append(f"| 行数完整性(Σtotal_chunks==总块数) | {'✅ 通过' if q['row_count_integrity_ok'] else '❌'} |")
    m.append(f"| 超模型上限(>8192 tok) | {q['chunks_over_model_limit_8192']} |")
    m.append(f"| 超 chunk_size(>512 tok) | {q['chunks_over_chunk_size_512']} |")
    m.append(f"| source_title 完整率 | {q['source_title_present_pct']}% |")
    m.append(f"| 超短块(<10 tok) | {q['tiny_chunks_lt10tok']:,}（{q['tiny_chunks_pct']}%）|\n")
    m.append("## 四、多块文献深度抽查\n")
    m.append(f"- 抽查来源包：{', '.join(d['deep_files'])}\n")
    m.append(f"- 抽查多块文献 **{d['sampled_multi_chunk_docs']}** 篇 / **{d['sampled_chunks']:,}** 块\n")
    m.append(f"- 乱码块：{d['garbage_char_chunks']}；结尾落在句子边界：{d['ends_at_sentence_boundary_pct']}%\n")
    m.append(f"- **同章节相邻块 token 重叠**：中位 {d['overlap_tokens_median']} / "
             f"均值 {d['overlap_tokens_mean']}（目标 {OVERLAP}）；有效重叠(≥32)占 "
             f"{d['overlap_present_ge32tok_pct']}%\n")
    m.append(f"- 跨章节相邻块重叠中位：{d['cross_section_overlap_median']}（设计上应≈0）\n")
    (REPORT / "切块质量报告.md").write_text("\n".join(m), encoding="utf-8")


if __name__ == "__main__":
    main()
