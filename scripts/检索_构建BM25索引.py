# -*- coding: utf-8 -*-
"""第六阶段 · 构建 BM25 关键词索引（多路检索的关键词一路）

从建库产物 merged_4m.parquet 读取 chunk_id + text，用 bm25s 建一个磁盘化、可内存
映射的 BM25 索引。chunk_id 与 4M Chroma 集合的 id 完全对齐，所以关键词命中可以直接
和向量命中按 id 融合，命中后的正文/元数据统一从 Chroma 取。

为什么是 bm25s 而不是 rank_bm25：rank_bm25 是纯 Python、全部驻留内存，4M 块要占
几十 GB，32GB 机器放不下；bm25s 把 BM25 分数预计算进 scipy 稀疏矩阵，可 save 到磁盘、
查询时 mmap 载入，几乎不额外占 RAM。

用法：
  # 先建 50 万子集（几分钟，用于端到端打通验证）
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_构建BM25索引.py --limit 500000 --out E:\\rag\\data\\bm25_index_500k
  # 验证无误后建全量
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_构建BM25索引.py --full --out E:\\rag\\data\\bm25_index_4m
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 按路径导入 BM25 分词公共约定（中文文件名）
_spec = importlib.util.spec_from_file_location(
    "bm25_common", os.path.join(os.path.dirname(__file__), "检索_BM25公共.py"))
_bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bc)
bm25_tokenize = _bc.bm25_tokenize
BM25_TOKENIZER_META = _bc.BM25_TOKENIZER_META
BM25_METHOD, BM25_K1, BM25_B = _bc.BM25_METHOD, _bc.BM25_K1, _bc.BM25_B

import bm25s

MERGED = os.path.join(ROOT, "data", "vectors", "merged_4m.parquet")


def load_corpus(path, limit, seed):
    """流式读取 chunk_id + text（列投影跳过 12GB 向量列），可按比例伯努利抽样。

    返回 (ids, texts)。抽样在全部 row group 上均匀发生，保住年份/期刊分布，
    不是简单取前 N 行（前 N 行都是早期 PMCID，偏老）。
    """
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    frac = 1.0 if not limit else min(1.0, limit / total)
    rng = np.random.default_rng(seed)
    print(f"[读取] 全量 {total:,} 行，目标 {'全量' if frac >= 1 else f'{limit:,}'}，"
          f"抽样比例 {frac:.4%}", flush=True)

    ids, texts = [], []
    t0 = time.time()
    seen = 0
    for batch in pf.iter_batches(batch_size=65536, columns=["chunk_id", "text"]):
        n = batch.num_rows
        seen += n
        cid = batch.column("chunk_id").to_pylist()
        txt = batch.column("text").to_pylist()
        if frac >= 1.0:
            ids.extend(cid)
            texts.extend(t or "" for t in txt)
        else:
            mask = rng.random(n) < frac
            for k in np.flatnonzero(mask):
                ids.append(cid[k])
                texts.append(txt[k] or "")
        if seen % (65536 * 10) == 0 or seen >= total:
            print(f"  扫描 {seen:,}/{total:,}，已收 {len(ids):,} 条 "
                  f"（{time.time() - t0:.0f}s）", flush=True)
    print(f"[读取] 完成，实收 {len(ids):,} 条，用时 {time.time() - t0:.0f}s", flush=True)
    return ids, texts


def build(args):
    os.makedirs(args.out, exist_ok=True)
    limit = None if args.full else args.limit

    ids, texts = load_corpus(args.merged, limit, args.seed)
    n = len(ids)
    if n == 0:
        raise SystemExit("没有读到任何文本，检查 merged parquet 路径")

    print(f"[分词] bm25s 分词 {n:,} 篇（{BM25_TOKENIZER_META['pipeline']}）...", flush=True)
    t0 = time.time()
    corpus_tokens = bm25_tokenize(texts, show_progress=True)
    print(f"[分词] 完成，用时 {time.time() - t0:.0f}s", flush=True)
    del texts  # 及时释放，4M 时正文占 ~6GB

    print(f"[索引] 建 BM25（method={BM25_METHOD}, k1={BM25_K1}, b={BM25_B}）...", flush=True)
    t0 = time.time()
    retriever = bm25s.BM25(method=BM25_METHOD, k1=BM25_K1, b=BM25_B)
    retriever.index(corpus_tokens, show_progress=True)
    print(f"[索引] 完成，用时 {time.time() - t0:.0f}s", flush=True)

    print(f"[保存] -> {args.out}", flush=True)
    retriever.save(args.out)  # 存 bm25s 索引本体（data/indices/vocab 等）

    # 位置 -> chunk_id 映射：检索命中的是行号，要还原成 id 才能跟向量结果融合、去 Chroma 取正文
    id_tbl = pa.table({"chunk_id": pa.array(ids, type=pa.string())})
    pq.write_table(id_tbl, os.path.join(args.out, "doc_ids.parquet"))

    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.merged,
        "n_docs": n,
        "full_corpus": bool(args.full),
        "sample_limit": None if args.full else args.limit,
        "seed": args.seed,
        "tokenizer": BM25_TOKENIZER_META,
        "note": "命中返回行号；doc_ids.parquet 给出行号->chunk_id；正文/元数据统一从 Chroma 按 id 取",
    }
    with open(os.path.join(args.out, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 落一个大小报告
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(args.out) for fn in fns)
    print(f"[完成] {n:,} 篇 BM25 索引 -> {args.out}  "
          f"（磁盘 {total_bytes / 1e9:.2f} GB）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=MERGED)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500000,
                    help="子集抽样目标条数（--full 时忽略）")
    ap.add_argument("--full", action="store_true", help="对全量 4M 建索引")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
