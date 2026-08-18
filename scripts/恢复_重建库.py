# -*- coding: utf-8 -*-
"""恢复_重建库.py — 事故恢复第 2 步（上一步 `恢复_合并向量元数据.py`）。

从 merged_4m.parquet(含预计算向量)重建一个干净的 Chroma 库。约 4.5 小时。
不重嵌入、不用 GPU。**请在你自己的终端窗口运行**（放后台跑的长任务会被系统回收）。

用法：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\恢复_重建库.py
  若中途 Windows 重启，续跑加 --resume：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\恢复_重建库.py --resume
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os, sys, json, time, argparse
from datetime import datetime
sys.stdout.reconfigure(encoding="utf-8")
import pyarrow.parquet as pq
import numpy as np
import chromadb

MERGED = os.path.join(ROOT, "data", "vectors", "merged_4m.parquet")
CHROMA = os.path.join(ROOT, "data", "chroma_db_4m")     # 全新干净目录，不碰坏掉的 chroma_db
COLL = "medrag_bge_base"
META_FIELDS = ["doc_id", "chunk_index", "total_chunks", "source_title", "token_count",
               "section", "pmcid", "pmid", "journal", "pub_year"]
DIM = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="断点续跑：不删库，从已有条数继续")
    ap.add_argument("--add-batch", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(CHROMA, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA)
    exists = COLL in [c.name for c in client.list_collections()]
    start_offset = 0
    if args.resume and exists:
        col = client.get_collection(COLL)
        start_offset = col.count()
        print(f"[续跑] 已有 {start_offset:,} 条，从第 {start_offset:,} 行继续（不删库）", flush=True)
    else:
        if exists:
            print(f"[库] 已存在集合 {COLL}，删除重建", flush=True)
            client.delete_collection(COLL)
        col = client.create_collection(
            name=COLL, metadata={"hnsw:space": "cosine", "embedding_model": "BAAI/bge-base-en-v1.5"})

    pf = pq.ParquetFile(MERGED)
    total = pf.metadata.num_rows
    tok_sum = tok_max = tok_min = None
    done = start_offset
    buf = {"ids": [], "emb": [], "docs": [], "metas": []}
    t0e = time.time()

    def flush():
        nonlocal done
        if not buf["ids"]:
            return
        col.add(ids=buf["ids"], embeddings=buf["emb"], documents=buf["docs"], metadatas=buf["metas"])
        done += len(buf["ids"])
        rate = (done - start_offset) / (time.time() - t0e + 1e-9)
        print(f"  入库 {done:,}/{total:,} ({rate:.0f} 块/s)", flush=True)
        for k in buf:
            buf[k].clear()

    print(f"[库] 从预计算向量重建：{total:,} 块 -> {COLL} @ {CHROMA}", flush=True)
    row_idx = 0
    for b in pf.iter_batches(batch_size=args.add_batch):
        d = b.to_pydict()
        for i in range(b.num_rows):
            tc = d["token_count"][i]
            tok_sum = (tok_sum or 0) + tc
            tok_max = tc if tok_max is None else max(tok_max, tc)
            tok_min = tc if tok_min is None else min(tok_min, tc)
            if row_idx < start_offset:
                row_idx += 1
                continue
            buf["ids"].append(d["chunk_id"][i])
            buf["emb"].append(np.frombuffer(d["vector"][i], dtype=np.float32).tolist())
            buf["docs"].append(d["text"][i])
            meta = {}
            for k in META_FIELDS:
                v = d[k][i]
                meta[k] = "" if v is None else (v if isinstance(v, (int, float)) else str(v))
            buf["metas"].append(meta)
            row_idx += 1
        flush()

    built = col.count()
    stats = {
        "collection_name": COLL, "total_chunks": built,
        "embedding_model": "BAAI/bge-base-en-v1.5", "embedding_dimension": DIM,
        "index_built_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_size_stats": {"mean": round(tok_sum / built, 2) if built else None,
                             "max": tok_max, "min": tok_min},
        "metadata_fields": META_FIELDS, "distance_metric": "cosine",
        "note": "从损坏 Chroma 段恢复的预计算向量重建（未重嵌入）",
        "chroma_path": CHROMA, "build_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.join(ROOT, "report_data"), exist_ok=True)
    sp = os.path.join(ROOT, "report_data", "向量库统计_medrag_bge_base.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 索引 {built:,} 块 | 用时 {stats['build_seconds']}s | 统计 -> {sp}", flush=True)


if __name__ == "__main__":
    main()
