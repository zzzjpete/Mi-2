"""
切块_修复ID.py — 把已生成的切块产物按 pmcid 重新生成 doc_id / chunk_id，保证全库唯一。

背景：首轮 doc_id 用 pmid 优先，但 pmid 并非唯一（勘误记录与原文共用 pmid），
导致 ~70 篇 doc_id 冲突、chunk_id 重复。pmcid 是 PMC 唯一登录号（100% 完整、全库唯一）。
本脚本不重新切块（每块已带 pmcid），只按 batch 流式改写两列：
  doc_id  = pmcid（pmcid 为空时保留原 doc_id 兜底）
  chunk_id = doc_id            (total_chunks==1)
           = f"{doc_id}#{chunk_index}"  (拆分块)
其余列不动。写临时文件后原子改名。

输入 / 输出：原地改写 data/chunks/*.parquet（先写临时文件再原子改名）。

用法::

    & $py scripts\切块_修复ID.py
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT_PATH as ROOT

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

CHUNK_DIR = ROOT / "data" / "chunks"
BATCH = 200_000


def fix_file(f):
    tmp = f.with_name(f.name + ".tmp")
    n = 0
    # 必须在 replace 前释放读句柄，否则 Windows 会 WinError 5（占用中无法覆盖）
    with pq.ParquetFile(f) as pf:
        writer = pq.ParquetWriter(str(tmp), pf.schema_arrow, compression="zstd")
        for batch in pf.iter_batches(batch_size=BATCH):
            cols = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
            pmcid = cols["pmcid"]
            old_doc = cols["doc_id"]
            pmcid_empty = pc.equal(pc.utf8_length(pc.cast(pmcid, pa.string())), 0)
            new_doc = pc.if_else(pmcid_empty, old_doc, pmcid)
            idx_str = pc.cast(cols["chunk_index"], pa.string())
            combined = pc.binary_join_element_wise(new_doc, idx_str, "#")
            is_whole = pc.equal(cols["total_chunks"], 1)
            new_chunk_id = pc.if_else(is_whole, new_doc, combined)
            cols["doc_id"] = new_doc
            cols["chunk_id"] = new_chunk_id
            new_batch = pa.record_batch([cols[nm] for nm in batch.schema.names],
                                        schema=batch.schema)
            writer.write_batch(new_batch)
            n += batch.num_rows
        writer.close()
    tmp.replace(f)
    return n


def main():
    files = sorted(CHUNK_DIR.glob("chunks_PMC*.parquet"))
    for f in files:
        n = fix_file(f)
        print(f"[fixed] {f.name}: {n:,} 行", flush=True)
    print("ALL FIXED")


if __name__ == "__main__":
    main()
