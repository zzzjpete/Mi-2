# -*- coding: utf-8 -*-
"""恢复_合并向量元数据.py — 事故恢复第 1 步（上一步 `恢复_导出向量.py`，下一步 `恢复_重建库.py`）。

把恢复的向量(recovered_vectors.parquet)按 chunk_id 合并回子集的原文+元数据
(subset_4000000_s42.parquet) -> merged_4m.parquet。不涉及 Chroma。
内存策略：向量 dict 常驻(~12GB)，流式扫子集写出，峰值 ~14GB。
输出列：subset 全列 + vector(binary 3072B)
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import sys
sys.stdout.reconfigure(encoding="utf-8")
import pyarrow as pa
import pyarrow.parquet as pq

REC = os.path.join(ROOT, "data", "vectors", "recovered_vectors.parquet")
SUB = os.path.join(ROOT, "data", "vectors", "subset_4000000_s42.parquet")
OUT = os.path.join(ROOT, "data", "vectors", "merged_4m.parquet")

print("载入恢复向量到内存 dict ...", flush=True)
id2vec = {}
for b in pq.ParquetFile(REC).iter_batches(batch_size=200_000, columns=["chunk_id", "vector"]):
    d = b.to_pydict()
    for cid, v in zip(d["chunk_id"], d["vector"]):
        id2vec[cid] = v
print(f"  向量 dict 就绪：{len(id2vec):,} 条", flush=True)

sub_pf = pq.ParquetFile(SUB)
out_schema = sub_pf.schema_arrow.append(pa.field("vector", pa.binary(3072)))
writer = pq.ParquetWriter(OUT, out_schema)
kept = skipped = 0
for b in sub_pf.iter_batches(batch_size=50_000):
    d = b.to_pydict()
    cids = d["chunk_id"]
    mask = [c in id2vec for c in cids]
    if not any(mask):
        continue
    idx = [i for i, m in enumerate(mask) if m]
    cols = {name: [d[name][i] for i in idx] for name in d}
    cols["vector"] = [id2vec[cids[i]] for i in idx]
    writer.write_table(pa.table(cols, schema=out_schema))
    kept += len(idx)
    skipped += len(cids) - len(idx)
print(f"  已合并 {kept:,}（跳过无向量 {skipped:,}）", flush=True) if kept % 500000 < 50000 else None
writer.close()
print(f"[完成] 合并 {kept:,} 条（无向量跳过 {skipped:,}）-> {OUT}", flush=True)
