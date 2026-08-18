# -*- coding: utf-8 -*-
"""恢复_导出向量.py — 事故恢复第 0 步：从损坏的 Chroma HNSW 段直接抢出向量。

阶段四向量库被回收进程杀坏后写的三步恢复链的头一步
（导出 → `恢复_合并向量元数据.py` → `恢复_重建库.py`）。
从 data_level0.bin 直接解析出 399.8 万条向量 -> 干净 parquet。
绕过会段错误的 Chroma 内核；向量数据本身完整(已验证范数=1)。
输出: data/vectors/recovered_vectors.parquet (chunk_id, vector[binary 3072B fp32])
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import sys, os, pickle
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEG = os.path.join(ROOT, "data", "chroma_db", "192b2cae-46d5-4bf9-a21e-a12ca70fc33a")
OUT = os.path.join(ROOT, "data", "vectors", "recovered_vectors.parquet")
ESZ, DIM = 3212, 768
VOFF, VEND, LOFF = 132, 132 + DIM * 4, 3204   # [132 链接][3072 向量][8 标签]

with open(os.path.join(SEG, "index_metadata.pickle"), "rb") as f:
    meta = pickle.load(f)
label_to_id = meta["label_to_id"]
N = int(meta["total_elements_added"])
print(f"待恢复 {N:,} 条向量", flush=True)

schema = pa.schema([("chunk_id", pa.string()), ("vector", pa.binary(DIM * 4))])
writer = pq.ParquetWriter(OUT, schema)
CH = 200_000
done = missing = bad_norm = 0
with open(os.path.join(SEG, "data_level0.bin"), "rb") as f:
    for start in range(0, N, CH):
        n = min(CH, N - start)
        arr = np.frombuffer(f.read(ESZ * n), dtype=np.uint8).reshape(n, ESZ)
        labels = np.ascontiguousarray(arr[:, LOFF:LOFF + 8]).view(np.uint64).reshape(n)
        ids, blobs = [], []
        for i in range(n):
            cid = label_to_id.get(int(labels[i]))
            if cid is None:
                missing += 1
                continue
            ids.append(cid)
            blobs.append(arr[i, VOFF:VEND].tobytes())
        writer.write_table(pa.table({"chunk_id": pa.array(ids, pa.string()),
                                     "vector": pa.array(blobs, pa.binary(DIM * 4))}))
        done += len(ids)
        print(f"  已恢复 {done:,}/{N:,} (缺失映射 {missing})", flush=True)
writer.close()
print(f"[完成] 恢复 {done:,} 条 -> {OUT}", flush=True)
