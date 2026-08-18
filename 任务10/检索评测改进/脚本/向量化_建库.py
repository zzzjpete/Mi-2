# -*- coding: utf-8 -*-
"""
第四阶段 · 向量化与索引构建（建库）

输入：data/chunks/chunks_PMC000..010.parquet（第三阶段切块产物，9243 万块）
流程：分层随机抽样子集 → BGE 批量生成文档向量 → 灌入 ChromaDB 持久化余弦索引 → 保存统计

用法（用 conda medrag 的 python 跑）：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\向量化_建库.py \
      --model bge-base --target-chunks 1500000 --seed 42

坑（见 rag-env-gotchas）：
  * pyarrow 必须在 torch 之前 import，否则 Windows 上 DLL 冲突段错误。
  * HF_HOME 用户级持久值指向旧路径，脚本里硬覆盖到 E:\\rag\\hf-cache。
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")          # 硬覆盖，早于任何 HF 导入
os.environ["HF_HUB_OFFLINE"] = "1"                   # 权重已缓存，离线加载
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import glob
import json
import time
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq                          # 必须早于 torch
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb

# ---- 模型登记：key -> (HF 名, 维度) ----
MODELS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", 384),
    "bge-base":  ("BAAI/bge-base-en-v1.5", 768),
    "bge-large": ("BAAI/bge-large-en-v1.5", 1024),
    "bge-m3":    ("BAAI/bge-m3", 1024),
}
# BGE 检索：查询加指令前缀效果更好；文档不加
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CHUNKS_GLOB = os.path.join(ROOT, "data", "chunks", "chunks_PMC*.parquet")
META_FIELDS = ["doc_id", "chunk_index", "total_chunks", "source_title",
               "token_count", "section", "pmcid", "pmid", "journal", "pub_year"]


def pick_device(prefer="cuda"):
    """挑一个真实可用的推理设备：cuda → mps → cpu。

    ⚠ 原来写的是 `device if torch.cuda.is_available() else "cpu"`。在 Mac 上
    `cuda.is_available()` 恒为假，于是**直接掉到 CPU，Metal 永远用不上**——
    不报错、不告警，只是慢，而"慢"很容易被当成"Mac 本来就慢"接受下来。

    ⚠ **mps 上必须用 float32**：`fp16` 那条分支原本只在 cuda 下走，这里保持同样口径。
    半精度在 mps 上对 BGE 这类模型会引入可见的数值偏差，而本项目的检索分数是要
    跨机器对照的——省那点显存不值得。

    `检索_多路检索.py` 的重排器也用这个函数，**设备选择只有一个来源**。
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEEmbedder:
    """BGE 嵌入模型：CLS pooling + L2 归一化。文档与查询分别走 embed_documents / embed_query。

    ⚠ **新建任何 collection 都必须复用这个类，不许换成 chromadb 自带的
    `SentenceTransformerEmbeddingFunction`。**

    本库的约定是三条一起成立的：CLS pooling、L2 归一化、**文档不加前缀而查询加
    `QUERY_INSTRUCTION` 前缀**。换一个包装器就等于换一套池化与归一化约定，
    于是两个 collection 的余弦分**不再可比**——而多路融合正是拿它们直接比大小。

    这类错**不抛异常、不报警、不体现在任何断言上**：库建得成、查得动、分数看着也正常，
    只是排序悄悄失去意义，等发现时往往已经在它上面做了一堆调参和结论。
    建 collection 时**只传 `embeddings=`，不要挂 embedding_function**。

    ⚠ 反向的一条：检索侧若已经自己加过前缀，就直接走 `_encode`，
    不能再调 `embed_query`（会二次加前缀）。
    """

    def __init__(self, model_key, device="cuda", fp16=True, max_length=512):
        name, dim = MODELS[model_key]
        self.key, self.name, self.dim, self.max_length = model_key, name, dim, max_length
        self.device = pick_device(device)
        print(f"[模型] 加载 {name} (dim={dim}) 到 {self.device} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.trunc_hits = 0        # 累计：分词后长度达到 max_length 的条数（截断上界）

    @torch.inference_mode()
    def _encode(self, texts, batch_size, count_trunc=False):
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            b = texts[i:i + batch_size]
            enc = self.tok(b, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt")
            if count_trunc:
                # attention_mask 每行和 == max_length → 该条被填满/截断（截断比例的上界）
                self.trunc_hits += int((enc["attention_mask"].sum(1) >= self.max_length).sum())
            enc = {k: v.to(self.device) for k, v in enc.items()}
            vecs = self.model(**enc).last_hidden_state[:, 0]     # CLS = BGE 句向量
            vecs = torch.nn.functional.normalize(vecs, dim=-1)
            out[i:i + len(b)] = vecs.float().cpu().numpy()
        return out

    def embed_documents(self, texts, batch_size=128):
        return self._encode(texts, batch_size, count_trunc=True)

    def embed_query(self, query, batch_size=1):
        return self._encode([QUERY_INSTRUCTION + query], batch_size)[0]


def stratified_sample(target, seed, cache_path):
    """流式扫全部包，按统一比例伯努利抽样，写 cache_path（分层：跨全部 11 包，保年份/期刊分布）。"""
    files = sorted(glob.glob(CHUNKS_GLOB))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    frac = min(1.0, target / total)
    print(f"[抽样] 全量 {total:,} 块，目标 {target:,}，抽样比例 {frac:.4%}，seed={seed}", flush=True)

    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    schema = pq.ParquetFile(files[0]).schema_arrow
    writer = pq.ParquetWriter(cache_path, schema)
    kept_total, per_pkg = 0, {}
    try:
        for f in files:
            pkg = os.path.basename(f).replace("chunks_", "").replace(".parquet", "")
            kept_pkg = 0
            for batch in pq.ParquetFile(f).iter_batches(batch_size=65536):
                n = batch.num_rows
                mask = rng.random(n) < frac
                if mask.any():
                    idx = pa.array(np.flatnonzero(mask))
                    sub = batch.take(idx)
                    writer.write_table(pa.Table.from_batches([sub]))
                    kept_pkg += len(idx)
            per_pkg[pkg] = kept_pkg
            kept_total += kept_pkg
            print(f"  {pkg}: 抽出 {kept_pkg:,}", flush=True)
    finally:
        writer.close()
    print(f"[抽样] 完成，共 {kept_total:,} 块 -> {cache_path}", flush=True)
    return {"total_corpus_chunks": total, "sample_fraction": frac,
            "sampled_chunks": kept_total, "per_package": per_pkg}


def build(args):
    t0 = time.time()
    cache = args.subset_cache or os.path.join(
        ROOT, "data", "vectors", f"subset_{args.target_chunks}_s{args.seed}.parquet")

    # 1) 子集：有缓存直接用，否则抽样
    if os.path.exists(cache):
        n = pq.ParquetFile(cache).metadata.num_rows
        print(f"[抽样] 复用已有子集 {cache}（{n:,} 块）", flush=True)
        sample_info = {"sampled_chunks": n, "reused_cache": True}
    else:
        sample_info = stratified_sample(args.target_chunks, args.seed, cache)

    # 2) 模型
    embedder = BGEEmbedder(args.model, fp16=not args.fp32)

    # 3) Chroma 持久化集合（余弦）；--resume 时不删、续建
    os.makedirs(args.chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=args.chroma_path)
    exists = args.collection in [c.name for c in client.list_collections()]
    start_offset = 0
    if args.resume and exists:
        col = client.get_collection(args.collection)
        start_offset = col.count()
        print(f"[续跑] 集合 {args.collection} 已有 {start_offset:,} 条，从第 {start_offset:,} 行继续（不删库）", flush=True)
    else:
        if exists:
            print(f"[库] 已存在集合 {args.collection}，先删除重建", flush=True)
            client.delete_collection(args.collection)
        col = client.create_collection(
            name=args.collection,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.name},
        )

    # 4) 流式：读子集 -> 批量向量 -> 批量入库
    pf = pq.ParquetFile(cache)
    total = pf.metadata.num_rows
    tok_sum = tok_max = tok_min = None
    done = start_offset                     # 续跑时从已有条数起算
    add_bs = args.add_batch                # Chroma 一次 add 的行数
    buf = {"ids": [], "texts": [], "metas": []}

    def flush():
        nonlocal done
        if not buf["ids"]:
            return
        vecs = embedder.embed_documents(buf["texts"], batch_size=args.batch_size)
        col.add(ids=buf["ids"], embeddings=vecs.tolist(),
                documents=buf["texts"], metadatas=buf["metas"])
        done += len(buf["ids"])
        rate = (done - start_offset) / (time.time() - t0_embed)
        print(f"  入库 {done:,}/{total:,}  ({rate:.0f} 块/s, 累计截断上界 {embedder.trunc_hits:,})", flush=True)
        buf["ids"].clear(); buf["texts"].clear(); buf["metas"].clear()

    print(f"[库] 开始建库：{total:,} 块 -> 集合 {args.collection}", flush=True)
    t0_embed = time.time()
    row_idx = 0
    for batch in pf.iter_batches(batch_size=add_bs):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            tc = d["token_count"][i]        # token 统计对所有行累计（含跳过的）
            tok_sum = (tok_sum or 0) + tc
            tok_max = tc if tok_max is None else max(tok_max, tc)
            tok_min = tc if tok_min is None else min(tok_min, tc)
            if row_idx < start_offset:      # 续跑：跳过已入库的前 start_offset 行
                row_idx += 1
                continue
            buf["ids"].append(d["chunk_id"][i])
            buf["texts"].append(d["text"][i])
            meta = {}
            for k in META_FIELDS:
                v = d[k][i]
                meta[k] = "" if v is None else (v if isinstance(v, (int, float)) else str(v))
            buf["metas"].append(meta)
            row_idx += 1
        flush()

    # 5) 统计
    built_n = col.count()
    stats = {
        "collection_name": args.collection,
        "total_chunks": built_n,
        "embedding_model": embedder.name,
        "embedding_dimension": embedder.dim,
        "index_built_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_size_stats": {
            "mean": round(tok_sum / built_n, 2) if built_n else None,
            "max": tok_max,
            "min": tok_min,
        },
        "metadata_fields": META_FIELDS,
        # --- 扩展信息（非需求模板要求，便于溯源/复现）---
        "distance_metric": "cosine",
        "query_instruction": QUERY_INSTRUCTION,
        "bert_truncated_upper_bound": embedder.trunc_hits,
        "bert_truncated_pct": round(100 * embedder.trunc_hits / built_n, 3) if built_n else None,
        "subset": sample_info,
        "subset_cache": cache,
        "seed": args.seed,
        "chroma_path": args.chroma_path,
        "build_seconds": round(time.time() - t0, 1),
        "embed_seconds": round(time.time() - t0_embed, 1),
    }
    os.makedirs(os.path.join(ROOT, "report_data"), exist_ok=True)
    stats_path = os.path.join(ROOT, "report_data", f"向量库统计_{args.collection}.json")
    with open(stats_path, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    print(f"\n[完成] 索引 {built_n:,} 块 | 用时 {stats['build_seconds']}s | 统计 -> {stats_path}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bge-base", choices=list(MODELS))
    ap.add_argument("--target-chunks", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--chroma-path", default=os.path.join(ROOT, "data", "chroma_db"))
    ap.add_argument("--subset-cache", default=None)
    ap.add_argument("--batch-size", type=int, default=128, help="GPU 向量化 batch")
    ap.add_argument("--add-batch", type=int, default=2000, help="Chroma 每次 add 行数")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：不删库，从集合已有条数继续（应对建库中途被强制重启）")
    args = ap.parse_args()
    if args.collection is None:
        args.collection = f"medrag_{args.model.replace('-', '_')}"
    build(args)


if __name__ == "__main__":
    main()
