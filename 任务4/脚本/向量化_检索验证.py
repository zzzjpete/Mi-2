# -*- coding: utf-8 -*-
"""
第四阶段 · 检索与质量验证

加载建库产物（ChromaDB 集合 + BGE 查询模型），实现需求指定的 query() 接口，并做：
  1. 基础统计验证（向量数量、样本元数据）
  2. 相似性检索验证（从索引摘取文本作查询，测自相似性）
  3. 边界情况验证（空查询、超长查询）
  4. 产出验收：真实医学问题检索返回相关片段 + 元数据过滤功能

用法：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\向量化_检索验证.py --model bge-base
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import importlib.util
import json
import sys
import textwrap

import pyarrow.parquet as pq          # 早于 torch
import numpy as np
import chromadb

# Windows 控制台是 GBK，会编不了 ✓ 等字符；统一 UTF-8，并把全部输出 tee 到报告文件
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
_REPORT = open(os.path.join(ROOT, "report_data", "检索验证报告.txt"), "w", encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


sys.stdout = _Tee(sys.stdout, _REPORT)

# 复用建库脚本里的 BGEEmbedder / MODELS / QUERY_INSTRUCTION（中文文件名 -> 按路径导入）
# 先找本脚本同目录、再退回 scripts\：这样连同依赖被拷进 任务4\脚本\ 单独发出去也能跑
_sib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "向量化_建库.py")
_spec = importlib.util.spec_from_file_location(
    "jianku", _sib if os.path.exists(_sib) else os.path.join(ROOT, "scripts", "向量化_建库.py"))
_jk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_jk)
BGEEmbedder = _jk.BGEEmbedder


def query(query_text, embedding_model, collection, n_results=5, where_filter=None):
    """需求指定的查询接口。

    Args:
        query_text:      查询文本
        embedding_model: 嵌入模型（BGEEmbedder，内部对查询加 BGE 指令前缀）
        collection:      Chroma 集合
        n_results:       返回结果数量
        where_filter:    元数据过滤条件（如 {"pub_year": {"$gte": 2020}}）
    Returns:
        Chroma 查询结果（含 ids/documents/metadatas/distances）
    """
    qv = embedding_model.embed_query(query_text)
    return collection.query(
        query_embeddings=[qv.tolist()],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )


def _show(res, max_text=160):
    """打印一次查询的命中列表（相似度 = 1 - 余弦距离）。"""
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    if not ids:
        print("    （无命中）")
        return
    for r, (cid, doc, m, dist) in enumerate(zip(ids, docs, metas, dists), 1):
        sim = 1 - dist
        src = f"{m.get('pmcid','?')} · {m.get('journal','?')} {m.get('pub_year','?')} · {m.get('section','?')}"
        title = (m.get("source_title") or "")[:70]
        snippet = textwrap.shorten(doc.replace("\n", " "), width=max_text, placeholder=" …")
        print(f"    [{r}] sim={sim:.3f}  {cid}")
        print(f"        {src}")
        print(f"        《{title}》")
        print(f"        {snippet}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bge-base")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--chroma-path", default=os.path.join(ROOT, "data", "chroma_db"))
    args = ap.parse_args()
    if args.collection is None:
        args.collection = f"medrag_{args.model.replace('-', '_')}"

    client = chromadb.PersistentClient(path=args.chroma_path)
    col = client.get_collection(args.collection)
    embedder = BGEEmbedder(args.model)

    print("\n" + "=" * 70)
    print("① 基础统计验证")
    print("=" * 70)
    n = col.count()
    print(f"  向量数量：{n:,}")
    peek = col.peek(limit=3)
    for i, (cid, m) in enumerate(zip(peek["ids"], peek["metadatas"]), 1):
        print(f"  样本{i} id={cid}  meta={ {k: m[k] for k in ('pmcid','journal','pub_year','section','token_count')} }")
    # 元数据完整性抽查
    dim = len(peek["embeddings"][0]) if peek.get("embeddings") is not None else "?"
    print(f"  向量维度：{dim}")

    print("\n" + "=" * 70)
    print("② 相似性检索验证（从索引摘文本作查询，测自相似性）")
    print("=" * 70)
    sample = col.peek(limit=5)
    ranks = []
    for cid, doc in zip(sample["ids"], sample["documents"]):
        res = query(doc, embedder, col, n_results=5)
        hit_ids = res["ids"][0]
        rank = hit_ids.index(cid) + 1 if cid in hit_ids else -1
        top_sim = 1 - res["distances"][0][0]
        ranks.append(rank)
        flag = "✓" if rank == 1 else ("· top5" if rank > 0 else "✗ 未进top5")
        print(f"  {cid}: 自身命中排名={rank} (top1 sim={top_sim:.3f}) {flag}")
    hit1 = sum(1 for r in ranks if r == 1)
    print(f"  → 自身排第1：{hit1}/{len(ranks)}（查询加指令前缀属非对称检索，排名靠前即正常）")

    print("\n" + "=" * 70)
    print("③ 边界情况验证")
    print("=" * 70)
    # 空查询
    try:
        res = query("", embedder, col, n_results=3)
        print(f"  空查询：返回 {len(res['ids'][0])} 条（未报错，向量仍可算）")
    except Exception as e:
        print(f"  空查询：异常 {type(e).__name__}: {e}")
    # 超长查询（远超 512 token）
    long_q = "myocardial infarction treatment guidelines " * 400
    try:
        res = query(long_q, embedder, col, n_results=3)
        print(f"  超长查询({len(long_q)}字符)：返回 {len(res['ids'][0])} 条（截断到512 token，未报错）")
    except Exception as e:
        print(f"  超长查询：异常 {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print("④ 产出验收 A：真实医学问题检索返回相关片段")
    print("=" * 70)
    med_queries = [
        "What is the mechanism of action of metformin in type 2 diabetes?",
        "How does CRISPR-Cas9 achieve targeted genome editing?",
        "What causes cytokine storm in severe infections?",
        "Role of tumor microenvironment in cancer immunotherapy",
    ]
    for q in med_queries:
        print(f"\n  Q: {q}")
        _show(query(q, embedder, col, n_results=3))

    print("\n" + "=" * 70)
    print("⑤ 产出验收 B：元数据过滤功能")
    print("=" * 70)
    q = "gene expression analysis using microarray"
    print(f"  Q: {q}")
    print("  · 不过滤（top3）:")
    _show(query(q, embedder, col, n_results=3))
    print("  · 过滤 pub_year >= 2020（top3）:")
    _show(query(q, embedder, col, n_results=3, where_filter={"pub_year": {"$gte": 2020}}))
    # 验证过滤确实生效：命中年份都 >= 2020
    res = query(q, embedder, col, n_results=10, where_filter={"pub_year": {"$gte": 2020}})
    yrs = [m["pub_year"] for m in res["metadatas"][0]]
    print(f"  · 过滤后命中年份：{sorted(set(yrs))} -> 全部 >=2020: {all(y >= 2020 for y in yrs)}")

    print("\n[验证完成]")


if __name__ == "__main__":
    main()
