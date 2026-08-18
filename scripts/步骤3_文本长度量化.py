"""
步骤3_文本长度量化.py — 阶段二第三步：文本特征量化分析（对应任务第3点）

复用 parsed_tok.parquet 里的 token_len（bge-m3 分词器算的），做：
  · 完整 token 长度分布（分位数、均值、标准差）
  · 超过 512 及其它候选 chunk_size 的比例 -> 判断要不要切、切多少
  · 按滑窗估算切块后总 chunk 数（给向量库规模打个底）
  · 画一张分布图（直方图 + 超阈值比例）

产物：
  - E:\\rag\\report_data\\token_hist.png
  - E:\\rag\\report_data\\step3.json
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import json
import math
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(os.path.join(ROOT, "data", "pubmed"))
REPORT = Path(os.path.join(ROOT, "report_data"))
IMG = REPORT / "token_hist.png"


def chunks_for(L, size, overlap):
    if L <= size:
        return 1
    return max(1, math.ceil((L - overlap) / (size - overlap)))


def main():
    df = pd.read_parquet(DATA / "parsed_tok.parquet")
    tl = df["token_len"]
    n = len(tl)

    # ---- 分布 ----
    dist = {
        "n": n,
        "min": int(tl.min()), "max": int(tl.max()),
        "mean": int(tl.mean()), "std": int(tl.std()),
        "p25": int(tl.quantile(.25)), "median": int(tl.median()),
        "p75": int(tl.quantile(.75)), "p90": int(tl.quantile(.90)),
        "p95": int(tl.quantile(.95)), "p99": int(tl.quantile(.99)),
    }

    # ---- 超阈值比例 ----
    thresholds = [512, 768, 1024, 2048, 4096, 8192]
    within = {t: round(100 * (tl <= t).mean(), 1) for t in thresholds}
    over_512 = round(100 * (tl > 512).mean(), 2)

    # ---- 切块数量估算（滑窗）----
    configs = [(512, 64), (384, 64), (256, 32)]
    chunk_est = {}
    for size, ov in configs:
        cs = tl.apply(lambda L: chunks_for(L, size, ov))
        chunk_est[f"{size}/{ov}"] = {
            "total_chunks": int(cs.sum()),
            "mean_per_doc": round(cs.mean(), 1),
            "max_per_doc": int(cs.max()),
        }

    summary = {"distribution": dist, "within_pct": within,
               "over_512_pct": over_512, "chunk_estimate": chunk_est,
               "img": str(IMG)}
    REPORT.mkdir(exist_ok=True)
    (REPORT / "step3.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 画图（英文标签，避免中文缺字方块）----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    p99 = dist["p99"]
    clipped = tl.clip(upper=p99)
    ax1.hist(clipped, bins=60, color="#4C78A8", edgecolor="white", linewidth=.3)
    ax1.axvline(512, color="#E45756", ls="--", lw=1.4,
                label="512 (embed target)")
    ax1.axvline(dist["median"], color="#54A24B", ls="--", lw=1.4,
                label=f"median {dist['median']}")
    ax1.set_title("Token length per document (bge-m3)", fontsize=10)
    ax1.set_xlabel("tokens (clipped at p99)")
    ax1.set_ylabel("# documents")
    ax1.legend(fontsize=8)

    labels = [str(t) for t in thresholds]
    vals = [within[t] for t in thresholds]
    bars = ax2.bar(labels, vals, color="#72B7B2", edgecolor="white")
    ax2.set_title("% of docs that fit within a size", fontsize=10)
    ax2.set_xlabel("chunk size (tokens)")
    ax2.set_ylabel("% docs ≤ size")
    ax2.set_ylim(0, 100)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%",
                 ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG, dpi=150)
    print("图已存:", IMG)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
