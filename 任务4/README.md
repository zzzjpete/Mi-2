# 阶段四 · 向量化与索引构建

> 嵌入模型加载 → ChromaDB 持久化索引 → 质量验证，并按硬件实测容量上限确定向量化规模。

## 接手卡

**状态**：✅ 完成 ｜ 2026-07-16 ｜ 四项周产出全部达成，质量验证全绿

| 项 | 内容 |
|---|---|
| **输入** | `data/chunks/`（9243 万块）→ 跨 11 包分层随机抽样 **3,998,000** 块（seed 42，可复现） |
| **产物** | `data/chroma_db_4m` —— 集合 `medrag_bge_base`，3,998,000 × 768 维，余弦，10 个元数据字段 |
| **保险文件** | `data/vectors/merged_4m.parquet`（15 GB，向量+正文+元数据）—— 可几小时重建库，**勿删** |
| **核心脚本** | `scripts/向量化_建库.py`（`BGEEmbedder` + 分层抽样 + 批量向量 + 灌库 + 统计，带 `--resume`） |
| **复跑** | `& $py scripts\向量化_建库.py --model bge-base --target-chunks 4000000 --seed 42 --chroma-path E:\rag\data\chroma_db_4m` |
| **验证** | `& $py scripts\向量化_检索验证.py --model bge-base --chroma-path E:\rag\data\chroma_db_4m` |
| **下游调用** | `BGEEmbedder`（中文文件名须按路径 `importlib` 导入）；`embed_query()` 会自动加 BGE 指令前缀，若查询**已带**前缀请走 `embedder._encode([q])[0]` 绕开重复添加 |
| **遗留** | 约 **4.3%** 块超 512 BERT token 被截尾（切块按 bge-m3/XLM-R 分词器，embedding 用 bge-base/BERT）；真全量需换 FAISS IVF-PQ |

> `$py` = `E:\rag\conda\envs\medrag\python.exe`
> ⚠️ 建库是数小时任务，跑之前确认终端不会被中途关闭：Chroma 大库在写 HNSW 段时被硬杀会损坏（本阶段发生过一次，靠 `merged_4m.parquet` 恢复）。

---

## 一、对照任务书（三步）
> 本阶段任务书原文见 [`任务书.txt`](任务书.txt)（逐字照录，未改写）。


| 任务书要求 | 实现 |
|---|---|
| 嵌入模型选择与加载 | ✅ `BAAI/bge-base-en-v1.5`（768 维），CLS pooling + L2 归一化，fp16，查询加指令前缀 |
| 向量库配置与索引构建（ChromaDB） | ✅ 持久化 `data/chroma_db_4m`，集合 `medrag_bge_base`，创建时指定余弦相似度 |
| 质量验证 | ✅ 基础统计 / 自相似性 / 边界情况三类全绿 |
| 根据硬件重新估计可处理的最大文献量，并按此量向量化 | ✅ 实测每条向量占内存后定规模（见「三」） |

**周产出验收**

| 产出要求 | 结果 |
|---|---|
| 向量数据库文件 | ✅ `data/chroma_db_4m`（集合 `medrag_bge_base`，3,998,000 条） |
| 正确向量数量的索引统计 | ✅ 3,998,000 块 / 768 维 / 10 个元数据字段，见统计 JSON |
| 测试查询返回相关医学文献片段 | ✅ 二甲双胍机制 / CRISPR / 细胞因子风暴 / 肿瘤微环境 4 题均命中，带完整出处 |
| 元数据过滤功能正常工作 | ✅ `pub_year>=2020` 过滤后仅返回 2021–2023 |

---

## 二、核心结果

- **索引**：`data/chroma_db_4m`，集合 `medrag_bge_base`，**3,998,000 条 / 768 维 / 余弦相似度**。
- **唯一 id**：直接用切块产物的 `chunk_id`（即 `doc_id#chunk_index`，如 `PMC212698#15`）。
- **10 个元数据字段**：`doc_id / chunk_index / total_chunks / source_title / token_count / section / pmcid / pmid / journal / pub_year`。
- **质量验证**：① 基础统计 3,998,000 向量 / 768 维 / 样本元数据完整；② 自相似性 —— 摘 5 个块回查 **5/5 自身排第 1**（sim 0.975–0.988）；③ 边界情况 —— 空查询、17,200 字符超长查询均优雅处理（截断到 512 token，不报错）。
- **检索样例**：
  - Q「二甲双胍在 2 型糖尿病中的作用机制」→ top1 sim **0.785**，命中《Metformin Protects against Podocyte Injury in Diabetic Kidney Disease》的「Mechanisms Whereby Metformin Reduces Hyperglycaemia」章节。
  - Q「肿瘤微环境在癌症免疫治疗中的作用」→ top1 sim **0.842**，命中《Immunosuppressive Signaling Pathways as Targeted Cancer Therapies》的「The Tumor Microenvironment」章节。

---

## 三、关键设计取舍

**1. 嵌入模型选型** —— 从候选（BGE small/base/large、OpenAI 付费、clinicalBERT 需微调）中选 **`BAAI/bge-base-en-v1.5`（768 维）**：语料是纯英文 PubMed oa_comm，英文专用模型比多语言 m3 更贴；BGE 列表里质量/速度的均衡档，RTX 3080 上快且省显存。采用 CLS pooling + L2 归一化；查询加指令前缀 `Represent this sentence for searching relevant passages:`（非对称检索官方做法，文档侧不加）。

**2. 硬件容量评估与规模确定（诚实说明）** —— 需求要求「根据硬件重新估计可处理的最大文献量，并按此量向量化」。本机实测：

- **瓶颈是内存**（不是 GPU 也不是硬盘）：ChromaDB 需把全部向量常驻内存做相似度检索。**实测每条向量占 3.64 KB**（768 维 fp32 约 3.0KB + HNSW 图约 0.6KB；原文与元数据存磁盘 sqlite，不占内存）。
- 32GB 本机留给索引约 14–20GB → **上限约 400–550 万块**。
- 据此定规模：**跨全部 11 包分层随机抽样约 400 万块（3,998,000，seed 42 可复现）**，占全量 9,243 万块的 **4.33%**；分层抽样保留年份/期刊分布，使检索与过滤演示具代表性。
  ⚠ **更正（2026-08-18 实测）**：本行原写「约 17.3 万篇」，那是按整篇等比推算的（4,000,645 × 4.33% ≈ 173,041），**前提是整篇整篇地抽，而实际是按块抽**。
  离线建文献目录后逐条数过：入库的 3,998,000 块实际**覆盖 2,274,167 篇**，平均每篇只有 **1.76 块**（原文平均 28.64 块），54.2% 的文献只有 1 块。
  即**覆盖面比当时估的宽 13 倍，但每篇的完整度远低于「整篇入库」**——「一篇文献在库里」不等于「它的关键段落可检索」。详见 `docs/工程笔记.md` 三·13。
- 早期用 20 万小样本粗估 6.2KB/条，本次在 150 万、400 万规模上**实测修正为 3.64KB/条**，容量比初估宽近一倍 —— 这也是能把规模从演示级子集提到约 400 万的依据。

**3. 扩展路径（若需真全量）** —— 把索引从 Chroma 换 **FAISS IVF-PQ**（乘积量化压到约 128 字节/条，9,243 万块约 12GB 内存可行，代价是召回略降）。嵌入向量与索引选型**解耦**，向量算好后灌 Chroma 或 FAISS 均可，扩展无需推倒重来。

> ⚠ 一处已知小尾巴（无害）：切块按 bge-m3(XLM-R) 分词器切、embedding 用 bge-base(BERT) 分词器，导致约 **4.3%** 的块超 512 BERT token 被截尾（多为薄记录/超长句，仅丢尾部）。

---

## 四、复现命令

```powershell
$py = "E:\rag\conda\envs\medrag\python.exe"
# 建库（分层抽样约 400 万块 → bge-base → Chroma 余弦索引 + 统计）
& $py E:\rag\scripts\向量化_建库.py --model bge-base --target-chunks 4000000 --seed 42 --chroma-path E:\rag\data\chroma_db_4m
# 检索与质量验证
& $py E:\rag\scripts\向量化_检索验证.py --model bge-base --chroma-path E:\rag\data\chroma_db_4m
```

---

## 五、下一步

用 LangChain 串「Chroma 检索 → qwen3:8b 生成」的 RAG 主链路（带出处作答），再用准备阶段留下的「模型爱答错的具体事实题」做**接入前后对比验收**。（后续阶段五、六先把检索侧做深：查询理解、多路检索与重排。）

---

## 六、本包内容

- **相关代码**（在顶层 `scripts/`）：`向量化_建库.py`（嵌入模型类 + 分层抽样 + 批量向量 + 灌 Chroma + 统计）、`向量化_检索验证.py`（query 接口 + 三类质量验证 + 两项产出验收）。
- **本地留存产物**（未上交，可由脚本复现）：向量库本体 `data/chroma_db_4m`、`向量库统计_medrag_bge_base.json`（索引统计）、`检索验证报告.txt`（三类验证 + 检索/过滤验收完整输出）。
