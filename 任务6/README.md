# 阶段六 · 检索系统（二）多路检索 + 多准则重排 + 完整流水线

> 在阶段五「查询理解与增强」之上，接一条真正的混合检索链路：向量 + 关键词双路检索 → 融合 → 多准则重排 → 带出处的证据列表。

## 接手卡

**状态**：✅ 完成 ｜ 2026-07-22 ｜ 全量 4M 库端到端验证 **21/21** 通过；测出并修复了一个 ~100s 的过滤时延问题

| 项 | 内容 |
|---|---|
| **输入** | 阶段五的 `EnhancedQuery` + `data/chroma_db_4m`（400 万向量）+ `data/bm25_index_4m` |
| **产物** | `data/bm25_index_4m`（3,998,000 块 / 3.55 GB）；一条可直接调用的端到端检索流水线 |
| **核心模块** | `scripts/检索_多路检索.py` —— `MultiPathRetriever`（向量+BM25 两路，三种融合）/ `MultiCriteriaReranker`（`bge-reranker-base` × 时效 × 权威 = 0.6/0.25/0.15）/ `RetrievalPipeline` |
| **下游调用**<br>（生成层用这个） | `pipe = RetrievalPipeline(bm25_dir=r"E:\rag\data\bm25_index_4m")`<br>`out = pipe.search(query, top_k=10)` → `out["results"]` 为 `List[Candidate]`<br>首次加载 ~15.8 GB RSS + 数分钟（HNSW 载入）；之后单次 `search()` ~0.5s；只要检索不要重排传 `load_reranker=False` |
| **前置构建** | `& $py scripts\检索_构建BM25索引.py --full --out E:\rag\data\bm25_index_4m` —— 约 13 min（读 19s + 分词 479s + 建索引 261s） |
| **验证** | `& $py scripts\检索_多路检索_验证.py --bm25 E:\rag\data\bm25_index_4m` —— 21 项断言 |
| **下游** | 阶段七生成层：LangChain 组织 `out["results"]` 的证据 → `qwen3:8b` 带出处作答 |
| **遗留** | 检索**质量**未量化（无人工标注集）——`rrf` vs `weighted`、重排权重是否最优均未判定；高选择性过滤未压测；期刊权威表是启发式清单 |

> `$py` = `E:\rag\conda\envs\medrag\python.exe`
> ⚠️ 向量路默认 `vector_filter_mode='postfilter'`，**不要**改回给 Chroma 下推 `where`——4M 集合上单次向量查询会从 ~1.2ms 涨到 ~108s（详见「五」）。

---

## 一、对照任务书（逐条）
> 本阶段任务书原文见 [`任务书.txt`](任务书.txt)（逐字照录，未改写）。


| 任务书要求 | 实现 | 位置 |
|---|---|---|
| 应用多路检索 `MultiPathRetriever` | ✅ 向量 + 关键词两路 + 融合 | `检索_多路检索.py :: MultiPathRetriever` |
| 检索初始化：向量检索 + 关键词检索(BM25) | ✅ 注入 Chroma 集合、BGE 嵌入器、bm25s 索引 | `__init__` |
| **构建 BM25 索引** | ✅ `bm25s`（磁盘化、可 mmap），从 4M 建库产物构建 | `检索_构建BM25索引.py` |
| 向量检索：ChromaDB 需把查询转为嵌入向量 | ✅ 消费查询理解层已带前缀的 `vector_queries`，编码后查 Chroma | `_vector_search` |
| 关键词检索：分词 | ✅ 建库/查询共用一套分词（英文停用词 + snowball 词干化） | `检索_BM25公共.py` |
| 关键词检索：计算 BM25 分数并取 top_k | ✅ `bm25s.retrieve`，行号→chunk_id 还原 | `_keyword_search` |
| 关键词检索：生成格式化结果 | ✅ 统一成 `Candidate`，命中后从 Chroma 补正文+元数据 | `_hydrate_and_filter` |
| 融合：简单合并去重 | ✅ `simple`：并集去重，命中两路 > 一路，最好排名 tiebreak | `_fuse` |
| 融合：RRF（学术检索适用） | ✅ `rrf`：Σ w/(k+rank)，向量各变体带查询理解层权重 | `_fuse` |
| 融合：加权融合（向量权重更高） | ✅ `weighted`：vw·norm(cos)+(1-vw)·norm(bm25)，默认 vw=0.7 | `_fuse` |
| `retrieve(query_info, top_k_vector, top_k_keyword, fusion_strategy)` | ✅ 签名一致 | `MultiPathRetriever.retrieve` |
| **重排序器**（可考虑 BAAI/bge-reranker-base） | ✅ 用的正是 `bge-reranker-base` | `MultiCriteriaReranker` |
| `AutoTokenizer` + `AutoModelForSequenceClassification` 加载 | ✅ 完全按任务书方式加载 | `__init__` |
| tokenize → 推理 → 转概率分数 | ✅ 交叉编码器 logit 过 sigmoid → [0,1] | `_relevance` |
| 多准则重排 relevance/recency/authority = 0.6/0.25/0.15 | ✅ 默认权重一致 | `MultiCriteriaReranker` |
| 　· recency：提取年份，随年份线性衰减 | ✅ current_year→1，往前 span(默认20)年→0；缺年份记 0.5 | `_recency` |
| 　· authority：按期刊加不同权重 | ✅ 顶刊/开放获取巨刊/未知刊分级 | `_authority` |
| **结合上周内容，完成完整检索流水线** | ✅ `process_query → retrieve → rerank` 一条 `search()` | `RetrievalPipeline` |

---

## 二、系统架构

```
原始查询 ─ MedicalQueryProcessor.process_query ─▶ EnhancedQuery（阶段五产物）
                                                     │
      ┌───────────────────────────────────────────────┤
      ▼ 向量路：BGE 编码(多变体加权) + Chroma        ▼ 关键词路：bm25s
   稠密候选                                        稀疏候选
      └────────── 融合 simple / rrf / weighted ──────────┐
                                                         ▼
                                        统一 hydrate 正文+元数据 + 两路一致过滤
                                                         ▼
                       MultiCriteriaReranker（bge-reranker-base 相关性 × 时效 × 权威）
                                                         ▼
                                                   最终证据 top_k
```

检索器直接消费 `EnhancedQuery`：向量路吃 `vector_queries`（已带 BGE 指令前缀、含缩写消歧变体与权重），关键词路把 `keyword_groups` 平铺成词袋喂 BM25，`filters`/`post_filters` 决定过滤。中文查询在阶段五已被译成英文，所以 BM25 只需英文分词。

---

## 三、核心结果

- **端到端验证 21/21**（全量 4M 向量 + 4M BM25，每条结论由数据计算、非硬编码）。
- **BM25 索引全量 3,998,000 块 / 3.55 GB**，与 4M 向量库同规模，是真·混合检索（BM25 有独立召回，实测一条查询 BM25 top-50 里 48 条是向量 top-50 没有的）。
- **单条查询 ~0.5s**。全量索引端到端冒烟（`RCT evidence for pembrolizumab in NSCLC published since 2020`，含 `pub_year>=2020` 过滤）：3 变体×500 向量 0.47s + BM25 0.08s + 交叉编码器重排 50 候选 0.49s，整条 **~1.1s**（含首查冷缓存）；top-5 全为 pembrolizumab/NSCLC 肿瘤论文、全部 ≥2020、相关性 0.97→0.52 合理排序。

---

## 四、关键设计取舍

**1. 为什么向量与关键词要"两路"。** 双塔向量检索擅长语义相近，但对罕见专有名词、缩写、精确字面容易漏（模型没见过就召不回）。BM25 按词频/逆文档频率打分，专有名词命中极稳，正好补这一块。两路融合才是"混合检索"的价值所在。

**2. BM25 为什么用 `bm25s` 不用 `rank_bm25`。** `rank_bm25` 是纯 Python、全部驻留内存，4M 块要占几十 GB，32GB 机器放不下。`bm25s` 把 BM25 分数预计算进 scipy 稀疏矩阵（CSC），可 `save` 到磁盘、查询时 `mmap` 载入，几乎不额外占 RAM。BM25 语料从建库产物 `merged_4m.parquet` 读，`chunk_id` 与 Chroma 集合 id 完全对齐，命中后正文/元数据统一从 Chroma 取；建库与查询的分词参数集中在 `检索_BM25公共.py` 保证两侧一致（否则同词被切成不同 token，命中率会莫名其妙地掉）。

**3. 三种融合策略的真实区别**（不是三个花名）：
- `simple`（简单合并去重）：并集去重，命中两路者恒排在只命中一路者之前，同档看最好排名。实现最简单，但**基本忽略各路内部的分数/排名差异**——这正是它的弱点。
- `rrf`（倒数排名融合，默认）：`score = Σ w/(k+rank)`。跨路只看排名不看分数量纲，避免余弦相似度与 BM25 分不可比的问题，稳健，学术检索常用。
- `weighted`（加权分数融合）：候选集内余弦相似度与 BM25 分各自 min-max 归一后加权求和，默认 `vw=0.7` 让向量权重更高；缺一路的分量记 0。

**4. 多准则重排为什么是交叉编码器。** 双塔向量把 query 和 passage 各自独立编码，快但错过词级交互；交叉编码器（`bge-reranker-base`）把二者拼起来过一遍 Transformer，判别更准，代价是每个候选都要过模型，所以只对融合后的小候选池（默认 50 条）重排。最终分是三准则加权和：

| 准则 | 权重 | 怎么算 |
|---|---|---|
| relevance 相关性 | 0.60 | 交叉编码器 logit 过 sigmoid → [0,1]，基础要求 |
| recency 时效性 | 0.25 | 从 `pub_year` 线性衰减：今年→1，往前 20 年→0；缺年份记 0.5 |
| authority 权威性 | 0.15 | 按期刊查权威性分级表；未知刊记 0.5 |

---

## 五、⚠ 关键实测发现：`where` 过滤把向量查询拖慢约 5 个数量级（已修复）

一句话结论：**在 4M 集合上给 Chroma 下推 `where` 过滤，单次向量查询从 ~1.2ms 暴涨到 ~108s**（带过滤的 HNSW 在 4M 规模退化）。修复方式是向量路默认**不下推 where**，改「无过滤检索 + 过量取样 + Python 后过滤」，实测把过滤查询的向量部分从 **111.67s 降到 0.027s**、整条 ~0.5s，且 21 项断言不变。这也回答了阶段五留下的开放问题「where 过滤是否显著拖慢查询」。

<details><summary>详细排查与修复过程（含前后时延对比表、代价与逃生阀）</summary>

**排查**：验证时注意到，带 `pub_year` 过滤的查询向量检索耗时 108–111s，而不带过滤的仅 0.03s。做了干净的微基准确认：无过滤向量查询中位 **1.2ms**（纯 HNSW），带过滤 ~108s——差约 5 个数量级，瓶颈完全在 Chroma 的带过滤 HNSW（4M 规模下的元数据过滤退化），BM25 一路、重排都不慢。

**修复**：向量路默认 `vector_filter_mode='postfilter'`——不给 Chroma 下推 `where`，改成"无过滤检索（~1ms）+ top_k 过量取样（×10，上限 500）+ Python 后过滤"。项目里本有 `match_where` 和 section 后过滤，hydrate 阶段对向量/关键词两路候选统一施加，语义与查询理解层解析出的过滤完全一致；BM25 侧同样过量取样。

| 查询（含 pub_year 过滤） | 修复前·向量 | 修复后·向量 | 修复后·整条 |
|---|---|---|---|
| `pub_year >= 2021` | 111.67 s | **0.027 s** | 0.547 s |
| `recent studies`（→ ≥2021） | 108.50 s | **0.013 s** | 0.414 s |

**代价与逃生阀**：后过滤对**高选择性**过滤（如 `pub_year>=2026`，命中占比极低）可能兜不住——过量取样窗口里存活太少。此时可传 `vector_filter_mode='where'` 换精确但慢的下推。默认选后过滤，是因为绝大多数医学查询的年份/章节过滤都是低选择性的，用 200 倍的时延换极少数边缘情况的召回不值。
</details>

---

## 六、验证：21/21，每条结论都由数据算出

`检索_多路检索_验证.py`，在全量 4M 向量 + 4M BM25 上跑，**每条 PASS/FAIL 都从真实数据计算并汇入总 ok，绝不硬编码**；过滤类断言特意构造成非空洞（必须确有 BM25 候选被过滤掉才算通过）。

- **A 多路检索**：向量/关键词各自返回非空；BM25 贡献了向量 top-k 之外的候选（独立召回，关键词独有 48/50）。
- **B 融合**：`rrf`/`weighted` 打分公式逐候选独立重算，最大误差 `0.00e+00`；`simple` 命中两路者全排在单路之前；三策略作用于同一候选并集；三者排序均单调递减。
- **C 过滤生效（非空洞）**：`pub_year>=阈值` 下 BM25 命中里 20 条低于阈值、最终泄漏 **0** 条；`methods` 章节后置过滤，BM25 命中里 44 条非-methods、最终泄漏 **0** 条，全部最终候选落在同一份 `canonical_to_raw['methods']` 写法表里。
- **D 多准则重排**：权重和为 1；时效性随年份单调（今年 1.00 / 前 10 年 0.50 / 前 100 年 0.00）；权威性 Nature 1.00 > PLoS ONE 0.62 > 未知 0.50；三准则均 ∈[0,1]；总分 = Σ 权重×准则分（误差 `0.00e+00`）；结果按总分单调递减；重排相对纯融合序确实改变了顺序。

---

## 七、BM25 索引规模

按「先子集验证再全量」推进：先建 50 万子集把整条链路跑通验证，确认无误后建全量。

| 索引 | 篇数 | 磁盘 | 用途 |
|---|---|---|---|
| `data/bm25_index_500k` | 500,248 | 445 MB | 端到端打通验证 |
| `data/bm25_index_4m` | 3,998,000 | 3.55 GB | 与 4M 向量库同规模，真·混合检索（读 19s + 分词 479s + 建索引 261s ≈ 13min） |

---

## 八、复现命令

```powershell
$py = "E:\rag\conda\envs\medrag\python.exe"

# 1) 构建 BM25 索引（子集 / 全量）
& $py E:\rag\scripts\检索_构建BM25索引.py --limit 500000 --out E:\rag\data\bm25_index_500k
& $py E:\rag\scripts\检索_构建BM25索引.py --full --out E:\rag\data\bm25_index_4m

# 2) 端到端验证（21 项）
& $py E:\rag\scripts\检索_多路检索_验证.py --bm25 E:\rag\data\bm25_index_4m

# 3) 单条查询演示（--fusion simple/rrf/weighted，--no-rerank，--translate llm 可切换）
& $py E:\rag\scripts\检索_多路检索.py --bm25 E:\rag\data\bm25_index_4m `
      --query "RCT evidence for pembrolizumab in NSCLC published since 2020" --fusion rrf
```

依赖：`bm25s`、`scipy`、`PyStemmer`；`BAAI/bge-reranker-base`（缓存到 `hf-cache`）。

---

## 九、已知局限与下一步

**已知局限（如实说明）**
1. ~~**质量未量化**：本阶段只证明了"算得对、过滤对、顺序会变、跑得快"，**没有证明检索结果更准**——`rrf` 是否优于 `weighted`、重排权重 0.6/0.25/0.15 是否最优、期刊权威表是否合理，都需人工标注查询才能判定，目前无标注集。~~

   > **⚠ 2026-08-12 更新：标注集已经有了，而且答案是「0.6/0.25/0.15 不最优，反而在伤害检索」。**
   >
   > 见 `任务10/检索评测改进/`（237 条 golden 检索评测集，只跑检索不生成答案，完全确定性）。
   > 实测：**融合+重排 R@10=0.425 / MRR=0.241，比它自己的向量腿（0.490 / 0.347）还差**，
   > 净毁掉 10 条命中，被毁的 26 条里 13 条是 ≤2015、被救的 16 条一条 ≤2015 都没有。
   > 病根是「饱和型相关性分 + 线性年份分**相加**」这个结构：交叉编码器认可的候选 rel 全挤在
   > 1.0 附近（池内前 20 的 std 只剩 ~0.08），而年份铺得很开（std ~0.18），
   > 于是 recency 的排序影响力几乎等于 rel（44.0% vs 47.7%）。
   > **归一化修不好**（池内 min-max 归一后 MRR 反降到 0.207）。
   >
   > **已修（P2a）**：`MultiCriteriaReranker` 默认改为 `mode="tiebreak"`、ε=0.02——
   > recency 不再当加法项，改当**同分裁决**（先按 ε 给 rel 分档，同档内才比 recency）。
   > 生产实测 **R@10 0.425→0.503、MRR 0.241→0.315**，四个年份桶没有一格更差。
   > 旧行为随时可用 `--rerank-mode weighted` 跑回来做对照，且已验证它**逐位复现**本阶段的原始数字。
   >
   > 本节下面第四小节那段「最终分是三准则加权和」描述的是**旧默认**，现在是同分裁决；
   > 三个准则的算法本身没变。`rrf` vs `weighted`、期刊权威表这两条**仍未量化**。
2. **高选择性过滤**未做压力测试，只在设计上留了 `where` 精确逃生阀。
3. **期刊权威表**是可调的启发式清单，非绝对排名；未知期刊落到中性默认值 0.5。
4. **BM25 双索引**：先建 50 万子集打通、再建全量 4M，两个索引都保留。

**下一步（生成层）**：LangChain 组织证据 → 本地 `qwen3:8b` 生成带出处的回答，在阶段一事实型问题上做前后对比评测；届时顺带建人工标注小评测集，把"哪种融合/哪组重排权重更准"量化掉。

---

## 十、本包内容

```
任务6\
├─ README.md                    本文件
├─ 多路检索与重排报告.docx        汇报稿
├─ 多路检索验证报告.txt           21/21 验证的原始输出（证据链）
└─ 脚本\                        交付拷贝；单一来源在顶层 scripts\
   ├─ 检索_多路检索.py           ⭐ 核心：MultiPathRetriever / MultiCriteriaReranker / RetrievalPipeline
   ├─ 检索_构建BM25索引.py       前置：建 BM25 索引
   ├─ 检索_BM25公共.py           建库/查询共用的分词约定
   ├─ 检索_多路检索_验证.py       21 项端到端验证
   ├─ 检索_查询理解.py           阶段五上游依赖（检索_多路检索.py 直接 import）
   ├─ 向量化_建库.py             阶段四上游依赖（提供 BGEEmbedder，同上）
   └─ 检索2_报告转word.py        生成上面那份 docx
```

> 两个上游依赖必须随包：`检索_多路检索.py` 启动时按**同目录**路径导入 `检索_查询理解.py` 与
> `向量化_建库.py`，少一个就 import 失败。

**本地留存产物**（可由脚本复现）：`data/bm25_index_4m`、`data/bm25_index_500k`。
