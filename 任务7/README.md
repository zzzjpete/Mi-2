# 阶段七 · 生成层：上下文组装 + 医学提示词工程 + 本地 LLM 生成流水线

> 检索侧已完成（阶段六）。本阶段分两部分：
> **之一** 把检索出的证据整理成可直接进提示词的上下文，并把"怎么问模型"固化成四段医学提示词；
> **之二** 真正调用本地 `qwen3:8b`，把四段串成一条端到端流水线，产出带出处、经审查的回答。

## 接手卡

**状态**：✅ 之一完成（2026-07-27，验证 65/65）｜ ✅ 之二完成（2026-07-31）
之二实测：JSON 容错 **14/14**、冒烟 **7/7**、离线链路 **4/4**、**真检索验收 4/4**、消融 **12/12**、**接入前后对比 4/4**（出处可溯源 RAG 22/22 vs 裸模型 0/18）
**未完**：把裸模型那 18 条引用逐条查实、人工标注评测集，见「九」

| 项 | 内容 |
|---|---|
| **输入** | 阶段六 `RetrievalPipeline.search()` 的 `out["results"]`（`List[Candidate]`） |
| **产物** | `data/tokenizer/qwen3_tokenizer.json`（11 MB，可重建）<br>`report_data/生成_上下文组装验证报告.txt`<br>`report_data/生成_流水线测试_{offline,live,ablation}.jsonl` + 同名 `报告_*.txt`<br>`report_data/检索快照_live.json`（检索结果固化，生成侧可反复复跑）<br>`report_data/生成_对比评测.jsonl` + `生成_对比评测报告.txt` |
| **核心模块**<br>（之一） | `scripts/生成_上下文组装.py` —— `DocumentChunk` / `ContextAssembler`<br>`scripts/生成_提示词模板.py` —— `PromptStage` / `PROMPT_STAGES`（四段）/ `MedicalPromptTemplates`<br>`scripts/生成_分词器.py` —— 从 Ollama GGUF 重建 qwen3:8b 分词器，精确算 token |
| **核心模块**<br>（之二） | `scripts/生成_LLM生成器.py` —— `LLMGenerator`（连接自检 / JSON 容错解析 / 批量生成）<br>`scripts/生成_流水线.py` —— `MedicalGenerationPipeline`（六步生成流程 + 消融开关）<br>`scripts/生成_流水线_测试.py` —— 测试查询集 + 指标统计 + 日志落盘<br>`scripts/生成_对比评测.py` —— 接入前后对比：RAG vs 裸 qwen3:8b |
| **下游调用** | `pipe = MedicalGenerationPipeline(retriever=RetrievalPipeline(bm25_dir=…))`<br>`res = pipe.generate(query, top_k=10)` → `res["answer"]` / `res["sources"]` / `res["generation_metrics"]`<br>不带检索器时传 `retrieved_docs=[...]`（阶段六 `Candidate` 列表），便于离线复跑 |
| **验证** | `& $py scripts\生成_上下文组装_验证.py`（65 项，加 `--ollama` / `--live` 为可选组）<br>`& $py scripts\生成_LLM生成器.py --smoke`（JSON 14 项 + 冒烟 7 项）<br>`& $py scripts\生成_流水线_测试.py --offline`（4 道题跑完整四段链）<br>`& $py scripts\生成_流水线_测试.py --ablation`（同题 3 种链路配置对照）<br>**真检索验收（两趟跑，避开内存冲突）**：<br>　`--dump-retrieval <快照.json> --bm25 …\bm25_index_4m` 只跑检索<br>　`--from-dump <快照.json>` 再跑生成<br>`& $py scripts\生成_对比评测.py`（接入前后对比，复用快照与已有 RAG 结果） |
| **遗留** | 硬证到的是**可验证性**（出处可溯源 RAG 22/22 vs 裸模型 0/18），不是**正确性**——内容谁更对仍只有人工核对，无标注集；裸模型那 18 条引用尚未逐条查实；消融只量化了成本侧（4.99× 耗时），"四段链是否更忠于证据"未判定；去重阈值 0.80、`diversity_decay` 0.75、`max_per_source` 3 依旧是未验证的默认值 |

> `$py` = `E:\rag\conda\envs\medrag\python.exe`
> ⚠ 两个必须显式设的开关，任缺一个都会**静默**得到坏结果：
> ① **`num_ctx` 默认只有 4096**，四段链装不下，超长不报错而是丢掉最前面的 system 与最相关证据 → 必须设 12288（`/api/ps` 实测生效，显存 6.33 GB）。
> ② **`think` 默认开着**，qwen3 是思考模型，Ollama ≥0.9 把推理过程放进 `message.thinking`，`content` 要等思考结束才写 → 实测思考段吃光整个 `num_predict`，**`content` 全空**。提示词里写 `/no_think` **关不掉**，必须传 API 层的 `think: false`（见「八」）。

---

## 一、对照任务书（第一部分 · 逐条）

> 两部分的任务书原文都在 [`任务书.txt`](任务书.txt)。

| 任务书要求 | 实现 | 位置 |
|---|---|---|
| 定义文档块数据类 `DocumentChunk`（text/metadata/relevance_score/source/chunk_id） | ✅ 五个字段与任务书完全一致；便捷读取一律做成 property，不加字段 | `生成_上下文组装.py :: DocumentChunk` |
| 定义上下文组装器 `ContextAssembler` | ✅ | `生成_上下文组装.py :: ContextAssembler` |
| 　· 加载 tokenizer | ✅ 与 qwen3:8b **同一套** BPE，从 Ollama GGUF 离线重建（本机无 Qwen HF 分词器） | `生成_分词器.py :: TokenCounter` |
| 　· 估算文本 token 数量 | ✅ `estimate_tokens()`；精确模式下不是"估算"而是精确值 | `ContextAssembler.estimate_tokens` |
| 　· 转换文档格式（DocumentChunk） | ✅ 兼容 `Candidate` / dict / `DocumentChunk`；顺手丢掉退化短块 | `to_document_chunks` |
| 　· 计算文本相似性并去重（Jaccard） | ✅ 词 3-gram shingle 的 Jaccard + 逐字重复快速通道，保留簇内最相关者 | `jaccard_similarity` / `deduplicate` |
| 　· 按照相关性排序 | ✅ 相关性降序，同分按 chunk_id，保证可复现 | `assemble_context` |
| 　· 优先高相关 + 考虑多样化（同来源太多则降优先级） | ✅ 有效分 = 相关性 × `diversity_decay^同源已选数`，另设同源硬上限 | `_effective_score` |
| 　· 构建上下文字符串并受长度限制（在完整段落处截断，如末 10% 内找句号） | ✅ 先按 token 硬截，再在末 10% 回退找段落分隔 → 句末标点；且跳过 `e.g.`/`et al.` 缩写点 | `truncate_at_boundary` |
| 　· 补充元数据（5 个字段） | ✅ 五个字段名称与任务书一致，另附引用清单等排查信息 | `assemble_context` |
| 　· 返回 `{context_text, metadata, selected_chunks}` | ✅ 键名一致 | `assemble_context` |
| 医学提示工程模板 `PromptStage`（name/system_prompt/user_prompt_template/temperature/max_tokens） | ✅ 五个字段一致，补充字段全部带默认值 | `生成_提示词模板.py :: PromptStage` |
| 　· `evidence_evaluator` 证据评估器 | ✅ T=0.0，max_tokens=1200，输出 JSON | `PROMPT_STAGES["evidence_evaluator"]` |
| 　· `answer_generator` 答案生成器 | ✅ T=0.3，max_tokens=1500，输出 Markdown（逐句挂 `[S#]`） | `PROMPT_STAGES["answer_generator"]` |
| 　· `critical_reviewer` 批判性审查器 | ✅ T=0.0，max_tokens=1200，输出 JSON（7 类问题分类） | `PROMPT_STAGES["critical_reviewer"]` |
| 　· `final_assembler` 最终组装器 | ✅ T=0.2，max_tokens=2000，输出 Markdown（含参考文献） | `PROMPT_STAGES["final_assembler"]` |

---

## 一之二、对照任务书（第二部分 · 逐条）

| 任务书要求 | 实现 | 位置 |
|---|---|---|
| **`LLMGenerator`**（model_name / base_url / timeout） | ✅ 三个参数名与任务书一致 | `生成_LLM生成器.py :: LLMGenerator` |
| 　· 初始化并测试链接 | ✅ `check_connection()` 打 `/api/tags`，并确认模型确实已安装；失败的报错里带"下一步该敲什么命令" | `check_connection` |
| 　· 温度和 token 数 | ✅ 实例级默认值 + 每次调用可覆盖；`num_predict=0` 在入口拦掉（会让 Ollama 挂住不返回） | `build_options` |
| **生成方法** | ✅ `generate(prompt, system_prompt, temperature, max_tokens, …)` | `generate` |
| 　· 用 prompt(+system) / temperature / max_tokens 构建完整提示 | ✅ 组装成 chat messages；另有 `generate_messages()` 直吃提示词模板的 `format_messages()` 输出 | `generate` / `generate_messages` |
| 　· 按模型格式加系统提示，并添加 JSON 格式要求 | ✅ `json_output=True` 时追加 `JSON_FORMAT_INSTRUCTION`（模板里已写过一次，实测仍有约 1/5 概率加代码块围栏，故再钉一次） | `JSON_FORMAT_INSTRUCTION` |
| 　· 从文本中提取 JSON 并编辑格式（添加缺失符号） | ✅ 四级递进解析：直接 loads → 剥 `<think>`/``` 围栏 → 扫平衡子串 → 补缺失括号并去尾随逗号。**14 项自检全过**（含实测最高频的「数值后多引号」） | `extract_json` / `_scan_balanced` / `_repair` |
| **批量生成** | ✅ `batch_generate()`，串行执行 + 单条错误隔离 + 进度回调 | `batch_generate` |
| **`MedicalGenerationPipeline`** | ✅ | `生成_流水线.py :: MedicalGenerationPipeline` |
| 　· 初始化组件（上下文 / 医学模板 / LLM 生成） | ✅ 三者均可注入，None 时按参数自建；检索器可选 | `__init__` |
| 　1. 上下文组装 | ✅ 调 `ContextAssembler.assemble_context()`；无证据直接走固定话术，**不调用模型** | `generate` |
| 　2. 检索结果评估（可选） | ✅ `evaluate=True` 时跑①，JSON 解析失败则降级为"不筛选"继续，不阻断链路 | `generate` |
| 　　· 用评估结果筛选上下文（按文档 id、检测标题行） | ✅ 按 `[S#]` 出处头**顺序锚定**切块（不能按分隔符 split，正文本身含空行），丢掉相关性 < 1 的块 | `split_context_blocks` / `filter_context_by_evaluation` |
| 　3. 提取评估结果并生成答案草稿 | ✅ 评估压成摘要喂给②；②只看筛后证据 | `format_evidence_summary` |
| 　4. 批判性审查（可选） | ✅ `review=True` 时跑③，输出 issues / verdict / bad_citations | `generate` |
| 　5. 生成最终答案（有审查用审查结果，没有则用草稿） | ✅ 与任务书一致：`review_obj is not None` 才跑④，否则草稿即答案 | `generate` |
| 　6. 后处理（引用标记 / 格式美化 / 免责声明） | ✅ 把答案里的 `[S#]` 与真实编号对账，**编造的编号直接标出来**；统一标题层级；附参考文献与免责声明 | `postprocess` |
| 　· `result` 字典（query/answer/context_metadata/generation_metrics/intermediate_results/sources/timestamp） | ✅ 七个键与任务书完全一致，`generation_metrics` 的四个子键也一致；另补 `citation_check` 等排查字段 | `_assemble_result` |
| 　· 计算总时间 | ✅ `total_time_seconds` + 每阶段 `stage_times` | `_assemble_result` |
| **测试完整流程** | ✅ 四种跑法：`--offline` 测链路 / `--live`（或两趟跑法）测四类验收错题 / `--ablation` 链路配置对照 / 另有独立的 `生成_对比评测.py` 做接入前后对比 | `生成_流水线_测试.py` |
| 　· 编写测试 query | ✅ 离线 4 道（对得上样例语料，含一道"必须拒答"）+ 在线 4 道（阶段一四类错题） | `OFFLINE_QUERIES` / `LIVE_QUERIES` |
| 　· 完善答案显示与统计信息 | ✅ 答案全文 + 分阶段耗时 + 阶段成功 + 证据漏斗 + 引用对账 + 出处清单 | `print_result` / `print_summary` |
| 　· log 测试结果及相关引用来源 | ✅ `report_data\生成_流水线测试_*.jsonl`（含 sources 与中间产物）+ 人读版 `报告_*.txt` | `log_records` |
| 　· log 关键指标（耗时 / 答案长度 / 阶段） | ✅ 三者都在 jsonl 的 `metrics` 里，另含 token 用量与链路配置 | `log_records` |

---

## 二、数据流

```
RetrievalPipeline.search() → List[Candidate]
        │
        ▼  DocumentChunk.from_candidate（相关性取 rerank_score → rel → fused → cos）
   统一格式 + 丢弃退化短块（正文 <20 字符）
        ▼  deduplicate：3-gram Jaccard ≥0.80 判重，簇内保留最相关的一条
   去重后候选
        ▼  按相关性降序排序
        ▼  贪心选取：max(相关性 × decay^同源已选数)，同源硬上限 3
        ▼  按 token 预算装块：装不下→在完整句/段处截断；仍装不下→跳过它继续看下一条
   context_text（每块一行出处头 [S1] PMCID · 期刊 (年份) · 章节 · "标题"）
        │
        ├─ metadata：5 个必需统计 + 引用清单 citations + 各类丢弃原因
        └─ selected_chunks：入选的 DocumentChunk（可渲染参考文献列表）
                │
                ▼
   PROMPT_STAGES  ①证据评估 → ②作答（挂 [S#]）→ ③批判审查 → ④按审查意见定稿
```

---

## 三、核心结果（均为实测）

- **token 计数是精确的，不是估算——而且是拿 Ollama 自己的计数对过的**：从 qwen3:8b 的 GGUF 里读出 151,936 词表 + 151,387 merges 重建 byte-level BPE，与 Ollama 实际分词（`/api/generate` `raw=true` 返回的 `prompt_eval_count`）逐条比对，9 个样本（2~1,143 tok，含中文、公式、多段落）**差值全为 0**。解析 GGUF 头 0.17 s，之后走 JSON 缓存加载 0.28 s。
  > ⚠ 单看"1000 条 `encode→decode` 往返一致"是**不够的**：byte-level BPE 的 decode 只是把 token 字符串拼起来，任何切分都能往返一致——它只能证明词表没载错，证明不了切分（从而 token 数）与推理端相同。所以才必须有上面那条与 Ollama 的实测对照。
- **实测语料的 chars/token**：中位 4.49，p5 3.05，最小 1.31（公式符号密集块），最大 6.50。
- **预算严格不超**：4 组预算实测 `200→150 tok/2 块`、`500→481/5（截断 1）`、`1200→1097/6`、`2800→2740/10`，均 ≤ 预算，且预算越大装得越多。
- **截断落在完整句/段**：200 条真实文本 × 3 档长度共触发 **544 次截断，越界 0 次**，且截断结果 100% 是原文前缀（不改写内容）、回退不越出末 10% 窗口。
- **去重零误删**：1000 条真实块上，被丢弃的**全部**是逐字重复；误删 0 条。
- **顺带发现的数据质量问题**：1000 条样例块里有 **8 条正文只有 `"xx"`**（占 0.8%，来自 PLoS Synopsis 类条目）。这类块永远不可能支撑结论却会白占一个引用位，已在 `to_document_chunks` 前置丢弃；也值得回头给阶段三的切块加一条最小长度过滤。
- **提示词开销实测**（不含证据）：证据评估器 345 tok / 答案生成器 496 / 批判审查器 418 / 最终组装器 580；参考文献列表每条约 **40 tok**（38.8 / 39.7 / 41.7）。
- **端到端真实跑通**（4M 向量 + 4M BM25，查询 `RCT evidence for pembrolizumab in NSCLC published since 2020`）：检索 10 条 → 入选 **5 条、来自 5 篇不同文献**、**2,670 / 2,800 tok**、1 条被截断；`relevance_score` 确实取自重排总分。整轮验证 76 s（含流水线加载 39 s）。

---

## 四、关键设计取舍

**1. 为什么去重用 3-gram shingle 而不是词袋。** 同一学科的两段不同正文，词袋 Jaccard 常有 0.3~0.5 的虚高，阈值很难定；3-gram 对词序敏感，只有真正复述同一段文字才会高度重合。验证里做了对照：同词逆序的两段文本，词袋 Jaccard **1.000**，3-gram **0.000**。

**2. 为什么必须去重。** PubMed 里同一段落会被多篇（勘误、综述引用）重复收录，阶段三切块本身还带 overlap。不去重的后果不是"浪费 token"，而是**把假的证据一致性喂给模型**——同一句话出现三次，模型会当成三项独立证据来加强结论。

**3. 多样性用软惩罚而不是硬轮转。** 同一篇文章连着几块常常确实最相关（Results 的连续段落），硬轮转会把明显更差的证据换进来。这里用 `有效分 = 相关性 × 0.75^同源已选数` 做衰减，再加同源硬上限 3：第 4 块同源自然让位给别的文献的次优块，但前 2~3 块该留还是留。

**4. 装不下就跳过，而不是停止。** 剩余预算 300 token 时，第 4 条 500 token 的证据放不下，但第 5 条 200 token 的仍可能有价值。贪心循环因此是"跳过继续"而非"遇阻即停"。

**5. 出处头里不放检索分数。** 把 `0.83` 这类数字给模型看，它会当成"可信度"去解释和引用。出处头只放模型判断证据质量、写参考文献真正需要的字段（PMCID / 期刊 / 年份 / 章节 / 标题），检索侧的原始信号原样留在 `metadata["_retrieval"]` 里备查。

**6. 为什么把生成拆成四段。** 阶段一压测裸 `qwen3:8b` 时的四类错（编造参考文献、罕见病药名张冠李戴、药物分类错误、时间线自相矛盾）都不是"没检索到"，而是**生成时越过了证据**。一次性生成里模型既当作者又当审稿人，几乎不会否定自己刚写的句子；拆开后 ② 只管写、③ 只管挑错、④ 只管收敛，③④ 面对的是"别人写的草稿"，否定成本低得多。
**代价要说清楚**：一次问答要跑 3~4 次模型。**这个代价是否值得，本部分给不出结论**——`MedicalPromptTemplates.chain(evaluate=False, review=False)` 留了消融开关，第二部分用同一批问题做 A/B 实测。

**7. 温度不是随手写的。** ①③ 要可解析、可复跑的结构化判断 → 0.0；② 需要组织语言的自由度但不能发散 → 0.3；④ 只做收敛式改写 → 0.2。

---

## 五、⚠ 关键点：token 预算必须精确，而本机原本没有 Qwen 分词器

**为什么不能糊弄。** Ollama 超出 `num_ctx` 时**不报错，而是静默丢掉最前面的内容**——通常正是 system 提示词和排在最前的高相关证据。也就是说预算低估的表现不是崩溃，而是"答得莫名其妙变差"，极难排查。所以代价不对称：高估只浪费一点窗口，低估直接毁掉这一轮生成。

**本机的问题。** `hf-cache` 里只有 `bge-base` / `bge-m3` / `bge-reranker-base`，没有 Qwen3 的 HF 分词器，而项目是全离线的。

**解法。** Ollama 的 qwen3:8b GGUF **自带完整分词器**（`tokenizer.ggml.tokens` 151,936 条 + `merges` 151,387 条，模型 `gpt2` 即 byte-level BPE，pre `qwen2`）。`生成_分词器.py` 只读 GGUF 头部的 KV 元数据（不碰 5 GB 张量），用 `tokenizers` 重建成同一套 BPE，再缓存成 JSON。正确性用 1000 条真实文本 `encode→decode` 往返一致来验证（往返一致对 byte-level BPE 是强校验）。

**降级路径**：显式 HF 分词器目录 → 字符启发式。启发式常数 3.0 chars/token 是实测标定的：真实块 **96.2% 被高估**、中位高估 1.50×，但公式密集的极端块仍会低估（最差 0.44×）——所以它只是降级路径，默认永远走精确分词器。

**num_ctx 该设多少**（`MedicalPromptTemplates.plan_budget` 按最坏情形算，`RECOMMENDED_NUM_CTX = 12288`）：

| num_ctx | 证据预算 | 说明 |
|---|---|---|
| 4096（Ollama 默认，已实测确认） | — | **装不下**，必须改 |
| 8192 | 2,312 tok | 可降级运行，但要显式把 `max_context_tokens` 调到 2312 以下 |
| **12288（建议）** | **6,408 tok** | **实测显存 6.33 GB**（`/api/ps` 的 `size_vram`），10 G 卡留有余量 |
| 16384 | 10,504 tok | 未实测；按 KV ≈0.14 MB/token 外推约 +0.6 GB |

预算公式：`证据预算 = num_ctx −（最坏段固定开销 + 该段输出上限）−（草稿上限 + 审查意见上限）− 参考文献预留 600`。全部按上限计，是刻意的最坏情形。实测最坏一段（默认 2800 证据预算 + 满载草稿/审查 + 最长输出）合计 **8,468 ≤ 12288**。

---

## 六、验证：65/65，每条结论都由数据算出

`生成_上下文组装_验证.py`，离线部分 7 秒跑完、不需要加载 4M 库；**没有任何一条是无条件打印的**（阶段五踩过"无条件 `print(✓)`"的坑）。凡是"不会误伤"类结论都在阶段三的 1000 条真实文本块上跑，而不是只用手造的小例子。

- **A 分词器（10 项）**：离线加载、词表 151,936、1000 条往返一致、计数为正且单调、空串为 0、截断不超上限（精确/启发式两种模式）、组装器与分词器计数一致、启发式保守性实测。
- **B 去重（9 项）**：自相似 1.0、不同正文低于阈值、3-gram 优于词袋（1.000 → 0.000）、逐字重复被丢、簇内保留最相关者、近重复（0.889）被丢、无关正文不误删、退化短块前置丢弃、真实语料零误删。
- **C 排序与多样性（5 项）**：关掉多样性时严格按相关性降序；硬上限生效；多样性确实让第 3 篇文献进入；同源第 4 条被排到别的文献之后；有效分公式手算核对（误差 `0.0e+00`）。
- **D 预算与截断（10 项）**：4 组预算都不超、装载量单调、元数据与实测一致、544 次截断全部落在完整句/段、截断是原文前缀、不越出末 10% 窗口、缩写点不误判（正反两向）、极小预算安全退化、大块跳过后小块仍入选。
- **E 元数据与引用（9 项）**：5 个必需字段齐全、计数自洽、`chunk_sources` 与入选块一致、`[S#]` 连续唯一且与入选块一一对应、引用元数据与块元数据一致、出处头对模型可见、参考文献编号对齐。
- **F 提示词模板（15 项）**：四段齐全、温度/长度合法、结构化两段温度为 0、模板变量自动推导正确、缺变量报错、JSON 花括号不被误替换、messages 形状、证据确实进入提示词、`/no_think` 附加、`max_tokens→num_predict` 映射、消融开关、三档 num_ctx 预算自洽、最坏一段落在 num_ctx 内。
- **G 端到端（4 项，`--live`）**：真实 `Candidate` 直接转 `DocumentChunk`、相关性取自重排总分、真实上下文不超预算（2,670/2,800）、证据来自 5 篇不同文献。
- **H 与 Ollama 实测对照（3 项，`--ollama`）**：本地重建的分词与 Ollama 实际分词逐条一致（9 样本差值集合 `[0]`）、Ollama 默认 `num_ctx` 实测为 4096、`num_ctx=12288` 时实测显存 6.33 GB。**这一组是本部分唯一真正碰到模型的地方**（只做分词与显存对照，不做生成）。

修 bug 的过程本身也留在这里：首轮跑出 4 项 FAIL，都是真问题——① 贪心循环把 `min_fragment_tokens`（截断片段下限）错当成整体预算下限，导致小块也进不来；② 缩写判定按词元取最后一个词，`e.g.` 的最后一个词元只是单字母 `g`，判不出来；③ `plan_budget` 漏算了参考文献列表的开销（实测每条 ~40 tok）；④ "真实语料零误删"的断言写成了"一条都不能丢"，而实际丢的 7 条是逐字重复的 `"xx"` 退化块——**是断言写松了，不是代码错了**，改成"被丢的必须逐字重复"后才是真检查。

---

## 七、复现命令

```powershell
$py = "E:\rag\conda\envs\medrag\python.exe"
$env:PYTHONIOENCODING = "utf-8"

# 1) 重建 Qwen3 分词器缓存并自检（首次会自动建，无需手动跑）
& $py E:\rag\scripts\生成_分词器.py --rebuild

# 2) 离线演示：用阶段三的 1000 条样例块组装一次上下文
& $py E:\rag\scripts\生成_上下文组装.py --demo --limit 40 --budget 1200

# 3) 看提示词
& $py E:\rag\scripts\生成_提示词模板.py --list
& $py E:\rag\scripts\生成_提示词模板.py --show critical_reviewer
& $py E:\rag\scripts\生成_提示词模板.py --budget 12288      # 按 num_ctx 算证据预算

# 4) 验证（离线 58 项，7 秒）
& $py E:\rag\scripts\生成_上下文组装_验证.py
# 4b) 追加真检索→组装端到端冒烟（加载 4M 库）+ 与 Ollama 的分词/显存实测对照
& $py E:\rag\scripts\生成_上下文组装_验证.py --live --ollama --bm25 E:\rag\data\bm25_index_4m

# ---- 以下为第二部分（要 Ollama 在跑；起服务见文末）----

# 5) LLM 生成器自检：JSON 容错 11 项（不调模型）+ 冒烟 7 项（真调模型）
& $py E:\rag\scripts\生成_LLM生成器.py --json-selftest
& $py E:\rag\scripts\生成_LLM生成器.py --smoke

# 6) 流水线单次演示（离线证据，完整四段链，约 25 s）
& $py E:\rag\scripts\生成_流水线.py --demo
& $py E:\rag\scripts\生成_流水线.py --demo --no-review        # 消融：2 段

# 7) 完整流程测试（产物落 report_data\）
& $py E:\rag\scripts\生成_流水线_测试.py --offline     # 4 题 × 四段链，约 2 min
& $py E:\rag\scripts\生成_流水线_测试.py --ablation    # 4 题 × 3 配置，约 3 min
# 8) 真检索验收 —— 两趟跑法（推荐）
#    检索器约 15.8GB、Ollama 的 llama-server 约 8.1GB，32G 机器同时开会换页。
#    但这两件事本不必同时发生：先把检索结果固化成快照，再单独跑生成。
& $py E:\rag\scripts\生成_流水线_测试.py --dump-retrieval E:\rag\report_data\检索快照_live.json `
      --bm25 E:\rag\data\bm25_index_4m      # 第一趟：只加载检索器，全程不连 Ollama
& $py E:\rag\scripts\生成_流水线_测试.py --from-dump E:\rag\report_data\检索快照_live.json
                                            # 第二趟：只加载 Ollama，不碰检索器

# 8b) 一趟跑完（内存充裕时才用）
& $py E:\rag\scripts\生成_流水线_测试.py --live --bm25 E:\rag\data\bm25_index_4m

# 9) 接入前后对比评测（复用上面的快照与已跑好的 RAG 结果，只补跑裸模型，约 1 分钟）
& $py E:\rag\scripts\生成_对比评测.py --snapshot E:\rag\report_data\检索快照_live.json
```

> 两趟跑法还有个副产品：检索结果固化成快照后，重跑生成或换链路配置不必再等检索器加载
> （实测 67.2s），评测迭代快得多。第一趟前建议先 `ollama stop qwen3:8b` 把模型从内存里放掉。

依赖：`tokenizers`（已装）；第二部分只用标准库 `urllib` 调 Ollama HTTP 接口，无新增依赖。
第 1~4 步**不需要** Ollama 在跑——分词器是从磁盘上的 GGUF 读的；第 5~7 步要连服务：

```powershell
$env:OLLAMA_MODELS = "E:\rag\ollama\models"      # 不设会去找改名前的旧路径
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList 'serve' -WindowStyle Hidden
```

（仓库根的一键启动脚本会顺带做这件事并激活 conda 环境，见阶段一交付包。）

---

## 八、第二部分实测结果

**A. `LLMGenerator` 自检**（`--smoke`，含 JSON 自检）

| 组 | 项数 | 结果 |
|---|---|---|
| JSON 容错解析（不调模型） | 11 | **11/11**：裸对象 / 代码块包裹 / 前后缀寒暄 / `<think>` 段 / 尾随逗号 / 截断缺右括号 / 截断在半个键值对 / 围栏只开不闭 / 全角引号 / 空输出 / 非 JSON |
| 真调模型冒烟 | 7 | **7/7**：文本非空 · 无 thinking 段 · 报告 token 用量 · JSON 解析成功 · `num_ctx=12288` 生效 · 批量两条成功 · `num_predict=0` 被拦截 |

`num_ctx` 生效是拿 `/api/ps` 实测的，不是看配置：`context_length=12288`，`size_vram=6.33 GB`——与阶段七之一按 KV cache 估算的 1.8 GB + 权重 5.2 GB 吻合，10 G 卡有余量。

**B. 完整链路**（`--offline`，4 道题各跑完整四段链，证据来自阶段三 1000 条样例块）

| 判定 | 结果 |
|---|---|
| 全部用例四段链无失败阶段 | **4/4** |
| 结构化阶段（①③）JSON 全部解析成功 | **8/8** |
| 无编造出处编号 | **0 例** |
| 每份答案都带真实引用 | 缺引用 **0 例** |

耗时 min 29.3s / 均 33.4s / max 36.2s；答案 min 1186 / 均 1857 / max 2508 字；共 16 次模型调用。

> ⚠ **这些数字每次跑都不一样**：②答案生成器温度 0.3、④定稿 0.2，本来就不是确定性的。
> 同一套代码前后两轮实测均耗时 30.9s → 33.4s、均字数 2301 → 1857。**上面四项判定（4/4、8/8、
> 0 例、0 例）两轮都成立，耗时与字数只作量级参考，不要当成可复现的定值引用。**
> 引这些数字时以 [`生成_流水线测试报告_offline.txt`](7.2_本地LLM集成与生成流水线/生成_流水线测试报告_offline.txt) 里的为准。

其中 `off-4` 是**故意问语料答不了的问题**（pembrolizumab 在 NSCLC 的 2024 指南起始剂量）。模型的回答是"现有证据中未提及…所有文献均发表于 2003 年，无法反映 2024 年指南"，**没有编出一个剂量**——这正是整个 RAG 系统要买的行为。原始输出见 [`生成_流水线测试报告_offline.txt`](7.2_本地LLM集成与生成流水线/生成_流水线测试报告_offline.txt)。

**C. 真检索验收**（`--live`，阶段一留下的四类错题，走 400 万向量库 + BM25）

检索侧：四道题各命中 10 条、来自 9~10 篇不同文献，年份 2014–2024（正是裸模型最容易编的近年文献）。

| 判定 | 结果 |
|---|---|
| 四段链无失败阶段 | **4/4** |
| 结构化阶段 JSON 全部解析成功 | **8/8** |
| 无编造出处编号 | **0 例** |
| 每份答案都带真实引用 | 缺引用 **0 例** |

耗时均 42.1s（检索器加载 67.2s 一次性，单次检索 0.56~1.27s）。逐题核对四类错：

| 用例 | 裸模型当时的错 | 本次 RAG 的回答 | 关键事实能否在检索原文核到 |
|---|---|---|---|
| ① CRISPR 脱靶 | 标题/期刊/卷页全编 | 参考文献全部来自检索证据，`[S#]` 与文末列表一一对应 | ✅ 编造编号 0 |
| ② 法布里病 ERT | 品牌名张冠李戴 | agalsidase-α / agalsidase-β / migalastat / 重组人 α-半乳糖苷酶 | ✅ 药名逐个命中 |
| ③ 多发性硬化 DMT 分类 | 单抗/S1P/干扰素混淆 | 按证据说"moderately/highly effective"，未硬套证据没有的分类 | ✅ |
| ④ 阿尔茨海默时间线 | 年份自相矛盾 | lecanemab 获批 2023-01-06、CLARITY-AD 报告 2022-11、aducanumab 2021 加速批准，前后自洽 | ✅ 含统计量 −0.45（p<0.001）原文可核 |

> ⚠ 这是**逐题人工核对**，不是自动评测。"关键事实能否核到"是我把答案里的药名、年份、试验名回到检索快照原文里 grep 的结果，可复核（快照在 `report_data\检索快照_live.json`）。
> **它证明的是"没编造"，不是"答得全面"**——比如③只覆盖了证据提到的分类维度，没有给出完整的 DMT 分类体系，因为证据里就没有。

**D. 接入前后对比评测**（`生成_对比评测.py`，同一道题：RAG 四段链 vs 裸 `qwen3:8b`）

公平性上做了两件事，否则这个对比不成立：① 裸模型用的是项目日常问答那套 system prompt
（**明确写了"不得编造参考文献"、"请给出可核对的出处（作者/期刊/年份/PMID）"**），不是随手写一个差的；
② 同模型、同 `num_ctx`、同 `think=False`、同 `max_tokens`，温度取 0.3 与 RAG 的答案生成段一致。

| | 出处可溯源 | 年份可核 | 药名可核 | 均字数 | 均耗时 |
|---|---|---|---|---|---|
| **RAG** | **22/22（100%）** | 10/17 | 12/13 | 2528 | 42.1 s |
| 裸模型 | **0/18（0%）** | 8/16 | 8/12 | 2557 | 9.8 s |

逐题看裸模型给出了什么形态的"出处"：

| 用例 | RAG | 裸模型 | 裸模型给的是什么 |
|---|---|---|---|
| ① CRISPR 脱靶 | 5/5 | 0/5 | 5 条"作者 (年份) – 期刊"，**一个 PMID/DOI 都没有** |
| ② 法布里病 ERT | 5/5 | 0/2 | 2 个 PMID |
| ③ 多发性硬化 DMT | 6/6 | 0/7 | **7 个 NEJM DOI** |
| ④ 阿尔茨海默 | 6/6 | 0/4 | 4 条"作者 (年份) – 期刊"，无标识符 |

**这张表真正说明的事**：裸模型会成规模地生产**引用形态的文本**——四道题一共 18 条，其中 9 条还带着
像模像样的 PMID / NEJM DOI——但**这套系统一条也验证不了**。RAG 侧的 22 条则条条能给出 PMCID 与
PMC 链接。字数几乎相同（2528 vs 2557），代价是 4.3 倍耗时。

> ⚠ **口径必须说清**：「可溯源 0/18」的意思是**无法用本系统核对**，**不等于这 18 条都是假的**——
> 其中一部分很可能是真论文（如 ① 的 Doudna 2012 Science）。RAG 买到的是**可验证性**，不是全知。
> 同理 B/C 两组的「核不到」也只说明它不在该题检索回的 top_k 条证据里，不代表错。
>
> ⚠ 另有一处**我读出来但机器没验证**的问题，供人工复核：③ 里裸模型把 **tofacitinib**（JAK 抑制剂，
> 类风湿关节炎用药）列为多发性硬化的 DMT 并配了一个 NEJM DOI。若属实，这正是阶段一记录的
> 「药物分类错误」原样复现。**这是我的判断，不是脚本的判定**，报告里不作为结论。

**E. 消融对照**（`--ablation`，同题同证据，只改链路配置）

| 配置 | 模型调用 | 均耗时 | 相对基线 | 均字数 | 编造出处 |
|---|---|---|---|---|---|
| 直接作答（对照基线） | 1 | 5.8 s | 1.00× | 1770 | 0 |
| 评估 + 作答 | 2 | 10.8 s | 1.87× | 1760 | 0 |
| 完整四段链 | 4 | 29.0 s | **4.99×** | 1742 | 0 |

12 组配置的模型调用次数与预期完全一致（12/12），三种配置都没出现编造出处。

**能从这张表得出的结论**：四段链的代价是 **约 5 倍耗时**，且答案长度基本不变（1742 vs 1770 字）——多花的时间没有变成更长的废话。
**不能从这张表得出的结论**：四段链是否**更忠于证据**。两个观察值得记下但都还不是证据：① 完整链的引用落地数普遍更高（4~5 条 vs 基线 1~3 条）；② 审查器在每份草稿里都挑出 3~6 个问题、`verdict` 多为 `revise`/`reject`。这只能说明③在做事，不能说明④改对了。**要判定必须有人工标注评测集**，见「九」。

---

## 九、已知局限与下一步

**已知局限（如实说明）**
1. **量到的是「可验证性」，不是「正确性」**：D 表能硬证的是"RAG 的出处 100% 可溯源、裸模型 0%"。至于两边**内容谁更对**，仍然只有人工逐题核对（C 表），没有标注集，也没有第三方评判。"答得更准"这句话目前只能说到"可核对的部分都核得上"。
2. **答案会中英混排**：④那道题的「回答」「证据要点」是英文，「证据强度与一致性」却切回了中文。提示词里写了"回答语言与用户提问语言保持一致"，模型没完全照做。不影响事实正确性，但交付观感上是缺陷。
3. **参数仍未经效果验证**：相似度阈值 0.80、`diversity_decay` 0.75、`max_per_source` 3、块上限 600 tok、`min_relevance_keep` 1，都只验证了"算得对、边界对"。
4. **多样性只按文献（pmcid）分组**：同一课题组的不同论文、同一试验的多篇报告仍会被当成不同来源。
5. **筛证据后编号不连续**：①判掉某条证据后其余块保留原编号（这是刻意的，见 `生成_流水线.py` 模块 docstring 取舍 1），最终答案里可能出现 `[S1][S3][S4]`。功能正确，但看起来会让人以为漏了。
6. **审查器偏严**：`verdict` 在离线测试里多为 `revise`/`reject`，`overall_grounded=False` 出现频繁。提示词里写的是"拿不准时按未被支持记，宁可多报"，这个倾向是设计使然，但**是否报得过多**没有标注集就无法判断。

**下一步**
1. **把裸模型那 18 条引用真的查一遍**（联网或对着 PubMed 本地元数据），把"无法用本系统核对"升级成"其中 N 条确实不存在"。这一步做完，"减少幻觉"才有硬数字，而不是只有可验证性对比。
3. **建人工标注小评测集**，把阶段五、六、七累积的未决问题一次量化掉：同义词扩展 B/C/D、`rrf` vs `weighted`、重排权重 0.6/0.25/0.15、去重/多样性参数、以及四段链 vs 直接作答（消融开关已就绪，缺的只是质量标注）。

---

## 十、本包内容

```
任务7\
├─ README.md                          本文件（两部分总览 + 两张对照任务书表）
├─ 任务书.txt                          任务书原文（两部分）
├─ 上下文组装与提示词工程报告.docx      阶段报告，14 节覆盖两部分
│                                     （数字由脚本从实测产物解析，不手写）
│
├─ 7.1_上下文组装与提示词工程\
│  ├─ 上下文组装验证报告.txt            65/65 验证的原始输出
│  └─ 新建 Microsoft Word 文档.DOCX     汇报稿
│
├─ 7.2_本地LLM集成与生成流水线\
│  ├─ 汇报稿.docx                      汇报稿
│  ├─ 生成_流水线测试报告_offline.txt   完整链路 4/4（含答案全文与出处清单）
│  ├─ 生成_流水线测试报告_live.txt      真检索验收 4/4（四类错题的答案与出处）
│  ├─ 生成_流水线测试报告_ablation.txt  消融对照 12 组
│  └─ 生成_对比评测报告.txt             RAG vs 裸模型并排全文 + 可溯源统计
│
└─ 脚本\                              交付拷贝；**单一来源在顶层 scripts\**，改代码改那边
   ├─ 生成_上下文组装.py               之一 核心：DocumentChunk / ContextAssembler
   ├─ 生成_提示词模板.py               之一 PromptStage / 四段医学提示词 / 预算规划
   ├─ 生成_分词器.py                   之一 从 Ollama GGUF 重建 qwen3:8b 分词器
   ├─ 生成_上下文组装_验证.py           之一 65 项验证（--ollama / --live 为可选组）
   ├─ 生成_LLM生成器.py                之二 LLMGenerator：连接自检 / JSON 容错 / 批量生成
   ├─ 生成_流水线.py                   之二 MedicalGenerationPipeline：六步生成流程
   ├─ 生成_流水线_测试.py               之二 测试查询集 + 指标统计 + 日志落盘
   ├─ 生成_对比评测.py                  之二 接入前后对比：RAG vs 裸 qwen3:8b
   └─ 生成_报告转word.py               生成上面那份 docx
```

> ⚠ **`脚本\` 刻意不按 7.1 / 7.2 拆开**：这 9 个模块互相按**同目录路径**导入
> （`_load_by_path(os.path.join(_HERE, "生成_上下文组装.py"))`），拆进两个子目录后
> 之二的脚本就找不到之一的模块，包内直接跑会 import 失败。报告和汇报稿按部分归类即可。

> 这一组文件是自洽的（互相按**同目录**路径导入），拷贝里直接跑也可以，已实测。
> 但**唯一真源是顶层 `scripts\`**：要改代码改那边，别改拷贝，否则两份会漂
> （仓库级的交付包校验脚本带 `--fix` 会做单向同步并校验 MD5）。
> 三点例外（都已实测，脚本对缺失均给可操作提示，不是裸异常）：
> ① 验证与测试脚本的报告固定写到 `E:\rag\report_data\`（绝对路径）；
> ② `--live` 那一组还依赖阶段六的多路检索一链（在阶段六交付包里），本包未重复收录；
> ③ `--demo` / `--offline` 用的样例证据文件是**阶段三的交付物**，本包同样未重复收录——
> 　 只拿到本包时这两个模式跑不了，改用 `--live`，或把该文件从阶段三包里取来。
>
> **包内拷贝已实测可独立运行**（2026-07-31，在 `任务7\脚本\` 目录内直接执行）：
> 冒烟 7/7、`生成_流水线.py --demo` 四段链跑通、`生成_流水线_测试.py --offline` 4/4。

**本地留存产物**（可由脚本重建）：`data/tokenizer/qwen3_tokenizer.json`（11 MB）、
`report_data/生成_上下文组装验证报告.txt`、`report_data/生成_流水线测试_{offline,ablation}.jsonl`。
