# 阶段五 · 检索系统（一）查询理解与增强

> 把用户的自然语言查询，变成检索层能直接消费的结构化增强查询（`EnhancedQuery`）。

## 接手卡

**状态**：✅ 完成 ｜ 2026-07-20 ｜ 离线自检 12/12，在线验证（400 万向量库实跑）全绿

| 项 | 内容 |
|---|---|
| **输入** | 原始自然语言查询（中文 / 英文均可） |
| **产物** | `EnhancedQuery` 结构体；词典 `data/dict/mesh_synonyms.json`（9.5 MB）、语料分布 `data/dict/corpus_meta.json` |
| **核心模块** | `scripts/检索_查询理解.py :: MedicalQueryProcessor.process_query()` —— 六步流水线（清洗 → 提过滤条件 → 中译英 → 实体识别 → 同义词扩展 → 生成查询版本） |
| **下游调用** | `eq.vector_queries`（已带 BGE 前缀，`[0]` 为主查询）· `eq.vector_query_weights` · `eq.keyword_query`（BM25 用）· `eq.filters`（直接给 `collection.query(where=…)`）· `eq.post_filters`（section 须检索后过滤）· `eq.entities` · `eq.notes` |
| **前置构建** | `& $py scripts\检索_构建同义词词典.py`（约 6s，需先下 `data/mesh/desc2026.xml`）；`& $py scripts\检索_扫描元数据分布.py`（约 17s） |
| **验证** | `& $py scripts\检索_查询理解_验证.py`（加 `--offline-only` 只跑功能自检，不加载 400 万向量库） |
| **下游** | 阶段六 `MultiPathRetriever` 直接消费 `EnhancedQuery` → `任务6/README.md` |
| **遗留** | `term-hit@10` 是代理指标且对平铺扩展有构造性偏袒，扩展策略 B/C/D 优劣**未判定**；未接 UMLS（需 UTS 审批）；商品名仅手工 32 条；期刊过滤要求字面精确 |

> `$py` = `E:\rag\conda\envs\medrag\python.exe`
> 中文文件名无法直接 `import`，下游一律用 `importlib.util.spec_from_file_location` 按路径加载（见「五、复现命令」末尾示例）。

---

## 一、对照任务书（逐条）
> 本阶段任务书原文见 [`任务书.txt`](任务书.txt)（逐字照录，未改写）。


| 任务书要求 | 实现 | 位置 |
|---|---|---|
| 静态医学同义词词典 `MEDICAL_SYNONYMS` | ✅ 保持扁平 `{词: [同义词]}` 接口，内部拆 4 张表便于维护 | `检索_查询理解.py` |
| 「实际应用中应更全面，可从 UMLS、MeSH 构建」 | ✅ **已构建**：解析 MeSH 2026 主题词表 → 26,694 主题词 / 107,944 词面 | `检索_构建同义词词典.py` |
| 医学实体 `MEDICAL_PATTERNS`（`\b` 单词边界） | ✅ 6 类正则 + MeSH 词典最长匹配双路 | `检索_查询理解.py` |
| 处理医学查询，返回增强的查询信息 | ✅ `MedicalQueryProcessor.process_query()` → `EnhancedQuery` | 同上 |
| 　· 基础清洗 | ✅ NFKC 归一、全角转半角、压空白、剥离客套引导语 | `_clean()` |
| 　· 识别医学实体 | ✅ 正则 + gazetteer，带类型/来源/置信度/歧义标记 | `extract_entities()` |
| 　· 同义词扩展 | ✅ 静态词典优先，MeSH 补覆盖，按实体和总量双重限额 | `expand_synonyms()` |
| 　· 生成向量查询 | ✅ 带 BGE 指令前缀；主查询 + 缩写消歧变体 + 融合权重 | `_build_vector_queries()` |
| 　· 生成关键词查询 | ✅ 组内 OR、组间 AND，可直接喂 BM25 | `_build_keyword_query()` |
| 　· 提取过滤条件（时间范围等） | ✅ `pub_year` / `section` / `journal`，中英文都支持 | `_extract_filters()` |
| BGE 最佳实践：查询加指令前缀 | ✅ 内置官方版与任务书版两条前缀，并做了 A/B 实测 | 验证 ③ |

---

## 二、核心结果（验证）

| 验证项 | 结果 |
|---|---|
| ① 功能自检（12 条查询，覆盖缩写/歧义/俗称/商品名/拼写/中文/时间/章节/无实体） | ✅ **12/12 通过**，0.1 秒 |
| ② 过滤条件下推（生成的 `where` 真丢给 Chroma 跑） | ✅ 4/4 返回非空，命中元数据逐条满足边界 |
| ③ BGE 指令前缀 `sentence`(官方) vs `question`(任务书) | 0.87 / 0.87，top10 重合 90% → 差异在噪声量级 |
| ④ 同义词扩展策略 | 见「四、4」 |
| ⑤ 中文 直接检索 vs 中译英 | **0.37 / 0.97** |

> **关于 term-hit@10**：本阶段没有人工标注集，用「top-k 命中块正文里是否出现目标医学术语」作相关性代理指标。它不等于真实相关性，但对「缩写有没有被理解」「中文有没有落到英文术语上」这类问题区分度足够，且完全可复现。真正的相关性评测放在第二部分。

---

## 三、两层词典：把任务书的「示例」做成能用的东西

任务书给的 `MEDICAL_SYNONYMS` 是示例，并注明「实际应用中应更全面，可从 UMLS、MeSH 等构建」。这条提示已落实。

| 层 | 规模 | 管什么 | 不可替代之处 |
|---|---|---|---|
| 静态精编 | 298 条（缩写 117 / 俗称 30 / 商品名 32 / 拼写 24 / 中英 95） | 缩写、患者用语、商品名、英美拼写、中英对照 | **MeSH 基本不收缩写**——"MI" 根本不是 MeSH 入口词，这层只能手工维护，但价值最高 |
| MeSH 2026 | 26,694 主题词 / 107,944 词面 | 术语规范化、覆盖面、实体类型 | 手工不可能覆盖到这个量级；实体类型直接由 MeSH 树号给出 |

**为什么选 MeSH 不用 UMLS**：UMLS 覆盖最全（400 万概念），但要注册 UTS 账号走 license 审批（通常数天），本周交不了；MeSH 是 NLM 官方叙词表、免费直接下载，且 PubMed 文献本来就用 MeSH 标引，与本项目语料同源。UMLS 已记为后续扩展项。

**MeSH 降噪**（不做的话查询会被扩散成噪声）：丢弃机器轮排词 116,350 条；丢弃非医学分支（卫生服务/社会学/情报学/出版类型/地理）4,361 个主题词；丢弃不合格词面 4,091 条；把倒装词面还原成自然语序（`Infarction, Myocardial` → `myocardial infarction`）。只保留 6 类树号：药物化学 10,649 / 疾病 5,193 / 生物 3,938 / 诊疗技术 3,090 / 生理过程 2,050 / 解剖 1,774。构建耗时 5.5 秒，产物 9.5MB。抽检：`metformin → glucophage`、`MI → heart attack`、`type 2 diabetes mellitus → NIDDM`。

---

## 四、关键设计取舍（都有实测或语料数据支撑）

### 1. 过滤短语先剥离，再送去做向量
`"metformin cardiovascular outcomes since 2020"` 里的 `since 2020` 已变成 `where` 子句，留在文本里只会污染语义向量。模块把命中的过滤短语从查询剥掉，剩下的「检索主体」才送 embedding。

### 2. 中文查询必须先转英文 —— 实测差距最大的一项
索引是 `bge-base-en-v1.5`（**纯英文模型**）+ 英文 PubMed 语料，而需求方给的示例查询正是中文。

| 平均 term-hit@10（3 条中文查询） | 直接中文检索 | 中译英后 |
|---|---|---|
| | **0.37** | **0.97** |

最直观的一例，Q「二甲双胍对心血管疾病有何影响？」：直接中文 → top1《TaiChi and Qigong for Depressive Symptoms in Patients with Chronic Heart Failure》（太极/气功治抑郁，sim 0.633）**完全跑题**；中译英 → top1《Protective effects of metformin in various cardiovascular diseases》（sim 0.803）**正中题意**。实现上给了两条路：**词典直译**（离线、零依赖、默认）和 **本地 qwen3:8b 整句翻译**（`--translate llm`，覆盖词典盲区）。

<details><summary>详细实测过程（为何 0.37 是"检索失败"而非"一般"，及受控对比说明）</summary>

3 条中文查询的直接检索得分是 **0.10 / 0.00 / 1.00**，均值 0.37 掩盖了一个规律：拿到 1.00 的那条是「CRISPR 基因编辑的脱靶效应」，它**含拉丁字母 token `CRISPR`**，英文模型抓住了这个锚点。也就是说，纯中文查询（前两条）基本等于检索失败，混入英文术语的才勉强可用——这反而更说明翻译层是必需的。

另一例：Q「肿瘤微环境在癌症免疫治疗中的作用」直接检索甚至命中了中文标题文献《铁死亡的发生机制及其在肺癌中的研究进展》——中文查询会被拉向语料里少量的中文内容，而不是主题相关的英文文献。

两臂用的是同一条英文指令前缀，唯一变量是查询正文的语言，属受控对比；且定性证据（太极治抑郁 / 中文标题文献）与代理指标独立互证，不是单靠指标下的结论。
</details>

### 3. `section` 过滤按语料实测自适应
全量扫描 400 万块发现 `section` 原始取值有 **391,164 种**写法。归一到 6 类规范章节后：

| 规范章节 | 覆盖 99% 所需写法数 | 实现方式 |
|---|---|---|
| abstract / introduction / discussion / results / conclusion | 1 / 5 / 7 / 24 / 47 | ✅ 下推 Chroma `$in` |
| **methods** | **7,726** | ⚠ `$in` 不现实 → 检索后用同一套归一化函数后置过滤 |

阈值 `SECTION_IN_LIMIT=60`，超过自动降级并在 `notes` 说明原因。另有 17.4% 的块章节名无法归类（`(no-section)`、`Case presentation`、`Main text` 等），章节过滤会漏掉这部分——已标注。

### 4. 同义词扩展方式 —— 结论：能救回缩写查询，但 B/C/D 谁更好下不了结论
一句话结论：**✅ 确定「扩展救回了缩写查询」**（不扩展时缩写查询 term-hit@10 只有 0.20，灾难级）；**❌ 但 B/C/D 三种扩展/融合方式谁更好，无法判定**——只有 6 条查询、且代理指标对「平铺扩展」有构造性偏袒。这个过程中我的判断被数据**修正了两次**。

<details><summary>详细实测过程（A/B/C/D 四臂、两次被数据修正的判断、term-hit@10 的构造性偏袒）</summary>

最初的判断是：「同义词拼进单条向量查询会把查询向量拉向几个词的质心、稀释主题，应该走多查询 + RRF」。实测（6 条查询，term-hit@10）：

| A 主查询（不扩展） | B 单查询平铺 | C 多查询等权 RRF | D 多查询加权 RRF |
|---|---|---|---|
| 0.87 | **0.95** | 0.90 | 0.90 |

**第一次修正：平铺（B）最高，与我的预设相反。** 但必须声明一个方法学偏差：term-hit@10 的目标术语**就是**同义词扩展项，把扩展项塞进查询天然更容易命中它们——**这个指标对 B 有构造性偏袒**，不能据此判 B 胜。

去掉饱和噪声看真信号：6 条里 5 条都是 1.00，唯一有区分度的是缩写查询 `"Does MI risk increase in patients with CKD?"` → **A=0.20 / B=0.70 / C=0.40 / D=0.40**。

**第二次修正（针对我自己的改动）：** 看到等权 RRF 里主查询是最差的一路（0.20），我推测「给主查询降权应该能拉高融合结果」，于是加了 `vector_query_weights`（主查询降权到 0.5）并重跑了一次完整验证。结果 **D 与 C 完全相同（0.40 / 0.90），没有任何可测量的改善——假设未被支持**。原因也清楚：RRF 的 `score = w/(k+rank)` 在 k=60 时对 2 倍权重变化并不敏感，排名位置压过了权重差异。

所以对这个权重的定位必须说清楚：它**不是**一项已验证的改进，而是把「主查询里含未展开的缩写」这个**信号**显式交给检索层，默认值 0.5 未经验证。真正该试的更强干预是**直接丢弃主查询**或**调小 k**——留给第二部分。

严格区分「确定的」和「不能下的」：
- ✅ **确定**：扩展救回了缩写查询。不扩展时 0.20 是灾难级——模型没理解 `MI`/`CKD`。
- ❌ **不能下的结论**：B / C / D 谁更好。6 条查询 + 有偏指标不足以判定，留给第二部分带人工标注的相关性评测。
- ❌ **被否掉的假设**：给主查询降权能改善融合结果（实测无差异）。
</details>

---

## 五、复现命令

```powershell
$py = "E:\rag\conda\envs\medrag\python.exe"

# 1) 构建 MeSH 同义词词典（需先下载 desc2026.xml 到 data\mesh\，298MB）
#    curl -L -o E:\rag\data\mesh\desc2026.xml https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml
& $py E:\rag\scripts\检索_构建同义词词典.py          # 约 6 秒 -> data\dict\mesh_synonyms.json (9.5MB)

# 2) 扫描语料元数据分布（章节写法映射 + 年份覆盖）
& $py E:\rag\scripts\检索_扫描元数据分布.py          # 约 17 秒 -> data\dict\corpus_meta.json

# 3) 单条查询 / 演示查询集
& $py E:\rag\scripts\检索_查询理解.py --query "Does MI risk increase in patients with CKD?"
& $py E:\rag\scripts\检索_查询理解.py --demo

# 4) 质量验证（--offline-only 只跑功能自检，不加载 400 万向量库）
& $py E:\rag\scripts\检索_查询理解_验证.py --offline-only
& $py E:\rag\scripts\检索_查询理解_验证.py
```

下游检索层这样调用（中文文件名按路径导入，与项目既有脚本一致）：

```python
import importlib.util
spec = importlib.util.spec_from_file_location("qu", r"E:\rag\scripts\检索_查询理解.py")
qu = importlib.util.module_from_spec(spec); spec.loader.exec_module(qu)

proc = qu.MedicalQueryProcessor()
eq = proc.process_query("二甲双胍对心血管疾病有何影响？")

eq.vector_queries          # 已带 BGE 指令前缀，[0] 为主查询
eq.vector_query_weights    # 与上一一对应，供加权 RRF
eq.keyword_query           # BM25 用，组内 OR、组间 AND
eq.filters                 # 直接作为 Chroma collection.query(where=...)
eq.post_filters            # 无法用 where 表达、需检索后过滤的条件
eq.entities, eq.expansions # 可解释性：识别到什么、扩展了什么
eq.notes                   # 处理过程中的提示与告警
```

---

## 六、已知局限与下一步

**已知局限（都已在代码 `notes` 里对用户可见）**
1. **中文词典直译会丢词**：如「CRISPR 基因编辑的**脱靶效应**」中「脱靶效应」未收录，译文丢失该约束。已内置 `--translate llm` 走本地 qwen3:8b 整句翻译作为补救，**已联机实测可用**（正确译出 `Off-target effects of CRISPR gene editing`）；代价是延迟——冷启动约 60 秒（5GB 权重进显存）、之后每次约 5 秒，已加超时重试避免冷启动被静默降级。
2. **term-hit@10 是代理指标**，且对平铺扩展有构造性偏袒，不足以判定策略优劣。
3. **MeSH 主表不含商品名全集**：商品名多在补充概念记录 `supp2026.xml`（约 3GB），本阶段未引入，常用的先手工维护了 32 条。
4. **小写缩写有歧义风险**：`mi` 等 2–4 字符缩写以小写出现时置信度记为 `medium`（大写为 `high`），目前仍会展开。
5. **期刊过滤要求字面精确**：库里是全称（`PLoS ONE`、`Sensors (Basel, Switzerland)`），写简称返回空，已在 `notes` 提示。
6. **模糊时间词是启发式**：`recent` / `最新` 按近 5 年处理，可通过 `recency_years` 调整或关闭。

**下一步（检索系统第二部分，见 `任务6/README.md`）**：接入检索层做多路 RRF + `where` 下推 + 后置过滤；混合检索（稠密 Chroma + 稀疏 BM25）；cross-encoder 重排；**建人工标注小评测集**把 B/C/D 之争与更强融合干预（直接丢弃主查询、调小 k）定下来；补测单条查询耗时与带 `where` 过滤是否显著更慢（本阶段只记了整轮耗时 1007s/635s，差异主要来自系统页缓存变热）。

---

## 七、本包内容

- **相关代码**（在顶层 `scripts/`）：`检索_查询理解.py`（**核心**：词典、实体正则、`MedicalQueryProcessor`）、`检索_查询理解_验证.py`（功能自检 + 4 项在线实测）、`检索_构建同义词词典.py`（MeSH XML → 词典）、`检索_扫描元数据分布.py`（语料章节/年份/期刊实测分布）。
- **本地留存产物**（可由脚本复现）：`data/dict/`、`data/mesh/`（词典与语料统计中间产物，约 310MB）、`查询理解验证报告.txt`（①–⑥ 完整输出）、`词典统计.json`。
