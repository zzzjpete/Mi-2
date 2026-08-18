# 阶段一 · 环境准备 + 本地 LLM 验证 + 引入数据源

> 准备阶段：配环境、验证本地大模型能跑、把数据源引进来跑通「下载 → 解析 → 结构化」。
> 这一阶段留下的**裸模型压测错题**，是后面整个 RAG 系统的验收标准。

## 接手卡

**状态**：✅ 完成 ｜ 2026-06-29

| 项 | 内容 |
|---|---|
| **输入** | 一台裸机（RTX 3080 10G / 32G 内存 / Ryzen 9 5900X），C 盘紧张，项目落 E 盘 |
| **产物** | conda 环境 `medrag`（Python 3.11）；Ollama + `qwen3:8b`；PubMed oa_comm baseline 首包（XML，43 MB / 3,028 篇）及其解析后的 parquet；4 个工具脚本 + 一键启动器 |
| **核心脚本** | `验证本地模型.py`（模型推理自检）、`解析数据管线.py`（下载→解析→结构化验证）、`命令行提问.py`（单次提问，中文安全）、`多轮问答.py`（带医学 system prompt + 全程日志） |
| **验证** | `& $py 脚本\验证本地模型.py` —— 推理通、`ollama ps` 显示 100% GPU；`& $py 脚本\解析数据管线.py` —— tar 流式 + lxml 解析 3,028 篇，0 失败 |
| **下游** | 阶段二拿这批 3,028 篇做数据尽调；阶段七拿本阶段的压测错题做接入前后对比评测 |
| **遗留** | ~~`start_medrag.ps1` 里的 `E:\medrag` 死路径~~ —— **已于 2026-07-27 修正**，全库死路径扫描确认无残留。仍需注意：用户级持久化的 `HF_HOME` 环境变量本身还指向旧路径，所以各脚本开头都硬覆盖 `os.environ["HF_HOME"]=r"E:\rag\hf-cache"`，新写脚本别漏 |

> `$py` = `E:\rag\conda\envs\medrag\python.exe`

---

## 一、对照任务书（逐条）
> 本阶段任务书原文见 [`任务书.txt`](任务书.txt)（逐字照录，未改写）。


| 任务书要求 | 实现 |
|---|---|
| 配置相关环境 | ✅ Miniconda + conda 环境 `medrag`（Python 3.11，**conda-forge** 源）；PyTorch 2.6.0+cu124，`torch.cuda.is_available()=True`；LangChain / Chroma / transformers / lxml 等 |
| 验证本地语言模型运行情况 | ✅ Ollama 0.30.11 + **Qwen3-8B（Q4_K_M，约 5 GB）**；`ollama ps` 100% GPU，约 **90–110 tokens/s**；「原生 Ollama」与「LangChain → Ollama」两条调用路径都验证过 |
| 引入数据源 | ✅ PubMed PMC OA Bulk 的 **oa_comm 子集**（CC BY，可商用）；下载 baseline 最小包（XML，43 MB，3,028 篇）；`tarfile` 流式 + `lxml` 解析，结构化存 parquet |
| 不开发 RAG 系统 | ✅ 本阶段只到「管线打通」为止 |

---

## 二、关键决策与理由

| 决策 | 理由 |
|---|---|
| conda-forge 源，不用 Anaconda 默认源 | 规避商用授权风险 |
| Python 3.11，不用系统自带的 3.14 | 3.14 太新，ML 库尚未跟上（这条至今有效：系统 PATH 上的 `python` 仍是 3.14，跑本项目脚本必须用 `conda\envs\medrag\python.exe`） |
| 环境 / 模型 / 缓存 / 数据全部落 E 盘 | C 盘空间紧张 |
| 数据取 **XML** 而非纯文本 | JATS XML 结构化、含 PMCID 与分节，是后面「带引用溯源」的前提 |
| 模型选 Qwen3-8B | 中文强；10 GB 显存跑 8B 量化最划算 |
| 只取 oa_comm 子集 | CC BY 允许商用，规避授权问题 |

---

## 三、⭐ 最重要的产出：裸模型的四类错，成了后面的验收用例

用 11 个不同难度的医学问题压测**未接 RAG 的** qwen3:8b：

- **概念 / 机制类答得扎实**：二甲双胍作用机制、细胞因子风暴、他汀 vs PCSK9 抑制剂等。
- **具体事实类会"自信地出错"**，且集中在四种情况：

| # | 错误类型 | 实例 |
|---|---|---|
| ① | **编造参考文献** | CRISPR 题：文献标题、期刊、卷页全是编的 |
| ② | **罕见病药名张冠李戴** | 法布里病：品牌名全错 |
| ③ | **药物分类错误** | 多发性硬化：单抗 / S1P 调节剂 / 干扰素混作一谈 |
| ④ | **最新药物、时间线混乱** | 阿尔茨海默题：日期自相矛盾，同一药列两遍 |

**结论**：错误集中在"可查证的具体事实"上——正是检索 + 溯源能修的部分。这四条从此成为整个项目的验收标准，写进了阶段七的评测计划（见 `任务7/README.md`）。

---

## 四、复现命令

```powershell
$py = "E:\rag\conda\envs\medrag\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:OLLAMA_MODELS = "E:\rag\ollama\models"     # ⚠ 必须设，否则 Ollama 找不到 E 盘上的模型

# 1) 验证本地模型推理（GPU 占用、生成速度）
& $py E:\rag\scripts\验证本地模型.py
ollama ps

# 2) 验证数据管线（tar 流式 + lxml 解析 → 结构化）
& $py E:\rag\scripts\解析数据管线.py

# 3) 日常问答
ollama run qwen3:8b                                   # 交互式
& $py E:\rag\scripts\命令行提问.py 他汀类药物的主要副作用是什么
& $py E:\rag\scripts\多轮问答.py                       # 带医学 system prompt + 日志
```

`多轮问答.py` 顶部「可调区」可改 `SYSTEM_PROMPT` / `temperature`（医学问答建议 0.2~0.3）/ `num_ctx` / `THINK`；对话日志写到 `logs\chat_log.jsonl`。

---

## 五、已知局限

1. **`start_medrag.ps1` 路径未更新**（见接手卡「遗留」），项目从 `E:\medrag` 改名为 `E:\rag` 后没跟着改。
2. **本阶段只用 43 MB 首包（3,028 篇）**验证管线，不代表全量；全量 76 GB 的下载与解析在阶段三完成。
3. **压测是人工阅读判定的**，11 题、单次采样，没有做多次采样或自动评分——它够用来定位"哪类问题会错"，但不构成量化基线；阶段七做前后对比时需要重新设计可复算的评测口径。

---

## 六、本包内容

```
任务1\
├─ README.md            本文件
└─ 脚本\                交付拷贝；单一来源在顶层 scripts\
   ├─ 验证本地模型.py    验证模型推理（GPU 占用、生成速度）
   ├─ 解析数据管线.py    验证「下载 → tar 流式 + lxml 解析 → 结构化」
   ├─ 命令行提问.py      单次提问（中文安全）
   ├─ 多轮问答.py        带医学 system prompt + 全程日志
   └─ start_medrag.ps1  一键启动器（拷贝；**实际使用的那份在仓库根目录**，
                        因为它按设计要被 `. E:\rag\start_medrag.ps1` 点源调用）
```

本阶段的数据产物（`data\pubmed\` 下的 43 MB 首包与解析后的 parquet）体积大且可重下，按 `.gitignore` 规则不入库，复现命令见「四」。
