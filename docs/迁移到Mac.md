# 把 medrag 迁到 MacBook Pro（M5 Pro / 48GB / 1TB）

> 结论：**能走局域网，要搬 ~72.8 GB，其余在 Mac 侧重建。**
> 本机 2.5GbE 网卡、E: 在 NVMe SSD 上，磁盘不是瓶颈，网络是唯一瓶颈。
> 48GB 内存能吃下主库检索（实测 15.8GB RSS），**全功能可跑**。
>
> 本文所有体积、路径、行号都来自 2026-08-16 实测，不是估的。

---

## 一、搬什么（~72.8 GB）

| 项 | 体积 | 为什么不能在 Mac 重建 |
|---|---|---|
| 代码 + 文档 + 报告（`scripts/` `任务1~10/` `report_data/` `docs/` `.git/` + 根目录文件） | 17 MB | **`git clone` 拿不到全部**：本地工作副本在 `.git/info/exclude`，个人工作日志、汇报草稿与 `report_data/` 在 `.gitignore`，各任务手写 DOCX 也被排除 |
| `hf-cache/`（bge-base / bge-m3 / bge-reranker） | 3.62 GB | 能从 HF 重下，但**必须是同一份权重才能复现历史分数**；且本项目是离线定位 |
| `data/chroma_db_4m/` | 64.58 GB | 重建要 `merged_4m.parquet`(15G) + 约 4.5 小时嵌入。**只有 6 个文件**（52.41G sqlite3 + 11.96G bin 打头），是最理想的传输形态，没有小文件惩罚 |
| `data/bm25_index_4m/` | 3.31 GB | 重建要 `data/chunks`(42G)，搬 3.3G 比搬 42G 划算 |
| `data/docs_catalog.db` | 1.05 GB | 重建只要 45 秒，但需要 `merged_4m.parquet`(15G)。不搬 parquet 就必须搬它 |
| `data/golden/` | 0.19 GB | 回归集，不可再生（已失去 held-out 资格，但仍是本项目唯一一把无方差的尺子） |
| `data/landmark/` | 60 KB | **全项目唯一一次联网请求的产物**。搬过去，Mac 永远不用联网 |
| `data/chroma_landmark/` `data/dict/` `data/tokenizer/` `data/index_stats.json` | ~25 MB | 前三个理论上可重建，但太小，搬了省事；`index_stats.json` 里的建库时间是历史事实，重跑还原不了 |

## 二、在 Mac 侧重建（不搬）

| 项 | 本机体积 | 怎么重建 |
|---|---|---|
| `conda/` | 5.63 GB | **Windows 二进制，物理上不能拷**。用 `requirements.txt` 重建，见第四节 |
| `ollama/` | 4.87 GB | `ollama pull qwen3:8b`，本地拉比网传快，还省得处理 blob 路径 |
| `pip-cache/` | 2.61 GB | 纯缓存，不要 |
| `data/service/*.db`、`logs/` | 运行期增长 | 服务起来自动建 |

## 三、留在 Windows（不搬）

`data/pubmed/`(76G)、`data/chunks/`(42G)、`data/vectors/`(15G+3.6G)、`data/mesh/`(299M)、`data/bm25_index_500k/`(0.42G) —— 只有重跑切块/建库管线才需要。

> ⚠ **一个真实的取舍**：不搬 `merged_4m.parquet`(15G) = **Mac 上永远无法重建 chroma 库和文献目录**。
> `docs/架构说明.md` 明写它是「建库保险文件，勿删」，阶段四库损坏就是靠它恢复的。
> Windows 这台还在 → 保险留那边可以；**Mac 要成为唯一机器 → 这 15G 必须搬**（总量变 87.8 GB），
> 脚本加 `-IncludeParquet`。

---

## 四、怎么传

### 路线 A：Mac 拉（推荐先试，Windows 侧零配置）

本机 SMB 服务在跑，管理共享 `E$` 已存在，可能直接就能连：

```
Finder → ⌘K → smb://192.168.1.165/E$   （用 Windows 管理员账号登录）
```

连上后在 Mac 终端用 rsync 拉，**`--partial` 给真正的断点续传**：

```bash
mkdir -p ~/rag
rsync -av --partial --progress --exclude='conda/' --exclude='pip-cache/' \
      --exclude='ollama/' --exclude='logs/' --exclude='data/' \
      /Volumes/E\$/rag/ ~/rag/

for d in chroma_db_4m bm25_index_4m golden landmark chroma_landmark dict tokenizer; do
  rsync -av --partial --progress /Volumes/E\$/rag/data/$d/ ~/rag/data/$d/
done
rsync -av --partial --progress /Volumes/E\$/rag/data/docs_catalog.db \
      /Volumes/E\$/rag/data/index_stats.json ~/rag/data/
```

⚠ 管理共享 `E$` 可能被 Windows 的远程 UAC 过滤挡掉（本地账号连 `C$`/`E$` 需要
`LocalAccountTokenFilterPolicy=1`，改注册表要管理员）。**试一下 30 秒就知道**，
连不上就走路线 B——不用去折腾注册表。

### 路线 B：Windows 推（Windows 侧不需要管理员）

Mac 上开 `系统设置 → 通用 → 共享 → 文件共享`，共享一个可写文件夹（例如 `~/rag`），
然后在 Windows 跑本仓库的脚本：

```powershell
$py = 'E:\rag\conda\envs\medrag\python.exe'   # 本脚本不用 python，列在这只为对照

# 先预演：只统计要搬什么、多大、估多久，不复制任何文件
& E:\rag\scripts\迁移_传输到Mac.ps1 -Dest \\Jinxis-MacBook-Pro\rag -DryRun

# 真传（可随时 Ctrl-C，重跑跳过已完成的文件、续传断掉的大文件）
& E:\rag\scripts\迁移_传输到Mac.ps1 -Dest \\Jinxis-MacBook-Pro\rag
```

脚本要点：搬运清单里每一项都写了「为什么必须搬」；`robocopy /Z` 给 52G 那个大文件断点续传；
`/COPY:DT` 不带 NTFS 属性（推到 macOS 共享上属性会失败）；开传前先做**目标可写性探测**，
免得四十分钟后才失败；退出码按 robocopy 语义判（`<8` 才算成功），**一项都没传也判失败**。

### 时间估（只算网络，SMB 实际效率约理论值 70~80%）

以下是 `-DryRun` 实算输出（合计 **72.79 GB / 10 项**）：

| 链路 | 72.79 GB |
|---|---|
| 2.5G 有线（本机就是 2.5GbE；Mac 要 USB-C 转接器，**交换机/路由器也得是 2.5G**） | 约 4~6 分钟 |
| 千兆有线 | 约 11~15 分钟 |
| Wi-Fi 6 | 约 12~36 分钟，波动大 |

M5 Pro 的 MacBook Pro 没有网口，走 Wi-Fi 也能传完，只是慢且不稳。

---

## 五、Mac 侧环境重建

```bash
# 1) conda 环境（不要拷 Windows 那份）
conda create -n medrag python=3.12 -y && conda activate medrag

# 2) ⚠ requirements.txt 里 torch==2.6.0+cu124 是 CUDA 专用轮子，Mac 装不了。
#    其余 127 项都能装。
grep -v '^torch==' requirements.txt > /tmp/req-mac.txt
pip install torch==2.6.0            # macOS arm64 轮子，自带 MPS 后端
pip install -r /tmp/req-mac.txt

# 3) Ollama
brew install ollama && ollama serve &
ollama pull qwen3:8b                # ~5GB

# 4) 项目根。放在 ~/rag 则可自动识别，不用设；放别处就显式指定
export MEDRAG_ROOT=~/rag
python scripts/_medrag_root.py      # 自检：打印解析到的根与四个子目录是否存在
```

> ⚠ **`requirements.txt` 原本缺 5 个真实依赖，已于 2026-08-16 补上**：
> `bm25s`、`PyStemmer`、`scipy`、`python-docx`、`matplotlib`。
> 前三个是 BM25 检索这条腿的核心——**照补之前那份建环境，检索会缺一路，
> 而多路融合少一路不报错，只是结果变差**。发现方式是拿 `pip list` 和
> `requirements.txt` 对差集，不是靠读代码。

## 六、代码改造 —— ✅ 已于 2026-08-16 全部完成并验证

> 下面记的是**做了什么、为什么这么做、怎么证明没改坏**。
> 不是待办清单。

### 6.0 核心决定：不改成 Mac 路径，改成自动解析

新增 `scripts/_medrag_root.py`（ASCII 文件名——本项目中文文件名必须按路径导入，
一个被 70 个脚本 import 的模块绝不能踩那个坑）。解析顺序：
`MEDRAG_ROOT` 环境变量 → 从 `__file__` 上溯找标志目录（同时含 `scripts/` 与
`requirements.txt`）→ 兜底 `E:\rag` 且仅当它真实存在。

**为什么不是 sed 成 Mac 路径**：出差三天回来要把活并回主机器，如果两台机器上
同一行路径永远不同，每个碰过的文件都会在**最没价值的那一行**上冲突。
现在两边源码逐字相同，git 只看见真正的改动。

⚠ **解析不出来直接抛 RuntimeError，不返回一个不存在的路径。**
这是本项目付过学费的地方：指向不存在目录的路径不会报错——Chroma 当场建个空库、
BM25 目录缺失只是把那一路静默关掉——最终表现成「库里没有证据」，
和真的没有证据在响应体里长得一模一样。宁可在 import 阶段响亮地失败。

改写由 `scripts/迁移_路径可移植.py` 机械完成（`--dry-run` / `--apply` / `--restore`），
**57 个文件 / 115 处**，跳过注释与三引号里的说明文字（那些改了纯属噪声）。
f-string 无法 `literal_eval`，工具**逐条列出而不是静默跳过**，4 处里 2 处真需手改
（都在 `向量化_建库.py`），已改。

> ⚠ 我一开始报的是「219 处」，那个数是错的：统计脚本把 docstring 内部的行
> 也算成了可执行行（它们不以 `"""` 开头）。真实数字是 115。

### 6.1 服务层：那个洞已修

`.env.example` 里 `MEDRAG_LOG_DIR / SNAPSHOT / BM25_DIR / CHROMA_DIR / SESSION_DB / CALLS_DB / DOCS_DB / INDEX_STATS` 都能配，
**写一份 Mac 的 `.env` 就能起 snapshot 模式的服务，零代码改动**。

⚠ **但 `MEDRAG_CHROMA_DIR` 对 live 模式不生效**（2026-08-16 查证）：
`服务_应用.py:564` 构造检索器时只传 `bm25_dir` / `translate` / `use_landmark`，**没传 `chroma_path`**；
第 167 行注释自己写着「只用于健康检查与占盘统计，不在这里加载」。
所以 live 模式实际打开的是 `检索_多路检索.py:60` 的模块常量 `CHROMA_PATH = r"E:\rag\data\chroma_db_4m"`。

**两种失败形态，一响一静默**：
- chroma：`PersistentClient` 会在 Mac 的 cwd 下建一个字面叫 `E:\rag\data\chroma_db_4m` 的空库，
  然后 `get_collection("medrag_bge_base")` 抛异常 → **会响**。
- landmark：`检索_多路检索.py:69` 的 `LANDMARK_PATH` 同样硬编码，而按设计
  「缺这个目录不报错，只是这一路自动关掉」→ **静默降级，P0 那条路无声消失**。
  这正是 `docs/工程笔记.md` 一·10 反复警告的形状：配了不生效、系统照常运行、结果慢慢变得没道理。

**已修**：`chroma_path` 与 `landmark_path` 都接进 `ServiceSettings` 并传给
`RetrievalPipeline`，新增 `MEDRAG_LANDMARK_DIR`，`chroma_dir` 那条撒谎的注释改掉了
——字段名必须承诺它真实的语义。

⚠ 同时注意：`landmark_dir` 的默认值**直接由 ROOT 拼出来，不从 `检索_多路检索.py`
取常量**——那个模块 import 阶段就要 chromadb / torch，而 snapshot 模式本来根本
不该加载它。

### 6.2 设备选择：cuda → mps → cpu

新增 `向量化_建库.py::pick_device()`，`BGEEmbedder` 与 `检索_多路检索.py` 的重排器
共用它——**设备选择只有一个来源**。放在 `向量化_建库.py` 是因为检索模块本来就
import 它取 `BGEEmbedder`，不引入新依赖。

原来两处都写 `device if torch.cuda.is_available() else "cpu"`：Mac 上恒假 →
**直接掉 CPU，Metal 永远用不上**。不报错、不告警，只是慢，而「慢」很容易被当成
「Mac 本来就慢」接受下来。

⚠ **mps 上保持 float32**：`fp16` 分支本来就只在 cuda 下走，这里维持同样口径。
半精度在 mps 上对 BGE 这类模型会引入可见的数值偏差，而本项目的检索分数是要**跨机器
对照**的——省那点内存不值得。

### 6.3 交付包

`_medrag_root.py` 已加进 `校验交付包.py` 的 `PACKAGES`——**用一个注入循环统一加，
不在 10 个清单里各抄一遍**（抄十遍等于给自己留十个忘改的机会）。
11 个交付包全部通过：清单齐全、MD5 一致、依赖闭合、README 无悬空引用、无死路径。

### 6.4 出发前闸门 —— 全部在 Windows 上真跑过

| 验证 | 结果 |
|---|---|
| `服务_验证.py` | **224/224**（4.3s） |
| `约束_验证.py` | **99/99** |
| `评估_验证.py` | **66/66** |
| `生成_上下文组装_验证.py` | **58/58** |
| `检索_查询理解_验证.py` | 通过 |
| `校验交付包.py` | **11/11 个包通过** |
| `golden_跑测.py --assert-baseline` | **14/14**；逐条名次 **235/235 × 2 配置 × 2 粒度**全同；聚合指标与四个年份桶**逐位复现**已发布数字 |
| `MEDRAG_ROOT` 覆盖测试 | 9 类路径（含 `HF_HOME`）**全部跟随**，无一处仍指向旧根 |

最后一条是这次改造的关键证据：**115 处改写在行为上是零影响**，
历史数字仍然可以引用。

---

## 七、怎么验证搬对了（别只看「跑起来了」）

按由便宜到贵的顺序：

```bash
python scripts/服务_验证.py          # 224 项，4 秒。不需要 Ollama、不需要向量库
python scripts/校验交付包.py          # 清单齐全 / MD5 一致 / import 闭合 / 无死路径
python scripts/landmark_探针.py      # 8 道题面，要 landmark collection
python scripts/golden_跑测.py --assert-baseline   # 检索层回归，11 分钟，不需要 Ollama
```

⚠⚠ **`--assert-baseline` 在 Mac 上不能按「逐位相同」读，必须先把判据放宽，否则你会把
设备差异误判成搬坏了。** 理由是实测事实：那条断言的强度来自「同一块 GPU 上三轮
`rel_score` 47,400/47,400 逐位相同」。Mac 上重排器跑的是 mps/CPU float32，
而 Windows 上是 cuda fp16 —— **query 侧编码与重排打分几乎不可能逐位复现**
（库内向量是已存的、不重算，所以只有 query 那一侧变）。

正确读法：**看逐条名次的一致率和聚合指标的差值，不看逐位相等**。
如果 R@5/R@10/R@20 和 T1·prod·MRR 与 Windows 那轮基本吻合、名次绝大多数复现 → 搬对了。
如果 R@k 明显掉一档 → 那不是浮点噪声，去查 6.1 那个 chroma 路径洞和 landmark 是否被静默关掉。

⚠ 端到端那 5 道题**不适合当迁移验收**：生成侧方差未量化（同一份代码、同一份证据、
`temperature=0` 连跑两轮，9 道应拒题里 3 道的拒答形态会翻转）。拿它判断「Mac 上是不是搬对了」，
测到的是噪声不是迁移。**迁移验收只用确定性那一侧：检索层 + 离线验证。**

---

---

## 七之二、写这份东西时新踩的一个坑

**含中文的 `.ps1` 必须存成 UTF-8 with BOM**（2026-08-16）。
Windows PowerShell 5.1 读 `.ps1` 时，没有 BOM 就按系统 ANSI（GBK）解码 —— 中文全变乱码，
而且乱码字节会**连带把语法读坏**（本次报的是 `Missing expression after ','`、
`The hash literal was incomplete`，一路指向根本没错的行）。
现象极具误导性：看起来像脚本写错了，实际是编码。
⚠ 与第二节铁律 `PYTHONIOENCODING=utf-8` 不是一回事——那条管 Python 的输出，
这条管 PowerShell 对脚本源文件的**解码**。
修法：`[System.IO.File]::WriteAllText($p, $c, (New-Object System.Text.UTF8Encoding($true)))`。

---

## 八、迁移后的不可逆风险

- **`data/landmark/entries.json` 是唯一一次联网的产物**。搬过去就不用再联网；
  但如果它在传输中损坏而 Windows 那份又被删了，就得再联一次网才能恢复。**先别删 Windows 这份。**
- **held-out 纪律不随迁移改变**：`golden_set.jsonl` 与改进任务清单附录 A 已进公开仓库，
  永久降级为回归集。在 Mac 上重跑它们**只能说「回归通过」，不能说「达到 X 分」**。
- **报告防降级判据跟着报告文件走**：Mac 上重跑验证脚本覆盖 `report_data/` 之前，
  确认新报告包含旧报告的每一类证据（尤其是 `--live` 真实端到端那几项），不是只比项数。

  ⚠⚠ **这条今天就被撞到了，而且是我撞的**（2026-08-16）：
  为了验证改造，我跑了 `生成_上下文组装_验证.py`（离线，58 项 / 6.5s），
  它**静默覆盖**了 `report_data\生成_上下文组装验证报告.txt` 里那份
  **65 项 / 88.8s 的 `--live` 版**（含「G. 端到端冒烟 —— 真实检索结果 → 上下文组装」）。
  `服务_验证.py` 有 `_refuse_downgrade()` 会拒绝这种覆盖，**这个脚本没有**。
  证据是靠 `任务7\7.1_…\上下文组装验证报告.txt` 那份包内副本救回来的
  ——**交付包规矩这次真的当了备份用**。
  **待办：给 `生成_上下文组装_验证.py`（以及其余没有该判据的验证脚本）补上同一条闸门。**
  这是「防降级判据只跟着报告文件走、不跟着脚本走」的第四种形态：
  不是判据写错了，是**这个脚本压根没有判据**。
