# -*- coding: utf-8 -*-
"""交付包完整性校验 —— 「任务N\\ 这个文件夹单独发出去，是不是一份完整、可用的交付？」

管的是**打包**，不是代码对错（代码对错由各阶段自己的 *_验证.py 负责）。
一次跑完检查五件事，每件都是实际出过问题的：

  A 清单齐全 + 版本一致 —— 该收的脚本在不在，且与顶层 scripts\\ 的真源 MD5 相同
                            （包内拷贝会随着改代码静默过期，人发现不了）
  B import 依赖闭包    —— 从代码里读出每个脚本动态导入的兄弟文件，检查是否都在同一包内
                            （任务6 曾缺 向量化_建库.py，单独发出去 import 就失败）
  C README 引用        —— README 提到的文件是否在包内（大数据/环境路径除外，那些给的是重建命令）
  D 死路径             —— 改名遗留的 E:\\medrag 之类，指向已不存在的目录
  E 报告版本一致       —— 包内的验证报告/统计产物与 report_data\\ 里的是不是同一版

用法：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\校验交付包.py
  加 --fix 自动修可安全自动修的部分：**单向**从 scripts\\ / report_data\\ 同步到包内。
  --fix 绝不反向覆盖真源，也不碰任何 .docx（汇报稿可能是手工润色过的，只报告不动手）。

新增阶段时：在 PACKAGES 里加一行即可；B、C、D 三项对新文件夹自动生效，不用配置。
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import argparse
import hashlib
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = os.path.join(ROOT, "scripts")
REPORT_DIR = os.path.join(ROOT, "report_data")

# ---------------------------------------------------------------------------
# 每个交付包该收哪些脚本 —— 这是唯一需要人工维护的清单（"该收什么"代码猜不出来）
# ---------------------------------------------------------------------------
PACKAGES = {
    "任务1": ("一 环境准备 + 本地 LLM 验证",
              ["验证本地模型.py", "解析数据管线.py", "命令行提问.py", "多轮问答.py",
               "start_medrag.ps1"]),
    "任务2": ("二 数据尽调 + 切块策略设计",
              ["步骤1_数据结构分析.py", "步骤2_领域内容理解.py", "步骤3_文本长度量化.py",
               "步骤4_分割策略分析.py", "生成报告.py"]),
    "任务3": ("三 文档解析与切块",
              ["下载数据.sh", "切块_主管线.py", "切块_修复ID.py", "切块_质量校验.py",
               "切块_报告转word.py"]),
    "任务4": ("四 向量化与索引构建",
              ["向量化_建库.py", "向量化_检索验证.py", "向量化_报告转word.py",
               "恢复_导出向量.py", "恢复_合并向量元数据.py", "恢复_重建库.py"]),
    "任务5": ("五 查询理解与增强",
              ["检索_查询理解.py", "检索_查询理解_验证.py", "检索_构建同义词词典.py",
               "检索_扫描元数据分布.py", "检索_报告转word.py",
               "向量化_建库.py"]),      # 阶段四上游依赖：验证脚本要 BGEEmbedder 做嵌入对照
    "任务6": ("六 多路检索 + 多准则重排",
              ["检索_多路检索.py", "检索_多路检索_验证.py", "检索_构建BM25索引.py",
               "检索_BM25公共.py", "检索2_报告转word.py",
               "检索_查询理解.py",      # 阶段五上游依赖：检索_多路检索.py import 阶段就要
               "向量化_建库.py"]),      # 阶段四上游依赖：同上（提供 BGEEmbedder）
    "任务7": ("七 上下文组装 + 医学提示词 + LLM 生成流水线",
              ["生成_上下文组装.py", "生成_提示词模板.py", "生成_分词器.py",
               "生成_上下文组装_验证.py", "生成_报告转word.py",
               # ——阶段七之二：本地 LLM 集成与完整生成流水线——
               "生成_LLM生成器.py", "生成_流水线.py", "生成_流水线_测试.py",
               "生成_对比评测.py"]),
    "任务8": ("八 答案评估 + 缓存策略 + 批量处理",
              ["评估_答案评估器.py", "生成_缓存.py", "生成_批量处理.py",
               "评估_验证.py", "评估_跑测试集.py", "评估_报告转word.py",
               # 阶段七上游依赖：评估_跑测试集.py import 阶段就要整条生成链
               "生成_流水线.py", "生成_上下文组装.py", "生成_提示词模板.py",
               "生成_LLM生成器.py", "生成_分词器.py"]),
    "任务9": ("九 强约束规则开发与幻觉抑制",
              ["约束_提示词层.py", "约束_格式校验器.py", "约束_受限流水线.py",
               "约束_对抗测试集.py", "约束_跑对抗测试.py", "约束_验证.py",
               "约束_报告转word.py",
               # 阶段七上游依赖：受限流水线继承阶段七的流水线，import 阶段就要整条链
               "生成_流水线.py", "生成_上下文组装.py", "生成_提示词模板.py",
               "生成_LLM生成器.py", "生成_分词器.py",
               # 阶段八上游依赖：跑对抗测试用 ParallelBatchProcessor 并发
               "生成_批量处理.py"]),
    # 不是新阶段：交付评审后提出的改进建议（测试量太小、要看命中分布），
    # 所以挂在 任务10 下面、不另起阶段号。
    "任务10\\检索评测改进": ("十之一（改进）golden 检索评测集",
               ["golden_构建.py", "golden_跑测.py",
                # 阶段七上游依赖：golden_构建.py 出题要 LLMGenerator
                "生成_LLM生成器.py",
                # 阶段六上游依赖：golden_跑测.py --run 必须走真检索链，不是可选功能
                "检索_多路检索.py", "检索_BM25公共.py",
                # 阶段五上游依赖：检索_多路检索.py 与 golden_跑测.py 都 import 阶段就要
                "检索_查询理解.py",
                # 阶段四上游依赖：检索_多路检索.py 要 BGEEmbedder
                "向量化_建库.py"]),
    "任务10": ("十 服务化与接口开发（FastAPI 骨架 + 问答 + 会话/统计/文档管理）",
               ["服务_错误码.py", "服务_模型.py", "服务_日志.py", "服务_会话.py",
                "服务_流式.py", "服务_应用.py", "服务_问答接口.py", "服务_验证.py",
                "服务_终端客户端.py", "服务_报告转word.py", "服务_汇报稿转word.py",
                # ——第二部分：文档管理 + 运营统计——
                "服务_文档目录.py", "服务_文档接口.py",
                # 阶段九上游依赖：服务默认走受约束流水线，import 阶段就要
                "约束_受限流水线.py", "约束_提示词层.py", "约束_格式校验器.py",
                # 阶段七上游依赖：受限流水线继承阶段七那条链
                "生成_流水线.py", "生成_上下文组装.py", "生成_提示词模板.py",
                "生成_LLM生成器.py", "生成_分词器.py"]),
}

# 全局依赖：`_medrag_root.py` 提供项目根解析，几乎每个脚本都在 import 阶段就要它。
# 写在这里统一注入，而不是在上面 10 个清单里各抄一遍——它不属于任何一个阶段，
# 抄十遍等于给自己留十个忘改的机会。
for _pkg, (_desc, _files) in PACKAGES.items():
    if "_medrag_root.py" not in _files:
        _files.insert(0, "_medrag_root.py")

# 允许缺席的跨阶段依赖：只有可选功能才用到，且脚本本身已对缺失做了降级
OPTIONAL_DEPS = {
    ("任务7", "检索_多路检索.py"):
        "阶段六脚本，仅 --live 组用；该组还需 65GB 向量库，脚本已做缺失降级提示",
    ("任务10", "检索_多路检索.py"):
        "阶段六脚本，仅 --mode live 用（默认 snapshot 模式不加载）；"
        "该模式还需 65GB 向量库 + 3.4GB BM25 索引，服务已做缺失降级并在健康检查里报出",
}

# 与 report_data\ 同名但**本就应该不同**的文件
EXPECT_DIFFER = {
    "RAG数据分析与设计说明.docx":
        "任务2 里那份是在脚本产出基础上又润色过的交付稿，report_data 里是脚本重跑的原始版",
}

# 包内报告 ←→ report_data 里的对应文件（文件名不一致的才需要在这写）
REPORT_ALIAS = {
    "上下文组装验证报告.txt": "生成_上下文组装验证报告.txt",
}

# 扫**代码 + 脚本产出的数据文件**。README / 工作记录 这类文档里出现旧路径是在讲"改名"
# 这件事本身，属正常，不扫。
# ⚠ `.json` 是 2026-07-31 补进来的：`任务2\step3.json`、`step4.json` 里的 `img` 字段一直
# 存着改名前的 `E:\medrag\report_data\*.png`，只扫代码扫不到。重跑脚本能自动修好（28/16
# 个统计值逐个 bit 相同，只有路径变了），但没人重跑就一直躺在交付包里。
DEAD_PATH = "E:\\medrag"
DEAD_PATH_EXT = (".py", ".ps1", ".sh", ".json")
DEAD_PATH_OK = {"切块_主管线.py",      # 注释描述的正是"用户级 HF_HOME 指向旧路径"
                "校验交付包.py"}       # 本文件把旧路径当常量存着

# 文档内「向下引用」：一句话说「看下面 X」，那 X 必须在本文件后续内容里真的出现。
# 只认**明确的向下指示词**（下面/下方/下表/后面/下文），避免把「见 docs/工程笔记.md 三·8」
# 这类跨文件引用误判。锚点取两种最可靠的形式：`配置 xxx` 与 `【xxx】`——
# 它们在本项目的报告里就是章节/表格的标题形式，判得准、几乎不会误报。
POINTER = re.compile(
    r"(?:看|见|详见|参见)\s*(?:下面|下方|下表|后面|下文)[^\n。；]{0,12}?"
    r"(?:配置\s*(?P<cfg>[A-Za-z_][A-Za-z0-9_]*)|【(?P<brk>[^】]{2,30})】)")

# README 里出现这些前缀的引用属于大数据/环境，本就不该进包
BULK_HINT = ("data\\", "data/", "conda", "hf-cache", "ollama\\", "ollama/", "logs", "pip-cache")
FILE_REF = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff\\/\.\-]+\.(?:py|ps1|sh|json|jsonl|txt|docx|png|parquet|md)")
# 动态导入：本项目的中文文件名一律按路径导入，形式固定
DYN_IMPORT = re.compile(r"(?:spec_from_file_location|_load_by_path|_load)\s*\([^)]*?[\"']([^\"']+\.py)[\"']",
                        re.S)

PROBLEMS = []
FIXED = []
#: 本轮**跳过、未验证**的检查项 [(名字, 原因)]。结论行据此如实措辞。
SKIPPED = []


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def source_of(name):
    """交付文件的真源位置：先找 scripts\\，再找仓库根（start_medrag.ps1 在根目录）。"""
    for cand in (os.path.join(SRC, name), os.path.join(ROOT, name)):
        if os.path.exists(cand):
            return cand
    return None


def bad(pkg, msg):
    PROBLEMS.append((pkg, msg))
    print(f"  ❌ {msg}")


def warn(pkg, msg):
    PROBLEMS.append((pkg, msg))
    print(f"  ⚠ {msg}")


def check_package(pkg, stage, wanted, fix=False):
    d = os.path.join(ROOT, pkg)
    sd = os.path.join(d, "脚本")
    print("=" * 92)
    print(f"{pkg}\\   {stage}")
    if not os.path.isdir(d):
        bad(pkg, f"文件夹不存在：{d}")
        return
    if not os.path.isdir(sd):
        if fix:
            os.makedirs(sd)
            print("  🔧 已建 脚本\\ 子目录")
        else:
            bad(pkg, "没有 脚本\\ 子目录")

    # ---- A 清单齐全 + 版本一致 ----
    have = set(os.listdir(sd)) if os.path.isdir(sd) else set()
    for name in wanted:
        src = source_of(name)
        dst = os.path.join(sd, name)
        if src is None:
            bad(pkg, f"{name}：真源不存在（清单写错，或文件已改名）")
            continue
        if not os.path.exists(dst):
            if fix:
                _copy(src, dst); FIXED.append(f"{pkg}\\脚本\\{name}（新增）")
            else:
                bad(pkg, f"{name}：缺，未收进包")
        elif md5(src) != md5(dst):
            if fix:
                _copy(src, dst); FIXED.append(f"{pkg}\\脚本\\{name}（更新）")
            else:
                bad(pkg, f"{name}：包内是**旧版本**，与 {os.path.relpath(src, ROOT)} 不一致")

    extra = sorted(have - set(wanted) - {"__pycache__"})
    if extra:
        warn(pkg, f"包内有清单外的文件：{extra}（要么加进 PACKAGES 清单，要么删掉）")
    if "__pycache__" in have:
        warn(pkg, "包内有 __pycache__（跑过脚本留下的，发包前删掉）")

    # ---- B import 依赖闭包（从代码读，不靠清单）----
    for f in sorted(x for x in have if x.endswith(".py")):
        src_text = open(os.path.join(sd, f), encoding="utf-8", errors="replace").read()
        for dep in {os.path.basename(x.replace("/", "\\")) for x in DYN_IMPORT.findall(src_text)}:
            if dep in have or dep == f:
                continue
            why = OPTIONAL_DEPS.get((pkg, dep))
            if why:
                print(f"  · {f} 可选依赖 {dep} 不在包内 —— {why}")
            else:
                bad(pkg, f"脚本\\{f} 运行时要 import {dep}，不在包内 → 单独发出去会 import 失败")

    # ---- C 文档引用（**所有交付文本文件**，不只 README）----
    # 2026-08-12 扩大范围：原先只扫 README.md，于是**报告正文里的悬空引用扫不到**。
    # 实际踩到的就是这个：`golden_命中分布表.txt` 里写着「看下面配置 wide 的对照表」，
    # 而那张表根本没写进文件——报告是单独发出去的独立文档，读的人只会看到一句指向空气的话。
    readme = os.path.join(d, "README.md")
    if not os.path.exists(readme):
        bad(pkg, "没有 README.md")
    inside = {x for _, _, fn in os.walk(d) for x in fn}
    docs = []
    for cur, dirs, fns in os.walk(d):
        dirs[:] = [x for x in dirs if x not in ("脚本", "__pycache__")]
        for fn in fns:
            if fn.lower().endswith((".md", ".txt")):
                docs.append(os.path.join(cur, fn))
    for doc in sorted(docs):
        rel = os.path.relpath(doc, d)
        text = open(doc, encoding="utf-8", errors="replace").read()
        # C-1 文件引用
        for ref in sorted({m.group(0) for m in FILE_REF.finditer(text)}):
            base = os.path.basename(ref.replace("/", "\\"))
            if base in inside or any(b in ref for b in BULK_HINT):
                continue
            if re.match(r"^任务\d", ref) or base.endswith(".md"):
                continue                      # 跨阶段互引（如"见 任务6/README.md"）属正常
            if (pkg, base) in OPTIONAL_DEPS:
                continue                      # 已在 OPTIONAL_DEPS 里说明过为什么不带
            if os.path.exists(os.path.join(SRC, base)) or os.path.exists(os.path.join(ROOT, base)):
                warn(pkg, f"{rel} 提到 {ref}，但没带进包（仓库别处有）")
        # C-2 文档内「向下引用」
        for m in POINTER.finditer(text):
            anchor = (m.group("cfg") or m.group("brk") or "").strip()
            if not anchor:
                continue
            # 锚点必须在**这句之后**的正文里真的出现过
            if anchor not in text[m.end():]:
                bad(pkg, f"{rel} 第 {text[:m.start()].count(chr(10))+1} 行写「{m.group(0).strip()}」，"
                         f"但「{anchor}」在本文件后续内容里根本不存在 → 悬空引用")

    # ---- E 报告 / 统计产物版本一致 ----
    # 递归找，不能只看包的顶层：阶段七按 7.1 / 7.2 分了子目录，报告都在子目录里，
    # 只 listdir 顶层的话这项检查会**静默失效**（一条都查不到却仍然打勾）。
    # 跳过 脚本\ ——那是 A 组按 MD5 管的，不是报告。
    pkg_files = []
    for cur, dirs, fns in os.walk(d):
        dirs[:] = [x for x in dirs if x not in ("脚本", "__pycache__")]
        for x in fns:
            pkg_files.append((x, os.path.join(cur, x)))
    for f, pkg_file in sorted(pkg_files):
        if f in EXPECT_DIFFER or f.lower().endswith((".md", ".docx")) and f not in REPORT_ALIAS:
            continue
        counterpart = os.path.join(REPORT_DIR, REPORT_ALIAS.get(f, f))
        if not os.path.exists(counterpart):
            continue
        if md5(counterpart) != md5(pkg_file):
            # 报告**一律不自动同步**：report_data\ 里是"最后一次跑的结果"，可能是只跑了离线组的
            # 部分结果（58/58），而包里那份可能是全开的完整结果（65/65）——自动覆盖会毁掉证据。
            warn(pkg, f"{f} 与 report_data\\{os.path.basename(counterpart)} 不是同一版，人工确认哪份该留：\n"
                      f"        包内         {_fingerprint(pkg_file)}\n"
                      f"        report_data  {_fingerprint(counterpart)}")

    if not any(p == pkg for p, _ in PROBLEMS):
        print("  ✅ 清单齐全、版本一致、依赖闭合、README 无悬空引用")


def _fingerprint(path):
    """给报告文件一个人能读的身份标签：优先用"总计 x/y 项通过"那行，否则用大小+时间。"""
    import time
    stamp = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
    if path.lower().endswith(".txt"):
        try:
            for line in reversed(open(path, encoding="utf-8", errors="replace").read().splitlines()):
                if re.search(r"总计\s*\d+\s*/\s*\d+", line):
                    return f"{line.strip()}   ({stamp})"
        except Exception:
            pass
    return f"{os.path.getsize(path):,} 字节   ({stamp})"


def _copy(src, dst):
    with open(src, "rb") as a, open(dst, "wb") as b:
        b.write(a.read())
    print(f"  🔧 已同步 {os.path.relpath(dst, ROOT)}")


def check_dead_paths():
    """D 死路径：改名遗留、指向已不存在目录的绝对路径。"""
    print("=" * 92)
    print("全库死路径扫描")
    hits = []
    for base in (SRC, ROOT):
        for f in sorted(os.listdir(base)):
            p = os.path.join(base, f)
            if not os.path.isfile(p) or not f.endswith(DEAD_PATH_EXT) or f in DEAD_PATH_OK:
                continue
            try:
                t = open(p, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            if DEAD_PATH in t or DEAD_PATH.replace("\\", "\\\\") in t:
                hits.append(os.path.relpath(p, ROOT))
    for h in sorted(set(hits)):
        PROBLEMS.append(("<全库>", f"{h} 里还有 {DEAD_PATH} 死路径"))
        print(f"  ❌ {h} 里还有 {DEAD_PATH} 死路径")
    if not hits:
        print(f"  ✅ 没有 {DEAD_PATH} 残留"
              f"（{'、'.join(sorted(DEAD_PATH_OK))} 里那处是描述旧路径的注释，已豁免）")



# ---------------------------------------------------------------------------
# F 悬空文档引用：引用了「本机有、但 git 没跟踪」的文档
# ---------------------------------------------------------------------------
# 起因：C 组那条 `if base.endswith(".md"): continue` 本意是放过跨阶段互引
# （「见 任务6/README.md」这类），结果**把所有 .md 引用一起豁免了**。
# 于是几十处指向某份**从不进仓库的本地工作文档**的交叉引用躺了两个月没人发现——
# 对 clone 仓库的人来说那是指向不存在文件的引用，而校验器每次都打绿灯。
#
# ⚠ 这和「空洞断言」同源：判据存在，但对特定输入永远不触发。
#   差别在于空洞断言是「条件太宽，永远为真」，这里是「显式豁免，永远不看」——
#   后者更隐蔽，因为豁免通常写得有理有据。
#
# 判据：扫所有 git 已跟踪的文本文件，找出其中提到的项目内文档名；
#       若同名文件在工作区里存在、却**不在 git 跟踪列表里**，就是悬空引用。
#       （只查本机能证实"确实没进仓库"的那些，不猜外部文件。）
DANGLING_EXT = (".md", ".txt", ".py", ".ps1", ".sh")
DOC_REF = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff\-]+\.md")

# 豁免：这两个脚本把文件名当成**被扫描的字面量**在用，不是文档指针。
# `提交前扫描.py` 的白名单里必须留着那个文件名，否则扫描器会第五次误报。
DANGLING_OK = {
    "scripts/提交前扫描.py": "把文件名当良性字面量白名单在用，不是文档指针",
    "scripts/提交前扫描_自测.py": "同上，自测用例里要出现这个名字才测得到白名单",
}


def _git_tracked_paths():
    """git 跟踪的**完整相对路径**集合（正斜杠）。拿不到 git 返回 None。

    与 `_git_tracked()`（只返回 basename）分开：G 组要解析相对路径，
    basename 集合在这里不够用——同名文件可能躺在别的目录下。
    """
    try:
        out = subprocess.run(["git", "-c", "core.quotepath=false", "ls-files"],
                             cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", timeout=60)
        if out.returncode != 0:
            return None
        return [x for x in out.stdout.splitlines() if x.strip()]
    except Exception:
        return None


def _git_tracked():
    """git 跟踪的文件名集合（basename）。拿不到 git 就返回 None，让本项自我跳过。"""
    try:
        out = subprocess.run(["git", "-c", "core.quotepath=false", "ls-files"],
                             cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", timeout=60)
        if out.returncode != 0:
            return None
        return {os.path.basename(x) for x in out.stdout.splitlines() if x.strip()}
    except Exception:
        return None


def check_dangling_docs():
    """F 悬空文档引用：引用了本机存在、但没进仓库的文档。"""
    print("=" * 92)
    print("全库悬空文档引用扫描")
    tracked = _git_tracked()
    if tracked is None:
        print("  · 跳过：这里不是 git 仓库（或 git 不可用），本项无从判断")
        SKIPPED.append(("悬空文档引用", "非 git 仓库"))
        return

    # 工作区里所有 .md 的 basename（不含被忽略的大目录）
    SKIP_DIRS = {"conda", "data", "hf-cache", "ollama", "pip-cache", "logs",
                 ".git", "__pycache__", "node_modules"}
    on_disk = {}
    for cur, dirs, fns in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in fns:
            if fn.endswith(".md"):
                on_disk.setdefault(fn, os.path.relpath(os.path.join(cur, fn), ROOT))
    untracked_docs = {n: p for n, p in on_disk.items() if n not in tracked}

    scanned = 0
    hits = []
    for cur, dirs, fns in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in fns:
            if not fn.endswith(DANGLING_EXT):
                continue
            rel = os.path.relpath(os.path.join(cur, fn), ROOT).replace("\\", "/")
            if os.path.basename(rel) not in tracked:
                continue                      # 只查已进仓库的文件，本地草稿不算
            if rel in DANGLING_OK:
                continue
            try:
                text = open(os.path.join(cur, fn), encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            scanned += 1
            for ref in set(DOC_REF.findall(text)):
                if ref in untracked_docs:
                    hits.append((rel, ref))

    for rel, ref in sorted(set(hits)):
        msg = (f"{rel} 引用了 {ref}，而它**没有进仓库** → 对 clone 的人是悬空引用")
        PROBLEMS.append(("<全库>", msg))
        print(f"  ❌ {msg}")

    # ⚠ 非空洞：把「本轮实际扫了多少个已跟踪文件」算出来，为 0 即判不通过。
    #   只认 scanned == 0，**不要把 untracked_docs == 0 也算失灵**：
    #   干净检出（别人 git clone 之后、或从某个 ref 拉出来的工作区）天然没有未跟踪文档，
    #   而那恰恰是这条检查最该服务的场景。写成「两个数任一为 0 就红」会让它在那里恒红——
    #   恒红的判据和恒绿的判据一样没用，都会被当噪声忽略。
    #   「没有东西可比」不等于「比不了」，同三态拒答是一个道理：空不等于坏。
    if scanned == 0:
        PROBLEMS.append(("<全库>", "悬空引用检查没有生效：一个已跟踪文件都没扫到"
                                   "（git 拿不到，或工作目录不对）"))
        print("  ❌ 本项没有真正生效（扫到 0 个已跟踪文件），当作不通过")
    elif not untracked_docs:
        print(f"  ✅ 没有悬空文档引用（扫了 {scanned} 个已跟踪文件；"
              f"本机没有未进仓库的文档，无可比对项）")
    elif not hits:
        print(f"  ✅ 没有悬空文档引用"
              f"（扫了 {scanned} 个已跟踪文件，比对 {len(untracked_docs)} 份未进仓库的文档："
              f"{'、'.join(sorted(untracked_docs))}）")
        print(f"  · 豁免 {len(DANGLING_OK)} 个：{'、'.join(sorted(DANGLING_OK))}"
              f" —— 它们把文件名当被扫描的字面量在用")



# ---------------------------------------------------------------------------
# G markdown 相对链接：`](相对路径)` 的目标必须真实存在
# ---------------------------------------------------------------------------
# 起因：2026-08-18 全库扫出 5 处存量断链，全在 任务N\README.md 里
# （`](服务_实测.json)` 实际在 log\ 下、`](生成_流水线测试报告_offline.txt)` 实际在
# 7.2_ 子目录下）。它们不是这轮改出来的，是一直躺在那里没人查。
#
# ⚠ 这条与 C 组不重叠，别以为有了 C 就够了：
#   · C 组查的是「README 提到的**文件名**在不在包内」，按 basename 匹配，
#     `](服务_实测.json)` 的 basename 在包里（在 log\ 下），所以 C 组照样放行；
#   · G 组查的是「这个**链接点得开吗**」，按相对路径解析。
#   同一个文件，一个查存在、一个查可达——**读者点的是链接，不是文件名**。
#
# 覆盖面按「对外契约」取：所有 git 已跟踪的 .md，不只 README，也不只交付包内的。
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")


def check_md_links():
    """G markdown 相对链接可达性。"""
    print("=" * 92)
    print("全库 markdown 链接可达性扫描")
    tracked = _git_tracked_paths()
    if tracked is None:
        print("  · 跳过：这里不是 git 仓库（或 git 不可用），本项无从判断")
        SKIPPED.append(("markdown 链接可达性", "非 git 仓库"))
        return

    mds = [x for x in tracked if x.endswith(".md")]
    total = 0
    hits = []
    for rel in mds:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        base = os.path.dirname(p)
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for m in MD_LINK.finditer(text):
            target = m.group(1).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            total += 1
            if not os.path.exists(os.path.normpath(os.path.join(base, target))):
                hits.append((rel, m.group(1)))

    for rel, target in sorted(set(hits)):
        msg = f"{rel} 里的链接 `]({target})` 指向的文件不存在 → 点开是 404"
        PROBLEMS.append(("<全库>", msg))
        print(f"  ❌ {msg}")

    # ⚠ 非空洞：把实际扫到的链接数打出来，为 0 说明这条检查根本没在工作
    if total == 0:
        PROBLEMS.append(("<全库>", f"markdown 链接检查没有生效：{len(mds)} 份 .md 里一条相对链接都没扫到"))
        print(f"  ❌ 本项没有真正生效（{len(mds)} 份 .md 里扫到 0 条相对链接），当作不通过")
    elif not hits:
        print(f"  ✅ {len(mds)} 份 .md 里的 {total} 条相对链接全部可达")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true",
                    help="单向同步 scripts\\ / report_data\\ → 包内（不碰真源，不碰 .docx）")
    ap.add_argument("--only", default=None, help="只查某个包，如 --only 任务6")
    args = ap.parse_args()

    todo = [(p, s, w) for p, (s, w) in PACKAGES.items() if not args.only or p == args.only]

    if args.fix:                      # 第一遍：只管修
        print("=" * 92)
        print("交付包完整性校验（--fix：先同步）")
        for pkg, stage, wanted in todo:
            check_package(pkg, stage, wanted, fix=True)
        PROBLEMS.clear()              # 修完的状态已经变了，结论以下面的复查为准
        print("\n" + "=" * 92)
        print("同步完毕，重新完整复查")

    print("=" * 92)
    if not args.fix:
        print("交付包完整性校验")
    for pkg, stage, wanted in todo:   # 复查（或首次检查）一律 fix=False，看的是最终状态
        check_package(pkg, stage, wanted, fix=False)
    if not args.only:
        check_dead_paths()
        check_dangling_docs()
        check_md_links()

    print("=" * 92)
    if FIXED:
        print(f"已自动同步 {len(FIXED)} 个文件：")
        for x in FIXED:
            print(f"  · {x}")
    if PROBLEMS:
        print(f"❌ 仍有 {len(PROBLEMS)} 处问题（--fix 修不了的，需要人工处理）：")
        for p, m in PROBLEMS:
            print(f"  · [{p}] {m}")
        return 1
    # ⚠ 结论行只列**本轮真跑过**的检查。跳过的必须显式报出来，不许混进这句里——
    #   一条声称「已验证」而实际没跑的结论比不打这条更糟：它让人以为查过了。
    #   （本轮改动的起因：在非 git 目录下跑，F/G 两项跳过，结论却照样说「全部可达」。）
    done = ["清单齐全", "版本一致", "依赖闭合", "README 无悬空引用", "报告同版", "无死路径"]
    skipped_names = {n for n, _ in SKIPPED}
    if "悬空文档引用" not in skipped_names:
        done.append("无悬空文档引用")
    if "markdown 链接可达性" not in skipped_names:
        done.append("markdown 链接全部可达")
    n_pkg = len(PACKAGES) if not args.only else 1
    print(f"✅ {n_pkg} 个交付包全部通过：" + "、".join(done))
    if SKIPPED:
        print(f"⚠ 另有 {len(SKIPPED)} 项**跳过、未验证**（不计入上面那句）：")
        for name, why in SKIPPED:
            print(f"  · {name} —— {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
