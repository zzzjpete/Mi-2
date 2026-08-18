#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""golden 检索评测集 · 构建（抽样 + 生成 query）
================================================================================
需求方 2026-08-12 提出（优先级排在 P0 之前）：建一个 ≥100 条的 golden 检索评测集。
方法：从现有语料抽 N 篇文献 → 根据文献内容生成测试 query → 看检索能否命中原文献 →
统计命中分布。

**这是检索评测，不生成答案。** 跑测只调 `RetrievalPipeline.search()`，不调 LLM 作答，
所以整批是分钟级、而且**完全确定性**——不受 docs/工程笔记.md 三·1 那个温度 0 方差影响。
本脚本负责前两步（抽样 + 生成 query），跑测与统计在 `golden_跑测.py`。

--------------------------------------------------------------------------------
三个设计决定（都会影响怎么读最终那张表，先写在这里）
--------------------------------------------------------------------------------
1. **ground truth 从 `merged_4m.parquet` 抽，不开 Chroma。**
   这个 parquet 就是建库用的那份合并表，行数 3,998,000 与集合 `count()` 完全一致，
   即「它有的 = 库里有的」。抽样只需元数据与正文，开 Chroma 要付 15.8GB / 67s，没必要。

2. **ground truth 只从「临床相关」的 chunk 里抽，但检索仍然打满 400 万。**
   语料是 PubMed oa_comm 全量，里面有大量植物学 / 材料学 / 传感器文献
   （journal top: PLoS ONE、Scientific Reports、Materials、Frontiers in Plant Science）。
   从一段聚合物合成的正文里生成不出「临床医生自然提问」，硬生成只会得到一批
   本系统现实中永远不会收到的 query，测出来的数不代表真实使用。
   所以**抽样端做临床过滤，检索端不做**——干扰项依然是全部 400 万条，难度没有被放水。
   ⚠ 代价要说清楚：这批数**只代表临床类 query 上的检索表现**，不代表全语料。

3. **section 归一化复用检索层同一份 `canonical_to_raw`**（`data/dict/corpus_meta.json`）。
   语料里 section 原始写法有 391,164 种，自己写正则归一会和检索层的 `SectionPostFilter`
   对不上，分层报告就没法和检索行为对照着看。归一不了的（约 17%）直接不参与抽样。

--------------------------------------------------------------------------------
用法
--------------------------------------------------------------------------------
    $py = "E:\\rag\\conda\\envs\\medrag\\python.exe"

    # 步骤 1：分层抽样（单遍扫 parquet，不需要 Ollama / GPU）
    & $py scripts\\golden_构建.py --sample --per-cell 6

    # 步骤 2：用 qwen3 生成中文 query（需要 Ollama，不需要检索器）
    & $py scripts\\golden_构建.py --gen

    # 步骤 3：出人工复核清单（明确要求 human-in-the-loop，不要全自动）
    & $py scripts\\golden_构建.py --review

    # 步骤 4：按复核结果定稿
    & $py scripts\\golden_构建.py --finalize
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT


import os
import sys

# ── 环境铁律（docs/工程笔记.md 第一节）：HF_HOME 必须硬覆盖，且 pyarrow 要早于 torch ──
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pyarrow.parquet as pq          # noqa: E402  ← 必须在 torch 之前

import argparse                       # noqa: E402
import importlib.util                 # noqa: E402
import json                           # noqa: E402
import random                         # noqa: E402
import re                             # noqa: E402
import time                           # noqa: E402
from collections import Counter, defaultdict   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_PARQUET = os.path.join(ROOT, "data", "vectors", "merged_4m.parquet")
CORPUS_META = os.path.join(ROOT, "data", "dict", "corpus_meta.json")
OUT_DIR = os.path.join(ROOT, "data", "golden")

F_CHUNKS = os.path.join(OUT_DIR, "golden_chunks.jsonl")       # 步骤 1 产物
F_RAW = os.path.join(OUT_DIR, "golden_queries_raw.jsonl")     # 步骤 2 产物
F_REVIEW = os.path.join(OUT_DIR, "golden_复核清单.md")         # 步骤 3 产物（人看的）
F_DECISION = os.path.join(OUT_DIR, "golden_复核决定.jsonl")    # 步骤 3 产物（人改的）
F_FINAL = os.path.join(OUT_DIR, "golden_set.jsonl")           # 步骤 4 产物


def _load_by_path(mod_name: str, filename: str):
    """按路径导入中文文件名模块。

    ⚠ 必须先登记进 `sys.modules` 再 `exec_module`——见 docs/工程笔记.md 三·8 那条坑：
    不登记会产出多个互不相等的副本，跨模块 isinstance / 异常处理器会静默失配。
    """
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


# ==============================================================================
# 一、分层维度
# ==============================================================================
# 需求点名的 section 四类，外加 abstract——摘要块信息密度最高，和 methods 那种
# 「全是试剂型号」的块正好构成两端，分层拆解时是最有信息量的对照。
SECTIONS = ["abstract", "introduction", "methods", "results", "discussion"]

# 年份分桶。⚠ 重排的 recency 用 recency_span=20 线性衰减（2006 年→0 分），
# 所以「老文献是不是系统性吃亏」正好能被这一维量出来，别把桶切得太粗。
YEAR_BUCKETS = [
    ("≤2015", lambda y: y is not None and y <= 2015),
    ("2016-2019", lambda y: y is not None and 2016 <= y <= 2019),
    ("2020-2022", lambda y: y is not None and 2020 <= y <= 2022),
    ("2023+", lambda y: y is not None and y >= 2023),
]


def year_bucket(y):
    for name, fn in YEAR_BUCKETS:
        if fn(y):
            return name
    return None


# ==============================================================================
# 二、临床相关性过滤
# ==============================================================================
# 说明：这不是一个「医学 vs 非医学」的分类器，只是一道粗筛，目的是别把聚合物合成、
# 土壤微生物、天线设计的段落抽成 ground truth。宁可漏（语料够大），不可误收。
CLINICAL_CORE = [
    "patient", "clinical", "disease", "treatment", "therapy", "diagnosis",
    "trial", "symptom", "prognosis", "syndrome",
]
CLINICAL_TERMS = CLINICAL_CORE + [
    "therapeutic", "diagnostic", "randomized", "randomised", "cohort", "dose",
    "dosage", "efficacy", "adverse", "mortality", "morbidity", "survival",
    "incidence", "prevalence", "comorbid", "hospital", "surgery", "surgical",
    "drug", "medication", "placebo", "biopsy", "tumor", "tumour", "carcinoma",
    "infection", "chronic ", "acute ", "outcome", "guideline", "screening",
    "physician", "inpatient", "outpatient", "myocardial", "serum", "plasma",
    "blood pressure", "intravenous", "lesion", "remission", "relapse",
    "patholog", "epidemiolog", "vaccine", "antibiotic", "metastas",
    "chemotherapy", "radiotherapy", "immunotherapy", "biomarker", "prescri",
    "mortality", "follow-up", "questionnaire", "participants",
]
NONCLINICAL_TERMS = [
    # 农林与环境
    "plant", "soil", "crop", "leaf", "seedling", "agronom", "photosynth",
    "cultivar", "rhizosph", "wastewater", "sediment", "geolog", "astronom",
    "marine", "fishery", "concrete", "textile", "tourism", "supply chain",
    # 材料与工程
    "polymer", "catalyst", "monomer", "coating", "nanocomposite", "sensor",
    "antenna", "wireless", "semiconductor", "battery", "corrosion",
    "mechanical properties", "tensile", "thermal conductivity",
    # 兽医与纯基础研究——这两类会命中一堆临床词，但生成不出临床医生会问的问题
    "cattle", "bovine", "veterinar", "poultry", "swine", "piglet", "livestock",
    "broiler", "aquaculture", "zebrafish", "drosophila", "c. elegans",
    "arabidopsis", "yeast strain", "knockout mice", "transfected cells",
]

MIN_POS = 7          # 命中的**不同**临床概念数下限，见 clinical_score

# ============================================================================
# 硬否决 —— **与临床词多少无关，命中即出局**
# ============================================================================
# 2026-08-12 审计发现：`pos >= 2*neg` 这个**比例**判据有结构缺陷——
# 一篇兽医论文只要临床词够多（实测 14 个），2 个负向词根本压不住，照样通过。
# 所以「这篇文章根本不属于临床医学」这件事**不能用词频比例表达**，要用硬否决。
#
# 审计到的四类系统性漏网（400 条新样本 + 已抽中的 120 条）：
#   ① 医学教育  BMC Medical Education / Perspectives on Medical Education
#      —— G107 那条生成出「问题一/问题二」的，正是这类
#   ② 兽医      BMC Veterinary Research（小鼠与牛的免疫佐剂）
#   ③ 农学/食品 G033 生物强化小麦、G032 RSC Advances 的种子处理
#   ④ 材料/化学 RSC Advances 这类综合刊里的非医学部分
#
# ⚠⚠ 这两个正则**宁可漏、不可误杀**。语料有 28.7 万条合格候选，漏掉一些毫无代价；
#     杀掉一篇真临床论文却会让评测集失真。第一版写得太宽，实测误杀 3 条真临床：
#       · `ecolog` 匹配进了 "Gyn**ecolog**y" → 毙掉了多囊卵巢综合征试管婴儿研究
#       · `materials` → 毙掉了《Materials》上的麦卢卡蜂蜜牙周治疗 RCT
#                       和《Journal of Functional Bio**materials**》上的种植体骨结合 RCT
#         （MDPI 的材料类刊**确实**发牙科/种植体临床试验，这个刊名不能当否决信号）
#       · `problem-based learning` → 毙掉了 BMC Family Practice 上
#         「冠心病患者教育」的 RCT（那是临床试验，不是医学教育研究）
#     所以：**所有词都加词边界**，且只保留刊名/标题里几乎不可能属于临床研究的那些。
JOURNAL_VETO = re.compile(
    r"\b(veterinar\w*|medical education|medical teacher|nurse education|"
    r"agricultur\w*|agronom\w*|food science|food chemistry|"
    r"polymers?|macromolecul\w*|rsc advances|chemosphere|"
    r"environmental science|ecolog\w*|entomolog\w*|zoolog\w*|botan\w*|"
    r"photonics|catalysis)\b", re.I)

# 标题里出现即否决——标题是文章主旨，比正文里的零星提及可靠得多。
# ⚠ 只放**几乎不可能出现在临床论文标题里**的词。像 seed / composite / polymer /
#   problem-based learning 这些在临床标题里有正当含义的（碘125粒子植入、复合骨、
#   聚合物胶束、患者教育 RCT），**一律不放**。
TITLE_VETO = re.compile(
    r"\b(biofortif\w*|cultivar\w*|agronomic\w*|seed treatment|soil|"
    r"crop\w*|wheat|maize|barley|livestock|poultry|broiler|"
    r"cattle|bovine|swine|piglet|aquacultur\w*|fishery|"
    r"medical (student|education|curriculum|teaching)s?|nursing student|"
    r"photocatal\w*|wastewater|corrosion|thermoelectric)\b", re.I)


def clinical_score(text: str, title: str, journal: str = ""):
    """返回 (是否临床相关, 命中的临床词数, 命中的非临床词数)。

    两层判据：
      **① 硬否决**（与词频无关）：期刊名或标题命中 JOURNAL_VETO / TITLE_VETO → 直接出局。
      **② 词频判据**：去重后命中 ≥MIN_POS 个临床词、至少 1 个核心词，且临床词 ≥ 2× 非临床词。

    三个细节都是实测调出来的：
    · **必须去重**——一段反复说 20 次 "patient" 不比命中 7 个不同临床概念的段落更临床。
    · **阈值 5 太松**：实测放进来了抗体芯片蛋白组学、RECQ1 解旋酶、牛的结节性皮肤病。
    · **光有比例判据不够**：兽医/医学教育论文的临床词密度和真临床论文一样高，
      比例判据结构上就压不住它们——必须靠硬否决。见上面两个 VETO 的注释。
    """
    j = str(journal or "")
    t = str(title or "")
    if JOURNAL_VETO.search(j) or TITLE_VETO.search(t):
        return False, 0, 99
    blob = (text or "").lower() + " \n " + t.lower()
    pos = {x for x in CLINICAL_TERMS if x in blob}
    core = {x for x in CLINICAL_CORE if x in blob}
    neg = {x for x in NONCLINICAL_TERMS if x in blob}
    ok = len(pos) >= MIN_POS and len(core) >= 1 and len(pos) >= 2 * len(neg)
    return ok, len(pos), len(neg)


def looks_like_prose(text: str):
    """挡掉表格碎片。返回 (是否像散文, 数字占比, 字母占比)。

    切块是按字符窗口切的，Results 段常常切出整段表格：
      `001  Laboratory tests 409.0 (394.5) 2.4% 87.8 (214.9) 3.4% <0 . 001  ART 6,809.6 ...`
    这种块**确实在索引里**（所以检索评测拿它当 ground truth 并不算作弊），
    但从它生成不出一句像样的临床问题——出题人只能去抄那些数字，而抄数字正是
    需要防的词面泄漏。所以在抽样端就挡掉，不留给出题环节去纠结。
    """
    t = text or ""
    if not t:
        return False, 0.0, 0.0
    n = len(t)
    dig = sum(c.isdigit() for c in t) / n
    alp = sum(c.isalpha() for c in t) / n
    return (dig <= 0.15 and alp >= 0.55), dig, alp


# ==============================================================================
# 三、步骤 1：分层蓄水池抽样
# ==============================================================================
def load_section_map(path=CORPUS_META):
    """raw section 字符串 → canonical 类名。用检索层同一份表，保证口径一致。"""
    with open(path, encoding="utf-8") as f:
        cm = json.load(f)
    raw2canon = {}
    for canon, variants in cm.get("section", {}).get("canonical_to_raw", {}).items():
        if canon in SECTIONS:
            for v in variants:
                raw2canon[v] = canon
    return raw2canon


def do_sample(args):
    t0 = time.time()
    raw2canon = load_section_map()
    print(f"[抽样] section 归一表：{len(raw2canon):,} 种原始写法 → {len(SECTIONS)} 个规范类")

    pf = pq.ParquetFile(args.parquet)
    cols = ["chunk_id", "text", "doc_id", "chunk_index", "total_chunks",
            "source_title", "token_count", "section", "pmcid", "pmid",
            "journal", "pub_year"]
    total_rows = pf.metadata.num_rows
    print(f"[抽样] 扫描 {args.parquet}")
    print(f"       {total_rows:,} 行 / {pf.metadata.num_row_groups} 个 row group，"
          f"只读 {len(cols)} 列（不读 vector，省掉约 12GB IO）\n")

    rng = random.Random(args.seed)
    # 每个 (section, year_bucket) 一个蓄水池
    pool = defaultdict(list)      # cell -> [记录]
    seen = Counter()              # cell -> 见过多少个合格候选（蓄水池算法要用）
    stat = Counter()              # 各道过滤淘汰了多少
    scanned = 0

    for batch in pf.iter_batches(batch_size=args.batch, columns=cols):
        d = batch.to_pydict()
        n = len(d["chunk_id"])
        for i in range(n):
            scanned += 1
            sec = raw2canon.get(d["section"][i])
            if sec is None:
                stat["section 归一不了"] += 1
                continue
            yb = year_bucket(d["pub_year"][i])
            if yb is None:
                stat["年份缺失/超范围"] += 1
                continue
            tc = d["token_count"][i] or 0
            if tc < args.min_tokens:
                stat[f"正文过短(<{args.min_tokens} token)"] += 1
                continue
            if not d["pmcid"][i]:
                stat["缺 pmcid"] += 1
                continue
            text = d["text"][i] or ""
            title = d["source_title"][i] or ""
            prose, dig, alp = looks_like_prose(text)
            if not prose:
                stat["表格碎片/非散文"] += 1
                continue
            ok, npos, nneg = clinical_score(text, title, d["journal"][i])
            if not ok:
                stat["非临床"] += 1
                continue

            cell = (sec, yb)
            seen[cell] += 1
            rec = {
                "chunk_id": d["chunk_id"][i], "doc_id": d["doc_id"][i],
                "pmcid": d["pmcid"][i], "pmid": d["pmid"][i],
                "source_title": title, "journal": d["journal"][i],
                "pub_year": d["pub_year"][i], "section_raw": d["section"][i],
                "section": sec, "year_bucket": yb,
                "chunk_index": d["chunk_index"][i],
                "total_chunks": d["total_chunks"][i],
                "token_count": tc, "clinical_pos": npos, "clinical_neg": nneg,
                "text": text,
            }
            # 标准蓄水池抽样：前 k 个直接收，之后以 k/seen 的概率替换
            k = args.pool_per_cell
            if len(pool[cell]) < k:
                pool[cell].append(rec)
            else:
                j = rng.randrange(seen[cell])
                if j < k:
                    pool[cell][j] = rec

        if scanned % 500_000 < args.batch:
            el = time.time() - t0
            print(f"  ... 已扫 {scanned:,}/{total_rows:,} 行 "
                  f"（{scanned/total_rows*100:.1f}%，{el:.0f}s，"
                  f"合格 {sum(seen.values()):,}）", flush=True)

    print(f"\n[抽样] 扫描完成，用时 {time.time()-t0:.0f}s")
    print(f"       合格候选 {sum(seen.values()):,} / {scanned:,} "
          f"（{sum(seen.values())/scanned*100:.2f}%）")
    print("       各道过滤淘汰：")
    for k, v in stat.most_common():
        print(f"         {k:<28} {v:>10,}")

    # ---- 从蓄水池里定额取样：每格 per_cell 条，不足的格子用大格子补足总数 ----
    cells = [(s, y) for s in SECTIONS for y, _ in YEAR_BUCKETS]
    picked, deficit = [], 0
    for cell in cells:
        have = pool[cell]
        rng.shuffle(have)
        take = have[:args.per_cell]
        deficit += args.per_cell - len(take)
        picked.extend(take)
    if deficit:
        print(f"\n[抽样] {deficit} 个名额因格子太小没填满，从候选多的格子补足")
        used = {r["chunk_id"] for r in picked}
        spare = [r for cell in cells for r in pool[cell] if r["chunk_id"] not in used]
        rng.shuffle(spare)
        picked.extend(spare[:deficit])

    # 同一篇文献只留一个 chunk，避免 ground truth 之间互相当干扰项
    by_doc, dedup = set(), []
    for r in picked:
        if r["pmcid"] in by_doc:
            continue
        by_doc.add(r["pmcid"])
        dedup.append(r)
    if len(dedup) < len(picked):
        print(f"[抽样] 同篇文献去重：{len(picked)} → {len(dedup)}")
    picked = dedup

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(F_CHUNKS, "w", encoding="utf-8") as f:
        for i, r in enumerate(picked, 1):
            r["gid"] = f"G{i:03d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[抽样] 写出 {len(picked)} 条 ground truth → {F_CHUNKS}")
    print("\n       实际分层分布（行=section，列=年份桶）：")
    grid = Counter((r["section"], r["year_bucket"]) for r in picked)
    hdr = "".join(f"{y:>12}" for y, _ in YEAR_BUCKETS)
    print(f"         {'':<14}{hdr}{'   合计':>8}")
    for s in SECTIONS:
        row = "".join(f"{grid[(s, y)]:>12}" for y, _ in YEAR_BUCKETS)
        tot = sum(grid[(s, y)] for y, _ in YEAR_BUCKETS)
        print(f"         {s:<14}{row}{tot:>8}")
    print(f"       期刊数 {len({r['journal'] for r in picked})}，"
          f"中位 token {sorted(r['token_count'] for r in picked)[len(picked)//2]}")


# ==============================================================================
# 四、步骤 2：用 qwen3 生成中文 query
# ==============================================================================
SYS_PROMPT = "你是医学检索评测集的出题人。你出的题用来测试检索系统能否把指定的文献片段找回来。"

GEN_PROMPT = """下面是一段来自 PubMed 开放获取文献的英文正文片段。请基于**这段正文**出 {n} 条中文检索问题。

这些问题会被丢进一个有 399.8 万块（覆盖 227.4 万篇）的检索系统里，**系统只看得到问题这一句话，看不到上面这段正文**。

硬性要求（逐条都要满足）：
1. **必须自足**：问题里要写清**疾病名 / 人群 / 干预措施**的具体名称。
   严禁出现「该患者」「这种治疗方案」「本研究」「上述药物」「此类患者」这种指代词——
   离开原文就没人知道你在问谁。
   ✗ 反例：「该患者非愈合溃疡的诊断依据是什么？」（谁？什么病？）
   ✗ 反例：「这种组合治疗方案最常见的严重不良反应有哪些？」（哪种方案？什么病？）
   ✓ 正例：「紫杉醇联合吉西他滨治疗晚期非小细胞肺癌，主要的严重不良反应是什么？」
2. **必须只能靠这段正文回答**：问题要针对本段里具体的发现、数据、方法或结论。
   凡是查教科书、靠常识就能答的泛泛问题，一律不合格。
3. **不许照抄**：不能整段搬用文献标题，也不能直接写出本段里的基因位点、化合物代号、
   队列/试验缩写、登记号（例如 rs1801133、CLARITY-AD、NCT 编号 这类）。
   疾病名、药物通用名、常见临床缩写这些临床医生本来就会说的词**可以用，而且应该用**——
   第 1 条要求自足，靠的就是这些词。
4. **不许抄具体数字**：本段里出现的样本量、百分比、p 值、剂量数值都不许原样写进问题。
5. 写成**临床医生自然提问**的口吻，中文，每条 15~40 字。
6. {n} 条问题要从不同角度问（比如一条问疗效结论、一条问安全性或研究人群），不要互为改写。

【文献标题】{title}
【章节】{section}
【发表年份】{year}
【正文片段】
{text}

输出一个 JSON 数组，元素是字符串，形如：["问题一", "问题二"]"""

# 词面泄漏的自动初筛：把 query 里的「拉丁字母串 / 数字」抠出来，看是否在原文里逐字出现。
# 中文 query 打英文库，主要泄漏渠道就是这类原样搬运的标识符与数值。
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_NUM = re.compile(r"\d+(?:\.\d+)?%?")
# 找裸数字前用它把「带字母的整块 token」抠掉，避免把 HOXC10 的 10、BRAF V600E 的 600
# 当成"照搬原文数字"。要点两个：
#   ① 连字母带数字整块吃掉（含中间的 - 与 .）
#   ② **连斜杠续写的编号一起吃掉** —— `miR-23a/206/499` 里的 206、499 是 miRNA 编号，
#      不是研究数值；只吃 `miR-23a` 会把后两个当成照搬数字。
_TOKEN_FOR_NUM = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9\-\.]*\d[A-Za-z0-9\-\.]*"      # 字母开头、含数字
    r"|\d[A-Za-z0-9\-\.]*[A-Za-z][A-Za-z0-9\-\.]*)"       # 数字开头、含字母
    r"(?:\s*/\s*[A-Za-z0-9\-\.]+)*")                       # 斜杠续写的编号

# ⚠ 临床医生嘴里本来就带的缩写，不算泄漏。
# 不加这个白名单，「HIV 患者的合并症」「CT 能不能替代」这种完全正常的问法会被判成照搬原文，
# 而复核决定文件又按旗标预设 keep——等于自动丢掉一批好题。
# 真正要防的是**罕见标识符**：基因符号、试验缩写、登记号、化合物代号。
COMMON_ABBR = {
    "hiv", "aids", "copd", "ct", "mri", "pet", "ecg", "ekg", "icu", "bmi", "crp",
    "esr", "dna", "rna", "pcr", "hpv", "hbv", "hcv", "tb", "sars", "cov", "covid",
    "pd-1", "pd-l1", "ctla-4", "her2", "egfr", "braf", "kras", "brca", "vegf",
    "nsclc", "sclc", "aml", "cll", "cml", "all", "gvhd", "ckd", "esrd", "afib",
    "hfref", "hfpef", "stemi", "nstemi", "pci", "cabg", "dvt", "pe", "ards",
    "ptsd", "adhd", "ms", "als", "ra", "sle", "ibd", "gerd", "nafld", "t2dm",
    "hba1c", "ldl", "hdl", "tsh", "psa", "cea", "afp", "os", "pfs", "orr", "qol",
    "rct", "icd", "who", "nyha", "ecog", "kps", "mmse", "gcs", "apache", "sofa",
    "ppi", "nsaid", "ssri", "snri", "ace", "arb", "ccb", "doac", "lmwh", "tnf",
    "il-6", "cart", "car-t", "ici", "art", "hrt", "ivf", "cpap", "ecmo", "crrt",
}


# 指代词：出题人看着原文写题，很容易写出「该患者」「这种方案」这种离开原文就读不懂的句子。
# 检索系统只看得到问题这一句，这类 query 等于拿一句无锚点的话去查全库 399.8 万块，测不出任何东西。
# 实测第一版提示词 6 条里中 3 条，属系统性问题，已在提示词里加硬规则；这里留一道兜底。
_ANAPHORA = re.compile(r"该[患者研药方治病人受试]|这[种类项个]|本[研试study]|上述|此类|这些[药治方]")


def leak_flags(query: str, text: str, title: str):
    """返回 (可疑词表, 原因列表)。

    只是**初筛**，不是判决。真正的泄漏检查在跑测时做：比 BM25-only 与向量-only 的
    Recall@10——两者都逼近 1.0 才说明 query 抄了原文词（需求方案第 4 条）。

    ⚠ 数字旗标的精度上限（2026-08-12 修完之后实测，240 条上 19 → **5** 条）：
      剩下的 5 条里 2 条是真照抄（`存活率降低45%`、`卧床45天`），
      3 条仍是误报（`10年预后`、`30天内再入院`——标准临床终点，不是照抄原文）。
      **不再往下修。** 区分「这个数字是本研究的参数」还是「通用临床概念」是**语义判断**，
      不是词法能解的——同 docs/工程笔记.md 三·18「空洞的部分作答不能用规则解」是同一堵墙。
      当成给人看的提示就够了：精度从 1/19 提到 2/5，人工扫一眼的成本已经可以接受。
    """
    blob = (text or "") + " " + (title or "")
    low = blob.lower()
    leaked, why = [], []
    if _ANAPHORA.search(query):
        why.append(f"含指代词（离开原文读不懂）：{_ANAPHORA.search(query).group()}")
    for m in _LATIN.findall(query):
        if m.lower() in COMMON_ABBR:
            continue
        if m.lower() in low:
            leaked.append(m)
    if leaked:
        why.append(f"疑似照搬原文罕见词：{leaked[:5]}")

    # ⚠ 找裸数字之前**必须先把英文 token 整个抠掉**（2026-08-12 修）。
    #   否则抓到的"数字"全是标识符碎片：CXCL**11**、HOXC**10**、miR-**23**a、
    #   BRAF V**600**E、COVID-**19**、anti-Ro**60**——旧版 19 条旗标里几乎全是这种误报。
    #   真正要防的是照抄**样本量 / 百分比 / p 值 / 剂量**这类可作词面锚点的数值。
    stripped = _TOKEN_FOR_NUM.sub(" ", query)
    nums = [m for m in _NUM.findall(stripped) if len(m.rstrip('%')) >= 2 and m in blob]
    # 年份是临床问题里的正常成分（「2016 年 ACR/EULAR 标准」），不算照搬
    nums = [m for m in nums if not re.fullmatch(r"(19|20)\d{2}", m)]
    if nums:
        why.append(f"照搬原文数字：{nums[:5]}")
    n = len(query.strip())
    if not (12 <= n <= 48):
        why.append(f"长度 {n} 字，超出 15~40 的宽限范围")
    return leaked + nums, why


def do_gen(args):
    if not os.path.exists(F_CHUNKS):
        sys.exit(f"找不到 {F_CHUNKS}，先跑 --sample")
    rows = [json.loads(l) for l in open(F_CHUNKS, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    gl = _load_by_path("生成_LLM生成器", "生成_LLM生成器.py")
    gen = gl.LLMGenerator(model_name=args.model, temperature=args.temperature,
                          max_tokens=400, num_ctx=8192, timeout=240)

    done = set()
    if args.resume and os.path.exists(F_RAW):
        for l in open(F_RAW, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["gid"])
        print(f"[生成] 续跑：已有 {len(done)} 条，跳过")

    mode = "a" if (args.resume and os.path.exists(F_RAW)) else "w"
    t0, nq, nfail = time.time(), 0, 0
    with open(F_RAW, mode, encoding="utf-8") as fout:
        for i, r in enumerate(rows, 1):
            if r["gid"] in done:
                continue
            # 正文太长会挤掉指令，截到 ~4000 字符（约 1000 token），保留段首
            text = r["text"][:4000]
            prompt = GEN_PROMPT.format(n=args.n_per_chunk, title=r["source_title"],
                                       section=r["section"], year=r["pub_year"], text=text)
            res = gen.generate(prompt, system_prompt=SYS_PROMPT,
                               json_output=True, expect="array", json_retries=1)
            qs = res.get("json") if res.get("json_ok") else None
            if not isinstance(qs, list):
                nfail += 1
                print(f"  [{i}/{len(rows)}] {r['gid']} ✗ JSON 解析失败："
                      f"{str(res.get('json_error'))[:80]}", flush=True)
                continue
            qs = [str(q).strip() for q in qs if str(q).strip()][:args.n_per_chunk]
            out = dict(r)
            out["queries"] = []
            for j, q in enumerate(qs, 1):
                leaked, why = leak_flags(q, r["text"], r["source_title"])
                out["queries"].append({
                    "qid": f"{r['gid']}-{j}", "query": q,
                    "leaked_terms": leaked, "flags": why,
                })
                nq += 1
            out["gen_elapsed"] = round(res.get("elapsed", 0), 2)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            flag = sum(1 for q in out["queries"] if q["flags"])
            print(f"  [{i}/{len(rows)}] {r['gid']} {r['section']:<12} "
                  f"{len(qs)} 条{'（' + str(flag) + ' 条带旗标）' if flag else ''} "
                  f"{res.get('elapsed', 0):.1f}s", flush=True)

    el = time.time() - t0
    print(f"\n[生成] 完成：{nq} 条 query，失败 {nfail} 个 chunk，用时 {el/60:.1f} 分钟")
    print(f"       → {F_RAW}")
    print(f"       下一步：--review 出复核清单（要求 human-in-the-loop，别跳）")


# ==============================================================================
# 五、步骤 3 / 4：人工复核与定稿
# ==============================================================================
# ==============================================================================
# 四之二、三层分层（用户 2026-08-12 拍板的口径）
# ==============================================================================
# **不合成"总体 Recall"。** 三层各自独立报。
# 理由（用户原话）：合成的总体数带着必须解释的偏差，那个数你永远不会引用，
# 不如从一开始就不产生它。
#
#   T1 纯中文临床层    query 里没有任何英文 token          ← **主指标**，报告里的召回率用这个
#   T2 术语直穿层      嵌了标准临床缩写 / 药物通用名        ← 见下方长注释，它有独立价值
#   T3 标识符锚定层    嵌了基因符号 / miRNA / 化合物代号 /
#                     队列或项目名                        ← 词面锚点极强，视为召回**上界**
#
# ⚠ T2 不要只当成"轻度泄漏"。中文 query 走的是「qwen3 整句翻译 → 英文检索」，
#   而 duloxetine / G-CSF 这类词在翻译时是**原样穿过**的。所以这一层实际在测
#   **「翻译器遇到本来就是英文的术语时会不会把它搞坏」**——考虑到词典直译会把
#   词表外病名整词丢弃那个已知坑（docs/工程笔记.md 三·2「法布里病→蝴蝶名录」那条），
#   这条链路值得单独有个数。它到底算不算白送锚点，由跑测数据回答：
#   如果 T2 的 BM25-only 召回和 T1 差不多，说明根本没送出多少锚点。

TIER_NAMES = {
    1: "T1 纯中文临床层（主指标）",
    2: "T2 术语直穿层（中文问句嵌英文术语，翻译时原样穿过）",
    3: "T3 标识符锚定层（词面锚点极强，视为召回上界）",
}

# 标准临床缩写 / 药物通用名 / 通行医学词——临床医生确实这么写
STD_CLINICAL = {
    "dna", "rna", "pcr", "nm", "iii", "hiv", "tb", "copd", "ebv", "covid-19",
    "sars-cov-2", "pd-l1", "pd-1", "cd8", "cd3", "er", "braf", "v600e",
    "nsclc", "ptc", "ptld", "ad", "adrd", "pfps", "vap", "esd", "srp", "fsh",
    "g-csf", "acr", "eular", "nad", "hhs", "fdg-pet", "locf", "ifn-",
    "duloxetine", "rituximab", "filgrastim", "pegfilgrastim",
    "marfan", "sarcoma", "mirna", "nanopore", "kinesio", "manuka", "snuff",
}
# 罕见标识符：基因/蛋白符号、miRNA、化合物或构建体代号、队列/注册库/项目名
RARE_ID = {
    "hoxc10", "cxcl11", "mcm3", "tubg1", "chi3l1", "epha2", "nr6a1", "taar9",
    "oct4", "ro60", "mir-23a", "mir-223", "mir-643a-3p",
    "irfv", "fty720", "lu-if7", "if7", "ad5-", "nw-kla", "zincol-2016", "nr-421",
    "alhs", "arat", "survsarc", "biobank", "uk", "bounce", "back", "now",
    "nsc", "tk",
}

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-\.]*")

# 三条硬失败，直接删（用户 2026-08-12 确认）：
#   G107-1/2  模型照抄了提示词里的占位符，生成出来的字面就是「问题一」「问题二」
#   G033-1    生物强化小麦对巴基斯坦妇女膳食锌摄入——农业营养学论文混过了临床过滤
# ⚠ G033-1 顺带暴露：**抽样的临床过滤有洞**，语料里可能还混着别的非临床论文。
#   不影响本轮（就这一条），但 P0 抽样前要回头查一次过滤规则。
#   G032-1/2  发酵残渣做农药载体、小麦种子发芽率与白粉病防治——纯农业问题，
#             却落在 T1 主指标层里，直接污染对外引用的那个数。
#             ⚠ 它是 2026-08-12 补硬否决规则时回放发现的：G032 出自 RSC Advances，
#             新规则（期刊否决 + 标题 "seed treatment"）能挡住，但集合是用旧规则建的。
HARD_DROP = {"G107-1", "G107-2", "G033-1", "G032-1", "G032-2"}


def classify_tier(query: str):
    """返回 (层号, 该 query 里的英文 token, 未登记的 token)。

    未登记的 token **默认判进 T3**——宁可误标成"锚点强"，也不让一个没人看过的
    标识符悄悄混进主指标 T1/T2。调用方会把未登记 token 打出来提请人工归档。
    """
    toks = [t for t in _TOKEN.findall(query) if len(t) >= 2]
    if not toks:
        return 1, [], []
    unknown = [t for t in toks if t.lower() not in STD_CLINICAL and t.lower() not in RARE_ID]
    if unknown or any(t.lower() in RARE_ID for t in toks):
        return 3, toks, unknown
    return 2, toks, []


def do_review(args):
    if not os.path.exists(F_RAW):
        sys.exit(f"找不到 {F_RAW}，先跑 --gen")
    rows = [json.loads(l) for l in open(F_RAW, encoding="utf-8") if l.strip()]
    nq = sum(len(r["queries"]) for r in rows)
    flagged = [(r, q) for r in rows for q in r["queries"] if q["flags"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    # ⚠ keep 一律预设 True。旗标是**提请注意**，不是判决——
    #   按旗标预设 keep=False 会把「HIV 患者合并症」这种完全正常的问法自动丢掉。
    with open(F_DECISION, "w", encoding="utf-8") as f:
        for r in rows:
            for q in r["queries"]:
                f.write(json.dumps({"qid": q["qid"], "query": q["query"],
                                    "keep": True,
                                    "auto_flags": q["flags"]}, ensure_ascii=False) + "\n")

    with open(F_REVIEW, "w", encoding="utf-8") as f:
        f.write("# golden 集人工复核清单\n\n")
        f.write(f"- 来源 chunk：{len(rows)}｜生成 query：{nq}\n")
        f.write(f"- 自动旗标：{len(flagged)} 条（**旗标只是初筛，不是判决**）\n")
        f.write(f"- 改 `{os.path.basename(F_DECISION)}` 里的 `keep` 字段（默认全 true），"
                f"再跑 `--finalize`\n\n")
        f.write("判不合格的三条常见理由：泛泛常识题 / 照抄原文标识符或数字 / 两条互为改写。\n\n")
        if flagged:
            f.write(f"## 先看这 {len(flagged)} 条（带旗标）\n\n")
            for r, q in flagged:
                f.write(f"- `{q['qid']}` {q['query']}\n")
                for w in q["flags"]:
                    f.write(f"    - {w}\n")
            f.write("\n")
        f.write("---\n\n## 全部逐条\n\n")
        for r in rows:
            f.write(f"## {r['gid']}　{r['section']} · {r['pub_year']} · {r['pmcid']}\n\n")
            f.write(f"**{r['source_title'][:160]}**　*{r['journal']}*\n\n")
            for q in r["queries"]:
                mark = "⚠" if q["flags"] else "　"
                f.write(f"- {mark} `{q['qid']}` {q['query']}\n")
                for w in q["flags"]:
                    f.write(f"    - 旗标：{w}\n")
            f.write(f"\n<details><summary>原文片段（{r['token_count']} token）</summary>\n\n"
                    f"```\n{r['text'][:1500]}\n```\n\n</details>\n\n")

    print(f"[复核] 清单 → {F_REVIEW}")
    print(f"[复核] 决定文件 → {F_DECISION}（{nq} 条，其中 {len(flagged)} 条已预标 keep=false）")


def do_finalize(args):
    if not os.path.exists(F_RAW):
        sys.exit("缺 --gen 的产物")
    rows = [json.loads(l) for l in open(F_RAW, encoding="utf-8") if l.strip()]

    # 复核决定文件是**可选**的：没有就默认全留（除硬失败），有就以它为准。
    dec = {}
    if os.path.exists(F_DECISION):
        for l in open(F_DECISION, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                dec[d["qid"]] = bool(d.get("keep"))
        print(f"[定稿] 读到复核决定 {len(dec)} 条")
    else:
        print("[定稿] 没有复核决定文件，除硬失败外全部保留")

    kept, dropped, unknown_all = [], [], Counter()
    for r in rows:
        for q in r["queries"]:
            qid = q["qid"]
            if qid in HARD_DROP:
                dropped.append((qid, "硬失败"))
                continue
            if dec and not dec.get(qid, True):
                dropped.append((qid, "人工剔除"))
                continue
            tier, toks, unknown = classify_tier(q["query"])
            for u in unknown:
                unknown_all[u] += 1
            kept.append({
                "qid": qid, "gid": r["gid"], "query": q["query"],
                "tier": tier, "tier_name": TIER_NAMES[tier], "en_tokens": toks,
                "gt_chunk_id": r["chunk_id"], "gt_pmcid": r["pmcid"],
                "gt_pmid": r["pmid"], "gt_doc_id": r["doc_id"],
                "section": r["section"], "year_bucket": r["year_bucket"],
                "pub_year": r["pub_year"], "journal": r["journal"],
                "source_title": r["source_title"], "token_count": r["token_count"],
                "leaked_terms": q["leaked_terms"],
            })
    with open(F_FINAL, "w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    print(f"[定稿] 保留 {len(kept)} 条，剔除 {len(dropped)} 条 → {F_FINAL}")
    for qid, why in dropped:
        print(f"         剔除 {qid}（{why}）")
    print()
    for t in (1, 2, 3):
        sub = [k for k in kept if k["tier"] == t]
        print(f"  {TIER_NAMES[t]}")
        print(f"     {len(sub)} 条 / {len({k['gid'] for k in sub})} 篇文献")
        g = Counter(k["section"] for k in sub)
        y = Counter(k["year_bucket"] for k in sub)
        print(f"     section：" + "｜".join(f"{s} {g[s]}" for s in SECTIONS))
        print(f"     年份　：" + "｜".join(f"{yb} {y[yb]}" for yb, _ in YEAR_BUCKETS))
    t1 = sum(1 for k in kept if k["tier"] == 1)
    if t1 < 100:
        print(f"\n  ⚠ 主指标层 T1 只有 {t1} 条，不足要求的 100。"
              f"加大 --per-cell 重抽再补一轮。")
    else:
        print(f"\n  ✓ 主指标层 T1 = {t1} 条，满足 ≥100 的下限")
    if unknown_all:
        print(f"\n  ⚠ 有 {len(unknown_all)} 个 token 未登记进 STD_CLINICAL / RARE_ID，"
              f"已按保守规则判进 T3：")
        for t, c in unknown_all.most_common():
            print(f"      {t}  ×{c}")
        print("    （要改判就把它加进 golden_构建.py 顶部那两个集合，再跑一次 --finalize）")


def main():
    ap = argparse.ArgumentParser(description="golden 检索评测集 · 构建")
    ap.add_argument("--sample", action="store_true", help="步骤1：分层抽样 ground truth")
    ap.add_argument("--gen", action="store_true", help="步骤2：用 qwen3 生成中文 query")
    ap.add_argument("--review", action="store_true", help="步骤3：出人工复核清单")
    ap.add_argument("--finalize", action="store_true", help="步骤4：按复核结果定稿")

    ap.add_argument("--parquet", default=MERGED_PARQUET)
    ap.add_argument("--per-cell", type=int, default=6, help="每个 (section×年份) 格取几条")
    ap.add_argument("--pool-per-cell", type=int, default=40, help="每格蓄水池容量（>per-cell 才有得挑）")
    ap.add_argument("--min-tokens", type=int, default=120, help="正文太短撑不起一个具体问题")
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="出题要多样性，这里**故意不用 0**；出完就冻结，不影响跑测的确定性")
    ap.add_argument("--n-per-chunk", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 chunk（试跑用）")
    ap.add_argument("--resume", action="store_true", help="续跑，跳过已生成的")

    args = ap.parse_args()
    if args.sample:
        do_sample(args)
    elif args.gen:
        do_gen(args)
    elif args.review:
        do_review(args)
    elif args.finalize:
        do_finalize(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
