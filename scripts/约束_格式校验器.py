# -*- coding: utf-8 -*-
"""第九阶段 · 格式与约束校验器（format checker）：给出「这次到底守没守约束」的判定

提示词只能提高遵守率，**判定必须由代码给**。本模块把任务书 1.b / 1.c / 1.d 的每一条
落成一个可执行的检查项，输出统一的违规清单（`Violation`），供三处使用：

    ① `约束_受限流水线.py` —— 不合规就回灌违规清单给模型重写（层 D 的重试/修正）
    ② `约束_跑对抗测试.py` —— 统计幻觉率 / 引用准确率 / 格式合规率
    ③ 单独当工具用 —— 拿任何一份已经生成好的答案来体检（`--jsonl` 批量）

六组检查（括号内是违规码前缀）：

    citation     引用编号在不在给定范围内、格式规不规范、事实句有没有挂出处
    structure    必需章节标题是否齐全（核心答案 / 证据总结 / 参考文献）
    terminology  医学缩写首次出现有没有给全称（预定义术语表 + 正则兜底）
    reference    参考文献是否完整（至少标题、期刊、年份）、有没有被改写或凭空多出来
    numeric      答案里的数字能不能在证据里找到（禁止编造的可自动判定部分）
    refusal      该拒答的有没有拒答、拒答短语是否一字不差

**能自动判定什么、不能判定什么，必须说清楚**：
- 能判定：编号越界、章节缺失、缩写没展开、参考文献字段缺失、数字在证据里找不到、拒答短语缺失。
- 判定不了：措辞谨慎的事实性错误（"该药可能有效"而证据说无效）、逻辑错误、临床上的不当推荐。
  这些要靠 ③ 批判审查段与人工评审，本模块不假装能测。
- 数字溯源有已知的假阳性：模型把 12.63% 四舍五入成 12.6%、把 1,234 写成 1234（已归一）、
  或做了单位换算，都会被记为"证据里找不到"。**这类判定按 medium 报，并列出原文供人核。**

用法：
    import importlib.util
    spec = importlib.util.spec_from_file_location("fc", r"E:\\rag\\scripts\\约束_格式校验器.py")
    fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)

    ck = fc.FormatChecker()
    rep = ck.check(answer, citations=ctx["metadata"]["citations"],
                   evidence_text=ctx["context_text"], question=q, expect_refusal=False)
    rep["compliant"], rep["scores"]["citation_accuracy"], rep["violations"]
    print(ck.repair_prompt(rep))              # 回灌给模型的修正指令
    fixed, applied = ck.auto_fix(answer, rep, citations)   # 确定性修正（不调模型）

CLI：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\约束_格式校验器.py --demo
    ... --jsonl E:\\rag\\report_data\\评估_测试集.jsonl      # 体检一批已生成的答案
    ... --glossary                                         # 打印术语表
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cp = _load_by_path("yueshu_tishicing", "约束_提示词层.py")
REFUSAL_PHRASE = _cp.REFUSAL_PHRASE
PARTIAL_PHRASE = _cp.PARTIAL_PHRASE
PARTIAL_ANSWERABLE_HEAD = _cp.PARTIAL_ANSWERABLE_HEAD
PARTIAL_GAP_HEAD = _cp.PARTIAL_GAP_HEAD
REQUIRED_SECTIONS = list(_cp.REQUIRED_SECTIONS)
OPTIONAL_SECTIONS = list(_cp.OPTIONAL_SECTIONS)
CITATION_PREFIX = _cp.CITATION_PREFIX


# ============================================================================
# 一、预定义术语表（任务书 1.d：「通过预定义术语表或正则检测」）
# ============================================================================
#: 缩写 → (英文全称, 中文全称, 其它可接受的中文写法)
#: 收录原则：**本项目语料与答案里真的会出现**的缩写（肿瘤/神经/免疫/代谢/罕见病 + 循证方法学
#: 与统计量），而不是通用医学缩写大全。表越大误报越多——表里有但答案里没出现的条目不产生
#: 任何检查，所以宁可精确不求全。
TERM_GLOSSARY: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    # ---- 研究设计与方法学 ----
    "RCT": ("randomized controlled trial", "随机对照试验", ("随机对照研究",)),
    "ITT": ("intention-to-treat", "意向性治疗分析", ("意向治疗",)),
    "PP": ("per-protocol", "符合方案集", ()),
    "GWAS": ("genome-wide association study", "全基因组关联研究", ()),
    "WGS": ("whole-genome sequencing", "全基因组测序", ()),
    "NGS": ("next-generation sequencing", "二代测序", ("新一代测序",)),
    "QoL": ("quality of life", "生活质量", ()),
    "AE": ("adverse event", "不良事件", ("不良反应",)),
    "SAE": ("serious adverse event", "严重不良事件", ()),
    # ---- 统计量 ----
    "HR": ("hazard ratio", "风险比", ("危险比",)),
    "OR": ("odds ratio", "比值比", ("优势比",)),
    "RR": ("relative risk", "相对危险度", ("相对风险",)),
    "CI": ("confidence interval", "置信区间", ("可信区间",)),
    "NNT": ("number needed to treat", "需治疗人数", ()),
    "SD": ("standard deviation", "标准差", ()),
    "IQR": ("interquartile range", "四分位距", ()),
    # ---- 疗效终点 ----
    "OS": ("overall survival", "总生存期", ("总生存",)),
    "PFS": ("progression-free survival", "无进展生存期", ("无进展生存",)),
    "DFS": ("disease-free survival", "无病生存期", ("无病生存",)),
    "ORR": ("objective response rate", "客观缓解率", ("总缓解率",)),
    "DCR": ("disease control rate", "疾病控制率", ()),
    "EDSS": ("Expanded Disability Status Scale", "扩展残疾状态量表", ()),
    "ARR": ("annualized relapse rate", "年复发率", ()),
    "ADAS-Cog": ("Alzheimer's Disease Assessment Scale-Cognitive subscale",
                 "阿尔茨海默病评定量表认知分量表", ()),
    "CDR-SB": ("Clinical Dementia Rating-Sum of Boxes", "临床痴呆评定量表总和评分", ()),
    "MMSE": ("Mini-Mental State Examination", "简易精神状态检查", ()),
    # ---- 肿瘤 ----
    "NSCLC": ("non-small cell lung cancer", "非小细胞肺癌", ()),
    "SCLC": ("small cell lung cancer", "小细胞肺癌", ()),
    "TNBC": ("triple-negative breast cancer", "三阴性乳腺癌", ()),
    "HCC": ("hepatocellular carcinoma", "肝细胞癌", ()),
    "CRC": ("colorectal cancer", "结直肠癌", ()),
    "EGFR": ("epidermal growth factor receptor", "表皮生长因子受体", ()),
    "ALK": ("anaplastic lymphoma kinase", "间变性淋巴瘤激酶", ()),
    "PD-1": ("programmed cell death protein 1", "程序性死亡受体 1", ("程序性细胞死亡蛋白 1",)),
    "PD-L1": ("programmed death-ligand 1", "程序性死亡配体 1", ()),
    "CAR-T": ("chimeric antigen receptor T cell", "嵌合抗原受体 T 细胞", ()),
    "TMB": ("tumor mutational burden", "肿瘤突变负荷", ()),
    # ---- 神经与精神 ----
    "MS": ("multiple sclerosis", "多发性硬化", ("多发性硬化症",)),
    "RRMS": ("relapsing-remitting multiple sclerosis", "复发缓解型多发性硬化", ()),
    "SPMS": ("secondary progressive multiple sclerosis", "继发进展型多发性硬化", ()),
    "PPMS": ("primary progressive multiple sclerosis", "原发进展型多发性硬化", ()),
    "DMT": ("disease-modifying therapy", "疾病修正治疗", ("疾病修饰治疗",)),
    "AD": ("Alzheimer disease", "阿尔茨海默病", ("阿尔兹海默病",)),
    "MCI": ("mild cognitive impairment", "轻度认知障碍", ("轻度认知损害",)),
    "ARIA": ("amyloid-related imaging abnormalities", "淀粉样蛋白相关影像学异常", ()),
    "PD": ("Parkinson disease", "帕金森病", ()),
    "ALS": ("amyotrophic lateral sclerosis", "肌萎缩侧索硬化", ("渐冻症",)),
    "CSF": ("cerebrospinal fluid", "脑脊液", ()),
    "BBB": ("blood-brain barrier", "血脑屏障", ()),
    # ---- 免疫、代谢与其它 ----
    "ERT": ("enzyme replacement therapy", "酶替代治疗", ("酶替代疗法",)),
    "GLA": ("alpha-galactosidase A", "α-半乳糖苷酶 A", ("α半乳糖苷酶A",)),
    "Gb3": ("globotriaosylceramide", "三己糖神经酰胺", ("Gb-3",)),
    "T2DM": ("type 2 diabetes mellitus", "2 型糖尿病", ("2型糖尿病",)),
    "COPD": ("chronic obstructive pulmonary disease", "慢性阻塞性肺疾病", ("慢阻肺",)),
    "CKD": ("chronic kidney disease", "慢性肾脏病", ()),
    "eGFR": ("estimated glomerular filtration rate", "估算肾小球滤过率", ()),
    "HbA1c": ("glycated hemoglobin", "糖化血红蛋白", ()),
    "BMI": ("body mass index", "体重指数", ("体质指数",)),
    "CVD": ("cardiovascular disease", "心血管疾病", ()),
    "IL-6": ("interleukin-6", "白细胞介素 6", ("白介素6",)),
    "TNF": ("tumor necrosis factor", "肿瘤坏死因子", ()),
    "mAb": ("monoclonal antibody", "单克隆抗体", ("单抗",)),
    "S1P": ("sphingosine-1-phosphate", "1-磷酸鞘氨醇", ("鞘氨醇-1-磷酸",)),
    "IFN": ("interferon", "干扰素", ()),
    # ---- 基因编辑（阶段一验收错题 ① 的主题）----
    # 注：CRISPR / Cas9 **不在**表里，见下面 NO_EXPANSION_NEEDED 的说明
    "sgRNA": ("single guide RNA", "单向导 RNA", ("单链向导RNA",)),
    "gRNA": ("guide RNA", "向导 RNA", ()),
    "PAM": ("protospacer adjacent motif", "前间隔序列邻近基序", ()),
    "HDR": ("homology-directed repair", "同源定向修复", ()),
    "NHEJ": ("non-homologous end joining", "非同源末端连接", ()),
    "indel": ("insertion or deletion", "插入缺失", ()),
}

#: 通用到不必展开的缩写：展开它们只会让答案变啰嗦，判违规属于误报。
#: 收录标准是「在本领域文献里当专名用、几乎从不展开」。**CRISPR / Cas9 就属于这一类**——
#: 实测第一版把它俩放进术语表，结果每一份基因编辑答案都被判「缺全称」，要求模型写
#: 「规律成簇间隔短回文重复序列（CRISPR）」纯属制造噪声。sgRNA / PAM / HDR 这些留在术语表里，
#: 因为论文里首次出现确实会展开。
NO_EXPANSION_NEEDED: Set[str] = {
    "DNA", "RNA", "mRNA", "HIV", "AIDS", "WHO", "FDA", "EMA", "NIH", "NMPA",
    "CT", "MRI", "PET", "PCR", "ELISA", "USA", "US", "UK", "EU", "COVID",
    "PMID", "PMCID", "PMC", "DOI", "URL", "ID", "NCT", "ICU", "IV", "IM",
    "CRISPR", "Cas9", "Cas12a", "Cas", "RNA-seq",
}

#: 正则兜底：术语表之外的疑似缩写（2~8 位，含大写字母，可带数字与连字符）。
#: 只报 low，不判违规——它抓到的很可能是基因名、试验名或普通专有名词。
_MAYBE_ABBREV = re.compile(r"\b(?=[A-Za-z0-9\-]{2,10}\b)(?=[A-Za-z0-9\-]*[A-Z]{2,})[A-Za-z][A-Za-z0-9\-]*\b")
#: 罗马数字（III 期试验）、纯数字后缀等不算缩写
_ROMAN = re.compile(r"^[IVX]{1,5}$")


# ============================================================================
# 二、正则：引用编号、章节、数字事实
# ============================================================================
#: 标准编号 [S1]
_MARKER_STD = re.compile(rf"\[{CITATION_PREFIX}(\d{{1,3}})\]")
#: 各种不规范写法：[文献1] [文献 1] (S1) （S1） 【S1】 [s1] [S 1] [1]
#: 三条分支而不是一条带可选前缀的正则——**圆括号里的裸数字不能当引用**：
#: 「随访(24)周」「分组(3)」会被一条式正则吞掉，`auto_fix` 再把它当无效编号删掉，
#: 就成了改坏正文。裸数字只认方括号 `[1]` 这一种。
_MARKER_LOOSE = re.compile(
    r"[\[\(（【]\s*(?:文献|参考文献|来源|证据)\s*(\d{1,3})\s*[\]\)）】]"     # [文献1]
    r"|[\[\(（【]\s*[Ss]\s*(\d{1,3})\s*[\]\)）】]"                        # [S1] (S1) 【S1】 [s 1]
    r"|\[\s*(\d{1,3})\s*\]")                                            # [1]


def _loose_num(m: "re.Match") -> str:
    """取不规范写法里的编号数字（三条分支只会有一组命中）。"""
    return next(g for g in m.groups() if g)
#: 章节标题（任何 # 级别，允许前后空格与全角冒号）。
#: ⚠ 前后一律用 `[ \t]*` 而不是 `\s*`：`\s` 吃换行，会把标题后面的空行一起吞掉，
#: 替换标题时正文就粘到标题上了（本模块 auto_fix 实测踩过）。
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*([^\n#]+?)[ \t]*[:：]?[ \t]*$", re.MULTILINE)
#: 标题写成粗体或光秃秃一行（模型很常见的偏差）：`**核心答案**` / `核心答案：`
_PSEUDO_HEADING = re.compile(r"^[ \t]{0,3}(?:\*\*|__)?[ \t]*([^\n#*_：:]{2,14})[ \t]*"
                             r"(?:\*\*|__)?[ \t]*[:：]?[ \t]*$", re.MULTILINE)
#: 标题里的编号前缀：`## 1. 核心答案` / `## 一、核心答案`
_HEAD_NUM = re.compile(r"^\s*(?:\d+|[一二三四五六七八九十]+)\s*[.、)）]\s*")
#: 章节标题的常见同义改写 → 应该用的名字（可确定性改回来）
SECTION_SYNONYMS: Dict[str, str] = {
    "结论": "核心答案", "答案": "核心答案", "直接回答": "核心答案", "回答": "核心答案",
    "核心结论": "核心答案", "主要发现": "核心答案",
    "证据要点": "证据总结", "证据": "证据总结", "证据摘要": "证据总结",
    "关键发现": "证据总结", "支持证据": "证据总结",
    "引用文献": "参考文献", "参考资料": "参考文献", "文献列表": "参考文献",
    "参考": "参考文献", "references": "参考文献",
    "证据强度与一致性": "证据强度与局限", "局限与提示": "证据强度与局限",
    "局限性": "证据强度与局限", "证据强度": "证据强度与局限",
}

#: 数字事实的抽取模式。命名与阶段八 `评估_答案评估器.py` 的关键信息类别有意保持一致，
#: 但**目的不同**：那边算"证据里的信息答案覆盖了多少"（召回），这边算"答案里的数字证据里
#: 有没有"（溯源）。方向相反，所以不复用那份实现。
_NUM_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("percentage", re.compile(r"\d+(?:[.,]\d+)?\s*%")),
    ("dose", re.compile(r"\d+(?:[.,]\d+)?\s*(?:mg|g|kg|µg|μg|ug|mcg|ng|mL|ml|L|IU|U|"
                        r"mmol|mol|nmol|μmol|umol)\b(?:\s*/\s*(?:kg|d|day|h|hr|周|天|日|次|m2|m²))?",
                        re.IGNORECASE)),
    ("pvalue", re.compile(r"[pP]\s*[<>=≤≥＜＞]\s*0?[.,]\d+")),
    ("statistic", re.compile(r"\b(?:HR|OR|RR|SMD|WMD|MD)\s*[=＝:：]?\s*\d+(?:[.,]\d+)?",
                             re.IGNORECASE)),
    ("sample_size", re.compile(r"\b[nN]\s*[=＝]\s*\d[\d,]*")),
    ("count", re.compile(r"\d[\d,]*\s*(?:例|名患者|位患者|patients|participants|subjects)")),
    ("year", re.compile(r"(?<![\d.])(?:19|20)\d{2}(?![\d.])")),
]

#: 免责声明、系统提示行：不参与句子级引用检查
_SYSTEM_LINE = re.compile(r"^\s*>\s")


# ============================================================================
# 三、句子切分（本模块自带，刻意不复用阶段八的评估器）
# ============================================================================
# 复用会把 `rouge` 这个第三方依赖拖进校验器，而校验器要能在流水线里无条件跑。
# 规则与阶段七、八一致：句末标点后必须跟空白/行尾，天然排除 "0.45"、"Fig. 1"。
_ABBREV_DOT = ("e.g", "i.e", "et al", "al", "vs", "cf", "fig", "no", "ca", "approx",
               "dr", "prof", "sd", "se", "ci", "resp", "eq", "ref", "refs", "vol")
_SENT_SPLIT = re.compile(r"(?<=[。！？；;])\s*|(?<=[.!?])(?=\s)")


def split_sentences(text: str) -> List[str]:
    """切句：中文标点直接切；英文句点要求后接空白，且排除常见缩写点。"""
    out: List[str] = []
    for raw in re.split(r"\n+", text or ""):
        buf = ""
        for piece in _SENT_SPLIT.split(raw):
            if piece is None:
                continue
            buf += piece
            s = buf.strip()
            if not s:
                buf = ""
                continue
            tail = re.split(r"[\s]", s)[-1].rstrip(".").lower()
            if s.endswith(".") and tail in _ABBREV_DOT:
                continue                      # 缩写点，继续attach到下一段
            out.append(s)
            buf = ""
        if buf.strip():
            out.append(buf.strip())
    return [s for s in out if s]


# ============================================================================
# 四、违规条目
# ============================================================================
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

#: 哪些违规码算"格式问题"（任务书要的**格式合规率**只看这一组）
FORMAT_CODES = ("structure.", "terminology.", "reference.incomplete",
                "reference.missing_entry", "citation.nonstandard_form")
#: 哪些违规码算"幻觉信号"（**幻觉率**只看这一组）
HALLUCINATION_CODES = ("citation.invalid_number", "reference.fabricated",
                       "reference.altered", "numeric.ungrounded", "refusal.missing")


@dataclass
class Violation:
    code: str
    severity: str                     # high / medium / low
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 五、工具函数
# ============================================================================
def strip_markers(text: str) -> str:
    """去掉出处标记，避免 [S3] 里的 3 被数字检查当成事实。"""
    return _MARKER_STD.sub(" ", text or "")


ALL_SECTIONS = REQUIRED_SECTIONS + OPTIONAL_SECTIONS


def canon_section(name: str) -> Optional[str]:
    """标题名 → 标准章节名；不是章节标题就返回 None。

    容忍三类偏差（都是实测里模型真会犯的）：编号前缀「1. 核心答案」、粗体「**核心答案**」、
    同义改写「结论 / 证据要点 / 引用文献」。容忍不等于放过——`check_structure` 仍会把
    偏差记为违规，`auto_fix` 再确定性改回标准写法。
    """
    n = _HEAD_NUM.sub("", (name or "").strip().strip("*_ 　")).strip(" ：:")
    if n in ALL_SECTIONS:
        return n
    return SECTION_SYNONYMS.get(n) or SECTION_SYNONYMS.get(n.lower())


def normalize_headings(answer: str) -> Tuple[str, List[Tuple[str, str]]]:
    """把各种走样的章节标题统一成 `## 标准名`，返回 (新文本, [(原写法, 标准名)])。

    `section_map` 与 `auto_fix` 共用这一个函数：**判定与修正必须基于同一套归一规则**，
    否则会出现"校验说缺章节、修正后仍然缺"的死循环。
    """
    changes: List[Tuple[str, str]] = []

    def _fix(m: re.Match) -> str:
        raw = m.group(0).strip()
        canon = canon_section(m.group(1))
        if not canon:
            return m.group(0)
        want = f"## {canon}"
        if raw != want:
            changes.append((raw, canon))
        return want

    text = _HEADING.sub(_fix, answer or "")
    text = _PSEUDO_HEADING.sub(_fix, text)
    return text, changes


def section_map(answer: str) -> Dict[str, str]:
    """按标题把答案切成 {标准章节名: 正文}。"""
    text, _ = normalize_headings(answer)
    heads = [(m.start(), m.end(), m.group(1).strip()) for m in _HEADING.finditer(text)]
    out: Dict[str, str] = {}
    for i, (s, e, name) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        canon = canon_section(name) or name
        # 同名章节重复出现时拼接，不覆盖（模型偶尔会把参考文献写两遍）
        out[canon] = (out.get(canon, "") + "\n" + text[e:end].strip()).strip()
    return out


def found_headings(answer: str) -> List[str]:
    return [m.group(1).strip() for m in _HEADING.finditer(answer or "")]


def _norm_num(s: str) -> str:
    """数字事实归一：去空格与千分位逗号、统一小写与全角符号，便于跨写法比对。"""
    s = s.strip().lower()
    s = s.replace("＝", "=").replace("＜", "<").replace("＞", ">").replace("μ", "µ")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)       # 1,234 → 1234
    s = re.sub(r"\s+", "", s)
    return s


def extract_numeric_facts(text: str) -> List[Dict[str, str]]:
    """抽出文本里的数字事实（百分比/剂量/p 值/统计量/样本量/年份）。"""
    out: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for kind, pat in _NUM_PATTERNS:
        for m in pat.finditer(text or ""):
            raw = m.group(0)
            key = (kind, _norm_num(raw))
            if key in seen:
                continue
            seen.add(key)
            out.append({"kind": kind, "raw": raw.strip(), "norm": _norm_num(raw)})
    return out


def _en_norm(s: str) -> str:
    return re.sub(r"[\s\-‐‑–—_]", "", (s or "").lower())


# ============================================================================
# 六、校验器
# ============================================================================
class FormatChecker:
    """六组检查一次跑完，返回统一的报告 dict。

    Args:
        glossary:            预定义术语表（缩写 → 全称）
        required_sections:   必需章节
        expansion_window:    判"首次出现附近有没有全称"的字符窗口（前后各 N 字）
        min_citation_coverage: 事实句挂出处的比例下限，低于此值报 `citation.missing`
        check_numeric:       是否做数字溯源（无证据文本时自动跳过）
    """

    def __init__(self,
                 glossary: Optional[Dict[str, Tuple[str, str, Tuple[str, ...]]]] = None,
                 required_sections: Optional[Sequence[str]] = None,
                 optional_sections: Optional[Sequence[str]] = None,
                 expansion_window: int = 120,
                 min_citation_coverage: float = 0.8,
                 check_numeric: bool = True,
                 max_examples: int = 5):
        self.glossary = dict(glossary if glossary is not None else TERM_GLOSSARY)
        self.required_sections = list(required_sections if required_sections is not None
                                      else REQUIRED_SECTIONS)
        self.optional_sections = list(optional_sections if optional_sections is not None
                                      else OPTIONAL_SECTIONS)
        self.expansion_window = int(expansion_window)
        self.min_citation_coverage = float(min_citation_coverage)
        #: 存成 `enable_numeric` 而不是 `check_numeric`：后者与同名方法撞车，
        #: 属性会把方法覆盖掉，调用时报 'bool' object is not callable
        self.enable_numeric = bool(check_numeric)
        self.max_examples = int(max_examples)

    # ------------------------------------------------------------------ 引用
    def check_citations(self, answer: str, valid: Set[str],
                        sections: Dict[str, str]) -> Tuple[Dict[str, Any], List[Violation]]:
        """编号有效性 + 写法规范性 + 事实句覆盖率。

        三件事分开记，因为处置完全不同：越界编号是**幻觉**（必须删）、写法不规范是**格式**
        （可确定性改写）、事实句没挂出处是**覆盖**问题（只能让模型补）。
        """
        vio: List[Violation] = []
        # ⚠ 参考文献一节的行首编号**不是正文引用**：那一段是代码生成的，把它算进来
        # 会凭空拉高引用总数、稀释准确率（实测 4 份真答案 71 个编号里有 26 个来自文献列表）。
        ref_body = sections.get(REQUIRED_SECTIONS[2], "")
        body_text = (answer or "").replace(ref_body, " ") if ref_body else (answer or "")
        std = _MARKER_STD.findall(body_text)
        used_std = [f"{CITATION_PREFIX}{n}" for n in std]

        # 不规范写法：先把标准写法挖掉，剩下的才可能是不规范的
        holes = _MARKER_STD.sub(" ", body_text)
        nonstd = [(m.group(0), _loose_num(m)) for m in _MARKER_LOOSE.finditer(holes)]

        invalid = sorted({m for m in used_std if m not in valid},
                         key=lambda s: int(s[len(CITATION_PREFIX):]))
        nonstd_valid = [(raw, f"{CITATION_PREFIX}{n}") for raw, n in nonstd
                        if f"{CITATION_PREFIX}{n}" in valid]
        nonstd_invalid = [(raw, f"{CITATION_PREFIX}{n}") for raw, n in nonstd
                          if f"{CITATION_PREFIX}{n}" not in valid]

        if invalid:
            vio.append(Violation(
                "citation.invalid_number", "high",
                f"引用了证据范围外的编号 {'、'.join(invalid)}（可用编号共 {len(valid)} 条）",
                {"invalid": invalid, "available": sorted(valid, key=lambda s: int(s[1:]))},
                auto_fixable=True))
        if nonstd_valid:
            vio.append(Violation(
                "citation.nonstandard_form", "medium",
                f"出处编号写法不规范 {len(nonstd_valid)} 处（如 {nonstd_valid[0][0]}），"
                f"应写成 [{CITATION_PREFIX}#]",
                {"examples": [r for r, _ in nonstd_valid[:self.max_examples]]},
                auto_fixable=True))
        if nonstd_invalid:
            vio.append(Violation(
                "citation.invalid_number", "high",
                f"不规范写法且编号越界 {len(nonstd_invalid)} 处（如 {nonstd_invalid[0][0]}）",
                {"examples": [r for r, _ in nonstd_invalid[:self.max_examples]]},
                auto_fixable=True))

        # ---- 事实句覆盖：只查「核心答案」「证据总结」两节 ----
        need, cited, uncited = 0, 0, []
        for name in (REQUIRED_SECTIONS[0], REQUIRED_SECTIONS[1]):
            for s in split_sentences(sections.get(name, "")):
                if not self._needs_citation(s):
                    continue
                need += 1
                if _MARKER_STD.search(s) or _MARKER_LOOSE.search(s):
                    cited += 1
                else:
                    uncited.append(s[:80])
        coverage = (cited / need) if need else None
        if need and coverage < self.min_citation_coverage:
            vio.append(Violation(
                "citation.missing", "medium",
                f"{need - cited}/{need} 个事实句没有出处编号（覆盖率 {coverage:.2f} < "
                f"{self.min_citation_coverage:.2f}）",
                {"uncited_examples": uncited[:self.max_examples], "coverage": coverage},
                auto_fixable=False))

        total = len(used_std) + len(nonstd)
        accuracy = ((len(used_std) - len(invalid) + len(nonstd_valid)) / total) if total else None
        return ({"used": sorted(set(used_std), key=lambda s: int(s[1:])),
                 "available": sorted(valid, key=lambda s: int(s[1:])),
                 "invalid": invalid,
                 "nonstandard": [r for r, _ in nonstd][:self.max_examples],
                 "n_markers": total, "n_invalid": len(invalid) + len(nonstd_invalid),
                 "accuracy": accuracy,
                 "coverage": coverage, "sentences_needing_citation": need,
                 "uncited_examples": uncited[:self.max_examples]}, vio)

    @staticmethod
    def _needs_citation(sentence: str) -> bool:
        """这句话是不是"必须挂出处"的事实句。

        豁免的四类：拒答与"证据不足"类表述、纯列表标签（"研究类型："）、系统行（引用块）、
        过短的片段。**这是启发式**：它分不出"转折句"和"断言句"，所以覆盖率只作参考，
        阈值也留得宽（默认 0.8）。
        """
        s = sentence.strip()
        if len(s) < 12 or _SYSTEM_LINE.match(s):
            return False
        if REFUSAL_PHRASE in s:
            return False
        if re.search(r"(证据不足|未涉及|未提及|未报告|无法回答|没有.{0,6}证据|文献未|不在.{0,8}文献)", s):
            return False
        if re.fullmatch(r"[-*\d.、\s]*[\u4e00-\u9fffA-Za-z]{0,10}[：:]", s):
            return False
        return True

    # ------------------------------------------------------------------ 章节
    def check_structure(self, answer: str,
                        sections: Dict[str, str]) -> Tuple[Dict[str, Any], List[Violation]]:
        vio: List[Violation] = []
        raw_heads = found_headings(answer)
        missing = [s for s in self.required_sections if s not in sections]
        # 标题走样：写成「结论」「1. 核心答案」「**核心答案**」——归一后 sections 里已经有了，
        # 但交付的答案里标题不对，仍要报（且可确定性改回来）
        synonym_used = [(raw, canon) for raw, canon in normalize_headings(answer)[1]
                        if canon in self.required_sections]
        if missing:
            vio.append(Violation(
                "structure.missing_section", "high",
                f"缺少必需章节标题：{'、'.join('## ' + m for m in missing)}",
                {"missing": missing, "found": raw_heads},
                auto_fixable=False))
        if synonym_used:
            vio.append(Violation(
                "structure.wrong_section_name", "medium",
                "章节标题写法不对：" + "、".join(f"「{a}」应为「## {b}」" for a, b in synonym_used),
                {"pairs": synonym_used}, auto_fixable=True))
        missing_opt = [s for s in self.optional_sections if s not in sections]
        if missing_opt:
            vio.append(Violation(
                "structure.missing_optional", "low",
                f"建议补上章节：{'、'.join(missing_opt)}", {"missing": missing_opt}))
        return ({"required": self.required_sections, "found": raw_heads,
                 "missing": missing, "missing_optional": missing_opt,
                 "synonym_used": synonym_used,
                 "ok": not missing}, vio)

    # ------------------------------------------------------------------ 术语
    def check_terminology(self, answer: str,
                          sections: Dict[str, str]) -> Tuple[Dict[str, Any], List[Violation]]:
        """缩写首次出现是否给出全称。

        检查范围排除「参考文献」一节——文献标题里的缩写是别人写的，要求答案去展开它没有道理。
        """
        body = answer or ""
        ref = sections.get(REQUIRED_SECTIONS[2], "")
        if ref:
            body = body.replace(ref, " ")
        body = strip_markers(body)

        checked, missing = [], []
        for abbr, (en, zh, aliases) in self.glossary.items():
            m = re.search(rf"(?<![A-Za-z0-9\-]){re.escape(abbr)}(?![A-Za-z0-9\-])", body)
            if not m:
                continue
            lo = max(0, m.start() - self.expansion_window)
            hi = min(len(body), m.end() + self.expansion_window)
            win = body[lo:hi]
            expanded = (_en_norm(en) in _en_norm(win)
                        or zh in win
                        or any(a and a in win for a in aliases))
            checked.append({"abbr": abbr, "expanded": expanded, "pos": m.start()})
            if not expanded:
                missing.append({"abbr": abbr, "expected_zh": zh, "expected_en": en,
                                "context": win[:70].replace("\n", " ")})
        vio: List[Violation] = []
        if missing:
            vio.append(Violation(
                "terminology.missing_expansion", "medium",
                "缩写首次出现未给全称：" + "、".join(
                    f"{x['abbr']}（应为「{x['expected_zh']}」或 {x['expected_en']}）"
                    for x in missing[:self.max_examples]),
                {"missing": missing}, auto_fixable=True))

        # 正则兜底：表外的疑似缩写，只提示不判违规
        known = set(self.glossary) | NO_EXPANSION_NEEDED
        unknown: List[str] = []
        for m in _MAYBE_ABBREV.finditer(body):
            w = m.group(0)
            if w in known or _ROMAN.match(w) or w.upper() in known:
                continue
            if len(w) < 2 or w.lower() in ("ph", "mg", "ml"):
                continue
            unknown.append(w)
        unknown = sorted(set(unknown))
        if unknown:
            vio.append(Violation(
                "terminology.unknown_abbrev", "low",
                f"检出术语表之外的疑似缩写 {len(unknown)} 个（可能是基因名/试验名，仅提示）："
                + "、".join(unknown[:self.max_examples]),
                {"unknown": unknown[:20]}))
        return ({"checked": checked, "missing_expansion": missing,
                 "unknown_abbrev": unknown[:20],
                 "n_checked": len(checked), "n_missing": len(missing),
                 "ok": not missing}, vio)

    # ---------------------------------------------------------------- 参考文献
    def check_references(self, sections: Dict[str, str], citations: List[Dict[str, Any]],
                         used_markers: Sequence[str]) -> Tuple[Dict[str, Any], List[Violation]]:
        """参考文献是否完整（标题/期刊/年份），有没有被改写或凭空多出条目。

        **区分两种"不完整"**：语料元数据本来就缺（`cause="metadata"`，不是模型的错，
        但仍要报出来，因为交付的答案确实不完整）与模型自己删改（`cause="model"`）。
        """
        vio: List[Violation] = []
        body = sections.get(REQUIRED_SECTIONS[2], "")
        by_marker = {c["marker"]: c for c in (citations or [])}
        entries: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        if not body.strip():
            vio.append(Violation("reference.missing_section", "high",
                                 "没有参考文献内容", {}, auto_fixable=True))
            return ({"entries": [], "n_entries": 0, "n_complete": 0, "incomplete": [],
                     "fabricated": [], "altered": [], "missing_entry": sorted(set(used_markers)),
                     "ok": False}, vio)

        # 引用块（`> ⚠ 免责声明` / `> ⚠ 校验提示`）由流水线附加在参考文献之后，
        # 属于系统行不是文献条目——不排掉的话每份答案都会凭空多出一条"缺编号的参考文献"。
        lines_in = [l.strip() for l in body.splitlines()
                    if l.strip() and not _SYSTEM_LINE.match(l) and l.strip() not in ("（无）", "(无)")]
        for line in lines_in:
            m = _MARKER_STD.match(line) or _MARKER_LOOSE.match(line)
            marker = (f"{CITATION_PREFIX}{m.group(1) if m.re is _MARKER_STD else _loose_num(m)}"
                      if m else None)
            rest = line[m.end():].strip(" .·-—") if m else line
            src = by_marker.get(marker or "")
            has_year = bool(re.search(r"(?<![\d])(1[89]|20)\d{2}(?![\d])", line))
            # 期刊：优先与权威元数据比对；没有元数据时退到"年份前面有一段非纯数字文字"
            if src:
                jr = (src.get("journal") or "").strip()
                has_journal = bool(jr) and _en_norm(jr) in _en_norm(line)
                ti = (src.get("title") or "").strip()
                has_title = bool(ti) and _en_norm(ti[:40]) in _en_norm(line)
                meta_missing = [k for k, v in (("title", ti), ("journal", jr),
                                               ("year", src.get("pub_year"))) if not v]
            else:
                has_journal = bool(re.search(r"[A-Za-z\u4e00-\u9fff][^()]{3,}\((?:1[89]|20)\d{2}\)", line)) \
                    or bool(re.search(r"[A-Za-z]{4,}[^.]*\.\s*(?:1[89]|20)\d{2}", line))
                has_title = len(re.sub(r"[^\wA-Za-z\u4e00-\u9fff]", "", rest)) >= 12
                meta_missing = []
            entry = {"marker": marker, "line": line[:160], "has_title": has_title,
                     "has_journal": has_journal, "has_year": has_year,
                     "known": bool(src), "metadata_missing": meta_missing}
            entries.append(entry)
            if marker:
                seen.add(marker)

        fabricated = [e for e in entries if e["marker"] and not e["known"]] if citations else []
        unmarked = [e for e in entries if not e["marker"]]
        incomplete = [e for e in entries
                      if e["known"] and not (e["has_title"] and e["has_journal"] and e["has_year"])]
        # 元数据本来就缺 vs 模型改写：前者 metadata_missing 非空
        altered = [e for e in incomplete if not e["metadata_missing"]]
        meta_gap = [e for e in incomplete if e["metadata_missing"]]
        # 只报"本来就存在、却没列进文献表"的编号。越界编号（[S9] 而证据只到 S6）不在这里报——
        # 它已经被 citation.invalid_number 抓为幻觉，再报一条"文献表缺 S9"是噪声，
        # 而且会让一处错误同时污染幻觉率与格式合规率两个指标。
        missing_entry = sorted({m for m in used_markers if m not in seen and m in by_marker},
                               key=lambda s: int(s[1:]))

        if fabricated:
            vio.append(Violation(
                "reference.fabricated", "high",
                f"参考文献里有 {len(fabricated)} 条不在系统给定列表中的条目（疑似编造）",
                {"lines": [e["line"] for e in fabricated[:self.max_examples]]},
                auto_fixable=True))
        if altered:
            vio.append(Violation(
                "reference.altered", "high",
                f"{len(altered)} 条参考文献与系统给定的原文对不上（标题/期刊/年份被改写或删减）",
                {"lines": [e["line"] for e in altered[:self.max_examples]]},
                auto_fixable=True))
        if meta_gap:
            vio.append(Violation(
                "reference.incomplete", "medium",
                f"{len(meta_gap)} 条参考文献缺字段（标题/期刊/年份），"
                f"来源是语料元数据本身就缺，不是模型删的",
                {"lines": [e["line"] for e in meta_gap[:self.max_examples]],
                 "missing_fields": [e["metadata_missing"] for e in meta_gap[:self.max_examples]]},
                auto_fixable=False))
        if unmarked:
            vio.append(Violation(
                "reference.incomplete", "medium",
                f"{len(unmarked)} 行参考文献没有出处编号，无法与正文对应",
                {"lines": [e["line"] for e in unmarked[:self.max_examples]]}, auto_fixable=True))
        if missing_entry:
            vio.append(Violation(
                "reference.missing_entry", "medium",
                f"正文引用了 {'、'.join(missing_entry)} 但参考文献里没有对应条目",
                {"markers": missing_entry}, auto_fixable=True))

        n_complete = sum(1 for e in entries
                         if e["has_title"] and e["has_journal"] and e["has_year"])
        return ({"entries": entries, "n_entries": len(entries), "n_complete": n_complete,
                 "incomplete": [e["line"] for e in incomplete[:self.max_examples]],
                 "fabricated": [e["line"] for e in fabricated[:self.max_examples]],
                 "altered": [e["line"] for e in altered[:self.max_examples]],
                 "metadata_gap": len(meta_gap),
                 "missing_entry": missing_entry,
                 "ok": not (fabricated or altered or missing_entry or unmarked)}, vio)

    # ------------------------------------------------------------------ 数字
    def check_numeric(self, answer: str, sections: Dict[str, str],
                      grounding_text: str) -> Tuple[Dict[str, Any], List[Violation]]:
        """答案里的数字能不能在证据（+ 元数据 + 问题）里找到。

        比对方式是**两边用同一组正则抽、抽完归一后比集合**，不是拿数字去证据里做子串搜索——
        后者会把 "12" 匹配到 "3126" 上，假阴性一大片。
        """
        vio: List[Violation] = []
        body = answer or ""
        ref = sections.get(REQUIRED_SECTIONS[2], "")
        if ref:
            body = body.replace(ref, " ")          # 参考文献里的年份不是答案的断言
        body = strip_markers(body)
        body = re.sub(r"^\s*\d+[.)、]\s", " ", body, flags=re.MULTILINE)   # 列表序号不是数据

        facts = extract_numeric_facts(body)
        ground = {(f["kind"], f["norm"]) for f in extract_numeric_facts(grounding_text or "")}
        # 数值也可能以别的类别出现在证据里（答案写 "n=120"，证据写 "120 patients"），
        # 所以再补一层"纯数值集合"作为宽松兜底，避免把同一个数判成编造。
        loose = {re.sub(r"[^\d.]", "", n) for _, n in ground if re.search(r"\d", n)}
        ungrounded = []
        for f in facts:
            if (f["kind"], f["norm"]) in ground:
                continue
            bare = re.sub(r"[^\d.]", "", f["norm"])
            if bare and bare in loose:
                continue
            ungrounded.append(f)
        if ungrounded:
            vio.append(Violation(
                "numeric.ungrounded", "medium",
                f"{len(ungrounded)} 个数字在证据中找不到出处："
                + "、".join(f"{x['raw']}" for x in ungrounded[:self.max_examples]),
                {"items": ungrounded[:self.max_examples * 2],
                 "note": "四舍五入/单位换算也会落在这里，需人工复核"},
                auto_fixable=False))
        n = len(facts)
        return ({"n_facts": n, "n_grounded": n - len(ungrounded),
                 "ungrounded": ungrounded[:self.max_examples * 2],
                 "grounded_ratio": (n - len(ungrounded)) / n if n else None,
                 "ok": not ungrounded}, vio)

    # ------------------------------------------------------------------ 拒答
    @staticmethod
    def _question_years(question: str) -> List[int]:
        """问题里明确写出的年份（只认 19xx/20xx 四位数，避免把 1795 名患者当年份）。"""
        return [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", question or "")
                if 1900 <= int(y) <= 2100]

    @staticmethod
    def _evidence_max_year(citations: Optional[Sequence[Dict[str, Any]]]) -> Optional[int]:
        ys = []
        for c in citations or []:
            y = c.get("pub_year")
            try:
                ys.append(int(y))
            except (TypeError, ValueError):
                continue
        return max(ys) if ys else None

    def check_refusal(self, answer: str, sections: Dict[str, str],
                      expect_refusal: Optional[bool],
                      question: str = "",
                      citations: Optional[Sequence[Dict[str, Any]]] = None
                      ) -> Tuple[Dict[str, Any], List[Violation]]:
        """拒答判定，出三态 `state`：完全拒答 / 部分作答 / 实质作答。

        **作用域（这一段是本方法最容易写错的地方，实测踩过两次）**：

        | 判什么 | 扫哪一节 | 为什么 |
        |---|---|---|
        | `state` 三态 | **仅核心答案** | 证据总结里有带出处的句子是**合规行为**——强约束要求拒答时也要列证据。拿它判"部分作答"，实测 9 道应拒题里 8 道被误标 |
        | `conflict` 两侧 | **仅核心答案** | 同上；且拒答短语若出现在「证据强度与局限」里不该算冲突 |
        | `asserted_in_evidence` | 证据总结 | 只**报出来**不参与三态。它回答的是另一个问题："系统到底有没有交付内容"，误拒判定会用到 |

        `detected` / `partial_marked` 仍按**全文**判，因为它们回答的是"这份答案里出现过
        这个短语吗"，是给统计与调试看的原始信号，不直接决定三态。
        """
        vio: List[Violation] = []
        text = answer or ""
        detected = REFUSAL_PHRASE in text
        partial_marked = PARTIAL_PHRASE in text
        # 近义拒答：模型自己发挥的说法。检出它是为了区分"根本没拒答"与"拒答了但没用规定短语"。
        # ⚠ 必须同时排除 partial_marked：部分作答的答案里「文献未涉及的部分：……」是**规定写法**，
        #   不排除的话每一条合规的部分作答都会被记成"近义表述"（实测 4/4 全中）。
        near = (bool(re.search(r"(无法回答|不能回答|没有.{0,8}(相关|直接).{0,4}证据|"
                               r"文献.{0,6}(未|没有).{0,6}(提及|涉及|报告))", text))
                and not detected and not partial_marked)
        core = sections.get(REQUIRED_SECTIONS[0], "")
        first_sentence = (split_sentences(core) or [""])[0]

        # 带出处的实质结论有几条。**两节分开数**，因为它们回答的是两个不同的问题：
        #   核心答案里的 → 这一节自己就自相矛盾（首句说没答案，紧接着给结论）→ 判 conflict、可修
        #   证据总结里的 → 系统其实答了，只是把答案放在了下面 → 决定三态
        # ⚠ 只数核心答案会漏掉最典型的一类：阶段九 adv-E1 的核心答案只有拒答短语，
        #   三条带出处的检测方法全在证据总结里，于是被判成完全拒答、记了一次误拒。
        def _asserted(sec: str) -> List[str]:
            return [s for s in split_sentences(sec)
                    if REFUSAL_PHRASE not in s and PARTIAL_PHRASE not in s
                    and _MARKER_STD.search(s) and self._needs_citation(s)]

        asserted = _asserted(core)
        evidence_sec = sections.get(REQUIRED_SECTIONS[1], "")
        asserted_ev = _asserted(evidence_sec)

        # ---- 三态：完全拒答 / 部分作答 / 实质作答 ----------------------------
        # 判定只看两件可程序化的事：短语在不在、**核心答案里**带出处的实质句有没有。
        # ⚠ 阈值是 **1 不是 2**：实测撞到过「一条带出处的结论 + 一句拒答」的真答案，
        #    按 >=2 判会放行，而它恰恰是最误导人的形态（首句说没答案，正文其实答了）。
        # ⚠ **不能把证据总结算进来**（试过，8/9 道应拒题被误判成部分作答）：强约束本来就
        #    要求证据总结带出处地列证据，**拒答时也要列**。adv-A1 问「2026 年获批的药」，
        #    证据总结里那三条是 MS 背景；adv-D1 的证据总结逐条解释「这篇给不出你要的信息」。
        #    「证据总结里有带出处的句子」因此不是「答了一部分」的信号。
        #    证据总结里的条数仍单独报成 `asserted_in_evidence`，误拒判定会用到它。
        if partial_marked or (detected and asserted):
            state = "partial"
        elif detected:
            state = "full_refusal"
        else:
            state = "substantive"

        if expect_refusal and state == "substantive":
            vio.append(Violation(
                "refusal.missing", "high",
                f"该问题超出证据范围，应原样写出「{REFUSAL_PHRASE}」，"
                + ("实际给出了近义表述但不是规定短语" if near else "实际给出了实质性回答"),
                {"near_miss": near, "core_head": core[:120]}, auto_fixable=False))
        # ⚠ 过度拒答只算**完全拒答**。半可答的题上「部分作答」本来就是最优解，
        #    把它也记成误拒，测出来的是口径与提示词的矛盾（阶段九 adv-E1 那条 FAIL 的根因）。
        if expect_refusal is False and state == "full_refusal":
            vio.append(Violation(
                "refusal.over_refusal", "medium",
                "证据本可支持作答，却完全拒答",
                {"core_head": core[:120]}, auto_fixable=False))

        # 拒答短语与实质结论并存 —— 自相矛盾，且可以纯字符串修掉。
        # ⚠ **两侧都必须限定在「核心答案」一节之内**：
        #   · asserted 只数核心答案 —— 证据总结里有引用是**合规行为**（强约束要求拒答时
        #     也要带出处地列证据），拿它判冲突会让每一次合法的完全拒答都触发（实测 9/9）。
        #   · detected 也要按核心答案判 —— 否则核心答案正常作答、而「证据强度与局限」里
        #     写了一句「……因此无法回答此问题」，会被误判成冲突。
        conflict = bool(REFUSAL_PHRASE in core and asserted and not partial_marked)
        if conflict:
            vio.append(Violation(
                "refusal.conflict", "medium",
                f"既写了「{REFUSAL_PHRASE}」，又在核心答案里给出 {len(asserted)} 条带出处的结论；"
                f"应改用「{PARTIAL_PHRASE}」",
                {"asserted": asserted[:2]}, auto_fixable=True))
        # ---- 时间越界：问题问的年份晚于**本次证据**的最新年份 ----------------
        # 这一条不需要理解语义，纯元数据比较，所以能确定性地判。它专治这一类：
        # 问「2026 年最新获批的 MS 药物」，证据最新到 2024，模型却写「仅能部分回答」，
        # 然后把「MS 是自身免疫病」「1993 年以来有 15 种 DMT」当成"可以回答的部分"——
        # 那两条与所问年份毫无关系，是**空洞的部分作答**。
        # ⚠ 界限用**本次证据的最大年份**，不是语料的（corpus_meta 的 pub_year.max 是 2026，
        #   拿它当界限这一类根本判不出来）。
        q_years = self._question_years(question)
        ev_max = self._evidence_max_year(citations)
        beyond = bool(q_years and ev_max and max(q_years) > ev_max)
        if beyond and state != "full_refusal":
            vio.append(Violation(
                "refusal.beyond_evidence_year", "high",
                f"问题问到 {max(q_years)} 年，而本次证据最新只到 {ev_max} 年——"
                f"这种情况没有「可以回答的部分」，应当完全拒答",
                {"question_years": q_years, "evidence_max_year": ev_max, "state": state},
                auto_fixable=False))

        # 用了部分作答短语，却没给「可以回答的部分：」小标题 —— 只提示，不做确定性修正
        if partial_marked and PARTIAL_ANSWERABLE_HEAD not in text:
            vio.append(Violation(
                "refusal.partial_unmarked", "low",
                f"写了「{PARTIAL_PHRASE}」但缺「{PARTIAL_ANSWERABLE_HEAD}」小标题",
                {"core_head": core[:120]}))

        return ({"detected": detected, "partial_marked": partial_marked, "state": state,
                 "asserted_count": len(asserted), "asserted_in_evidence": len(asserted_ev),
                 "near_miss_phrase": near,
                 "conflict": conflict, "expected": expect_refusal,
                 "first_sentence": first_sentence[:120],
                 "beyond_evidence_year": beyond,
                 "question_years": q_years, "evidence_max_year": ev_max,
                 # 应拒题上「部分作答」同样算守住了边界：半可答的题里它才是最优解。
                 # 但时间越界那一类除外——那种题不存在"可回答的一半"。
                 "ok": ((not expect_refusal) or state in ("full_refusal", "partial"))
                       and not (beyond and state != "full_refusal")}, vio)

    # ------------------------------------------------------------------ 主入口
    def check(self, answer: str,
              citations: Optional[Sequence[Dict[str, Any]]] = None,
              evidence_text: str = "",
              question: str = "",
              expect_refusal: Optional[bool] = None,
              reference_list: str = "") -> Dict[str, Any]:
        """跑完六组检查。

        Args:
            citations:      权威出处列表（`ctx["metadata"]["citations"]`）。没有它时
                            编号有效性无从判定，`citation.invalid_number` 一项会跳过。
            evidence_text:  证据正文（数字溯源用）
            question:       原问题。**要算进溯源语料**——答案复述问题里的数字（"2025 年之后"）
                            不是编造。
            expect_refusal: True=这题应当拒答；False=这题应当能答；None=不判定。
        """
        answer = answer or ""
        secs = section_map(answer)
        cits = list(citations or [])
        valid = {c["marker"] for c in cits}
        vios: List[Violation] = []

        cit_rep, v = self.check_citations(answer, valid, secs)
        vios += v if cits else [x for x in v if x.code != "citation.invalid_number"]
        st_rep, v = self.check_structure(answer, secs)
        vios += v
        tm_rep, v = self.check_terminology(answer, secs)
        vios += v
        rf_rep, v = self.check_references(secs, cits, cit_rep["used"]) if cits or secs.get(
            REQUIRED_SECTIONS[2]) else ({"skipped": True, "ok": True}, [])
        vios += v
        if self.enable_numeric and (evidence_text or reference_list):
            ground = "\n".join([evidence_text or "", reference_list or "",
                                question or "",
                                " ".join(str(c.get(k) or "") for c in cits
                                         for k in ("title", "journal", "pub_year", "pmcid", "pmid"))])
            nm_rep, v = self.check_numeric(answer, secs, ground)
            vios += v
        else:
            nm_rep = {"skipped": True, "ok": True, "n_facts": 0, "n_grounded": 0}
        rr_rep, v = self.check_refusal(answer, secs, expect_refusal,
                                       question=question, citations=citations)
        vios += v

        vios.sort(key=lambda x: (SEVERITY_ORDER.get(x.severity, 9), x.code))
        blocking = [x for x in vios if x.severity in ("high", "medium")]
        fmt_vio = [x for x in blocking if x.code.startswith(FORMAT_CODES)]
        hall_vio = [x for x in blocking if x.code.startswith(HALLUCINATION_CODES)]

        return {
            "compliant": not blocking,
            "format_compliant": not fmt_vio,
            "hallucination_free": not hall_vio,
            "n_violations": {"high": sum(1 for x in vios if x.severity == "high"),
                             "medium": sum(1 for x in vios if x.severity == "medium"),
                             "low": sum(1 for x in vios if x.severity == "low")},
            "violations": [x.to_dict() for x in vios],
            "citation": cit_rep, "structure": st_rep, "terminology": tm_rep,
            "reference": rf_rep, "numeric": nm_rep, "refusal": rr_rep,
            "scores": {
                "citation_accuracy": cit_rep["accuracy"],
                "citation_coverage": cit_rep["coverage"],
                "structure_ok": st_rep["ok"],
                "terminology_ok": tm_rep["ok"],
                "reference_complete_ratio": (rf_rep.get("n_complete", 0) / rf_rep["n_entries"]
                                             if rf_rep.get("n_entries") else None),
                "numeric_grounded_ratio": nm_rep.get("grounded_ratio"),
                "refusal_ok": rr_rep["ok"],
                # 三态放进 scores，调用方不必自己按短语再判一次（服务层 refused 就读它）
                "refusal_state": rr_rep["state"],
            },
            "sections_found": list(secs),
        }

    # ------------------------------------------------------------------ 修正
    def repair_prompt(self, report: Dict[str, Any], max_items: int = 12) -> str:
        """把违规清单写成回灌给模型的指令。

        **只回灌 high/medium**：low 项（如未知缩写提示）本来就可能是误报，让模型去"修"
        它们只会诱发无谓改写，反而增加新违规的机会。
        """
        items = [v for v in report["violations"] if v["severity"] in ("high", "medium")]
        if not items:
            return ""
        lines = []
        for i, v in enumerate(items[:max_items], 1):
            lines.append(f"{i}. [{v['severity']}] {v['message']}")
            d = v.get("detail") or {}
            if v["code"] == "citation.invalid_number":
                lines.append(f"   → 删掉这些编号（或改挂真实支持该句的编号）；"
                             f"可用编号只有：{'、'.join(d.get('available', []))}")
            elif v["code"] == "citation.missing":
                for s in (d.get("uncited_examples") or [])[:3]:
                    lines.append(f"   → 这句缺出处：{s}")
            elif v["code"] == "structure.missing_section":
                lines.append(f"   → 补上标题（一字不改）：{'、'.join('## ' + m for m in d.get('missing', []))}")
            elif v["code"] == "terminology.missing_expansion":
                for x in (d.get("missing") or [])[:4]:
                    lines.append(f"   → {x['abbr']} 首次出现处改成：{x['expected_zh']}（{x['abbr']}）")
            elif v["code"] in ("reference.fabricated", "reference.altered"):
                lines.append("   → 参考文献一节整段替换成系统给出的列表原文，不要改一个字")
            elif v["code"] == "numeric.ungrounded":
                for x in (d.get("items") or [])[:4]:
                    lines.append(f"   → 证据里查不到这个数：{x['raw']}（删掉它，或改写成证据里的原值）")
            elif v["code"] == "refusal.missing":
                lines.append(f"   → 「核心答案」第一句改成：{REFUSAL_PHRASE}。并说明缺什么")
        return "\n".join(lines)

    def auto_fix(self, answer: str, report: Dict[str, Any],
                 citations: Optional[Sequence[Dict[str, Any]]] = None,
                 reference_list: str = "") -> Tuple[str, List[str]]:
        """确定性修正：只做**不需要理解内容**的那几件事，不调模型。

        做五件：① 不规范编号写法改写回 [S#] ② 删除越界编号 ③ 同义章节标题改回标准名
        ④ 参考文献一节被改写/编造时，整段换回系统给定的列表
        ⑤ 拒答短语与带出处的实质结论并存时，把拒答短语换成部分作答短语。

        ⑤ 只换短语、**不插「可以回答的部分：」小标题**：换短语是纯逻辑推论（写了拒答又给出
        带出处的结论，按定义就是部分作答），插小标题却要判断哪几句属于"能答的部分"——
        那需要理解内容。所以后半留给模型修正段，由 `refusal.partial_unmarked` 报出来。

        刻意**不做**的：补写缺失的「核心答案」正文、删掉未溯源的数字——这两件都需要理解
        句子，交给模型修正段（或让它保留，由报告如实记下）。
        """
        applied: List[str] = []
        text = answer or ""
        valid = {c["marker"] for c in (citations or [])}

        # ① 不规范写法 → 标准写法（只改编号有效的）
        if valid:
            def _fix_loose(m: re.Match) -> str:
                mk = f"{CITATION_PREFIX}{_loose_num(m)}"
                return f"[{mk}]" if mk in valid else ""
            new = _MARKER_LOOSE.sub(_fix_loose, text)
            if new != text:
                applied.append("统一出处编号写法 → [S#]（无效编号直接删除）")
                text = new
            # ② 越界的标准写法编号：删掉
            def _drop_invalid(m: re.Match) -> str:
                mk = f"{CITATION_PREFIX}{m.group(1)}"
                return m.group(0) if mk in valid else ""
            new = _MARKER_STD.sub(_drop_invalid, text)
            if new != text:
                applied.append("删除证据范围外的出处编号")
                text = new

        # ③ 走样的标题 → `## 标准名`（与 section_map 用同一个归一函数，见 normalize_headings）
        new, changed = normalize_headings(text)
        if changed:
            applied.append(f"章节标题改回标准写法（{len(changed)} 处）")
            text = new

        # ④ 参考文献被改写/编造 → 换回系统列表
        codes = {v["code"] for v in report.get("violations", [])}
        if reference_list and (codes & {"reference.fabricated", "reference.altered",
                                        "reference.missing_section", "reference.missing_entry"}):
            secs = section_map(text)
            head = f"## {REQUIRED_SECTIONS[2]}"
            if REQUIRED_SECTIONS[2] in secs and text.find(head) >= 0:
                idx = text.find(head)
                nxt = _HEADING.search(text, idx + len(head))
                end = nxt.start() if nxt else len(text)
                # 参考文献是最后一节时，它后面往往还挂着免责声明/校验提示这类引用块。
                # 整段替换若连它们一起扔掉，等于把免责声明删了——留下来重新接上。
                keep = [l for l in text[idx + len(head):end].splitlines()
                        if _SYSTEM_LINE.match(l.strip())]
                tail = ("\n\n" + "\n".join(keep) if keep else "")
                text = (text[:idx] + head + "\n" + reference_list.strip() + tail
                        + "\n\n" + text[end:])
            else:
                text = text.rstrip() + "\n\n" + head + "\n" + reference_list.strip()
            applied.append("参考文献一节整段换回系统给定列表")

        # ⑤ 拒答短语 + 带出处的实质结论并存 → 换成部分作答短语（见方法 docstring）
        if "refusal.conflict" in codes and REFUSAL_PHRASE in text:
            text = text.replace(REFUSAL_PHRASE, PARTIAL_PHRASE, 1)
            applied.append(f"「{REFUSAL_PHRASE}」→「{PARTIAL_PHRASE}」（正文有带出处的结论）")

        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text, applied


# ============================================================================
# 七、批量与汇总
# ============================================================================
def aggregate(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把一批报告汇总成任务书要的三个率 + 分项。

    **口径写在这里，报告里必须照抄**：
      幻觉率     = 至少命中一条 high/medium 幻觉类违规的用例数 / 用例数
                  （幻觉类 = 编号越界 / 参考文献编造或改写 / 数字无出处 / 该拒未拒）
      引用准确率 = Σ 有效编号 / Σ 全部编号（micro，跨用例合并，不是各用例平均）
      格式合规率 = 无 high/medium 格式类违规的用例数 / 用例数

    拒答**拆成三态**（2026-08-11）：完全拒答 / 部分作答 / 实质作答。
      full_refusal_rate  = 首句拒答且核心答案里没有带出处的实质句
      partial_rate       = 写了部分作答短语，或写了拒答短语但正文给了带出处的结论
      substantive_rate   = 两句都没写
    **不要再报一个合并的"拒答率"**：把部分作答并进拒答，会让「正文其实答了」的用例
    在统计里消失，而那正是最误导读者的一类。
    """
    n = len(reports)
    if not n:
        return {"n": 0}
    tot_mk = sum(r["citation"].get("n_markers", 0) or 0 for r in reports)
    bad_mk = sum(r["citation"].get("n_invalid", 0) or 0 for r in reports)
    hall = [r for r in reports if not r["hallucination_free"]]
    fmt_ok = [r for r in reports if r["format_compliant"]]
    all_ok = [r for r in reports if r["compliant"]]

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    codes: Dict[str, int] = {}
    for r in reports:
        for v in r["violations"]:
            if v["severity"] in ("high", "medium"):
                codes[v["code"]] = codes.get(v["code"], 0) + 1
    return {
        "n": n,
        "hallucination_rate": round(len(hall) / n, 4),
        "citation_accuracy": round((tot_mk - bad_mk) / tot_mk, 4) if tot_mk else None,
        "format_compliance_rate": round(len(fmt_ok) / n, 4),
        "full_compliance_rate": round(len(all_ok) / n, 4),
        "citation_coverage_mean": _mean([r["citation"].get("coverage") for r in reports]),
        "numeric_grounded_mean": _mean([r["numeric"].get("grounded_ratio") for r in reports]),
        "reference_complete_mean": _mean([r["scores"].get("reference_complete_ratio")
                                          for r in reports]),
        "structure_ok_rate": round(sum(1 for r in reports if r["structure"]["ok"]) / n, 4),
        "terminology_ok_rate": round(sum(1 for r in reports if r["terminology"]["ok"]) / n, 4),
        # 拒答三态（口径见 docstring）。老报告没有 state 字段时按实质作答计。
        "full_refusal_rate": round(sum(1 for r in reports
                                       if r["refusal"].get("state") == "full_refusal") / n, 4),
        "partial_rate": round(sum(1 for r in reports
                                  if r["refusal"].get("state") == "partial") / n, 4),
        "substantive_rate": round(sum(1 for r in reports
                                      if r["refusal"].get("state", "substantive")
                                      == "substantive") / n, 4),
        "markers_total": tot_mk, "markers_invalid": bad_mk,
        "violation_counts": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
    }


def format_report(rep: Dict[str, Any], indent: str = "  ") -> str:
    """人读版。"""
    s = rep["scores"]
    lines = [
        f"{indent}合规：{'是' if rep['compliant'] else '否'}"
        f"（格式 {'✓' if rep['format_compliant'] else '✗'} / "
        f"幻觉 {'✓' if rep['hallucination_free'] else '✗'}）  "
        f"违规 high {rep['n_violations']['high']} / medium {rep['n_violations']['medium']} / "
        f"low {rep['n_violations']['low']}",
        f"{indent}引用：用 {len(rep['citation']['used'])} 个编号，无效 {rep['citation']['n_invalid']}，"
        f"准确率 {s['citation_accuracy']}，事实句覆盖 {s['citation_coverage']}",
        f"{indent}章节：{'齐全' if rep['structure']['ok'] else '缺 ' + '、'.join(rep['structure']['missing'])}"
        f"　术语：查 {rep['terminology']['n_checked']} 个缩写，缺全称 {rep['terminology']['n_missing']}",
        f"{indent}参考文献：{rep['reference'].get('n_entries', 0)} 条，"
        f"完整 {rep['reference'].get('n_complete', 0)}，"
        f"编造 {len(rep['reference'].get('fabricated', []))}，改写 {len(rep['reference'].get('altered', []))}",
        f"{indent}数字：{rep['numeric'].get('n_grounded', 0)}/{rep['numeric'].get('n_facts', 0)} 可溯源",
        f"{indent}拒答：{'检出' if rep['refusal']['detected'] else '未检出'}"
        f"（期望 {rep['refusal']['expected']}）",
    ]
    for v in rep["violations"]:
        lines.append(f"{indent}  · [{v['severity']}] {v['code']}：{v['message']}")
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================
_DEMO_GOOD = """## 核心答案
在提供的文献中，随机对照试验（RCT）显示该疗法可降低年复发率（ARR）[S1][S2]。
证据以 2021 年的多中心研究为主 [S2]。

## 证据总结
- 一项 2021 年的 RCT 报告治疗组 ARR 为 0.26，对照组 0.58 [S1]。
- 另一项队列研究纳入 n=1,234 例患者，随访 24 周 [S2]。

## 证据强度与局限
证据以 RCT 与队列研究为主，样本量中等；未覆盖儿童人群。

## 参考文献
[S1] Off-target effects of therapy X. Nucleic Acids Research (2021). PMC7778913
[S2] Cohort study of therapy X. Frontiers in Immunology (2021). PMC8670230
"""

_DEMO_BAD = """## 结论
该疗法可完全治愈该病 [S1][文献2][S9]，有效率达 92.7%，推荐所有患者使用 EDSS 评分随访。

## 证据要点
- 治疗组 PFS 明显延长。
- 一项 III 期试验（NCT01234567）证实了这一点。

## 参考文献
[S1] Off-target effects of therapy X. Nucleic Acids Research (2021). PMC7778913
[S3] Smith J, et al. A landmark trial of therapy X. New England Journal of Medicine (2024).
"""

_DEMO_EVIDENCE = """[S1] Off-target effects of therapy X. In a randomized controlled trial the
annualized relapse rate was 0.26 in the treatment group versus 0.58 in controls (p<0.001).
[S2] A cohort study of 1,234 patients followed for 24 weeks reported similar findings.
"""

_DEMO_CITATIONS = [
    {"marker": "S1", "title": "Off-target effects of therapy X",
     "journal": "Nucleic Acids Research", "pub_year": 2021, "pmcid": "PMC7778913", "pmid": ""},
    {"marker": "S2", "title": "Cohort study of therapy X",
     "journal": "Frontiers in Immunology", "pub_year": 2021, "pmcid": "PMC8670230", "pmid": ""},
]
_DEMO_REFLIST = ("[S1] Off-target effects of therapy X. Nucleic Acids Research (2021). PMC7778913\n"
                 "[S2] Cohort study of therapy X. Frontiers in Immunology (2021). PMC8670230")


def _demo() -> int:
    ck = FormatChecker()
    for name, ans in (("合规样例", _DEMO_GOOD), ("违规样例", _DEMO_BAD)):
        rep = ck.check(ans, citations=_DEMO_CITATIONS, evidence_text=_DEMO_EVIDENCE,
                       question="该疗法的年复发率是多少？", expect_refusal=False,
                       reference_list=_DEMO_REFLIST)
        print("=" * 92)
        print(f"{name}")
        print("=" * 92)
        print(format_report(rep))
        if not rep["compliant"]:
            print("\n--- 回灌给模型的修正指令 ---")
            print(ck.repair_prompt(rep))
            fixed, applied = ck.auto_fix(ans, rep, _DEMO_CITATIONS, _DEMO_REFLIST)
            print(f"\n--- 确定性修正（{len(applied)} 项：{applied}）---")
            print(fixed)
            rep2 = ck.check(fixed, citations=_DEMO_CITATIONS, evidence_text=_DEMO_EVIDENCE,
                            question="该疗法的年复发率是多少？", expect_refusal=False,
                            reference_list=_DEMO_REFLIST)
            print("\n--- 修正后复检 ---")
            print(format_report(rep2))
        print()
    return 0


def _run_jsonl(path: str, limit: int) -> int:
    """体检一批已经生成好的答案（jsonl，每行含 answer / sources 等字段）。"""
    ck = FormatChecker()
    reports, rows = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ans = obj.get("answer") or (obj.get("rag") or {}).get("answer") or ""
            if not ans:
                continue
            cits = obj.get("citations") or [
                {"marker": s.get("marker"), "title": s.get("title"), "journal": s.get("journal"),
                 "pub_year": s.get("pub_year"), "pmcid": s.get("pmcid"), "pmid": s.get("pmid")}
                for s in (obj.get("sources") or [])]
            rep = ck.check(ans, citations=cits, evidence_text=obj.get("evidence_text", ""),
                           question=obj.get("query", ""))
            reports.append(rep)
            rows.append((obj.get("id") or obj.get("query", "")[:40], rep))
    for name, rep in rows:
        print(f"\n■ {name}")
        print(format_report(rep))
    print("\n" + "=" * 92)
    print(json.dumps(aggregate(reports), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--jsonl", default=None, help="体检一批已生成的答案")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--glossary", action="store_true", help="打印术语表")
    args = ap.parse_args()

    if args.glossary:
        print(f"预定义术语表 {len(TERM_GLOSSARY)} 条；无需展开的通用缩写 "
              f"{len(NO_EXPANSION_NEEDED)} 条\n")
        for k, (en, zh, alias) in TERM_GLOSSARY.items():
            print(f"  {k:<10} {zh:<24} {en}" + (f"   别名：{'、'.join(alias)}" if alias else ""))
        return 0
    if args.jsonl:
        return _run_jsonl(args.jsonl, args.limit)
    if args.demo:
        return _demo()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
