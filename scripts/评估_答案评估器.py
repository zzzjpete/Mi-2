# -*- coding: utf-8 -*-
"""第八阶段 · 生成答案的多维度评估器

四个维度，各回答一个不同的问题（**不要把它们的分数加起来看**，量的不是一回事）：

  ① 文本相似性（ROUGE）  —— 答案和参照文本有多少重合？
  ② 关键信息召回          —— 参照里的医学关键信息（剂量/百分比/时间/安全性…），答案覆盖了多少？
  ③ 幻觉信号              —— 有多少「无出处的绝对化表述」？
  ④ 可读性                —— 句子多长、结构如何、语言是否一致？

**参照物（reference）是什么，决定了 ①② 的口径**，这一点必须先说清楚，否则数字会被误读：

  reference_kind="gold"      人工标准答案。任务书里 `recall = overlap / gt_matches` 的本意。
                             本项目**目前没有**这份标注（阶段七就记着这条遗留），
                             脚本支持 `--refs` 传入，一旦有了就能直接用。
  reference_kind="evidence"  该题检索回来的证据原文。这是本项目**现在拿得到**的参照。
                             此时 ①② 量的是「答案对证据的覆盖/贴合程度」，
                             **是忠于证据的代理指标，不是正确性**。
  reference_kind="none"      不给参照，只跑 ③④（这两维本来就不需要参照）。

③④ 与参照无关，任何时候都能算。

⚠ 三条口径提醒（写进报告时请一并带上）：
  · ROUGE 高 ≠ 答得好。它只量词面重合；照抄证据能拿高分，换个说法说对了会被扣分。
  · 幻觉信号是**启发式**，只能说「这句话的措辞像是没有依据」，不能断定内容是假的。
    本模块对**同一句里带 [S#]/PMID/DOI 出处**的信号单独记为 mitigated（有出处），
    正是因为「研究表明[S3]」和光秃秃的「研究表明」完全是两回事。
  · 可读性分是**约定俗成的经验带**（见 READABILITY_BANDS），不是校准过的量表。

用法：
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", r"E:\\rag\\scripts\\评估_答案评估器.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

    e = ev.AnswerEvaluator()
    r = e.evaluate(answer, reference=gold_answer)              # 有标准答案
    r = e.evaluate(answer, evidence=evidence_blob)             # 只有证据原文
    r["rouge"]["rouge-l"]["f"] / r["key_info"]["recall"] / r["hallucination"]["risk_score"]
    r["readability"]["avg_sentence_length"]

CLI：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\评估_答案评估器.py --demo
    ... --jsonl E:\\rag\\report_data\\生成_流水线测试_live.jsonl     # 评一批已跑好的答案
"""
import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# rouge 库：任务书指定用它。装不上时退到本模块自带的等价实现，并在结果里标明用的是哪个。
# 自带实现刻意复刻 rouge 1.0.1 的语义（unique n-gram 集合 + F 值分母加 1e-8），
# 这样两者的数值可以直接比对——`评估_验证.py` 里就是这么逐条对的。
# ---------------------------------------------------------------------------
try:
    from rouge import Rouge as _RougeLib
    ROUGE_BACKEND = "rouge-lib"
except ImportError:                                    # pragma: no cover - 取决于环境
    _RougeLib = None
    ROUGE_BACKEND = "builtin"

ROUGE_METRICS = ("rouge-1", "rouge-2", "rouge-l")
_ZERO = {"r": 0.0, "p": 0.0, "f": 0.0}


# ============================================================================
# 零、分词：中英混排必须分开处理
# ============================================================================
# 本项目的答案是**中英混排**的（提示词要求跟随提问语言，模型没完全照做，见阶段七遗留 2）。
# rouge 库按空白切词——中文整段没有空格，一句话会被切成 1 个"词"，
# 于是任意两句中文的 ROUGE 几乎恒为 0 或 1，完全失去分辨力（`评估_验证.py` 里有实测对照）。
# 通行做法：中文按**字**、英文数字按**词**。下面这条正则就是干这个的。
_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z][A-Za-z\-']*|\d+(?:\.\d+)?|%")


def mixed_tokenize(text: str, lower: bool = True) -> List[str]:
    """中文按字 / 英文数字按词 的混合分词。ROUGE 与可读性都用它。"""
    s = (text or "").lower() if lower else (text or "")
    return _TOKEN.findall(s)


def sentence_tokens(text: str) -> List[List[str]]:
    """按句切分再逐句分词 → [[token,…], …]（空句丢掉）。ROUGE 的输入形态。"""
    out: List[List[str]] = []
    for s in split_sentences(text):
        toks = mixed_tokenize(s)
        if toks:
            out.append(toks)
    return out


def _tokenized_for_rouge(text: str) -> str:
    """喂给 rouge 库的形态：句内用空格分词，**句间用 "." 分隔**。

    两个坑都在这一行里躲掉了（都是实测撞出来的，不是预防性设计）：

    ① rouge 1.0.1 的 `_get_scores` 是 `hyp.split(".")` 切句的，而医学答案满是
       `0.45`、`p<0.001` 这类小数——直接拼接会把 "0.45" 劈成 "0" 和 "45" 两个"句子"。
       所以 token 内部的小数点换成 "·"，"." 只保留做句子分隔符。
    ② 反过来，如果把标点全去掉、整段当成一个句子喂进去，rouge 的 `_recon_lcs` 是
       **递归**回溯的，2000+ token 的答案直接 `RecursionError` 崩掉（实测本项目的真实答案
       就会崩）。给它真正的句子边界，每句几十个 token，递归深度自然就下来了。

    顺带一提：句子级切分本来就是 ROUGE-L summary level 的正确用法。
    """
    sents = sentence_tokens(text)
    return ".".join(" ".join(t.replace(".", "·") for t in toks) for toks in sents)


# ============================================================================
# 一、文本相似性：ROUGE
# ============================================================================
def _ngrams(tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    """unique n-gram 集合（与 rouge 1.0.1 的 exclusive=True 语义一致）。"""
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _lcs_token_set(x: Sequence[str], y: Sequence[str]) -> Set[str]:
    """回溯出一条最长公共子序列，返回其中**去重后的**词集合。

    为什么是集合而不是长度：rouge 1.0.1 的 `_recon_lcs` 末尾套了 `Ngrams(..., exclusive=True)`，
    也就是把 LCS 里重复出现的词折叠掉再计数。要和它对得上就必须照做——
    直接用 LCS 长度会系统性偏高（真实答案上实测偏高 0.17~0.26）。

    回溯顺序也与库一致（相等取对角，否则看上/左哪个大），保证选到同一条 LCS。
    """
    n, m = len(x), len(y)
    if not n or not m:
        return set()
    # 先把词映射成整数，内层循环比字符串比较快不少（真实答案约 1000~2500 词）
    vocab: Dict[str, int] = {}
    xi = [vocab.setdefault(t, len(vocab)) for t in x]
    yi = [vocab.setdefault(t, len(vocab)) for t in y]

    table: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, prev = table[i], table[i - 1]
        xv = xi[i - 1]
        for j in range(1, m + 1):
            row[j] = prev[j - 1] + 1 if xv == yi[j - 1] else (
                prev[j] if prev[j] >= row[j - 1] else row[j - 1])

    out: Set[str] = set()
    i, j = n, m
    while i > 0 and j > 0:
        if xi[i - 1] == yi[j - 1]:
            out.add(x[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] > table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return out


def _prf(overlap: float, n_hyp: int, n_ref: int) -> Dict[str, float]:
    """rouge 1.0.1 的 F 值口径：分母加 1e-8（所以完全一致时 f=0.999999995 而非 1.0）。"""
    p = overlap / n_hyp if n_hyp else 0.0
    r = overlap / n_ref if n_ref else 0.0
    f = (2 * p * r) / (p + r + 1e-8) if (p or r) else 0.0
    return {"r": r, "p": p, "f": f}


def rouge_builtin(hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
    """自带的 ROUGE-1/2/L 实现（不依赖 rouge 库）。

    语义刻意与 rouge 1.0.1 对齐：n-gram 取**唯一集合**，ROUGE-L 用整段 LCS 除以
    唯一词数。差异出现在何处、有多大，由 `评估_验证.py` 在真实答案上逐条实测，不靠断言。
    """
    hs, rs = sentence_tokens(hypothesis), sentence_tokens(reference)
    ht = [t for s in hs for t in s]                 # rouge_n 是把各句拼起来算的
    rt = [t for s in rs for t in s]
    out: Dict[str, Dict[str, float]] = {}
    for n in (1, 2):
        hg, rg = _ngrams(ht, n), _ngrams(rt, n)
        out[f"rouge-{n}"] = _prf(len(hg & rg), len(hg), len(rg))
    # ROUGE-L 走 summary level：对每个参照句取它与全部假设句的 LCS 词集，跨句取并集
    union: Set[str] = set()
    for r_sent in rs:
        for h_sent in hs:
            union |= _lcs_token_set(r_sent, h_sent)
    out["rouge-l"] = _prf(len(union), len(set(ht)), len(set(rt)))
    return out


def rouge_scores(hypothesis: str, reference: str,
                 backend: Optional[str] = None) -> Dict[str, Any]:
    """ROUGE-1/2/L。空文本不抛异常（rouge 库遇空会抛 ValueError），返回全 0 并说明。

    Returns: {"rouge-1":{r,p,f}, "rouge-2":…, "rouge-l":…, "backend":…, "note":…}
    """
    use = backend or ROUGE_BACKEND
    h, r = (hypothesis or "").strip(), (reference or "").strip()
    if not h or not r:
        return {**{m: dict(_ZERO) for m in ROUGE_METRICS}, "backend": use,
                "note": "假设或参照为空，ROUGE 无定义，记 0"}
    if use == "rouge-lib" and _RougeLib is not None:
        try:
            th, tr = _tokenized_for_rouge(h), _tokenized_for_rouge(r)
            if not th.strip() or not tr.strip():     # 分词后为空（纯标点/纯符号的答案）
                return {**{m: dict(_ZERO) for m in ROUGE_METRICS}, "backend": use,
                        "note": "分词后无有效 token，ROUGE 记 0"}
            scores = _RougeLib().get_scores(th, tr)[0]
            return {**{m: dict(scores[m]) for m in ROUGE_METRICS},
                    "backend": "rouge-lib", "note": "rouge 库（输入已按句切分并预分词）"}
        except (ValueError, ZeroDivisionError, KeyError, RecursionError) as e:
            # RecursionError 是真会发生的：rouge 的 LCS 回溯是递归的，超长单句会栈溢出。
            # 这里退到自带实现而不是让整轮评估崩掉。
            return {**rouge_builtin(h, r), "backend": "builtin",
                    "note": f"rouge 库报错（{type(e).__name__}: {e}），已退到自带实现"}
    return {**rouge_builtin(h, r), "backend": "builtin", "note": "自带实现"}


# ============================================================================
# 二、关键信息抽取与召回
# ============================================================================
# 任务书点名的六类字段（前六项）。第七项「统计量」是本项目补充的——阶段七验收时
# 人工核对的关键事实里就有 `−0.45（p<0.001）` 这类数字，它和剂量一样是"编不得"的硬信息。
# 每条 pattern 后面配一个 normalizer，把不同写法归一，否则 "10 mg" 与 "10mg" 会被当成两条。
_NUM = r"\d+(?:[.,]\d+)?"
_DOSE_UNIT = (r"(?:mg|g|kg|[μµu]g|mcg|ng|m[lL]|L|IU|U|mmol|mol|mEq|"
              r"毫克|克|微克|毫升|国际单位)")
_TIME_UNIT = (r"(?:weeks?|months?|years?|days?|hours?|minutes?|min|wks?|yrs?|"
              r"周|月|年|日|天|小时|分钟)")

_ZH_NUM = "零一二三四五六七八九十两半"

KEY_INFO_PATTERNS: List[Tuple[str, str, str]] = [
    # (类别, 正则, 说明)
    ("percentage", rf"(?<![\d.]){_NUM}\s*%|百分之[{_ZH_NUM}百分点\d]+",
     "百分比符号"),
    ("dosage", rf"(?<![\d.]){_NUM}\s*{_DOSE_UNIT}(?:\s*/\s*(?:kg|m2|m²|日|天|d|day|week|周))?",
     "剂量信息（数值+单位，可带 /kg、/day 等）"),
    # 中文写法要吃下量词「个」与中文数字：「18 个月」「每两周」都要能抽出来
    ("duration", rf"(?<![\d.])(?:{_NUM}|[{_ZH_NUM}]{{1,3}})\s*"
                 rf"(?:[-–~至到]\s*(?:{_NUM}|[{_ZH_NUM}]{{1,3}})\s*)?个?\s*{_TIME_UNIT}",
     "时间范围（疗程/随访/时长）"),
    ("safety", None, "安全信息：风险 / 副作用 / 不良反应"),
    ("recommendation", None, "治疗建议：建议 / 治疗 / 方案"),
    ("mechanism", None, "作用机制：机制 / 原理 / 作用"),
    # ↓ 本项目补充，不在任务书六项内，报告里单列
    ("statistic", r"\bp\s*[<=>≤≥]\s*0?\.\d+|\b(?:HR|OR|RR|CI|95%\s*CI)\b\s*[:=]?\s*"
                  rf"{_NUM}?|(?<![\d.])[-−]{_NUM}\s*(?:分|points?)\b|\bn\s*=\s*\d+",
     "统计量（p 值 / HR / OR / 95%CI / 样本量）—— 本项目补充项"),
]

# ---------------------------------------------------------------------------
# 关键词三类走**概念表**而不是单条正则，原因是本项目绕不开的一件事：
# 证据是英文（PubMed oa_comm），答案常是中文或中英混排。若按字面比对，
# 「不良反应」永远匹配不上 "adverse events"，这三类的召回率会恒等于 0——
# 那不是模型漏了信息，是尺子坏了。所以中英同义写法一律映射到同一个 concept id。
# ---------------------------------------------------------------------------
CONCEPT_MAP: Dict[str, List[Tuple[str, str]]] = {
    "safety": [
        ("adverse_effect", r"副作用|不良反应|不良事件|\badverse\s+(?:events?|reactions?|"
                           r"effects?)\b|\bside[-\s]?effects?\b|\bAEs?\b"),
        ("risk", r"风险|\brisks?\b|\brisky\b"),
        ("safety", r"安全性|安全(?:性)?评价|\bsafety\b|\bwell[-\s]tolerated\b|耐受性|\btolerabilit\w*"),
        ("toxicity", r"毒性|\btoxicit(?:y|ies)\b|\btoxic\b"),
        ("contraindication", r"禁忌|\bcontraindicat\w*"),
        ("warning", r"警告|黑框|\bwarnings?\b|\bboxed\s+warning\b|\bcaution\b"),
        ("mortality", r"死亡|病死|\bmortality\b|\bdeaths?\b|\bfatal\w*"),
    ],
    "recommendation": [
        ("recommendation", r"建议|推荐|\brecommend\w*|\badvis\w*"),
        ("treatment", r"治疗|疗法|\btreatments?\b|\btherap(?:y|ies|eutic)\b|\btreated?\b"),
        ("regimen", r"方案|用药方案|\bregimens?\b|\bschedules?\b|\bprotocols?\b"),
        ("guideline", r"指南|共识|\bguidelines?\b|\bconsensus\b"),
        ("management", r"处置|管理|监测|\bmanagement\b|\bmonitor\w*"),
        ("indication", r"适应证|适应症|\bindicated\s+for\b|\bindications?\b|\bapproved\s+for\b"),
        ("intervention", r"干预|\binterventions?\b"),
    ],
    "mechanism": [
        ("mechanism", r"机制|原理|\bmechanis\w*|\bmode\s+of\s+action\b|\bMOA\b"),
        ("action", r"作用(?!机制)|\bactions?\b|\beffects?\s+on\b"),
        ("pathway", r"通路|信号(?:通路|转导)|\bpathways?\b|\bsignal\w*\b|\bcascade\b"),
        ("target", r"靶点|靶向|\btargets?\b|\btargeting\b"),
        ("receptor", r"受体|\breceptors?\b"),
        ("inhibition", r"抑制|\binhibit\w*|\bblock(?:s|ade|ing)?\b|\bsuppress\w*"),
        ("agonism", r"激动(?:剂|作用)?|\bagonis\w*|\bactivat\w*"),
        ("antagonism", r"拮抗|\bantagonis\w*"),
        ("binding", r"结合|\bbind(?:s|ing)?\b|\baffinity\b"),
        ("mediation", r"介导|\bmediat\w*"),
        ("clearance", r"清除|降解|\bclearance\b|\bclear(?:s|ing|ed)?\b|\bdegrad\w*"),
    ],
}
CONCEPT_CATS = set(CONCEPT_MAP)
_COMPILED_CONCEPTS = {cat: [(cid, re.compile(p, re.IGNORECASE)) for cid, p in items]
                      for cat, items in CONCEPT_MAP.items()}

KEY_INFO_CATEGORIES = [c for c, _, _ in KEY_INFO_PATTERNS]
TASKBOOK_CATEGORIES = KEY_INFO_CATEGORIES[:6]      # 任务书点名的六类
_COMPILED = [(c, re.compile(p, re.IGNORECASE) if p else None, d)
             for c, p, d in KEY_INFO_PATTERNS]

#: 数值类字段（要做单位归一）；概念类见 CONCEPT_MAP
_NUMERIC_CATS = {"percentage", "dosage", "duration", "statistic"}
_SPACE = re.compile(r"\s+")
#: 中文单位/数字 → 英文，避免"10 毫克"和"10 mg"、"两周"和"2 weeks"被算成不同的信息
_UNIT_ALIAS = {"毫克": "mg", "克": "g", "微克": "ug", "μg": "ug", "µg": "ug", "mcg": "ug",
               "毫升": "ml", "国际单位": "iu", "周": "week", "月": "month", "年": "year",
               "日": "day", "天": "day", "小时": "hour", "分钟": "minute", "wks": "week",
               "wk": "week", "yrs": "year", "yr": "year", "weeks": "week", "months": "month",
               "years": "year", "days": "day", "hours": "hour", "minutes": "minute",
               "min": "minute", "个": ""}
_ZH_DIGIT = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}
_ZH_NUM_RUN = re.compile(rf"[{_ZH_NUM}]{{1,3}}")


def zh_numeral_to_arabic(s: str) -> str:
    """中文数字 → 阿拉伯数字：两→2、半→0.5、十二→12、二十四→24。

    只处理 100 以内（医学里的疗程/随访基本都在这个量级），处理不了的原样返回。
    """
    if s == "半":
        return "0.5"
    if "十" in s:
        head, _, tail = s.partition("十")
        if (head and head not in _ZH_DIGIT) or (tail and tail not in _ZH_DIGIT):
            return s
        return str((_ZH_DIGIT[head] if head else 1) * 10 + (_ZH_DIGIT[tail] if tail else 0))
    if all(c in _ZH_DIGIT for c in s) and len(s) == 1:
        return str(_ZH_DIGIT[s])
    return s


def normalize_key_info(cat: str, raw: str) -> str:
    """把同一条信息的不同写法归一到同一个字符串，供集合求交。"""
    s = _SPACE.sub("", raw.strip().lower()).replace(",", "")
    if cat in _NUMERIC_CATS:
        s = _ZH_NUM_RUN.sub(lambda m: zh_numeral_to_arabic(m.group(0)), s)   # 两周 → 2周
        for k in sorted(_UNIT_ALIAS, key=len, reverse=True):
            if k in s:
                s = s.replace(k, _UNIT_ALIAS[k])                             # 2周 → 2week
        s = re.sub(r"(\d)\.0+\b", r"\1", s)           # 10.0mg → 10mg
        s = s.replace("−", "-").replace("–", "-").replace("~", "-").replace("至", "-")
    return s


class MedicalKeyInfoExtractor:
    """用正则从文本里抽医学关键信息，并算「答案覆盖了参照里多少关键信息」。

    召回率按任务书的公式：`recall = overlap / gt_matches`，其中
    overlap = 参照与答案**归一后**关键信息集合的交集大小，gt_matches = 参照里的信息条数。
    另给每类的分项召回，便于看清是哪类信息丢了（实测最常丢的是剂量与统计量）。
    """

    def __init__(self, categories: Optional[Sequence[str]] = None):
        self.categories = list(categories) if categories else list(KEY_INFO_CATEGORIES)
        self._pats = [(c, p, d) for c, p, d in _COMPILED if c in self.categories]

    def extract(self, text: str) -> Dict[str, List[str]]:
        """→ {类别: [归一后的信息, …]}（同类内去重、保持首次出现顺序）。

        数值三类抽的是「值+单位」，概念三类抽的是 concept id（中英同义已合并）。
        """
        t = text or ""
        out: Dict[str, List[str]] = {}
        for cat, pat, _ in self._pats:
            seen: Dict[str, None] = {}
            if cat in CONCEPT_CATS:
                for cid, cpat in _COMPILED_CONCEPTS[cat]:
                    if cpat.search(t):
                        seen.setdefault(cid, None)
            elif pat is not None:
                for m in pat.finditer(t):
                    v = normalize_key_info(cat, m.group(0))
                    if v:
                        seen.setdefault(v, None)
            out[cat] = list(seen)
        return out

    def extract_raw(self, text: str) -> Dict[str, List[str]]:
        """同 extract，但保留原始写法，用于人读报告。"""
        t = text or ""
        out: Dict[str, List[str]] = {}
        for cat, pat, _ in self._pats:
            if cat in CONCEPT_CATS:
                out[cat] = [f"{cid}:{m.group(0)}" for cid, cpat in _COMPILED_CONCEPTS[cat]
                            for m in [cpat.search(t)] if m]
            elif pat is not None:
                out[cat] = [m.group(0).strip() for m in pat.finditer(t)]
        return out

    def recall(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """关键信息召回率。ground_truth 为空时返回 recall=None（**不是 0**）。

        这个区分很重要：没有参照 ≠ 一条都没召回。混为一谈会让汇总均值被无参照的题拉垮。
        """
        gt = self.extract(ground_truth)
        gen = self.extract(generated)
        per: Dict[str, Any] = {}
        overlap_total = gt_total = 0
        for cat in self.categories:
            g, a = set(gt.get(cat, [])), set(gen.get(cat, []))
            ov = g & a
            overlap_total += len(ov)
            gt_total += len(g)
            per[cat] = {
                "gt_matches": len(g), "generated_matches": len(a), "overlap": len(ov),
                "recall": (len(ov) / len(g)) if g else None,
                "missed": sorted(g - a)[:12],
            }
        taskbook_gt = sum(per[c]["gt_matches"] for c in TASKBOOK_CATEGORIES if c in per)
        taskbook_ov = sum(per[c]["overlap"] for c in TASKBOOK_CATEGORIES if c in per)
        return {
            "recall": (overlap_total / gt_total) if gt_total else None,
            "overlap": overlap_total, "gt_matches": gt_total,
            "recall_taskbook_6": (taskbook_ov / taskbook_gt) if taskbook_gt else None,
            "per_category": per,
        }


# ============================================================================
# 三、幻觉信号检测
# ============================================================================
# 每条信号 = (名称, 正则, 权重, 为什么算信号)。权重是**约定**，不是校准值：
# 绝对化断言（100% / 完全安全）比含糊的"研究表明"危害大，所以给更高权重。
HALLUCINATION_SIGNALS: List[Tuple[str, str, float, str]] = [
    ("vague_citation",
     r"研究表明|研究显示|研究证实|大量研究|多项研究|有研究发现|文献报道|众所周知|"
     r"\bstudies\s+(?:have\s+)?(?:show|shown|suggest|demonstrate|indicate)\w*\b|"
     r"\bresearch\s+(?:has\s+)?(?:shown|demonstrated|indicates?)\b|"
     r"\bit\s+is\s+well[-\s]known\b",
     1.0, "「研究表明」类断言但没给具体出处"),
    ("unqualified_proof",
     r"已被证明|已证实|业已证明|毫无疑问|确凿无疑|"
     r"\b(?:has|have)\s+been\s+proven\b|\bproven\s+to\b|\bundoubtedly\b|"
     r"\bwithout\s+(?:a\s+)?doubt\b|\bconclusively\s+(?:shown|proven)\b",
     1.5, "「已被证明」缺乏限定条件（人群/终点/证据等级）"),
    ("absolute_percentage",
     r"(?<![\d.])100\s*%|(?<![\d.])0\s*%\s*(?:的)?(?:风险|副作用|不良反应)|"
     r"\b100\s*percent\b",
     2.0, "医学里极少有 100% / 0% 的结论"),
    ("over_absolute",
     r"完全(?:安全|有效|无害|无副作用|治愈)|绝对(?:安全|有效|可靠)|"
     r"百分之百|无任何副作用|没有任何风险|"
     r"\bcompletely\s+(?:safe|effective|harmless)\b|\btotally\s+safe\b|"
     r"\bno\s+side\s+effects?\s+at\s+all\b|\bguaranteed\s+to\b",
     2.0, "过度绝对化（完全安全/有效/无害）"),
    ("universal_quantifier",
     r"所有患者(?:都|均)|每一位患者|无一例外|总是能|从不会|必然(?:会|能)|一定(?:能|会)治愈|"
     r"\ball\s+patients\s+(?:will|respond|benefit)\b|\balways\s+(?:works?|effective)\b|"
     r"\bnever\s+(?:fails?|causes?)\b",
     1.5, "全称量化：所有患者 / 总是 / 从不"),
]

_COMPILED_SIGNALS = [(n, re.compile(p, re.IGNORECASE), w, d)
                     for n, p, w, d in HALLUCINATION_SIGNALS]

#: 句内出现这些，说明该断言**是挂了出处的**（RAG 的核心产出），单独记为 mitigated
_CITED = re.compile(r"\[S\d+\]|\bPMID[:\s]*\d{6,9}|\bPMC\d{5,9}\b|\b10\.\d{4,9}/\S+")
#: 句子切分：中英文终止符 + 换行。英文缩写点（e.g. / et al. / Fig.）不当句号。
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])|\n+")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|et\s+al|vs|Fig|Dr|No|approx|cf|Inc|Ltd)\.$", re.I)

#: 风险分刻度：risk = 1 - exp(-每千字加权信号数 / SCALE)。
#: 为什么不是"除以上限再截断"——那样短文本一两个信号就顶到 1.0，之后再多信号分数也不变，
#: **失去单调性**（实测：110 字的文本 1 个信号和 4 个信号都是 1.0，分不出轻重）。
#: 指数形式恒在 [0,1) 且严格单调：信号每多一个分数一定上升，只是越往上升得越慢。
#: SCALE=2.0 的含义：每千字 1 个加权信号 ≈ 0.39，2 个 ≈ 0.63，4 个 ≈ 0.86。
#: **这是约定的刻度，不是校准值**——它让不同长度的答案可比，但"多高算危险"仍需人判断。
RISK_DENSITY_SCALE = 2.0


def split_sentences(text: str) -> List[str]:
    """按中英文句末标点切句，跳过英文缩写点。可读性与幻觉检测共用。"""
    raw = [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]
    out: List[str] = []
    for seg in raw:
        # 英文句号：逐个看，缩写点不切
        parts, buf = [], ""
        for piece in re.split(r"(?<=\.)\s+", seg):
            buf = (buf + " " + piece).strip() if buf else piece
            if not _ABBREV.search(buf.rstrip()):
                parts.append(buf)
                buf = ""
        if buf:
            parts.append(buf)
        out.extend(p.strip() for p in parts if p.strip())
    return out


class HallucinationDetector:
    """无依据的绝对化表述检测。信号越多、越绝对、越没出处，风险分越高。

    ⚠ 这是**措辞层面**的启发式：它发现的是"这句话说得太满且没给出处"，
    不能证明内容为假；反过来，一句编造的事实如果措辞谨慎，它也抓不到。
    """

    def __init__(self, count_cited_as_signal: bool = False,
                 density_scale: float = RISK_DENSITY_SCALE):
        #: True = 带出处的断言也照样计入风险（更严）；默认 False，见模块 docstring
        self.count_cited_as_signal = bool(count_cited_as_signal)
        self.density_scale = float(density_scale)

    def detect(self, text: str) -> Dict[str, Any]:
        sents = split_sentences(text)
        hits: List[Dict[str, Any]] = []
        for i, s in enumerate(sents):
            cited = bool(_CITED.search(s))
            for name, pat, weight, why in _COMPILED_SIGNALS:
                for m in pat.finditer(s):
                    hits.append({"signal": name, "matched": m.group(0), "weight": weight,
                                 "why": why, "cited": cited, "sentence_index": i,
                                 "sentence": s[:160]})
        unmitigated = [h for h in hits if self.count_cited_as_signal or not h["cited"]]
        weighted = sum(h["weight"] for h in unmitigated)
        chars = max(1, len(text or ""))
        density = weighted / (chars / 1000.0)
        # 严格单调、恒在 [0,1)：多一个信号分数一定上升，不会像截断式那样提前顶死
        risk = (1.0 - math.exp(-density / self.density_scale)) if self.density_scale > 0 else 0.0
        by_signal: Dict[str, int] = {}
        for h in unmitigated:
            by_signal[h["signal"]] = by_signal.get(h["signal"], 0) + 1
        return {
            "risk_score": round(risk, 4),
            "risk_level": "低" if risk < 0.2 else ("中" if risk < 0.5 else "高"),
            "signals_total": len(hits),
            "signals_unmitigated": len(unmitigated),
            "signals_with_citation": len(hits) - len([h for h in hits if not h["cited"]]),
            "weighted_score": round(weighted, 2),
            "density_per_1k_chars": round(density, 3),
            "by_signal": by_signal,
            "hits": hits[:40],
            "note": ("带 [S#]/PMID/DOI 的断言记为 mitigated，不计入风险"
                     if not self.count_cited_as_signal else "带出处的断言也计入风险"),
        }


# ============================================================================
# 四、可读性
# ============================================================================
#: 经验带：句长落在 [lo, hi] 记满分，越界按距离线性衰减到 0（跨度 span）。
#: 中文医学写作 15~45 字、英文 10~28 词是常见的舒适区。**是约定，不是实证阈值。**
READABILITY_BANDS = {
    "avg_sentence_chars": (15.0, 45.0, 40.0),
    "avg_sentence_tokens": (8.0, 32.0, 30.0),
    "long_sentence_ratio": (0.0, 0.25, 0.5),
}
LONG_SENTENCE_CHARS = 80

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_HEADING = re.compile(r"^#{1,6}\s+\S", re.M)
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S", re.M)


def _band_score(v: float, lo: float, hi: float, span: float) -> float:
    if lo <= v <= hi:
        return 1.0
    d = (lo - v) if v < lo else (v - hi)
    return max(0.0, 1.0 - d / span) if span > 0 else 0.0


class ReadabilityEvaluator:
    """可读性：平均句子长度为主，另附结构与语言一致性。

    多测一项**语言一致性**，是因为阶段七留下的已知缺陷正是"答案中英混排"
    （英文提问，某一节切回中文）。既然是已知问题，就该量出来而不是只在文档里写一句。
    """

    def analyze(self, text: str) -> Dict[str, Any]:
        t = text or ""
        sents = split_sentences(t)
        n = len(sents)
        char_lens = [len(s) for s in sents]
        tok_lens = [len(mixed_tokenize(s)) for s in sents]
        avg_chars = (sum(char_lens) / n) if n else 0.0
        avg_toks = (sum(tok_lens) / n) if n else 0.0
        long_ratio = (sum(1 for c in char_lens if c > LONG_SENTENCE_CHARS) / n) if n else 0.0

        # 语言一致性：逐句判主语言，看少数派占比
        zh = sum(1 for s in sents if len(_CJK.findall(s)) > len(_LATIN.findall(s)))
        en = sum(1 for s in sents if len(_LATIN.findall(s)) > len(_CJK.findall(s)))
        dominant = "zh" if zh >= en else "en"
        minority = min(zh, en)
        mixed_ratio = (minority / n) if n else 0.0

        paragraphs = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
        headings = len(_HEADING.findall(t))
        list_items = len(_LIST_ITEM.findall(t))
        structured = bool(headings or list_items)

        sub = {
            "sentence_length": _band_score(avg_chars, *READABILITY_BANDS["avg_sentence_chars"]),
            "token_length": _band_score(avg_toks, *READABILITY_BANDS["avg_sentence_tokens"]),
            "long_sentence": _band_score(long_ratio, *READABILITY_BANDS["long_sentence_ratio"]),
            "structure": 1.0 if structured else 0.5,
            "language_consistency": 1.0 - mixed_ratio,
        }
        score = sum(sub.values()) / len(sub)
        return {
            "readability_score": round(score, 4),
            "avg_sentence_length": round(avg_chars, 2),        # 任务书点名的指标
            "avg_sentence_tokens": round(avg_toks, 2),
            "sentences": n,
            "max_sentence_length": max(char_lens) if char_lens else 0,
            "long_sentence_ratio": round(long_ratio, 4),
            "long_sentence_threshold_chars": LONG_SENTENCE_CHARS,
            "chars": len(t),
            "paragraphs": len(paragraphs),
            "headings": headings,
            "list_items": list_items,
            "structured": structured,
            "dominant_language": dominant,
            "mixed_language_ratio": round(mixed_ratio, 4),
            "sub_scores": {k: round(v, 4) for k, v in sub.items()},
            "note": "句长经验带见 READABILITY_BANDS，是约定不是校准量表",
        }


# ============================================================================
# 五、总评估器
# ============================================================================
class AnswerEvaluator:
    """把四个维度合成一次调用。

    Args:
        categories:  关键信息类别子集（默认全部七类，含本项目补充的 statistic）
        count_cited_as_signal: 幻觉检测是否把带出处的断言也算风险（默认否）
    """

    def __init__(self, categories: Optional[Sequence[str]] = None,
                 count_cited_as_signal: bool = False):
        self.key_info = MedicalKeyInfoExtractor(categories)
        self.hallucination = HallucinationDetector(count_cited_as_signal)
        self.readability = ReadabilityEvaluator()

    def evaluate(self, answer: str,
                 reference: Optional[str] = None,
                 evidence: Optional[str] = None,
                 case_id: str = "") -> Dict[str, Any]:
        """评一份答案。

        reference 优先于 evidence；两者都没有时只出 ③④，且 ①② 明确记为 None，
        **不记 0** —— 没参照和"零重合"必须区分得开。
        """
        if reference and reference.strip():
            ref, kind = reference, "gold"
        elif evidence and evidence.strip():
            ref, kind = evidence, "evidence"
        else:
            ref, kind = "", "none"

        out: Dict[str, Any] = {
            "case_id": case_id,
            "reference_kind": kind,
            "reference_chars": len(ref),
            "hallucination": self.hallucination.detect(answer),
            "readability": self.readability.analyze(answer),
        }
        if kind == "none":
            out["rouge"] = None
            out["key_info"] = None
            out["note"] = "无参照文本，只评幻觉信号与可读性"
        else:
            out["rouge"] = rouge_scores(answer, ref)
            out["key_info"] = self.key_info.recall(answer, ref)
            out["note"] = ("参照 = 人工标准答案" if kind == "gold" else
                           "参照 = 该题检索到的证据原文；量的是对证据的覆盖，不是正确性")
        out["key_info_in_answer"] = {k: len(v) for k, v in
                                     self.key_info.extract(answer).items()}
        return out

    # ---------------- 汇总 ----------------
    @staticmethod
    def aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """多份评估结果的汇总。均值只在**有定义**的那些上取（None 跳过）。"""
        def mean(vals: Sequence[Optional[float]]) -> Optional[float]:
            xs = [v for v in vals if v is not None]
            return round(sum(xs) / len(xs), 4) if xs else None

        n = len(results)
        rl = [r["rouge"]["rouge-l"]["f"] if r.get("rouge") else None for r in results]
        r1 = [r["rouge"]["rouge-1"]["f"] if r.get("rouge") else None for r in results]
        r2 = [r["rouge"]["rouge-2"]["f"] if r.get("rouge") else None for r in results]
        rec = [r["key_info"]["recall"] if r.get("key_info") else None for r in results]
        rec6 = [r["key_info"]["recall_taskbook_6"] if r.get("key_info") else None
                for r in results]
        risk = [r["hallucination"]["risk_score"] for r in results]
        unmit = [r["hallucination"]["signals_unmitigated"] for r in results]
        read = [r["readability"]["readability_score"] for r in results]
        slen = [r["readability"]["avg_sentence_length"] for r in results]
        mixed = [r["readability"]["mixed_language_ratio"] for r in results]
        by_signal: Dict[str, int] = {}
        for r in results:
            for k, v in r["hallucination"]["by_signal"].items():
                by_signal[k] = by_signal.get(k, 0) + v
        return {
            "cases": n,
            "rouge_1_f": mean(r1), "rouge_2_f": mean(r2), "rouge_l_f": mean(rl),
            "key_info_recall": mean(rec), "key_info_recall_taskbook_6": mean(rec6),
            "hallucination_risk": mean(risk),
            "hallucination_signals_unmitigated": sum(unmit),
            "hallucination_by_signal": by_signal,
            "readability_score": mean(read),
            "avg_sentence_length": mean(slen),
            "mixed_language_ratio": mean(mixed),
            "reference_kinds": sorted({r["reference_kind"] for r in results}),
        }


def format_evaluation(r: Dict[str, Any], indent: str = "  ") -> str:
    """人读的一段式摘要。每个数字都直接来自 r，不做二次估算。"""
    lines = []
    if r.get("rouge"):
        g = r["rouge"]
        lines.append(f"{indent}① 相似性  ROUGE-1 f={g['rouge-1']['f']:.4f}  "
                     f"ROUGE-2 f={g['rouge-2']['f']:.4f}  ROUGE-L f={g['rouge-l']['f']:.4f}"
                     f"   [{g['backend']}，参照={r['reference_kind']}]")
    else:
        lines.append(f"{indent}① 相似性  —（无参照文本）")
    if r.get("key_info"):
        k = r["key_info"]
        rec = "—" if k["recall"] is None else f"{k['recall']:.4f}"
        per = "  ".join(
            f"{c}={k['per_category'][c]['overlap']}/{k['per_category'][c]['gt_matches']}"
            for c in k["per_category"] if k["per_category"][c]["gt_matches"])
        lines.append(f"{indent}② 关键信息召回 {rec}（{k['overlap']}/{k['gt_matches']}）  {per}")
    else:
        lines.append(f"{indent}② 关键信息召回 —（无参照文本）")
    h = r["hallucination"]
    lines.append(f"{indent}③ 幻觉风险 {h['risk_score']:.4f}（{h['risk_level']}）  "
                 f"信号 {h['signals_unmitigated']} 个无出处 / 共 {h['signals_total']} 个  "
                 f"{h['by_signal'] or ''}")
    d = r["readability"]
    lines.append(f"{indent}④ 可读性 {d['readability_score']:.4f}  平均句长 "
                 f"{d['avg_sentence_length']} 字（{d['sentences']} 句，长句占比 "
                 f"{d['long_sentence_ratio']:.2f}）  中英混排 {d['mixed_language_ratio']:.2f}")
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================
_DEMO_ANSWER = """## 回答
lecanemab 是抗淀粉样蛋白单克隆抗体，用于早期阿尔茨海默病 [S1]。
研究表明该药可延缓认知下降 [S2]。CLARITY-AD 试验中，18 个月时 CDR-SB 差值为 -0.45 分（p<0.001）[S2]。
剂量为 10 mg/kg，每两周静脉给药一次 [S3]。
需注意 ARIA 等不良反应，发生率约 12.6% [S3]。
该疗法已被证明完全安全，所有患者都能获益。
"""
_DEMO_REFERENCE = """Lecanemab is an anti-amyloid monoclonal antibody for early Alzheimer disease.
In the CLARITY-AD trial, the adjusted mean difference in CDR-SB at 18 months was -0.45 (p<0.001).
The approved dose is 10 mg/kg intravenously every 2 weeks.
ARIA-E occurred in 12.6% of participants; ARIA-H in 17.3%. Infusion reactions were common.
Treatment recommendations emphasize MRI monitoring for adverse events.
The mechanism involves binding to amyloid protofibrils and promoting clearance.
"""


def _run_demo() -> int:
    e = AnswerEvaluator()
    r = e.evaluate(_DEMO_ANSWER, reference=_DEMO_REFERENCE, case_id="demo")
    print("=" * 92)
    print("答案评估器演示（参照 = 手写的标准答案样例）")
    print("=" * 92)
    print("【答案】\n" + _DEMO_ANSWER)
    print("【参照】\n" + _DEMO_REFERENCE)
    print("-" * 92)
    print(format_evaluation(r))
    print("-" * 92)
    print("命中的幻觉信号：")
    for h in r["hallucination"]["hits"]:
        flag = "有出处" if h["cited"] else "无出处"
        print(f"  · [{h['signal']}] {h['matched']!r}（{flag}，权重 {h['weight']}）—— {h['why']}")
    print("\n参照里的关键信息（按类）：")
    ext = MedicalKeyInfoExtractor()
    for cat, vals in ext.extract(_DEMO_REFERENCE).items():
        if vals:
            print(f"  · {cat:<15} {vals}")
    print(f"\nROUGE 后端：{ROUGE_BACKEND}")
    return 0


def _run_jsonl(path: str, limit: int) -> int:
    """评一批已跑好的答案（生成_流水线测试_*.jsonl）。参照取该题的证据原文（若能取到）。"""
    if not os.path.exists(path):
        print(f"找不到 {path}")
        return 2
    e = AnswerEvaluator()
    rows, results = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            r = e.evaluate(rec.get("answer", ""), case_id=rec.get("case_id", str(i)))
            rows.append(rec)
            results.append(r)
    for rec, r in zip(rows, results):
        print(f"\n[{r['case_id']}] {rec.get('query', '')[:80]}")
        print(format_evaluation(r))
    print("\n" + "=" * 92)
    print("汇总：" + json.dumps(AnswerEvaluator.aggregate(results), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用内置样例跑一次四维评估")
    ap.add_argument("--jsonl", default=None, help="评一批已跑好的答案（*.jsonl，需含 answer 字段）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.jsonl:
        return _run_jsonl(args.jsonl, args.limit)
    if args.demo:
        return _run_demo()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
