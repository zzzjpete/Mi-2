# -*- coding: utf-8 -*-
"""第九阶段 · 验证：强约束提示层 / 格式校验器 / 修正机制 / 受限流水线

**不调用模型**，秒级跑完。每一条 PASS/FAIL 都由实际算出的变量比对得出——
不存在任何无条件 `print("✓")`（阶段五踩过这个坑）。

能用真实产物就用真实产物：`report_data\\评估_测试集.jsonl`（阶段八留下的真答案）、
`检索快照_live.json`（真检索证据）。缺文件时该组**跳过并在报告里写明**，不静默算通过。

用法：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\约束_验证.py
    加 --quiet 只看汇总
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT, evidence_path

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "report_data")
REPORT_PATH = os.path.join(REPORT_DIR, "约束_验证报告.txt")
LIVE_JSONL = evidence_path("评估_测试集.jsonl")
SNAPSHOT = evidence_path("检索快照_live.json")


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cp = _load("yueshu_tishicing", "约束_提示词层.py")
fc = _load("yueshu_jiaoyanqi", "约束_格式校验器.py")
cpipe = _load("yueshu_liushuixian", "约束_受限流水线.py")
ac = _load("yueshu_yongli", "约束_对抗测试集.py")
tpl = _load("shengcheng_tishici", "生成_提示词模板.py")

R = cp.REQUIRED_SECTIONS
PHRASE = cp.REFUSAL_PHRASE
PARTIAL = cp.PARTIAL_PHRASE


# ============================================================================
# 记账
# ============================================================================
class Checker:
    def __init__(self, quiet: bool = False):
        self.rows: List[Tuple[str, str, bool, str]] = []
        self.skipped: List[Tuple[str, str]] = []
        self.group = ""
        self.quiet = quiet
        self.lines: List[str] = []

    def head(self, title: str):
        self.group = title
        self._out("\n" + "=" * 92)
        self._out(title)
        self._out("=" * 92)

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.rows.append((self.group, label, ok, detail))
        self._out(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        return ok

    def skip(self, label: str, why: str):
        self.skipped.append((self.group, f"{label}：{why}"))
        self._out(f"  [SKIP] {label}   {why}")

    def note(self, text: str):
        self._out(f"        {text}")

    def _out(self, s: str):
        self.lines.append(s)
        if not self.quiet:
            print(s)

    def summary(self) -> Tuple[int, int]:
        return sum(1 for _, _, ok, _ in self.rows if ok), len(self.rows)


# ============================================================================
# 公共小夹具
# ============================================================================
#: 两套夹具：`CITS`/`REFLIST` 干净可通过，用于"合规基线"；`CITS3`/`REFLIST3` 里的 S3
#: **元数据本身就缺期刊与年份**（真语料里确实有这种记录），专供参考文献完整性那一组。
#: 分开是必要的：把 S3 混进合规基线，会让所有"应当合规"的断言都因为一条元数据缺陷而失败，
#: 测出来的全是夹具的问题，不是被测代码的问题。
CITS = [
    {"marker": "S1", "title": "Off-target effects of therapy X",
     "journal": "Nucleic Acids Research", "pub_year": 2021, "pmcid": "PMC7778913", "pmid": "1"},
    {"marker": "S2", "title": "Cohort study of enzyme replacement therapy",
     "journal": "Frontiers in Immunology", "pub_year": 2021, "pmcid": "PMC8670230", "pmid": ""},
]
REFLIST = ("[S1] Off-target effects of therapy X. Nucleic Acids Research (2021). PMC7778913 / PMID 1\n"
           "[S2] Cohort study of enzyme replacement therapy. Frontiers in Immunology (2021). PMC8670230")
CITS3 = CITS + [{"marker": "S3", "title": "Untitled preprint record",
                 "journal": "", "pub_year": None, "pmcid": "PMC9999999", "pmid": ""}]
REFLIST3 = REFLIST + "\n[S3] Untitled preprint record. PMC9999999"
EVIDENCE = ("[S1] In a randomized controlled trial the annualized relapse rate was 0.26 versus "
            "0.58 in controls (p<0.001); 1,234 patients were followed for 24 weeks.\n\n"
            "[S2] Enzyme replacement therapy (ERT) with recombinant human alpha-galactosidase "
            "reduced globotriaosylceramide by 45.2%.")


def good_answer() -> str:
    return (f"## {R[0]}\n"
            "随机对照试验（RCT）显示该疗法可降低年复发率 [S1]。\n\n"
            f"## {R[1]}\n"
            "- 治疗组年复发率 0.26，对照组 0.58（p<0.001）[S1]。\n"
            "- 酶替代治疗（ERT）使三己糖神经酰胺下降 45.2% [S2]。\n\n"
            f"## {cp.OPTIONAL_SECTIONS[0]}\n"
            "证据以随机对照试验与队列研究为主，样本量中等。\n\n"
            f"## {R[2]}\n" + REFLIST + "\n")


# ============================================================================
# A 强约束提示层
# ============================================================================
def group_prompt_layer(c: Checker):
    c.head("A 强约束提示层（层 A / 层 C）")
    blocks = cp.CONSTRAINT_BLOCKS
    stages = cp.build_constrained_stages()

    c.check("五个硬约束块齐全且各不相同",
            len(blocks) == 5 and len({b.text for b in blocks.values()}) == 5,
            f"{list(blocks)}")

    # 校验器实际会产出的违规码：从六组检查的常量里取
    known_prefixes = tuple(sorted(set(fc.FORMAT_CODES) | set(fc.HALLUCINATION_CODES) |
                                  {"citation.", "structure.", "terminology.", "reference.",
                                   "numeric.", "refusal."}))
    bad = [ck for b in blocks.values() for ck in b.checks if not ck.startswith(known_prefixes)]
    c.check("每条硬约束都声明了兜底的校验项，且校验项属于校验器的六组之一",
            all(b.checks for b in blocks.values()) and not bad,
            f"未识别的校验码：{bad}" if bad else
            f"共 {sum(len(b.checks) for b in blocks.values())} 条映射")

    sys_ag = stages["answer_generator"].system_prompt
    c.check("硬约束层排在系统提示词最前面（不是附在末尾）",
            sys_ag.startswith(blocks["kb_boundary"].text[:20]),
            f"开头：{sys_ag[:22]!r}")
    c.check("答案生成段挂满五块约束",
            all(blocks[b].text in sys_ag for b in cp.DEFAULT_LAYER_A),
            f"层A={cp.STAGE_LAYER_A['answer_generator']}")
    c.check("证据评估段（输出 JSON）不挂输出格式约束",
            blocks["output_format"].text not in stages["evidence_evaluator"].system_prompt,
            "章节骨架对 JSON 段有害，故只挂知识库边界与禁止编造")
    c.check("自检清单只出现在链上两个写答案的段（修正段另算，它本就该带自检）",
            all((cp.SELF_CHECK in stages[k].system_prompt) == (k in cp.STAGE_SELF_CHECK)
                for k in tpl.PROMPT_STAGES)
            and cp.SELF_CHECK in stages["format_fixer"].system_prompt,
            f"自检段={sorted(cp.STAGE_SELF_CHECK)} + format_fixer")

    c.check("拒答短语在提示词里一字不差出现",
            PHRASE in blocks["kb_boundary"].text and PHRASE in cp.SELF_CHECK,
            f"「{PHRASE}」")
    ut = stages["final_assembler"].user_prompt_template
    c.check("层 C 输出骨架把三个必需章节写进用户消息",
            all(f"## {s}" in ut for s in R), f"{R}")

    # 不能污染阶段七的对象：对照实验依赖两份提示词各自独立
    base_before = tpl.PROMPT_STAGES["answer_generator"].system_prompt
    _ = cp.build_constrained_stages()
    c.check("生成受约束模板不修改阶段七的 PromptStage 对象",
            tpl.PROMPT_STAGES["answer_generator"].system_prompt == base_before
            and stages["answer_generator"] is not tpl.PROMPT_STAGES["answer_generator"],
            "两组提示词互不影响")

    fx = stages["format_fixer"]
    c.check("修正段变量齐全且温度为 0",
            set(fx.required_vars) == {"question", "context", "answer", "violations",
                                      "reference_list"} and fx.temperature == 0.0,
            f"vars={sorted(fx.required_vars)} temp={fx.temperature}")

    # token 预算：约束层变长了，证据预算不能被吃穿
    T = cp.ConstrainedPromptTemplates()
    tk = _load("shengcheng_fenciqi", "生成_分词器.py").TokenCounter()
    budget = T.plan_budget(tk, num_ctx=tpl.RECOMMENDED_NUM_CTX)
    base_budget = tpl.MedicalPromptTemplates().plan_budget(tk, num_ctx=tpl.RECOMMENDED_NUM_CTX)
    c.check("加了约束层之后，证据上下文预算仍 ≥ 默认的 2800 token",
            budget["recommended_context_tokens"] >= 2800,
            f"阶段七 {base_budget['recommended_context_tokens']} → "
            f"阶段九 {budget['recommended_context_tokens']} token")

    na = cp.no_evidence_answer("测试原因", REFLIST)
    rep = fc.FormatChecker().check(na, citations=CITS, reference_list=REFLIST,
                                   expect_refusal=True)
    c.check("无证据话术自带拒答短语且章节齐全（阶段七那版没有短语）",
            PHRASE in na and rep["structure"]["ok"] and rep["refusal"]["detected"]
            and PHRASE not in tpl.NO_EVIDENCE_NOTICE,
            "阶段七 NO_EVIDENCE_NOTICE 不含固定短语，统计拒答率时会漏算")


# ============================================================================
# B 引用编号校验
# ============================================================================
def group_citation(c: Checker):
    c.head("B 引用来源校验（任务书 1.b）")
    ck = fc.FormatChecker()

    rep = ck.check(good_answer(), citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("合规答案：无效引用 0、准确率 1.0",
            rep["citation"]["n_invalid"] == 0 and rep["citation"]["accuracy"] == 1.0,
            f"用了 {rep['citation']['used']}")

    a = good_answer().replace("[S1]。", "[S1][S9]。", 1)
    rep = ck.check(a, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("越界编号 [S9] 被判为 high 幻觉类违规",
            "S9" in rep["citation"]["invalid"]
            and any(v["code"] == "citation.invalid_number" and v["severity"] == "high"
                    for v in rep["violations"]),
            f"invalid={rep['citation']['invalid']}")

    a = good_answer().replace("[S1]。", "[文献1]。", 1)
    rep = ck.check(a, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("[文献1] 判为「写法不规范」而不是编造（编号本身有效）",
            any(v["code"] == "citation.nonstandard_form" for v in rep["violations"])
            and not rep["citation"]["invalid"],
            f"nonstandard={rep['citation']['nonstandard']}")

    a = good_answer().replace("24 周", "随访(24)周")
    a = a.replace("- 治疗组", "- 分组(3)后治疗组")
    rep = ck.check(a, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    fixed, _ = ck.auto_fix(a, rep, CITS, REFLIST)
    c.check("圆括号里的裸数字不当引用（(3) 不会被删）", "(3)" in fixed,
            "一条式正则会把 (3) 当无效编号删掉，改写正文")

    # 准确率是手算得出的：3 个标准编号 + 1 个不规范，其中 1 个越界
    a = (f"## {R[0]}\n结论一 [S1]，结论二 [S9]，结论三 [文献2] 与 [S2]。\n\n"
         f"## {R[1]}\n- 要点 [S1]。\n\n## {R[2]}\n{REFLIST}\n")
    rep = ck.check(a, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    exp = (5 - 1) / 5          # 正文 5 个编号（S1/S9/文献2/S2/S1），其中 S9 越界
    c.check("引用准确率 = 有效编号 / 全部编号，与手算一致",
            abs(rep["citation"]["accuracy"] - exp) < 1e-9,
            f"{rep['citation']['n_markers']} 个编号，越界 {rep['citation']['n_invalid']}，"
            f"准确率 {rep['citation']['accuracy']:.4f}（手算 {exp:.4f}）")
    c.check("参考文献一节的行首编号不计入正文引用（否则准确率被代码生成的列表稀释）",
            rep["citation"]["n_markers"] == 5,
            "文献列表里另有 2 个 [S#]，未计入")

    a = (f"## {R[0]}\n该疗法可以显著降低患者的年复发率并改善长期结局。\n\n"
         f"## {R[1]}\n- 治疗组年复发率 0.26，对照组 0.58 [S1]。\n\n"
         f"## {R[2]}\n{REFLIST}\n")
    rep = ck.check(a, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("缺出处的事实句被抓出来（覆盖率 0.5）",
            abs((rep["citation"]["coverage"] or 0) - 0.5) < 1e-9
            and any(v["code"] == "citation.missing" for v in rep["violations"]),
            f"覆盖 {rep['citation']['coverage']}，例：{rep['citation']['uncited_examples'][:1]}")

    a = (f"## {R[0]}\n{PHRASE}。提供的文献未涉及该药物的儿童用药数据。\n\n"
         f"## {R[1]}\n（无可用证据）\n\n## {R[2]}\n{REFLIST}\n")
    rep = ck.check(a, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST,
                   expect_refusal=True)
    c.check("拒答句与「证据未涉及」句豁免引用要求",
            not any(v["code"] == "citation.missing" for v in rep["violations"]),
            f"需挂出处的句子 {rep['citation']['sentences_needing_citation']} 句")

    rep = ck.check(good_answer(), citations=None, evidence_text=EVIDENCE)
    c.check("没有权威编号列表时跳过越界判定（不误报）",
            not any(v["code"] == "citation.invalid_number" for v in rep["violations"]),
            "citations=None 时只做写法与覆盖检查")

    rep = ck.check("", citations=CITS)
    c.check("空答案不抛异常，且被判为不合规",
            rep["compliant"] is False, f"违规 {len(rep['violations'])} 条")


# ============================================================================
# C 章节结构
# ============================================================================
def group_structure(c: Checker):
    c.head("C 输出格式 · 必需章节（任务书 1.d）")
    ck = fc.FormatChecker()

    rep = ck.check(good_answer(), citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("三个必需章节齐全时 structure.ok",
            rep["structure"]["ok"] and not rep["structure"]["missing"], f"{R}")

    a = good_answer().replace(f"## {R[1]}", "")
    rep = ck.check(a, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("缺章节判 high",
            any(v["code"] == "structure.missing_section" and v["severity"] == "high"
                for v in rep["violations"]),
            f"missing={rep['structure']['missing']}")

    for raw, canon in (("## 结论", R[0]), ("## 1. 核心答案", R[0]),
                       ("**核心答案**", R[0]), ("## 引用文献", R[2])):
        text = good_answer().replace(f"## {canon}", raw, 1)
        secs = fc.section_map(text)
        rep = ck.check(text, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
        c.check(f"标题走样「{raw}」仍能归到「{canon}」，且记为格式违规",
                canon in secs and any(v["code"] == "structure.wrong_section_name"
                                      for v in rep["violations"]),
                f"归一后章节：{sorted(secs)}")

    # 判定与修正必须基于同一套归一规则，否则会「修完还是报同一条」
    text = good_answer().replace(f"## {R[0]}", "## 结论", 1)
    rep = ck.check(text, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    fixed, _ = ck.auto_fix(text, rep, CITS, REFLIST)
    rep2 = ck.check(fixed, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("确定性修正后同一条违规不再出现（归一规则一致，不会死循环）",
            any(v["code"] == "structure.wrong_section_name" for v in rep["violations"])
            and not any(v["code"] == "structure.wrong_section_name" for v in rep2["violations"]),
            "normalize_headings 同时供 section_map 与 auto_fix 使用")

    c.check("空文本不崩且判不合规", ck.check("")["compliant"] is False, "")


# ============================================================================
# D 术语规范
# ============================================================================
def group_terminology(c: Checker):
    c.head("D 术语规范 · 缩写首次出现给全称（任务书 1.d）")
    ck = fc.FormatChecker()
    body = f"## {R[0]}\n{{}}\n\n## {R[1]}\n- 要点 [S1]。\n\n## {R[2]}\n{REFLIST}\n"

    rep = ck.check(body.format("患者的 PFS 明显延长 [S1]。"), citations=CITS,
                   reference_list=REFLIST)
    c.check("表内缩写没给全称 → medium 违规",
            any(v["code"] == "terminology.missing_expansion" and v["severity"] == "medium"
                for v in rep["violations"]),
            f"缺 {[x['abbr'] for x in rep['terminology']['missing_expansion']]}")

    for txt, why in (("无进展生存期（PFS）明显延长 [S1]。", "中文全称"),
                     ("progression-free survival (PFS) 明显延长 [S1]。", "英文全称"),
                     ("无进展生存（PFS）明显延长 [S1]。", "别名")):
        rep = ck.check(body.format(txt), citations=CITS, reference_list=REFLIST)
        c.check(f"给了{why}即通过", rep["terminology"]["ok"], txt[:28])

    far = "无进展生存期是一个终点。" + "另外证据还讨论了别的问题。" * 12 + "PFS 明显延长 [S1]。"
    rep = ck.check(body.format(far), citations=CITS, reference_list=REFLIST)
    c.check("全称离首次出现太远不算展开（窗口约束真的生效）",
            not rep["terminology"]["ok"], f"窗口 ±{ck.expansion_window} 字")

    a = (f"## {R[0]}\n结论 [S1]。\n\n## {R[1]}\n- 要点 [S1]。\n\n"
         f"## {R[2]}\n[S1] Progression-free survival (PFS) in NSCLC. Lancet (2021). PMC1\n")
    rep = ck.check(a, citations=[CITS[0]], reference_list=REFLIST)
    c.check("参考文献标题里的缩写不参与检查（那是别人写的标题）",
            rep["terminology"]["n_checked"] == 0,
            f"检查了 {rep['terminology']['n_checked']} 个缩写")

    rep = ck.check(body.format("CRISPR/Cas9 与 DNA 修复相关 [S1]。"), citations=CITS,
                   reference_list=REFLIST)
    c.check("通用专名（CRISPR/Cas9/DNA）不要求展开",
            rep["terminology"]["ok"],
            "收录标准：本领域当专名用、几乎从不展开")

    rep = ck.check(body.format("使用 GUIDE-seq 方法检测 [S1]。"), citations=CITS,
                   reference_list=REFLIST)
    c.check("表外疑似缩写只报 low，不影响合规",
            any(v["code"] == "terminology.unknown_abbrev" and v["severity"] == "low"
                for v in rep["violations"]) and rep["terminology"]["ok"],
            f"unknown={rep['terminology']['unknown_abbrev']}")


# ============================================================================
# E 参考文献完整性
# ============================================================================
def group_reference(c: Checker):
    c.head("E 参考文献完整性（任务书 1.d：至少含标题、期刊、年份）")
    ck = fc.FormatChecker()
    # 这一组专用带元数据缺陷的三条夹具：S3 的期刊与年份在语料里本来就是空的
    CITS, REFLIST = CITS3, REFLIST3
    head = f"## {R[0]}\n结论 [S1]。\n\n## {R[1]}\n- 要点 [S1][S2]。\n\n## {R[2]}\n"

    rep = ck.check(head + REFLIST, citations=CITS, reference_list=REFLIST)
    ent = rep["reference"]["entries"]
    c.check("完整条目（标题+期刊+年份）被判完整",
            ent[0]["has_title"] and ent[0]["has_journal"] and ent[0]["has_year"],
            f"S1：{ent[0]['has_title']}/{ent[0]['has_journal']}/{ent[0]['has_year']}")

    c.check("元数据本身缺期刊年份的条目：报 incomplete 且归因于元数据而非模型",
            any(v["code"] == "reference.incomplete" for v in rep["violations"])
            and rep["reference"]["metadata_gap"] >= 1
            and not rep["reference"]["altered"],
            f"S3 缺字段 {ent[2]['metadata_missing']}")

    fake = head + REFLIST + "\n[S4] A landmark trial. New England Journal of Medicine (2024)."
    rep = ck.check(fake, citations=CITS, reference_list=REFLIST)
    c.check("凭空多出的条目判 reference.fabricated（high）",
            any(v["code"] == "reference.fabricated" and v["severity"] == "high"
                for v in rep["violations"]),
            f"{rep['reference']['fabricated']}")

    altered = head + REFLIST.replace("Nucleic Acids Research (2021)",
                                     "Nature Biotechnology (2019)")
    rep = ck.check(altered, citations=CITS, reference_list=REFLIST)
    c.check("期刊/年份被改写判 reference.altered（high）",
            any(v["code"] == "reference.altered" for v in rep["violations"]),
            f"{rep['reference']['altered']}")

    partial = head + "[S1] Off-target effects of therapy X. Nucleic Acids Research (2021). PMC7778913 / PMID 1"
    rep = ck.check(partial, citations=CITS, reference_list=REFLIST)
    c.check("正文引用了 S2 但列表没有 → missing_entry",
            "S2" in rep["reference"]["missing_entry"],
            f"{rep['reference']['missing_entry']}")

    with_disc = head + REFLIST + "\n\n" + cpipe.DISCLAIMER
    rep = ck.check(with_disc, citations=CITS, reference_list=REFLIST)
    c.check("免责声明行不被当成参考文献条目",
            rep["reference"]["n_entries"] == 3,
            f"条目数 {rep['reference']['n_entries']}（3 条文献 + 1 行免责声明）")

    rep = ck.check(f"## {R[0]}\n结论 [S1]。\n\n## {R[1]}\n- 要点 [S1]。\n",
                   citations=CITS, reference_list=REFLIST)
    c.check("完全没有参考文献一节 → high",
            any(v["severity"] == "high" and v["code"].startswith("reference.")
                or v["code"] == "structure.missing_section" for v in rep["violations"]),
            f"{[v['code'] for v in rep['violations'] if v['severity'] == 'high']}")


# ============================================================================
# F 数字溯源（禁止编造的可自动判定部分）
# ============================================================================
def group_numeric(c: Checker):
    c.head("F 禁止编造 · 数字溯源（任务书 1.c）")
    ck = fc.FormatChecker()
    head = f"## {R[0]}\n{{}}\n\n## {R[1]}\n- 要点 [S1]。\n\n## {R[2]}\n{REFLIST}\n"

    rep = ck.check(head.format("年复发率为 0.26，p<0.001 [S1]。"), citations=CITS,
                   evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("证据里有的数字判为可溯源",
            rep["numeric"]["ok"], f"{rep['numeric']['n_grounded']}/{rep['numeric']['n_facts']}")

    rep = ck.check(head.format("有效率高达 92.7% [S1]。"), citations=CITS,
                   evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("证据里没有的百分比被抓出来",
            any(v["code"] == "numeric.ungrounded" for v in rep["violations"])
            and any(x["raw"].startswith("92.7") for x in rep["numeric"]["ungrounded"]),
            f"{[x['raw'] for x in rep['numeric']['ungrounded']]}")

    rep = ck.check(head.format("共纳入 1234 例患者，随访 24 周 [S1]。"), citations=CITS,
                   evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("千分位与写法差异被归一（证据 1,234 ↔ 答案 1234）",
            not any("1234" in x["raw"] for x in rep["numeric"]["ungrounded"]),
            f"未溯源项：{[x['raw'] for x in rep['numeric']['ungrounded']]}")

    rep = ck.check(head.format("剂量为 10mg/kg [S1]。"), citations=CITS,
                   evidence_text="[S1] the dose was 10 mg/kg weekly", reference_list=REFLIST)
    c.check("剂量的空格差异被归一（10 mg/kg ↔ 10mg/kg）",
            rep["numeric"]["ok"], f"未溯源：{[x['raw'] for x in rep['numeric']['ungrounded']]}")

    rep = ck.check(head.format("结论见下 [S1]。"), citations=CITS,
                   evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("参考文献里的年份（2021）不算答案的数字断言",
            rep["numeric"]["n_facts"] == 0, f"抽到 {rep['numeric']['n_facts']} 个数字事实")

    rep = ck.check(head.format("1. 第一点 [S1]。\n2. 第二点 [S1]。"), citations=CITS,
                   evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("列表序号不被当成数据",
            rep["numeric"]["n_facts"] == 0, f"抽到 {rep['numeric']['n_facts']} 个")

    rep = ck.check(head.format("2026 年之后的数据不在证据中 [S1]。"), citations=CITS,
                   evidence_text=EVIDENCE, question="2026 年最新获批的药物有哪些？",
                   reference_list=REFLIST)
    c.check("复述问题里的年份不算编造（问题算进溯源语料）",
            rep["numeric"]["ok"], f"未溯源：{[x['raw'] for x in rep['numeric']['ungrounded']]}")

    rep = ck.check(head.format("共 n=1234 例 [S1]。"), citations=CITS,
                   evidence_text=EVIDENCE, reference_list=REFLIST)
    c.check("跨类别宽松兜底：答案 n=1234 ↔ 证据 1,234 patients 不误报",
            rep["numeric"]["ok"], f"未溯源：{[x['raw'] for x in rep['numeric']['ungrounded']]}")


# ============================================================================
# G 知识库边界（拒答）
# ============================================================================
def group_refusal(c: Checker):
    c.head("G 知识库边界 · 拒答判定（任务书 1.a）")
    ck = fc.FormatChecker()
    refuse = (f"## {R[0]}\n{PHRASE}。提供的文献片段未涉及 2026 年之后的审批信息。\n\n"
              f"## {R[1]}\n（无可用证据）\n\n## {R[2]}\n{REFLIST}\n")

    rep = ck.check(refuse, citations=CITS, expect_refusal=True, reference_list=REFLIST)
    c.check("该拒答且拒答了 → 无 refusal 违规",
            rep["refusal"]["detected"] and rep["refusal"]["ok"], "")

    near = refuse.replace(PHRASE, "文献中似乎没有提到相关内容，无法回答")
    rep = ck.check(near, citations=CITS, expect_refusal=True, reference_list=REFLIST)
    c.check("近义表述不算拒答（短语必须一字不差），但会标记 near_miss",
            not rep["refusal"]["detected"] and rep["refusal"]["near_miss_phrase"]
            and any(v["code"] == "refusal.missing" for v in rep["violations"]),
            "拒答要能被程序识别，才谈得上统计拒答率")

    one_char = refuse.replace(PHRASE, PHRASE.replace("此问题", "该问题"))
    rep = ck.check(one_char, citations=CITS, expect_refusal=True, reference_list=REFLIST)
    c.check("改一个词就不算命中固定短语",
            not rep["refusal"]["detected"], "「…此问题」→「…该问题」")

    rep = ck.check(refuse, citations=CITS, expect_refusal=False, reference_list=REFLIST)
    c.check("本可作答却拒答 → over_refusal（对照组靠它抓过度拒答）",
            any(v["code"] == "refusal.over_refusal" for v in rep["violations"]), "")

    conflict = (f"## {R[0]}\n{PHRASE}。但该疗法可显著降低年复发率 [S1]。"
                f"酶替代治疗可降低底物水平 [S2]。另有证据支持长期获益 [S1]。\n\n"
                f"## {R[1]}\n- 要点 [S1]。\n\n## {R[2]}\n{REFLIST}\n")
    rep = ck.check(conflict, citations=CITS, expect_refusal=True, reference_list=REFLIST)
    c.check("既拒答又给多条结论 → conflict，且三态判为部分作答",
            rep["refusal"]["conflict"] and rep["refusal"]["state"] == "partial", "")

    # ⚠ 阈值是 1 不是 2：实测撞到过「一条带出处的结论 + 一句拒答」的真答案，
    #    按 >=2 判会放行，而它恰恰是最误导人的形态（首句说没答案，正文其实答了）。
    one_assert = (f"## {R[0]}\n酶替代治疗可降低底物水平 [S2]。{PHRASE}。"
                  f"提供的片段未报告长期随访数据。\n\n"
                  f"## {R[1]}\n- 要点 [S1]。\n\n## {R[2]}\n{REFLIST}\n")
    rep = ck.check(one_assert, citations=CITS, expect_refusal=True, reference_list=REFLIST)
    c.check("只有一条带出处结论 + 拒答短语 → 同样判 conflict（阈值 1）",
            rep["refusal"]["conflict"] and rep["refusal"]["asserted_count"] == 1, "")

    # 旧写法（拒答短语 + 「其中可以回答的部分」）在新规格下不再合规：
    # 首句仍是"无法回答"，扫一眼摘要的人照样被误导。应判冲突并被确定性修正。
    old_partial = conflict.replace("但该疗法", "其中可以回答的部分：该疗法")
    rep = ck.check(old_partial, citations=CITS, expect_refusal=True, reference_list=REFLIST)
    c.check("旧写法（拒答短语 + 可以回答的部分）仍判冲突：首句没变，读者照样被误导",
            rep["refusal"]["conflict"], "")
    fixed, applied = ck.auto_fix(old_partial, rep, CITS, REFLIST)
    c.check("确定性修正把拒答短语换成部分作答短语（纯字符串，不调模型）",
            PARTIAL in fixed and PHRASE not in fixed
            and any("→" in a for a in applied), f"applied={applied}")
    rep2 = ck.check(fixed, citations=CITS, expect_refusal=True, reference_list=REFLIST)
    c.check("修正后不再冲突，三态仍为部分作答，且应拒题上仍算守住边界",
            (not rep2["refusal"]["conflict"]) and rep2["refusal"]["state"] == "partial"
            and rep2["refusal"]["ok"], "")

    # 部分作答短语本身：可答题上不算过度拒答（阶段九 adv-E1 那条 FAIL 的根因）
    partial_ok = (f"## {R[0]}\n{PARTIAL}\n可以回答的部分：该疗法可降低年复发率 [S1]。\n"
                  f"文献未涉及的部分：长期随访数据。\n\n"
                  f"## {R[1]}\n- 要点 [S1]。\n\n## {R[2]}\n{REFLIST}\n")
    rep = ck.check(partial_ok, citations=CITS, expect_refusal=False, reference_list=REFLIST)
    c.check("可答题上「部分作答」不记过度拒答（只有完全拒答才记）",
            not any(v["code"] == "refusal.over_refusal" for v in rep["violations"])
            and rep["refusal"]["state"] == "partial", "")

    rep = ck.check(refuse, citations=CITS, expect_refusal=True, reference_list=REFLIST)
    c.check("完全拒答（无带出处结论）三态判为 full_refusal",
            rep["refusal"]["state"] == "full_refusal"
            and rep["refusal"]["asserted_count"] == 0, "")

    # ---- 时间越界：问题年份晚于本次证据的最新年份 → 只能完全拒答 ----------------
    # 这一类不需要理解语义，纯元数据比较。它专治「空洞的部分作答」里可判的那一半：
    # 问 2026 年的新药、证据只到 2023，却把疾病背景当成"可以回答的部分"。
    YCITS = [dict(c, pub_year=2023) for c in CITS]
    partial_2026 = (f"## {R[0]}\n{PARTIAL}\n可以回答的部分：该病是一种遗传病 [S1]。\n"
                    f"文献未涉及的部分：2026 年获批情况。\n\n"
                    f"## {R[1]}\n- 要点 [S1]。\n\n## {R[2]}\n{REFLIST}\n")
    rep = ck.check(partial_2026, citations=YCITS, question="2026 年最新获批的药物有哪些？",
                   expect_refusal=None, reference_list=REFLIST)
    c.check("问 2026、证据只到 2023，却写部分作答 → 判 refusal.beyond_evidence_year",
            any(v["code"] == "refusal.beyond_evidence_year" for v in rep["violations"])
            and rep["refusal"]["evidence_max_year"] == 2023, "")

    full_2026 = (f"## {R[0]}\n{PHRASE}。提供的文献片段最新为 2023 年。\n\n"
                 f"## {R[1]}\n- 无相关证据。\n\n## {R[2]}\n{REFLIST}\n")
    rep = ck.check(full_2026, citations=YCITS, question="2026 年最新获批的药物有哪些？",
                   expect_refusal=True, reference_list=REFLIST)
    c.check("同一个问题上完全拒答 → 不判越界（行为正确就不该报）",
            not any(v["code"] == "refusal.beyond_evidence_year" for v in rep["violations"]), "")

    rep = ck.check(partial_2026, citations=YCITS, question="该病有哪些治疗方式？",
                   expect_refusal=None, reference_list=REFLIST)
    c.check("问题里没有年份 → 不判越界（不误伤正常的部分作答）",
            not any(v["code"] == "refusal.beyond_evidence_year" for v in rep["violations"]), "")

    rep = ck.check(partial_2026, citations=YCITS, question="2019 年的试验结果如何？",
                   expect_refusal=None, reference_list=REFLIST)
    c.check("问题年份早于证据最新年份 → 不判越界",
            not any(v["code"] == "refusal.beyond_evidence_year" for v in rep["violations"]), "")

    rep = ck.check(partial_2026, citations=YCITS,
                   question="CLARITY-AD 入组 1795 名患者的主要终点是什么？",
                   expect_refusal=None, reference_list=REFLIST)
    c.check("四位数但不是年份（入组 1795 人）→ 不当成年份",
            not rep["refusal"]["question_years"], "只认 19xx/20xx")

    rep = ck.check(good_answer(), citations=CITS, expect_refusal=None, reference_list=REFLIST)
    c.check("expect_refusal=None 时不产生任何 refusal 违规（不预设立场）",
            not any(v["code"].startswith("refusal.") for v in rep["violations"]), "")


# ============================================================================
# H 修正机制
# ============================================================================
def group_repair(c: Checker):
    c.head("H 修正机制（任务书 1.b：无效引用或缺失引用 → 触发重试或修正）")
    ck = fc.FormatChecker()
    broken = (f"## 结论\n该疗法有效 [S1][S9][文献2]。\n\n"
              f"## 证据要点\n- 年复发率 0.26 [S1]。\n\n"
              f"## {R[2]}\n{REFLIST}\n[S7] Fabricated study. Nature (2030).\n\n"
              + cpipe.DISCLAIMER)
    rep = ck.check(broken, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    n_before = rep["n_violations"]["high"] + rep["n_violations"]["medium"]
    fixed, applied = ck.auto_fix(broken, rep, CITS, REFLIST)
    rep2 = ck.check(fixed, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    n_after = rep2["n_violations"]["high"] + rep2["n_violations"]["medium"]

    c.check("确定性修正删掉越界编号 [S9]", "[S9]" not in fixed, f"applied={applied}")
    c.check("确定性修正把 [文献2] 改写成 [S2]",
            "[文献2]" not in fixed and "[S2]" in fixed, "")
    c.check("确定性修正把被污染的参考文献整段换回系统列表",
            "[S7]" not in fixed and REFLIST.splitlines()[0] in fixed, "")
    c.check("整段替换不会连免责声明一起删掉",
            "不构成临床诊疗建议" in fixed, "阶段九实测踩过：替换到文末把声明也扔了")
    c.check("修正后违规数严格下降", n_after < n_before, f"{n_before} → {n_after}")

    again, applied2 = ck.auto_fix(fixed, rep2, CITS, REFLIST)
    c.check("确定性修正幂等（再修一次不再改动）",
            again == fixed and not applied2, f"第二次 applied={applied2}")

    hint = ck.repair_prompt(rep)
    lows = [v for v in rep["violations"] if v["severity"] == "low"]
    c.check("回灌给模型的修正指令只含 high/medium",
            hint and all(v["message"] not in hint for v in lows),
            f"{len(hint.splitlines())} 行；low 项 {len(lows)} 条未回灌")
    c.check("完全合规时不产生修正指令（不触发无谓重试）",
            ck.repair_prompt(ck.check(good_answer(), citations=CITS, evidence_text=EVIDENCE,
                                      reference_list=REFLIST)) == "", "")


# ============================================================================
# I 受限流水线集成（用假生成器，不调模型）
# ============================================================================
class _FakeGen:
    """假生成器：只按调用次数返回预置文本，用来验证层 D 的控制流。"""

    def __init__(self, texts: List[str]):
        self.texts = list(texts)
        self.calls = 0

    def generate_messages(self, messages, **kw):
        self.calls += 1
        t = self.texts[min(self.calls - 1, len(self.texts) - 1)]
        return {"text": t, "ok": True, "prompt_eval_count": 10, "eval_count": 20}


def _pipe(gen, **kw):
    p = cpipe.ConstrainedGenerationPipeline(generator=gen, verbose=False, **kw)
    p._state.cur = {"question": "该疗法的年复发率是多少？", "context": EVIDENCE,
                    "metrics": {"stage_times": {}, "token_counts": {}},
                    "expect_refusal": kw.pop("_expect", None),
                    "check": None, "repair": None, "all_dropped": False}
    return p


def group_pipeline(c: Checker):
    c.head("I 受限流水线集成（假生成器，不调模型）")
    bad = f"## 结论\n该疗法有效 [S1][S9]。\n\n## 证据要点\n- 年复发率 0.26 [S1]。\n"

    gen = _FakeGen(["irrelevant"])
    p = _pipe(gen, max_repair_rounds=0)
    post = p.postprocess(bad, CITS, REFLIST)
    c.check("层 D 确定性修正在流水线里生效（越界编号被删、参考文献被补）",
            "[S9]" not in post["answer"] and f"## {R[2]}" in post["answer"]
            and post["constraint_check"]["compliant"],
            f"修正记录 {[r['applied'] for r in p._st()['repair']['rounds']]}")
    c.check("max_repair_rounds=0 时不调用模型修正段",
            gen.calls == 0, f"底层生成器被调用 {gen.calls} 次")

    gen2 = _FakeGen(["irrelevant"])
    p2 = _pipe(gen2, max_repair_rounds=0, deterministic_fix=False)
    post2 = p2.postprocess(bad, CITS, REFLIST)
    c.check("对照组（层 D 全关）不会被偷偷修好——标题与越界编号保持原样",
            "[S9]" in post2["answer"] and "## 结论" in post2["answer"]
            and not post2["constraint_check"]["compliant"],
            "否则基线的合规率是我们自己抬上去的")

    good_fix = (f"## {R[0]}\n该疗法可降低年复发率 [S1]。\n\n"
                f"## {R[1]}\n- 治疗组年复发率 0.26 [S1]。\n\n"
                f"## {cp.OPTIONAL_SECTIONS[0]}\n证据为随机对照试验。\n\n## {R[2]}\n{REFLIST}\n")
    gen3 = _FakeGen([good_fix])
    p3 = _pipe(gen3, max_repair_rounds=1, deterministic_fix=False)
    post3 = p3.postprocess(bad, CITS, REFLIST)
    rounds = p3._st()["repair"]["rounds"]
    c.check("不合规时触发模型修正段，且修正后判为合规",
            gen3.calls == 1 and post3["constraint_check"]["compliant"]
            and rounds and rounds[-1]["mode"] == "llm",
            f"调用 {gen3.calls} 次，违规 {rounds[-1]['before']}→{rounds[-1]['after']}")

    worse = f"## 结论\n该疗法有效 [S1][S8][S9]，有效率 99.9%。\n"
    gen4 = _FakeGen([worse])
    p4 = _pipe(gen4, max_repair_rounds=1, deterministic_fix=False)
    post4 = p4.postprocess(bad, CITS, REFLIST)
    rd = p4._st()["repair"]["rounds"][-1]
    c.check("模型把答案改得更糟时回退到修正前的版本",
            rd.get("rolled_back") is True and "[S8]" not in post4["answer"],
            f"违规 {rd['before']}→{rd['after']}，已回退")

    # 兜底边界：①判定全部证据不相关、模型却照样作答 → 整段换成拒答（默认关）
    gen_h = _FakeGen(["x"])
    ph = _pipe(gen_h, max_repair_rounds=0, hard_refuse_on_all_irrelevant=True)
    ph._st()["all_dropped"] = True
    post_h = ph.postprocess(good_answer(), CITS, REFLIST)
    c.check("兜底边界开启且全部证据被判不相关时，最终答案被换成固定拒答",
            PHRASE in post_h["answer"] and gen_h.calls == 0
            and ph._st()["repair"]["forced_refusal"] is True,
            "模型那一版答案仍留在 model_answer_before_force 里备查")
    gen_h2 = _FakeGen(["x"])
    ph2 = _pipe(gen_h2, max_repair_rounds=0)          # 默认关
    ph2._st()["all_dropped"] = True
    post_h2 = ph2.postprocess(good_answer(), CITS, REFLIST)
    c.check("兜底边界默认关：同样的输入不改写（对抗测试量的是模型守不守约束）",
            PHRASE not in post_h2["answer"]
            and ph2._st()["repair"]["forced_refusal"] is False, "")

    gen5 = _FakeGen(["x"])
    p5 = _pipe(gen5, max_repair_rounds=0)
    p5._st()["expect_refusal"] = True
    post5 = p5.postprocess(good_answer(), CITS, REFLIST)
    c.check("expect_refusal 传到校验报告（该拒未拒被记为 high）",
            any(v["code"] == "refusal.missing" for v in post5["constraint_check"]["violations"]),
            "")

    # 无证据路径：不调用模型，直接给带固定短语的结构化拒答
    gen6 = _FakeGen(["x"])
    p6 = cpipe.ConstrainedGenerationPipeline(generator=gen6, verbose=False)
    ctx = {"metadata": {"citations": [], "estimated_tokens": 0}, "selected_chunks": []}
    p6._state.cur = {"question": "q", "context": "", "metrics": {}, "expect_refusal": True,
                     "check": None, "repair": None, "all_dropped": False}
    res = p6._assemble_result("q", "", ctx, {"stage_times": {}, "token_counts": {},
                                             "stage_success": {}}, {}, time.time(),
                              no_evidence=True)
    c.check("无证据时返回带固定拒答短语的结构化答案，且没调用模型",
            PHRASE in res["answer"] and gen6.calls == 0
            and res["constraint_check"]["refusal"]["detected"],
            "")

    p7 = cpipe.ConstrainedGenerationPipeline(generator=_FakeGen(["x"]), verbose=False)
    c.check("受约束流水线默认装的是阶段九提示词，且带独立的修正段",
            cp.CONSTRAINT_BLOCKS["citation"].text in
            p7.templates.get("answer_generator").system_prompt
            and p7.fixer_stage.name == cp.FORMAT_FIXER.name
            and "format_fixer" not in p7.templates.stages,
            "修正段不塞进注册表，避免污染对照组的模板")


# ============================================================================
# J 汇总口径
# ============================================================================
def group_aggregate(c: Checker):
    c.head("J 指标汇总口径")
    ck = fc.FormatChecker()
    r_ok = ck.check(good_answer(), citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    bad1 = good_answer().replace("[S1]。", "[S1][S9]。", 1)          # 越界 → 幻觉类
    r_bad = ck.check(bad1, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)
    bad2 = good_answer().replace(f"## {R[0]}", "## 结论", 1)          # 标题 → 格式类
    r_fmt = ck.check(bad2, citations=CITS, evidence_text=EVIDENCE, reference_list=REFLIST)

    agg = fc.aggregate([r_ok, r_bad, r_fmt])
    # 汇总值按 4 位小数存盘（报告要读），所以比对容差取 5e-5 而不是 1e-9
    c.check("幻觉率 = 命中幻觉类违规的用例数 / 用例数（手算 1/3）",
            abs(agg["hallucination_rate"] - 1 / 3) < 5e-5, f"{agg['hallucination_rate']}")
    c.check("格式合规率 = 无格式类违规的用例数 / 用例数（手算 2/3）",
            abs(agg["format_compliance_rate"] - 2 / 3) < 5e-5,
            f"{agg['format_compliance_rate']}")
    tot = sum(r["citation"]["n_markers"] for r in (r_ok, r_bad, r_fmt))
    bad_n = sum(r["citation"]["n_invalid"] for r in (r_ok, r_bad, r_fmt))
    c.check("引用准确率是 micro 口径（Σ有效/Σ全部），与手算一致",
            abs(agg["citation_accuracy"] - (tot - bad_n) / tot) < 5e-5,
            f"{tot - bad_n}/{tot} = {agg['citation_accuracy']}")
    c.check("越界编号只算幻觉、标题走样只算格式（两类不混）",
            not r_bad["hallucination_free"] and r_bad["format_compliant"]
            and r_fmt["hallucination_free"] and not r_fmt["format_compliant"], "")
    c.check("空输入不崩", fc.aggregate([])["n"] == 0, "")

    # 对抗用例集本身的自洽
    st = ac.stats()
    c.check("对抗用例集含对照组（否则「全拒答」也能拿满分）",
            st["by_category"].get("E_control", 0) >= 2 and st["expect_answer"] >= 2,
            f"{st['n']} 道：应拒 {st['expect_refusal']} / 可答 {st['expect_answer']}")
    c.check("五类攻击面齐全",
            set(st["by_category"]) == set(ac.CATEGORIES), f"{sorted(st['by_category'])}")


# ============================================================================
# K 真实产物
# ============================================================================
def group_real(c: Checker):
    c.head("K 真实产物上的行为（阶段八留下的真答案 / 阶段七真检索证据）")
    if not os.path.exists(LIVE_JSONL):
        c.skip("阶段八真答案体检", f"缺 {LIVE_JSONL}")
        return
    ck = fc.FormatChecker()
    reps, rows = [], []
    with open(LIVE_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not o.get("answer"):
                continue
            cits = [{"marker": s.get("marker"), "title": s.get("title"),
                     "journal": s.get("journal"), "pub_year": s.get("pub_year"),
                     "pmcid": s.get("pmcid"), "pmid": s.get("pmid")}
                    for s in (o.get("sources") or [])]
            r = ck.check(o["answer"], citations=cits, question=o.get("query", ""))
            reps.append(r)
            rows.append((o.get("case_id", "?"), r))
    if not reps:
        c.skip("阶段八真答案体检", "文件里没有可用答案")
        return

    agg = fc.aggregate(reps)
    c.check("校验器能吃下真实答案而不抛异常", len(reps) >= 1, f"{len(reps)} 份")
    n_sec_bad = sum(1 for r in reps
                    if any(v["code"].startswith("structure.") and v["severity"] in ("high", "medium")
                           for v in r["violations"]))
    c.check("阶段七/裸模型写出的答案，没有一份符合本阶段的章节规范（这正是要改的）",
            n_sec_bad == len(reps),
            f"{n_sec_bad}/{len(reps)} 份章节不合规（用「回答/证据要点」或干脆没有标题），"
            f"应为「{R[0]}/{R[1]}/{R[2]}」")
    c.check("这些真答案里没有越界编号（阶段七的引用校验本来就在管这件事）",
            agg["markers_invalid"] == 0,
            f"{agg['markers_total']} 个编号，越界 {agg['markers_invalid']}")
    c.note(f"真答案基线：幻觉率 {agg['hallucination_rate']}、"
           f"格式合规率 {agg['format_compliance_rate']}、"
           f"引用准确率 {agg['citation_accuracy']}、"
           f"术语合规率 {agg['terminology_ok_rate']}")

    if os.path.exists(SNAPSHOT):
        items = ac.build_items(SNAPSHOT)
        c.check("每道对抗题都能配到真实检索证据",
                all(len(it["candidates"]) > 0 for it in items),
                f"{len(items)} 道题，每题 "
                f"{sorted({len(it['candidates']) for it in items})} 条证据")
    else:
        c.skip("对抗用例配证据", f"缺 {SNAPSHOT}")


# ============================================================================
# main
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    c = Checker(quiet=args.quiet)
    c._out("=" * 92)
    c._out("第九阶段验证 · 强约束提示层 / 格式校验器 / 修正机制 / 受限流水线")
    c._out("=" * 92)
    c._out(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　不调用模型")
    c._out("每条 PASS/FAIL 均由实际算出的变量比对得出，无一条是无条件打印。")

    group_prompt_layer(c)
    group_citation(c)
    group_structure(c)
    group_terminology(c)
    group_reference(c)
    group_numeric(c)
    group_refusal(c)
    group_repair(c)
    group_pipeline(c)
    group_aggregate(c)
    group_real(c)

    passed, total = c.summary()
    c._out("\n" + "=" * 92)
    c._out("汇总")
    c._out("=" * 92)
    by_group: Dict[str, List[bool]] = {}
    for g, _, ok, _ in c.rows:
        by_group.setdefault(g, []).append(ok)
    for g, oks in by_group.items():
        c._out(f"  {sum(oks)}/{len(oks)}   {g}")
    if c.skipped:
        c._out("\n  跳过（未静默当作通过）：")
        for g, why in c.skipped:
            c._out(f"    · [{g}] {why}")
    failed = [(g, l, d) for g, l, ok, d in c.rows if not ok]
    if failed:
        c._out("\n  未通过：")
        for g, l, d in failed:
            c._out(f"    · [{g}] {l}   {d}")
    c._out(f"\n总计 {passed}/{total} 项通过   用时 {time.time() - t0:.1f}s")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(c.lines) + "\n")
    print(f"\n报告：{REPORT_PATH}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
