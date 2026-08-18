# -*- coding: utf-8 -*-
"""第九阶段 · 强约束系统提示层：把硬约束写成模型照着做得到的指令语言

阶段七的四段链已经在提示词里写了"引用规则"和"安全边界"，但那是**建议式**的散文：
"只能引用上下文中真实出现过的编号"这句话，模型读得懂，却没有任何机制保证它照做——
阶段七之二实测里，四道验收题仍出现过引用编号漂移、答案中英混排、章节标题时有时无。
本阶段做的是把这些约束**分层写死 + 生成后逐条校验 + 不合规触发修正**。

四层，各管一段，缺一不可：

    层 A 硬约束层（本模块）   ── 系统提示词最前面的不可协商条款，四类：
                                知识库边界 / 引用来源 / 禁止编造 / 术语规范与输出格式
    层 B 角色与写作要求（阶段七原文）── 每段各自的职责（评估员 / 作者 / 审稿人 / 定稿人）
    层 C 输出骨架（用户消息）  ── 必需章节标题写在用户消息里，给模型一个可填的模板
    层 D 生成后校验与修正      ── `约束_格式校验器.py` + `约束_受限流水线.py`

**为什么层 D 不能省**：提示词是概率性的，不是编译期约束。层 A 只能提高遵守率，
层 D 才能给出"这次到底守没守"的判定，并在没守时触发重试/修正。本阶段的实测就是量这两件事
分别贡献了多少。

排版上的两个刻意选择：
  1. 硬约束放在系统提示词**最前面**——指令跟随对位置敏感，靠前的约束更难被后面的写作
     要求稀释。⚠ 但要记得阶段七踩过的坑：Ollama 超出 `num_ctx` 时**静默丢掉最前面的内容**，
     所以 `num_ctx=12288` 与上下文预算规划必须同时成立，否则第一个被丢的就是这一层。
  2. 自检清单放在**最后**——输出前的最后一眼，紧邻生成起点。

用法：
    import importlib.util
    spec = importlib.util.spec_from_file_location("cp", r"E:\\rag\\scripts\\约束_提示词层.py")
    cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)

    stages = cp.build_constrained_stages()          # 与阶段七同形状的 PromptStage 字典
    tpl    = cp.ConstrainedPromptTemplates()        # 直接给流水线用
    cp.REFUSAL_PHRASE                               # "根据现有文献无法回答此问题"

CLI：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\约束_提示词层.py --list
    ... --show answer_generator      # 打印受约束后的完整提示词
    ... --diff answer_generator      # 只看相对阶段七多了什么
"""
import argparse
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(mod_name: str, filename: str):
    """中文文件名模块按路径导入（与阶段五~八一致）。"""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tpl_mod = _load_by_path("shengcheng_tishici", "生成_提示词模板.py")
PromptStage = _tpl_mod.PromptStage
BASE_STAGES: Dict[str, Any] = _tpl_mod.PROMPT_STAGES
MedicalPromptTemplates = _tpl_mod.MedicalPromptTemplates


# ============================================================================
# 一、约束常量（校验器与流水线都从这里取，避免两处写死后漂移）
# ============================================================================
#: 知识库边界的**固定拒答短语**。任务书指定原文，一字不改；校验器按精确匹配判定。
#: 之所以要求"精确"而不是"语义相近"：拒答必须是可被下游程序识别的信号，
#: "文献中似乎没有提到"这类自由发挥没法用正则判，也没法统计拒答率。
REFUSAL_PHRASE = "根据现有文献无法回答此问题"

#: **部分作答短语**（2026-08-11 新增）。原来只有一句拒答短语，规格要求"只能答一部分时
#: 也先写它"——模型照做，结果是**摘要说没答案、正文全是答案**的自相矛盾。临床场景下医生
#: 扫一眼摘要就走，会漏掉下面正确的内容；而这种误报恰好集中在阴性结果与证据冲突两类问题上，
#: 也就是最需要 RAG 帮忙的场景。所以把"完全答不了"与"能答一部分"拆成两句独立信号。
PARTIAL_PHRASE = "现有文献仅能部分回答此问题"

#: 部分作答时用来分隔"答得了"与"答不了"的两个小标题，校验器与确定性修正共用同一份字面量。
PARTIAL_ANSWERABLE_HEAD = "可以回答的部分："
PARTIAL_GAP_HEAD = "文献未涉及的部分："

#: 必需章节标题（任务书 1.d）。顺序即回答顺序。
REQUIRED_SECTIONS: List[str] = ["核心答案", "证据总结", "参考文献"]

#: 推荐但不强制的章节：缺了只提示不判违规。保留它是因为阶段七的安全边界要落在这里。
OPTIONAL_SECTIONS: List[str] = ["证据强度与局限"]

#: 出处编号方案。任务书举例 `[文献1]`，本项目自阶段七起用 `[S1]`——两者都满足"唯一的临时
#: 编号"，改前缀要连带改上下文组装器（`assemble_context` 里 `marker = f"S{n}"`）、
#: 阶段七/八已落盘的全部快照与缓存键。**保持 `S` 前缀，把兼容性放到校验器一侧**：
#: 模型若写成 `[文献3]`/`[3]`/`【S3】`，校验器识别为"格式不规范但编号有效"，可确定性改写回
#: `[S3]`，不算编造。这比全局改前缀风险小得多，也顺手覆盖了模型的实际漂移。
CITATION_PREFIX = "S"


def marker_of(i: int) -> str:
    """第 i 条证据的出处标记（1 起）。"""
    return f"[{CITATION_PREFIX}{i}]"


# ============================================================================
# 二、硬约束块
# ============================================================================
@dataclass
class ConstraintBlock:
    """一条硬约束：给模型看的 `text`，给人看的 `why`，给校验器对账的 `checks`。

    `checks` 里写的是"这条约束由哪些校验项兜底"，与 `约束_格式校验器.py` 的 violation code
    一一对应。**约束和校验成对存在**是本模块的设计前提：写不进校验器的约束等于没有约束，
    只是许愿。
    """
    id: str
    title: str
    text: str
    why: str
    checks: List[str] = field(default_factory=list)


CONSTRAINT_BLOCKS: Dict[str, ConstraintBlock] = {

    # ---------------------------------------------------------------- 1.a
    "kb_boundary": ConstraintBlock(
        id="kb_boundary",
        title="知识库边界",
        checks=["refusal.missing", "refusal.conflict", "refusal.partial_unmarked"],
        why="裸模型对超出知识范围的问题会顺势编一个答案；固定短语让「答不了」变成"
            "可被程序识别、可被统计的信号，而不是一段模棱两可的话。"
            "拆成完全/部分两句，是因为合成一句会让「正文明明答了、摘要却说没答」"
            "——读的人扫一眼第一句就走了。",
        text=f"""【硬约束 1／知识库边界】
- 你能使用的全部知识，就是本次提供的文献片段。你自己"记得"的医学知识一律不作数。
- 先判断这些片段属于哪一种情况，**三选一**，第一句写法完全不同：

  ① **完全答不了**（没有相关内容，或只有沾边内容，答不出问题的任何一部分）：
     「核心答案」第一句原样写：{REFUSAL_PHRASE}
     然后用 1-2 句说明**缺的是什么**（如：片段中未涉及该药物、未报告该指标、无该年份之后的数据）。
     ⚠ 写了这一句，后面就**不许再给带出处的实质结论**——那是自相矛盾。有结论就该走 ②。

  ② **能答一部分**（问题的某些方面有证据，另一些没有）：
     第一句原样写：{PARTIAL_PHRASE}
     接着写「{PARTIAL_ANSWERABLE_HEAD}」，只写有证据支撑的内容，每句带出处编号；
     再写「{PARTIAL_GAP_HEAD}」，列出文献没能覆盖的那几点。
     ⚠ 这种情况**不要**写「{REFUSAL_PHRASE}」。

  ③ **能完整回答**：正常作答，两句都不要写。

- **问题问的年份晚于所有证据的发表年份时，一律属于情况 ①**（完全答不了），不得用 ②。
  这种题没有"可以回答的部分"——把该疾病的背景介绍或更早年份的进展写成"可以回答的部分"，
  等于用无关内容冒充答案。
- 这两句话都必须一字不差，不得改写成"文献中似乎没有提到""暂无定论"之类。
- 反例（禁止）：问题问 2025 年后的新疗法，片段里只有 2019 年的数据，却写"目前主流方案是……"。
- 反例（禁止）：第一句写"{REFUSAL_PHRASE}"，下面却列出三条带 [{CITATION_PREFIX}1] 的具体结论。
  ——有结论就属于 ②，第一句应该是"{PARTIAL_PHRASE}"。
- 正例①：{REFUSAL_PHRASE}。提供的文献片段最新为 2019 年，且未涉及该类新药的注册试验。
- 正例②：{PARTIAL_PHRASE}
  {PARTIAL_ANSWERABLE_HEAD}该药的主要不良反应包括…… [{CITATION_PREFIX}2]
  {PARTIAL_GAP_HEAD}片段中未报告长期随访数据与儿童人群的用量。"""),

    # ---------------------------------------------------------------- 1.b
    "citation": ConstraintBlock(
        id="citation",
        title="引用来源",
        checks=["citation.invalid_number", "citation.nonstandard_form", "citation.missing"],
        why="出处是本系统唯一可核对的东西。编号必须落在给定范围内，否则整条溯源链断掉；"
            "无引用的事实句与裸模型的输出没有区别。",
        text=f"""【硬约束 2／引用来源】
- 每一条证据在上下文里都有唯一编号，形如 [{CITATION_PREFIX}1]、[{CITATION_PREFIX}2]。
- 每一个事实性句子后面必须紧跟它的来源编号；一句话有多个来源就并列写 [{CITATION_PREFIX}1][{CITATION_PREFIX}3]。
- **只能写上下文中真实出现过的编号**。上下文只给到 [{CITATION_PREFIX}6] 就绝不能出现 [{CITATION_PREFIX}7]。
- 编号格式固定为方括号 + 大写 {CITATION_PREFIX} + 数字，不要写成 [文献1]、（{CITATION_PREFIX}1）、【{CITATION_PREFIX}1】或 [1]。
- 不确定某句来自哪一条证据时，说明该点缺少直接证据，而不是随手挂一个编号。""",),

    # ---------------------------------------------------------------- 1.c
    "no_fabrication": ConstraintBlock(
        id="no_fabrication",
        title="禁止编造",
        checks=["numeric.ungrounded", "reference.fabricated", "reference.altered"],
        why="阶段一压测裸模型的四类错（编造参考文献、药名张冠李戴、分类错误、时间线矛盾）"
            "全部属于「越过证据自行补齐」。这一条把「补齐」明确列为违规行为。",
        text="""【硬约束 3／禁止编造】
- 严禁添加文献片段中没有的数据、结论或细节：具体包括药名、剂量、给药方案、发生率、
  百分比、样本量、p 值、置信区间、风险比、试验名称/编号、期刊、年份、PMID/PMCID、作者。
- 片段里没有数字，就不要给数字；不要"估计""大约""通常在 10%~20% 之间"。
- 不得把片段中较弱的表述升格：片段写"可能相关"就不能写成"可显著改善"。
- 不得补充片段之外的参考文献。参考文献列表由系统给出，你只能原样使用，
  不得增加条目、改写标题、补全期刊或年份。
- 用户要求给出片段中没有的信息时（例如"文献未提及的副作用是什么"），
  正确做法是指出该信息不在所提供的文献中，而不是从常识里补一个。""",),

    # ---------------------------------------------------------------- 1.d
    "terminology": ConstraintBlock(
        id="terminology",
        title="术语规范",
        checks=["terminology.missing_expansion"],
        why="缩写不展开会让答案只对本领域的人可读；而「首次出现给全称」恰好是可以用"
            "预定义术语表 + 正则自动判定的一条规范。",
        text="""【硬约束 4／术语规范】
- 医学与统计缩写**首次出现时必须给出全称**，格式：全称（缩写）或 缩写（全称）。
  例：随机对照试验（RCT）、非小细胞肺癌（NSCLC）、无进展生存期（PFS）、风险比（HR）。
- 同一缩写只需在首次出现时展开，之后直接使用缩写。
- 全称以文献片段中的写法为准；片段里没有写全称、你也不能确定的，
  保留英文原文并注明"（文献未给出全称）"，不要自己编一个中文译名。
- 若问题要求解释一个上下文中没有出现的术语，按硬约束 1 处理（拒答），不要凭印象解释。""",),

    # ---------------------------------------------------------------- 1.d
    "output_format": ConstraintBlock(
        id="output_format",
        title="输出格式",
        checks=["structure.missing_section", "reference.incomplete"],
        why="固定章节让答案可被程序切开、可被逐节校验，也让人一眼找到结论与出处；"
            "章节名固定为任务书指定的三个，不接受同义改写。",
        text=f"""【硬约束 5／输出格式】
- 回答必须使用下列 Markdown 二级标题，名称一字不改、顺序不变：
  ## {REQUIRED_SECTIONS[0]}
  ## {REQUIRED_SECTIONS[1]}
  ## {OPTIONAL_SECTIONS[0]}
  ## {REQUIRED_SECTIONS[2]}
- 不要把标题改成"结论""总结""引用文献"等同义词，也不要额外加编号（如"## 1. 核心答案"）。
- 「{REQUIRED_SECTIONS[2]}」一节原样抄录系统给出的列表，每条至少含标题、期刊、年份。
- 回答语言与提问语言一致（中文提问用中文回答）；药名、试验名、专有名词保留英文原文，
  不要整句中英混写。"""),
}

#: 层 A 的默认组合。顺序 = 出现在系统提示词里的顺序（重要性递减）。
DEFAULT_LAYER_A: List[str] = ["kb_boundary", "citation", "no_fabrication",
                              "terminology", "output_format"]

#: 自检清单：放在系统提示词最后，紧邻生成起点。
SELF_CHECK = f"""【输出前自检】逐条确认，任何一条不成立就改到成立再输出：
1. 每个 [{CITATION_PREFIX}#] 都能在上文证据里找到同号的块？
2. 答案里的每个数字（剂量/百分比/年份/样本量/p 值）都能在证据原文中逐字找到？
3. 每个缩写的首次出现都带了全称？
4. 四个章节标题齐全且名称一字不差？
5. 如果证据其实答不了这个问题，第一句是不是「{REFUSAL_PHRASE}」？"""


def layer_a_text(block_ids: Sequence[str] = DEFAULT_LAYER_A) -> str:
    """拼出层 A 的完整文本。"""
    return "\n\n".join(CONSTRAINT_BLOCKS[b].text for b in block_ids)


# ============================================================================
# 三、层 C：输出骨架（写在用户消息里，给模型一个可填的模板）
# ============================================================================
def answer_skeleton(with_references: bool) -> str:
    """必需章节的骨架。`with_references=False` 用于草稿段（参考文献由代码统一附加）。"""
    lines = [
        f"## {REQUIRED_SECTIONS[0]}",
        f"（2-4 句直接给出结论，每句末尾带 [{CITATION_PREFIX}#]；"
        f"若证据不足，第一句原样写「{REFUSAL_PHRASE}」并说明缺什么）",
        "",
        f"## {REQUIRED_SECTIONS[1]}",
        f"（分条列出支持结论的关键发现，每条带 [{CITATION_PREFIX}#]、研究类型与年份，"
        f"数字必须来自证据原文）",
        "",
        f"## {OPTIONAL_SECTIONS[0]}",
        "（证据以何种研究类型为主、是否一致、证据未覆盖的部分、需临床医师判断之处）",
    ]
    if with_references:
        lines += ["", f"## {REQUIRED_SECTIONS[2]}",
                  "（原样抄录下面给定的参考文献列表，不增删、不改写）"]
    return "\n".join(lines)


#: 无证据可用时由代码直接产出的回答（不调用模型）。
#: 阶段七这里用的是一段"建议换检索词"的话术，**不含固定拒答短语**——统计拒答率时
#: 会漏掉这一类，所以本阶段换成同一套结构与短语。
def no_evidence_answer(reason: str = "", reference_list: str = "") -> str:
    why = reason or "本次检索没有返回任何可用的文献片段。"
    parts = [f"## {REQUIRED_SECTIONS[0]}",
             f"{REFUSAL_PHRASE}。{why}",
             "",
             f"## {REQUIRED_SECTIONS[1]}",
             "（无可用证据）",
             "",
             f"## {OPTIONAL_SECTIONS[0]}",
             "本次没有证据支撑任何结论。建议换用更具体的检索词（药物通用名、疾病标准名、"
             "研究类型），或放宽年份与章节过滤条件后重试。",
             "",
             f"## {REQUIRED_SECTIONS[2]}",
             reference_list.strip() or "（无）"]
    return "\n".join(parts)


# ============================================================================
# 四、把层 A/C 装进阶段七的四段提示词
# ============================================================================
#: 哪一段挂哪些约束。① 只做证据评估、不写答案，挂"知识库边界 + 禁止编造"就够了；
#: 给它挂输出格式反而有害——它的输出是 JSON，不是带章节的回答。
STAGE_LAYER_A: Dict[str, List[str]] = {
    "evidence_evaluator": ["kb_boundary", "no_fabrication"],
    "answer_generator":   DEFAULT_LAYER_A,
    "critical_reviewer":  ["kb_boundary", "citation", "no_fabrication"],
    "final_assembler":    DEFAULT_LAYER_A,
}

#: 哪些段要在最后附自检清单：只有真正写答案的两段。
STAGE_SELF_CHECK = {"answer_generator", "final_assembler"}


def _constrained_system(base: str, block_ids: Sequence[str], self_check: bool) -> str:
    """层 A（最前）+ 阶段七原文（层 B）+ 自检（最后）。"""
    parts = [layer_a_text(block_ids), base.strip()]
    if self_check:
        parts.append(SELF_CHECK)
    return "\n\n".join(parts)


#: 层 C：② 与 ④ 的用户模板要换成带必需章节的骨架。其余两段（JSON 输出）不动。
def _user_template_answer_generator() -> str:
    return f"""问题：{{question}}

可用证据：
{{context}}

证据评估结论（来自上一步，可为空）：
{{evidence_summary}}

请**只**根据上述证据作答，按下面的骨架输出（章节标题一字不改）：

{answer_skeleton(with_references=False)}

这一步不要写参考文献列表。"""


def _user_template_final_assembler() -> str:
    return f"""问题：{{question}}

证据：
{{context}}

草稿答案：
{{draft_answer}}

审查意见：
{{review}}

参考文献列表（由系统给出，原样使用，不得增删或改写其中的标题、期刊、年份与编号）：
{{reference_list}}

请按审查意见修订后输出最终回答，按下面的骨架（章节标题一字不改）：

{answer_skeleton(with_references=True)}"""


#: 层 D 的修正段：校验器发现违规后，把**具体违规条目**回灌给模型重写一遍。
#: 温度 0.0——修正是收敛动作，不需要多样性；且阶段七实测过"温度 0 时原样重发无效"，
#: 所以这一段的输入里必须带上违规清单（变量 `violations`），它就是那个"改变了的输入"。
FORMAT_FIXER = PromptStage(
    name="格式修正器",
    description="按校验器给出的违规清单修正回答；只删改不新增，不引入证据之外的内容",
    temperature=0.0,
    max_tokens=2000,
    output_format="markdown",
    enable_thinking=False,
    system_prompt=layer_a_text(DEFAULT_LAYER_A) + """

你是格式与约束修正器。你拿到一份已经写好的回答、它所依据的证据、系统给出的参考文献列表，
以及一份**自动校验器列出的违规清单**。你的任务是让这份回答通过全部校验。

修正原则（顺序即优先级）：
1. 逐条消除违规：删掉无效的出处编号（而不是改成别的编号来蒙混）、把不规范的编号写法改成
   [S#]、给缺全称的缩写补上全称、补齐缺失的章节标题、删掉证据里查不到的数字与细节。
2. **只能删改，不得新增任何证据之外的内容**。修不好的地方就删掉那句话，
   删到答不了问题时，按硬约束 1 改写成拒答。
3. 保留原回答中合法的 [S#] 标注与被证据支持的内容，不要重写风格。
4. 直接输出修正后的**完整回答**，不要输出说明、diff 或"已修改如下"之类的话。

""" + SELF_CHECK,
    user_prompt_template="""问题：{question}

证据：
{context}

参考文献列表（原样使用）：
{reference_list}

待修正的回答：
{answer}

自动校验器发现的问题（每条都必须处理）：
{violations}

请输出修正后的完整回答。""",
)


def build_constrained_stages(
        layer_a: Optional[Dict[str, List[str]]] = None,
        self_check_stages: Optional[Sequence[str]] = None,
        include_fixer: bool = True) -> Dict[str, Any]:
    """在阶段七四段提示词之上叠加层 A / 层 C，返回同形状的 `PromptStage` 字典。

    **不修改阶段七的对象**（`dataclasses.replace` 出新实例）：同一个进程里可能同时跑
    基线组与受约束组做对照，共享可变对象会让两组悄悄相互污染。
    """
    la = layer_a or STAGE_LAYER_A
    sc = set(self_check_stages if self_check_stages is not None else STAGE_SELF_CHECK)
    user_tpl = {"answer_generator": _user_template_answer_generator(),
                "final_assembler": _user_template_final_assembler()}

    out: Dict[str, Any] = {}
    for key, st in BASE_STAGES.items():
        blocks = la.get(key, DEFAULT_LAYER_A)
        out[key] = replace(
            st,
            system_prompt=_constrained_system(st.system_prompt, blocks, key in sc),
            user_prompt_template=user_tpl.get(key, st.user_prompt_template),
            required_vars=[],          # 让 __post_init__ 按新模板重新推导
        )
    if include_fixer:
        out["format_fixer"] = replace(FORMAT_FIXER, required_vars=[])
    return out


class ConstrainedPromptTemplates(MedicalPromptTemplates):
    """受约束版模板注册表：与阶段七的 `MedicalPromptTemplates` 接口完全一致，可直接替换。

    注册表里**只有链上的四段**（`include_fixer` 默认 False）——`format_fixer` 不是链上的
    一段，而是校验不通过时才触发的修正段，由 `约束_受限流水线.py` 自己持有并按需调用。
    分开放的好处：注册表就等于"这条链会跑哪几段"，看一眼就知道，不用去猜哪些是可选的。
    """

    def __init__(self, include_fixer: bool = False, **kwargs: Any):
        super().__init__(stages=build_constrained_stages(include_fixer=include_fixer, **kwargs))


# ============================================================================
# 五、给报告用的统计
# ============================================================================
def layer_stats() -> Dict[str, Any]:
    """层 A 相对阶段七加了多少字 / 多少 token（token 用分词器精确算，拿不到就退到估算）。"""
    try:
        tk = _load_by_path("shengcheng_fenciqi", "生成_分词器.py").TokenCounter()
        count = tk.count
        exact = True
    except Exception:                      # 分词器不可用时退到粗估，不让统计功能挡住主流程
        count = lambda s: max(1, len(s) // 3)      # noqa: E731
        exact = False
    con = build_constrained_stages()
    rows = {}
    for key, st in con.items():
        base = BASE_STAGES.get(key)
        b_sys = base.system_prompt if base else ""
        b_usr = base.user_prompt_template if base else ""
        rows[key] = {
            "system_chars": (len(b_sys), len(st.system_prompt)),
            "system_tokens": (count(b_sys), count(st.system_prompt)),
            "user_chars": (len(b_usr), len(st.user_prompt_template)),
            "layer_a": STAGE_LAYER_A.get(key, DEFAULT_LAYER_A) if base else DEFAULT_LAYER_A,
            "new_stage": base is None,
        }
    return {"exact_tokens": exact, "stages": rows,
            "blocks": {b: len(CONSTRAINT_BLOCKS[b].text) for b in CONSTRAINT_BLOCKS}}


# ============================================================================
# CLI
# ============================================================================
def _diff(base: str, new: str) -> str:
    """把"新增的段落"挑出来（按空行切段做集合差；够用且比 difflib 输出好读）。"""
    old_paras = {p.strip() for p in re.split(r"\n\s*\n", base) if p.strip()}
    added = [p.strip() for p in re.split(r"\n\s*\n", new)
             if p.strip() and p.strip() not in old_paras]
    return "\n\n".join(added)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="列出约束块与各段挂载情况")
    ap.add_argument("--show", default=None, help="打印受约束后的某一段完整提示词")
    ap.add_argument("--diff", default=None, help="只打印该段相对阶段七新增的内容")
    ap.add_argument("--blocks", action="store_true", help="打印五个硬约束块的原文")
    ap.add_argument("--stats", action="store_true", help="打印字数/token 增量")
    args = ap.parse_args()

    stages = build_constrained_stages()

    if args.show:
        st = stages[args.show]
        print("=" * 92)
        print(f"{args.show} · {st.name} — {st.description}")
        print(f"temperature={st.temperature} max_tokens={st.max_tokens} "
              f"output_format={st.output_format} 变量={st.required_vars}")
        print("=" * 92)
        print("--- system ---\n" + st.system_prompt)
        print("\n--- user template ---\n" + st.user_prompt_template)
        return 0

    if args.diff:
        base = BASE_STAGES.get(args.diff)
        st = stages[args.diff]
        print(f"=== {args.diff}：相对阶段七新增的 system 内容 ===")
        print(_diff(base.system_prompt if base else "", st.system_prompt))
        if base and base.user_prompt_template != st.user_prompt_template:
            print(f"\n=== {args.diff}：user 模板已整体替换（层 C 输出骨架）===")
            print(st.user_prompt_template)
        return 0

    if args.blocks:
        for b in DEFAULT_LAYER_A:
            blk = CONSTRAINT_BLOCKS[b]
            print("=" * 92)
            print(f"{blk.id} · {blk.title}    ← 由 {', '.join(blk.checks)} 兜底")
            print(f"为什么：{blk.why}")
            print("-" * 92)
            print(blk.text)
        return 0

    if args.stats:
        s = layer_stats()
        print(f"token 计数{'精确（qwen3 分词器）' if s['exact_tokens'] else '为估算（分词器不可用）'}")
        for k, v in s["stages"].items():
            tag = "（本阶段新增）" if v["new_stage"] else ""
            print(f"- {k:<20}{tag} system {v['system_chars'][0]}→{v['system_chars'][1]} 字 / "
                  f"{v['system_tokens'][0]}→{v['system_tokens'][1]} token   层A={v['layer_a']}")
        return 0

    # 默认：--list
    print(f"拒答短语：{REFUSAL_PHRASE}")
    print(f"必需章节：{REQUIRED_SECTIONS}  推荐章节：{OPTIONAL_SECTIONS}")
    print(f"出处编号：[{CITATION_PREFIX}#]（任务书举例 [文献#]，见模块说明）\n")
    print("硬约束块：")
    for b in DEFAULT_LAYER_A:
        blk = CONSTRAINT_BLOCKS[b]
        print(f"  - {blk.id:<16}{blk.title:<8} {len(blk.text):>4} 字   兜底校验项：{blk.checks}")
    print("\n各段挂载：")
    for k, st in stages.items():
        print(f"  - {k:<20} 层A={STAGE_LAYER_A.get(k, DEFAULT_LAYER_A)} "
              f"自检={'是' if k in STAGE_SELF_CHECK else '否'} "
              f"temp={st.temperature}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
