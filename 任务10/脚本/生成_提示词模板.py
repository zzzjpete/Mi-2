# -*- coding: utf-8 -*-
"""第七阶段（一）· 医学提示词工程模板

四段式生成链（每段一个 `PromptStage`，各自的温度与输出长度独立配置）：

    ① evidence_evaluator 证据评估器 —— 先判证据能不能用（相关性/研究类型/是否直接回答）
    ② answer_generator   答案生成器 —— 只依据可用证据作答，逐句挂 [S#] 出处
    ③ critical_reviewer  批判性审查器 —— 逐条断言核对是否被引用证据支持，抓幻觉
    ④ final_assembler    最终组装器 —— 按审查意见修订，产出带出处与局限说明的定稿

为什么要拆四段而不是一个大提示词：阶段一压测裸 qwen3:8b 时的四类错（编造参考文献、罕见病
药名张冠李戴、药物分类错误、时间线自相矛盾）都不是"没检索到"，而是**生成时越过了证据**。
一次性生成里，模型既当作者又当审稿人，几乎不会否定自己刚写的句子；拆开后，②只管写、③只管
挑错、④只管收敛，③④看到的是"别人写的草稿"，否定成本低得多。代价是一次问答要跑 3~4 次
模型（本机 qwen3:8b 每次数秒到十几秒）——**这个代价是否值得，属于阶段七第二部分要用
"接入前后对比评测"实测判定的事，本模块只提供可关停的开关（`chain(...)` 可选段）。**

温度取值也不是随手写的：①③要的是**可解析的结构化判断**，温度 0.0 保证同一输入同一输出，
便于复跑与验证；②需要一点组织语言的自由度但不能发散，0.3；④只做收敛改写，0.2。

用法：
    from 生成_提示词模板 import PROMPT_STAGES, MedicalPromptTemplates
    stage = PROMPT_STAGES["answer_generator"]
    msgs  = stage.format_messages(question=q, context=ctx["context_text"], evidence_summary="")
    opts  = stage.to_ollama_options()      # {'temperature':0.3,'num_predict':1500}

CLI：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_提示词模板.py --list
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_提示词模板.py --show answer_generator
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 模板变量：只替换 {标识符} 形式，JSON 示例里的 {"key": ...} 不会被误伤
_VAR = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_template(template: str, **values: Any) -> str:
    """把 {var} 替换成实际值；模板里出现但未提供的变量直接报错（不静默留占位符）。"""
    missing = sorted({m.group(1) for m in _VAR.finditer(template)} - set(values))
    if missing:
        raise KeyError(f"缺少模板变量：{missing}")
    return _VAR.sub(lambda m: str(values[m.group(1)]), template)


# ============================================================================
# 一、阶段数据类
# ============================================================================
@dataclass
class PromptStage:
    """一段提示词的完整配置：角色设定 + 用户模板 + 采样参数。

    温度与 max_tokens 绑在阶段上而不是全局：结构化判断（温度 0）和自然语言作答（温度 0.3）
    对采样的要求相反，用一套全局参数必然有一头将就。
    """
    name: str
    system_prompt: str
    user_prompt_template: str
    temperature: float
    max_tokens: int

    # ---- 以下为带默认值的补充配置，不影响任务书要求的五个字段 ----
    description: str = ""
    output_format: str = "text"          # "json" | "markdown" | "text"
    enable_thinking: bool = False        # Qwen3 思考模式；结构化阶段一律关掉
    top_p: float = 0.9
    required_vars: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.required_vars:       # 从模板里自动推导需要哪些变量
            seen, order = set(), []
            for m in _VAR.finditer(self.user_prompt_template):
                if m.group(1) not in seen:
                    seen.add(m.group(1))
                    order.append(m.group(1))
            self.required_vars = order
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"{self.name}: temperature 越界 {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"{self.name}: max_tokens 必须为正")

    # ---------------- 渲染 ----------------
    def render_user(self, **values: Any) -> str:
        text = render_template(self.user_prompt_template, **values)
        # Qwen3 的思考模式软开关：/no_think 关闭 <think> 段，省 token 也让 JSON 更好解析
        if not self.enable_thinking:
            text += "\n\n/no_think"
        return text

    def format_messages(self, **values: Any) -> List[Dict[str, str]]:
        """渲染成 chat messages（Ollama / LangChain 通用形状）。"""
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.render_user(**values)}]

    def to_ollama_options(self, **extra: Any) -> Dict[str, Any]:
        """采样参数 → Ollama options（max_tokens 在 Ollama 里叫 num_predict）。"""
        opts = {"temperature": self.temperature, "num_predict": self.max_tokens,
                "top_p": self.top_p}
        opts.update(extra)
        return opts

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "output_format": self.output_format, "enable_thinking": self.enable_thinking,
                "required_vars": list(self.required_vars)}


# ============================================================================
# 二、公共片段
# ============================================================================
# 出处规则单独抽出来：四段里有三段要引用同一套规则，写三遍必然会漂
CITATION_RULES = """引用规则（强制）：
- 每一个事实性断言后面，紧跟支持它的证据编号，形如 [S1] 或 [S2][S5]；一句话若有多个来源就并列标注。
- 只能引用上下文中真实出现过的编号，不得杜撰 [S9] 这种不存在的编号。
- 证据里没有的内容一律不写：不得补充上下文之外的药名、剂量、试验名称、期刊、年份、PMID 或统计数字。
- 上下文不足以回答时，明确写出"现有证据不足以回答……"，并说明缺什么，不要用常识补齐。"""

SAFETY_CLAUSE = """安全与边界：
- 输出面向专业检索场景，不构成诊疗建议；涉及具体用药、剂量、个体化处置时，注明需由临床医师结合患者情况判断。
- 不确定就明说不确定，禁止用"显然""公认""大量研究表明"等无出处的措辞掩盖不确定性。
- 回答语言与用户提问语言保持一致（中文提问用中文回答），但证据中的药名、试验名、专有名词保留英文原文。"""

# 检索为空 / 全部证据不相关时，代码走这条，不要再让模型自由发挥
NO_EVIDENCE_NOTICE = ("本次检索没有找到可支持回答的文献证据。为避免编造，这里不给出结论。"
                      "建议：换用更具体的检索词（药物通用名、疾病标准名、研究类型），或放宽年份/章节过滤条件。")

#: 建议的 num_ctx。qwen3:8b 的架构上限是 40960，但本机是 10G 显存的 3080：
#: 权重(Q4_K_M) 约 5.2GB，KV cache 约 0.14MB/token（36 层 × 8 KV 头 × 128 × 2 × fp16），
#: 12288 上下文约 1.8GB，加上计算缓冲仍留有余量；再往上（16384→2.4GB）会开始逼近显存。
#: ⚠ Ollama 默认 num_ctx 只有 4096，四段链装不下，第二部分接入时必须显式设置。
RECOMMENDED_NUM_CTX = 12288

#: 参考文献列表的预留 token。实测每条约 40 token（38.8 / 39.7 / 41.7，见验证报告 F 组），
#: 按上限 15 条留 600。
REFERENCE_RESERVE_TOKENS = 600


# ============================================================================
# 三、四段提示词
# ============================================================================
PROMPT_STAGES: Dict[str, PromptStage] = {

    # ------------------------------------------------------------------ ①
    "evidence_evaluator": PromptStage(
        name="证据评估器",
        description="逐条判定检索证据的相关性、研究类型与可用性，为后续作答筛掉噪声证据",
        temperature=0.0,          # 结构化判断：要可复现、可解析
        max_tokens=1200,
        output_format="json",
        enable_thinking=False,
        system_prompt="""你是一名循证医学的证据评估员。你的唯一任务是评估给定文献片段对特定问题的可用性，不回答问题本身。

评估时严格遵守：
- 只依据片段中实际出现的文字判断，不使用任何外部知识，也不脑补片段没写的内容。
- 相关性 relevance 用 0-3 分：3=直接回答问题；2=与问题同一主题、可作旁证；1=仅沾边（同一疾病或同一药物但问的不是这件事）；0=不相关。
- 研究类型 study_type 只能从以下取值中选：meta_analysis / systematic_review / rct / cohort / case_control / case_report / in_vitro / animal / review / guideline / unclear。片段没有交代研究设计就填 unclear，不要从期刊名或语气推测。
- key_finding 必须是片段中可核对的原文要点（≤40 字），不得改写成更强的结论。
- 只输出 JSON，不要输出解释、前后缀或代码块标记。""",
        user_prompt_template="""问题：{question}

待评估证据：
{context}

请对上面每一条证据（按其编号）输出一个 JSON 对象，汇总成一个 JSON 数组，字段如下：
marker（证据编号，如 "S1"）、relevance（0-3 整数）、study_type（枚举值）、directly_answers（true/false）、key_finding（≤40 字，原文要点）、caveats（局限，如样本小/动物实验/年代久远/仅摘要，没有则填 ""）。

只输出 JSON 数组本身。""",
    ),

    # ------------------------------------------------------------------ ②
    "answer_generator": PromptStage(
        name="答案生成器",
        description="仅基于筛选后的证据生成带 [S#] 出处的草稿答案",
        temperature=0.3,          # 需要组织语言的自由度，但不能发散
        max_tokens=1500,
        output_format="markdown",
        enable_thinking=False,
        system_prompt="""你是一名严谨的医学文献助理，基于检索到的文献证据回答问题。你的回答必须完全建立在给定证据之上——这是本系统存在的意义：没有证据支持的话，宁可不说。

""" + CITATION_RULES + """

写作要求：
- 先直接回答问题，再展开证据；不要写"根据您的问题"这类开场白。
- 证据之间若相互矛盾，如实并列呈现双方及其出处，不要挑一个当结论。
- 区分证据强度：随机对照试验/系统综述 与 病例报告/体外实验 不能用同样的口吻陈述。
- 注意证据的年份：较早的结论若被更新的证据修正，以更新的为准并说明变化。

""" + SAFETY_CLAUSE,
        user_prompt_template="""问题：{question}

可用证据：
{context}

证据评估结论（来自上一步，可为空）：
{evidence_summary}

请基于上述证据作答。结构：
1. **直接回答**：2-4 句给出结论，每句带 [S#]。
2. **证据要点**：分条列出支持该结论的关键发现，每条带 [S#] 与研究类型/年份。
3. **不确定与空白**：证据没覆盖到、或证据间有冲突的地方。
不要在这一步写参考文献列表。""",
    ),

    # ------------------------------------------------------------------ ③
    "critical_reviewer": PromptStage(
        name="批判性审查器",
        description="逐条核对草稿断言是否被引用证据支持，标出幻觉、越界与引用错误",
        temperature=0.0,          # 审查结论要稳定可复现
        max_tokens=1200,
        output_format="json",
        enable_thinking=False,
        system_prompt="""你是一名批判性的医学审稿人。你面前是别人写的草稿答案和它所依据的证据。你的任务是找出草稿中**不被证据支持**的地方，而不是把答案改好，也不是补充你自己的医学知识。

逐条检查每一个事实性断言，按以下问题类型归类：
- unsupported：断言没有引用，或引用的证据其实没说这件事。
- overstated：证据只支持较弱的说法（如"可能相关"），草稿写成了更强的说法（如"可显著改善"）。
- bad_citation：引用了上下文中不存在的编号，或编号与内容对不上。
- fabricated_detail：出现了证据中没有的药名、剂量、试验名、期刊、年份、PMID 或数字。
- outdated：结论依赖较早证据，而上下文中有更新的证据与之不符。
- missing_caveat：证据有明显局限（动物实验、样本极小、仅体外），草稿未提示。
- scope：超出证据范围的推广（如把某亚组结论说成对所有患者成立）。

判定原则：拿不准时按"未被支持"记，宁可多报也不放过；但不要为找问题而找问题，无问题就返回空数组。
只输出 JSON，不要输出解释或代码块标记。""",
        user_prompt_template="""问题：{question}

证据：
{context}

待审查的草稿答案：
{draft_answer}

请输出一个 JSON 对象，字段如下：
issues（数组，每项含 claim=有问题的原句片段、issue_type=上述枚举、severity=high/medium/low、evidence_check=对照证据的说明、suggestion=建议怎么改）、
unsupported_claim_count（整数）、bad_citations（数组，草稿中引用了但上下文没有的编号）、
overall_grounded（true/false，草稿整体是否忠于证据）、verdict（accept / revise / reject 三选一）。

只输出 JSON 对象本身。""",
    ),

    # ------------------------------------------------------------------ ④
    "final_assembler": PromptStage(
        name="最终组装器",
        description="按审查意见修订草稿，输出带出处、证据强度与局限说明的最终回答",
        temperature=0.2,          # 只做收敛式改写
        max_tokens=2000,
        output_format="markdown",
        enable_thinking=False,
        system_prompt="""你是最终定稿人。你拿到：原问题、证据、草稿答案、审查意见。你的任务是按审查意见修订草稿，产出可直接交付的最终回答。

修订原则（顺序即优先级）：
1. 审查意见指出的每个问题都必须处理：删除不被支持的断言、把夸大的说法降调到证据实际支持的强度、修正或删除错误引用、补上缺失的局限提示。
2. **只能删改与降调，不得新增证据之外的内容**——修订不是重写，更不是补充你自己的知识。
3. 保留草稿中被证据支持的部分及其 [S#] 标注，不要在修订中把出处丢掉。
4. 如果修订后剩下的内容已不足以回答问题，就如实说明证据不足，而不是硬凑一个答案。

""" + CITATION_RULES + """

""" + SAFETY_CLAUSE,
        user_prompt_template="""问题：{question}

证据：
{context}

草稿答案：
{draft_answer}

审查意见：
{review}

参考文献列表（由系统给出，原样使用，不得增删或改写其中的标题、期刊、年份与编号）：
{reference_list}

请输出最终回答，结构如下：
## 回答
（2-4 句结论，每句带 [S#]）

## 证据要点
（分条，每条带 [S#]，注明研究类型与年份）

## 证据强度与一致性
（说明证据以何种研究类型为主、是否一致、是否存在冲突或年份跨度问题）

## 局限与提示
（证据未覆盖的部分、适用边界、需临床医师判断之处）

## 参考文献
（原样附上给定的参考文献列表）""",
    ),
}


# ============================================================================
# 四、模板管理
# ============================================================================
class MedicalPromptTemplates:
    """四段提示词的注册表 + 渲染/预算工具。"""

    #: 默认链路顺序；关掉 ①③ 就退化成"检索→直接作答"，用于阶段七第二部分的对照实验
    DEFAULT_CHAIN = ["evidence_evaluator", "answer_generator", "critical_reviewer", "final_assembler"]

    def __init__(self, stages: Optional[Dict[str, PromptStage]] = None):
        self.stages = dict(stages or PROMPT_STAGES)

    def __getitem__(self, key: str) -> PromptStage:
        return self.get(key)

    def get(self, key: str) -> PromptStage:
        if key not in self.stages:
            raise KeyError(f"未知阶段 {key!r}，可用：{sorted(self.stages)}")
        return self.stages[key]

    def list_stages(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.stages.values()]

    def chain(self, evaluate: bool = True, review: bool = True) -> List[str]:
        """生成链路。evaluate/review 关掉即为消融实验（少跑 1~2 次模型）。"""
        keys = list(self.DEFAULT_CHAIN)
        if not evaluate:
            keys.remove("evidence_evaluator")
        if not review:
            keys.remove("critical_reviewer")
        return keys

    def render(self, key: str, **values: Any) -> Dict[str, Any]:
        """渲染一段：messages + 采样参数，直接可喂 Ollama / LangChain。"""
        st = self.get(key)
        return {"stage": key, "name": st.name, "messages": st.format_messages(**values),
                "options": st.to_ollama_options(), "output_format": st.output_format}

    # ---------------- token 预算 ----------------
    def prompt_overhead_tokens(self, key: str, token_counter) -> int:
        """一段提示词里**除证据以外**的固定开销（system + 模板骨架 + 其它变量留白）。"""
        st = self.get(key)
        blank = {v: "" for v in st.required_vars}
        return (token_counter.count(st.system_prompt)
                + token_counter.count(st.render_user(**blank)))

    def plan_budget(self, token_counter, num_ctx: int = RECOMMENDED_NUM_CTX,
                    reference_reserve: int = REFERENCE_RESERVE_TOKENS,
                    reply_headroom: int = 0) -> Dict[str, Any]:
        """给定 num_ctx，算出证据上下文最多能放多少 token。

        Ollama 超出 num_ctx 不会报错而是**静默截断最前面的内容**（通常正是 system 提示词和
        排在最前的高相关证据），所以这里按最吃上下文的那一段留足余量：
            证据预算 = num_ctx − (该段固定开销 + 它的输出上限)
                                − (草稿上限 + 审查意见上限)      ← 都要一起塞进第 ④ 段
                                − 参考文献列表预留               ← 实测每条约 40 token
        全部按各段 max_tokens 的**上限**计，是刻意的最坏情形估计。
        """
        rows = {}
        for k in self.stages:
            st = self.get(k)
            oh = self.prompt_overhead_tokens(k, token_counter)
            rows[k] = {"overhead": oh, "max_tokens": st.max_tokens,
                       "reserved": oh + st.max_tokens}
        worst_key = max(rows, key=lambda k: rows[k]["reserved"])
        worst = rows[worst_key]
        # 除证据外，final_assembler 还要装下草稿与审查意见（按各自 max_tokens 上限计）
        extra = self.get("answer_generator").max_tokens + self.get("critical_reviewer").max_tokens
        context_budget = num_ctx - worst["reserved"] - extra - reference_reserve - reply_headroom
        return {"num_ctx": num_ctx, "per_stage": rows, "worst_stage": worst_key,
                "chained_inputs_reserved": extra,
                "reference_reserve": reference_reserve,
                "recommended_context_tokens": max(0, context_budget)}


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", default=None, help="打印某一阶段的完整提示词")
    ap.add_argument("--budget", type=int, default=0, help="按给定 num_ctx 算证据预算")
    args = ap.parse_args()

    T = MedicalPromptTemplates()
    if args.show:
        st = T.get(args.show)
        print("=" * 90)
        print(f"{args.show} · {st.name} — {st.description}")
        print(f"temperature={st.temperature} max_tokens={st.max_tokens} "
              f"output_format={st.output_format} 变量={st.required_vars}")
        print("=" * 90)
        print("--- system ---\n" + st.system_prompt)
        print("\n--- user template ---\n" + st.user_prompt_template)
        return

    if args.budget:
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "fenciqi", os.path.join(os.path.dirname(os.path.abspath(__file__)), "生成_分词器.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        print(json.dumps(T.plan_budget(m.TokenCounter(), num_ctx=args.budget),
                         ensure_ascii=False, indent=2))
        return

    for d in T.list_stages():
        print(f"- {d['name']:<8} temp={d['temperature']:<4} max_tokens={d['max_tokens']:<5} "
              f"out={d['output_format']:<8} vars={d['required_vars']}")
        print(f"    {d['description']}")


if __name__ == "__main__":
    main()
