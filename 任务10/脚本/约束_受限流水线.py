# -*- coding: utf-8 -*-
"""第九阶段 · 受约束生成流水线：层 A（提示词）+ 层 D（校验与修正）接进阶段七的四段链

    检索 ─▶ 上下文组装 ─▶ ①证据评估 ─▶ ②草稿 ─▶ ③审查 ─▶ ④定稿
                                                        │
                                             ┌──────────┴──────────┐
                                             │  层 D：校验 → 修正   │
                                             │  1. 确定性修正（免费）│
                                             │  2. 复检             │
                                             │  3. 仍不合规 → 回灌违规│
                                             │     清单给模型重写    │
                                             │  4. 再确定性修正 + 复检│
                                             └──────────┬──────────┘
                                                        ▼
                                          带 constraint_check 的最终结果

**继承而不是重写**：阶段七的 `MedicalGenerationPipeline` 一行不改，本类只覆写三个方法
（`_run_stage` 抓上下文、`postprocess` 做校验与修正、`_assemble_result` 换拒答话术）。
好处是阶段七、八的既有调用方（缓存包装器、批量处理、评测脚本）全部照常工作，
对照实验里两组走的是同一条主干代码，差异只在提示词与层 D。

修正的顺序是**先确定性、后模型**，不是反过来：
  - 编号写法、越界编号、章节标题、参考文献整段替换，这些不需要理解内容，正则改完就对，
    零成本零风险；
  - 剩下的（缺出处的句子、查不到出处的数字、缺全称的缩写）才值得花一次模型调用。
反过来做的话，每次都要多烧一次模型去修本来正则就能修的东西。

用法：
    import importlib.util
    spec = importlib.util.spec_from_file_location("cpipe", r"E:\\rag\\scripts\\约束_受限流水线.py")
    cpipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(cpipe)

    pipe = cpipe.ConstrainedGenerationPipeline(max_repair_rounds=1)
    res  = pipe.generate(q, retrieved_docs=docs, expect_refusal=False)
    res["constraint_check"]["compliant"]        # 本次是否合规
    res["constraint_repair"]["rounds"]          # 每一轮修了什么、修完还剩什么

CLI（离线用阶段三样例块跑一次，需 Ollama）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\约束_受限流水线.py --demo
    ... --baseline      # 同一批证据跑阶段七未加约束的链路，两边都用同一把尺子量
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")        # 硬覆盖：用户级 HF_HOME 指向改名前的旧路径
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional

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


_pipe_mod = _load_by_path("shengcheng_liushuixian", "生成_流水线.py")
_cp = _load_by_path("yueshu_tishicing", "约束_提示词层.py")
_fc = _load_by_path("yueshu_jiaoyanqi", "约束_格式校验器.py")

MedicalGenerationPipeline = _pipe_mod.MedicalGenerationPipeline
ContextAssembler = _pipe_mod.ContextAssembler
LLMGenerator = _pipe_mod.LLMGenerator
DISCLAIMER = _pipe_mod.DISCLAIMER
_MARKER = _pipe_mod._MARKER
_THINK_TAG = _pipe_mod._THINK_TAG
_NO_THINK = _pipe_mod._NO_THINK

ConstrainedPromptTemplates = _cp.ConstrainedPromptTemplates
REFUSAL_PHRASE = _cp.REFUSAL_PHRASE
REQUIRED_SECTIONS = _cp.REQUIRED_SECTIONS
FormatChecker = _fc.FormatChecker


class ConstrainedGenerationPipeline(MedicalGenerationPipeline):
    """带强约束提示层与生成后校验/修正的医学 RAG 流水线。

    Args（除阶段七全部参数外）:
        checker:            `FormatChecker`；None 则用默认配置
        max_repair_rounds:  模型修正的最大轮数（0 = 只做确定性修正，不调模型）
        deterministic_fix:  是否做确定性修正（默认开；关掉可用来单独量"提示词的效果"）
        hard_refuse_on_all_irrelevant:
            ①判定**全部证据都不相关**、而模型仍然给了实质性回答时，把最终答案整段替换成
            固定拒答。**默认关**：对抗测试要量的是模型守不守知识库边界这条约束，代码替它
            拒答会把这件事测没了。生产环境想要一条不依赖模型的兜底边界时再打开。
            （做成"事后改写"而不是"提前短路"，是为了链路里仍然留下模型自己那一版答案，
            报告里能对照"模型本来会怎么答"。）
        constrained_prompts: 是否用受约束提示词（默认开；关掉即阶段七基线，
            但仍然跑层 D 校验 —— 对照组就是这么来的）
    """

    def __init__(self, *args: Any,
                 checker: Optional[Any] = None,
                 max_repair_rounds: int = 1,
                 deterministic_fix: bool = True,
                 hard_refuse_on_all_irrelevant: bool = False,
                 constrained_prompts: bool = True,
                 **kwargs: Any):
        if constrained_prompts and kwargs.get("templates") is None:
            kwargs["templates"] = ConstrainedPromptTemplates()
        super().__init__(*args, **kwargs)
        # `checker or FormatChecker()` 在这里是错的：阶段八踩过同类坑（定义了 __len__ 的
        # 对象空时布尔为假，`or` 会把传进来的对象悄悄换掉）。统一写 is None 判断。
        self.checker = checker if checker is not None else FormatChecker()
        self.max_repair_rounds = int(max_repair_rounds)
        self.deterministic_fix = bool(deterministic_fix)
        self.hard_refuse_on_all_irrelevant = bool(hard_refuse_on_all_irrelevant)
        self.constrained_prompts = bool(constrained_prompts)
        # 修正段单独持有，不塞进 templates 注册表：基线组用的是阶段七那份注册表，
        # 往调用方传进来的对象里偷偷加一段，是那种"跑对照实验时两组悄悄互相污染"的经典来源。
        self.fixer_stage = _cp.build_constrained_stages(include_fixer=True)["format_fixer"]
        # 每次 generate 的中间状态（问题、证据文本、metrics 引用）按线程存：
        # 批量处理里一个线程一条流水线，但共享实例也不能串味。
        self._state = threading.local()

    # ---------------- 每次调用的状态 ----------------
    def _st(self) -> Any:
        s = getattr(self._state, "cur", None)
        if s is None:
            s = {"question": "", "context": "", "metrics": None, "expect_refusal": None,
                 "check": None, "repair": None, "all_dropped": False}
            self._state.cur = s
        return s

    # ---------------- 覆写：抓上下文与 metrics ----------------
    def _run_stage(self, key: str, values: Dict[str, Any],
                   metrics: Dict[str, Any]) -> Dict[str, Any]:
        st = self._st()
        if values.get("context"):
            st["context"] = values["context"]      # 最后一次进模型的证据即校验用的证据
        st["metrics"] = metrics
        return super()._run_stage(key, values, metrics)

    def filter_context_by_evaluation(self, context_text, citations, rows):
        out = super().filter_context_by_evaluation(context_text, citations, rows)
        self._st()["all_dropped"] = bool(out.get("all_dropped"))
        return out

    # ---------------- 主入口 ----------------
    def generate(self, query: str, *args: Any,
                 expect_refusal: Optional[bool] = None, **kwargs: Any) -> Dict[str, Any]:
        """比阶段七多一个参数 `expect_refusal`：这题**是否应当拒答**。

        它只影响校验口径（不满足时报 `refusal.missing` / `refusal.over_refusal`），
        不会被写进提示词——把答案偷偷告诉模型再去考它，测出来的数字没有意义。
        """
        self._state.cur = {"question": query, "context": "", "metrics": None,
                           "expect_refusal": expect_refusal, "check": None, "repair": None,
                           "all_dropped": False}
        res = super().generate(query, *args, **kwargs)
        st = self._st()
        if st["check"] is not None:
            res["constraint_check"] = st["check"]
            res["constraint_repair"] = st["repair"]
        return res

    # ---------------- 覆写：后处理 = 清洗 + 校验 + 修正 ----------------
    def postprocess(self, answer: str, citations: List[Dict[str, Any]],
                    reference_list: str) -> Dict[str, Any]:
        """**刻意不调用 `super().postprocess()`**。

        阶段七那版在发现编造编号时会往答案末尾追加一段"⚠ 校验提示"，而本阶段的做法是
        **把编造的编号删掉**。两者叠加会得到一份自相矛盾的交付物（正文里编号已经没了，
        末尾却写着"发现编造编号"）。所以这里重写清洗步骤，再接层 D。
        """
        st = self._st()
        text = _THINK_TAG.sub("", answer or "")
        text = _NO_THINK.sub("", text).strip()
        # ⚠ 标题归一属于**确定性修正**，必须受 deterministic_fix 管：对照组（层 D 全关）
        # 若在这里顺手把「## 回答」改成「## 核心答案」，基线的格式合规率就被我们自己抬上去了，
        # 对比表也就失去意义。
        if self.deterministic_fix:
            text, _ = _fc.normalize_headings(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # 参考文献与免责声明先补齐再校验：校验器要按"交付给用户的那份"来判，
        # 否则每次都会报一条"没有参考文献"的假违规。
        if REQUIRED_SECTIONS[2] not in _fc.section_map(text) and reference_list:
            text += f"\n\n## {REQUIRED_SECTIONS[2]}\n" + reference_list
        if self.add_disclaimer and "不构成临床诊疗建议" not in text:
            text += "\n\n" + DISCLAIMER

        # 兜底边界（默认关）：①判定全部证据不相关，模型却照样作答 → 整段换成拒答
        forced = False
        if self.hard_refuse_on_all_irrelevant and st["all_dropped"] and REFUSAL_PHRASE not in text:
            st["model_answer_before_force"] = text
            text = _cp.no_evidence_answer(
                "证据评估判定检索到的全部文献片段与问题不相关。", reference_list)
            if self.add_disclaimer:
                text += "\n\n" + DISCLAIMER
            forced = True
            self._log("    层D 兜底边界触发：全部证据被判不相关，最终答案改为拒答")

        text, report, repair = self.enforce(text, citations, reference_list)
        repair["forced_refusal"] = forced
        st["check"], st["repair"] = report, repair

        valid = {c["marker"] for c in citations}
        used = set(_MARKER.findall(text))
        return {"answer": text.strip(),
                "citations_used": sorted(used & valid, key=lambda s: int(s[1:])),
                "citations_available": sorted(valid, key=lambda s: int(s[1:])),
                "fabricated_markers": sorted(used - valid, key=lambda s: int(s[1:])),
                "has_citation": bool(used & valid),
                "constraint_check": report}

    # ---------------- 修正段的一次调用 ----------------
    def _run_fixer(self, values: Dict[str, Any], n: int) -> Dict[str, Any]:
        """跑一次格式修正段，并把耗时/token 按轮次单独记进 metrics（后一轮不覆盖前一轮）。"""
        st = self._st()
        stage = self.fixer_stage
        key = f"format_fixer_r{n}"
        t0 = time.time()
        try:
            r = self.generator.generate_messages(stage.format_messages(**values),
                                                 temperature=stage.temperature,
                                                 max_tokens=stage.max_tokens)
            r["ok"] = True
        except Exception as e:                      # 修正失败不能连累主答案，降级返回
            r = {"ok": False, "error": str(e), "text": "",
                 "prompt_eval_count": None, "eval_count": None}
        m = st.get("metrics") or {}
        m.setdefault("stage_times", {})[key] = round(time.time() - t0, 3)
        m.setdefault("token_counts", {})[key] = {"prompt": r.get("prompt_eval_count"),
                                                 "output": r.get("eval_count"),
                                                 "truncated": bool(r.get("truncated"))}
        self._log(f"  · {key:<20} {m['stage_times'][key]:>6.2f}s  "
                  f"in={r.get('prompt_eval_count')} out={r.get('eval_count')}"
                  + ("" if r.get("ok") else f"  ✗ {str(r.get('error'))[:80]}"))
        return r

    # ---------------- 层 D：校验 → 修正 → 复检 ----------------
    def enforce(self, answer: str, citations: List[Dict[str, Any]],
                reference_list: str) -> Any:
        """返回 (最终答案, 最终校验报告, 修正过程记录)。

        每一轮都记下"修之前有哪些违规、修了什么、修之后还剩哪些"，报告里可以直接引用——
        这是"重试/修正机制到底起没起作用"的唯一证据。
        """
        st = self._st()
        q, ctx = st["question"], st["context"]
        exp = st["expect_refusal"]

        def _check(a: str) -> Dict[str, Any]:
            return self.checker.check(a, citations=citations, evidence_text=ctx,
                                      question=q, expect_refusal=exp,
                                      reference_list=reference_list)

        t0 = time.time()
        report = _check(answer)
        rounds: List[Dict[str, Any]] = []
        # 完整留一份"修正前"的报告：它就是"只有提示词约束、没有层 D"那一组的观测值。
        # 对照实验因此不必为消融再跑一遍模型——同一次生成，取修正前后两个切面即可。
        first = {"compliant": report["compliant"],
                 "violations": [v["code"] for v in report["violations"]
                                if v["severity"] in ("high", "medium")],
                 "report": report}

        # ---- 第 0 步：确定性修正（不调模型）----
        if self.deterministic_fix and not report["compliant"]:
            fixed, applied = self.checker.auto_fix(answer, report, citations, reference_list)
            if applied:
                rep2 = _check(fixed)
                rounds.append({"round": 0, "mode": "deterministic", "applied": applied,
                               "before": len(first["violations"]),
                               "after": len([v for v in rep2["violations"]
                                             if v["severity"] in ("high", "medium")]),
                               "compliant": rep2["compliant"], "seconds": 0.0})
                answer, report = fixed, rep2
                self._log(f"    层D 确定性修正 {len(applied)} 项 → 违规 "
                          f"{rounds[-1]['before']}→{rounds[-1]['after']}")

        # ---- 第 1..N 轮：回灌违规清单给模型重写 ----
        n = 0
        while not report["compliant"] and n < self.max_repair_rounds:
            n += 1
            hint = self.checker.repair_prompt(report)
            if not hint:
                break
            t1 = time.time()
            before = len([v for v in report["violations"] if v["severity"] in ("high", "medium")])
            r = self._run_fixer({"question": q, "context": ctx, "answer": answer,
                                 "violations": hint, "reference_list": reference_list}, n)
            new = (r.get("text") or "").strip()
            if not (r.get("ok") and new):
                rounds.append({"round": n, "mode": "llm", "applied": [],
                               "before": before, "after": before, "compliant": False,
                               "error": r.get("error", "修正段无输出"),
                               "seconds": round(time.time() - t1, 3)})
                self._log("    层D 模型修正失败，保留上一版答案")
                break
            new = _THINK_TAG.sub("", new)
            new = _NO_THINK.sub("", new).strip()
            new, _ = _fc.normalize_headings(new)
            if REQUIRED_SECTIONS[2] not in _fc.section_map(new) and reference_list:
                new += f"\n\n## {REQUIRED_SECTIONS[2]}\n" + reference_list
            if self.add_disclaimer and "不构成临床诊疗建议" not in new:
                new += "\n\n" + DISCLAIMER
            applied = ["模型按违规清单重写"]
            if self.deterministic_fix:
                rep_tmp = _check(new)
                new2, ap2 = self.checker.auto_fix(new, rep_tmp, citations, reference_list)
                if ap2:
                    new, applied = new2, applied + ap2
            rep2 = _check(new)
            after = len([v for v in rep2["violations"] if v["severity"] in ("high", "medium")])
            rounds.append({"round": n, "mode": "llm", "applied": applied,
                           "before": before, "after": after, "compliant": rep2["compliant"],
                           "seconds": round(time.time() - t1, 3)})
            self._log(f"    层D 模型修正第 {n} 轮 → 违规 {before}→{after} "
                      f"{'（已合规）' if rep2['compliant'] else ''}")
            # 修坏了就退回上一版：修正只应减少违规，增加了说明这一轮是负作用
            if after > before:
                self._log("    层D 修正后违规反而变多，回退到修正前的版本")
                rounds[-1]["rolled_back"] = True
                break
            answer, report = new, rep2

        repair = {"rounds": rounds, "llm_rounds": n,
                  "initial": first,
                  "final_compliant": report["compliant"],
                  "seconds": round(time.time() - t0, 3)}
        return answer, report, repair

    # ---------------- 覆写：无证据时的拒答话术 ----------------
    def _assemble_result(self, query: str, answer: str, ctx: Dict[str, Any],
                         metrics: Dict[str, Any], inter: Dict[str, Any], t_start: float,
                         post: Optional[Dict[str, Any]] = None, no_evidence: bool = False,
                         **kwargs: Any) -> Dict[str, Any]:
        """无证据（检索为空 / 草稿为空）时，换成**带固定拒答短语的结构化回答**。

        阶段七这里给的是一段"建议换检索词"的话术，里面**没有那句固定短语**——统计拒答率时
        这一类会被算成"没拒答"，而它其实是最该拒答的一类。
        """
        if no_evidence:
            refs = ContextAssembler.render_reference_list(ctx.get("selected_chunks") or [],
                                                          ctx["metadata"].get("citations"))
            reason = ("本次检索没有返回任何可用的文献片段。" if not ctx.get("selected_chunks")
                      else "模型未能基于检索到的文献片段产出可用答案。")
            answer = _cp.no_evidence_answer(reason, refs)
            st = self._st()
            st["check"] = self.checker.check(answer, citations=ctx["metadata"].get("citations"),
                                             evidence_text="", question=query,
                                             expect_refusal=st.get("expect_refusal"),
                                             reference_list=refs)
            st["repair"] = {"rounds": [], "llm_rounds": 0, "final_compliant": st["check"]["compliant"],
                            "initial": {"compliant": st["check"]["compliant"], "violations": []},
                            "seconds": 0.0, "note": "无证据路径：答案由代码产出，未调用模型"}
        res = super()._assemble_result(query, answer, ctx, metrics, inter, t_start,
                                       post=post, no_evidence=no_evidence, **kwargs)
        st = self._st()
        if st.get("check") is not None:
            res["constraint_check"] = st["check"]
            res["constraint_repair"] = st["repair"]
        return res


# ============================================================================
# CLI 演示
# ============================================================================
def _demo(args: argparse.Namespace) -> int:
    asm_mod = _load_by_path("shengcheng_zuzhuang", "生成_上下文组装.py")
    docs = asm_mod.load_sample_chunks(limit=args.limit, keyword=args.keyword)
    if not docs:
        print(f"样例块里没有含 {args.keyword!r} 的内容，换个 --keyword")
        return 1
    print(f"载入样例证据 {len(docs)} 条")

    gen = LLMGenerator(model_name=args.model, num_ctx=_pipe_mod.RECOMMENDED_NUM_CTX,
                       verbose=True)
    layer_d = not args.no_layer_d
    pipe = ConstrainedGenerationPipeline(
        assembler=ContextAssembler(max_context_tokens=args.budget, verbose=True),
        generator=gen, max_repair_rounds=args.repair_rounds if layer_d else 0,
        deterministic_fix=layer_d,
        constrained_prompts=not args.baseline, verbose=True)
    print(f"提示词：{'阶段七基线（未加约束）' if args.baseline else '阶段九受约束'}"
          f"｜层 D：{'开（确定性修正 + 模型修正上限 %d 轮）' % args.repair_rounds if layer_d else '全关'}")
    if args.baseline and layer_d:
        print("  ⚠ 注意：这里只关了提示词约束，层 D 仍在工作（走样的标题会被改回来）。"
              "对抗测试里的 baseline 组是 --baseline --no-layer-d，两者不可混为一谈。")
    res = pipe.generate(args.query, retrieved_docs=docs, expect_refusal=args.expect_refusal)

    print("\n" + "=" * 92)
    print(res["answer"])
    print("=" * 92)
    print(_fc.format_report(res["constraint_check"]))
    print(f"\n修正过程：{json.dumps(res['constraint_repair'], ensure_ascii=False)[:600]}")
    gm = res["generation_metrics"]
    print(f"耗时 {gm['total_time_seconds']}s | 模型调用 {gm['llm_calls']} 次 | "
          f"分阶段 {gm['stage_times']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="用阶段七未加约束的提示词跑（**只关提示词，层 D 仍开着**）")
    ap.add_argument("--no-layer-d", dest="no_layer_d", action="store_true",
                    help="关掉生成后校验与修正，只校验不修；"
                         "配 --baseline 才等于对抗测试里的 baseline 组")
    ap.add_argument("--query", default="What does the evidence say about the role of "
                                       "inflammation in disease progression?")
    ap.add_argument("--keyword", default="inflammation")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--repair-rounds", type=int, default=1)
    ap.add_argument("--expect-refusal", dest="expect_refusal", action="store_true", default=None)
    ap.add_argument("--model", default="qwen3:8b")
    args = ap.parse_args()
    if not args.demo:
        ap.print_help()
        return 0
    return _demo(args)


if __name__ == "__main__":
    sys.exit(main())
