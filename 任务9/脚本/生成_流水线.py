# -*- coding: utf-8 -*-
"""第七阶段（二）· 医学生成流水线：把检索、上下文组装、四段提示词、本地 LLM 串成一条链

    查询 ─▶ [检索]* ─▶ 上下文组装 ─▶ ①证据评估 ─▶ 按评估筛证据 ─▶ ②生成草稿
         ─▶ ③批判审查 ─▶ ④按审查定稿 ─▶ 后处理（引用校验/格式/免责声明）─▶ 结果
    （*检索为可选：传 retrieved_docs 就跳过，便于离线验证与复跑）

为什么①③可以关：这两段是"用 3~4 倍耗时换忠于证据"的赌注，值不值得必须实测而不是断言。
`generate(evaluate=..., review=...)` 就是消融开关：

    evaluate=False, review=False  →  1 次模型调用（检索 + 直接作答，对照组基线）
    evaluate=True,  review=False  →  2 次
    evaluate=True,  review=True   →  4 次（完整四段链）

关掉审查时**不跑④**，直接拿草稿当答案——④的输入里"审查意见"是必填变量，没有审查就让它
空转只会多烧一次模型且无从修订。这一条与任务书"如果成功审查，使用审查结果；如果没有审查，
直接使用草稿"一致。

设计上的两个关键取舍：

  1. **筛证据时不重新编号**。①判定某条证据不相关后，它的块被移出上下文，但**其余块保留
     原有的 [S#] 标号**。重新编号会让 `metadata["citations"]` 里的映射全部失效，也让复跑
     日志对不上；代价只是最终答案里的编号可能不连续（[S1][S3][S4]），这完全可接受。
  2. **参考文献列表由代码生成，不让模型写**。阶段一压测裸模型时最典型的错就是编造参考
     文献（标题、期刊、卷页全错）。④的提示词里把列表作为既定事实给它并要求原样使用，
     后处理再校验一遍——模型没有任何机会自己拼一条出来。

用法：
    import importlib.util
    spec = importlib.util.spec_from_file_location("gp", r"E:\\rag\\scripts\\生成_流水线.py")
    gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)

    pipe = gp.MedicalGenerationPipeline()                    # 不带检索器（离线）
    res  = pipe.generate("问题", retrieved_docs=docs)         # docs 为阶段六 Candidate 列表
    print(res["answer"]); print(res["generation_metrics"])

    # 带检索的完整链路（首次加载约 15.8GB RSS + 数分钟）
    pipe = gp.MedicalGenerationPipeline(retriever=RetrievalPipeline(bm25_dir=...))
    res  = pipe.generate("RCT evidence for pembrolizumab in NSCLC since 2020", top_k=10)

CLI（离线用阶段三样例块跑一次完整链路，不加载 65GB 向量库）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_流水线.py --demo
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
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(mod_name: str, filename: str):
    """中文文件名模块按路径导入（与阶段五、六、七之一一致的做法）。"""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_asm_mod = _load_by_path("shengcheng_zuzhuang", "生成_上下文组装.py")
_tpl_mod = _load_by_path("shengcheng_tishici", "生成_提示词模板.py")
_llm_mod = _load_by_path("shengcheng_llm", "生成_LLM生成器.py")

ContextAssembler = _asm_mod.ContextAssembler
DocumentChunk = _asm_mod.DocumentChunk
MedicalPromptTemplates = _tpl_mod.MedicalPromptTemplates
PROMPT_STAGES = _tpl_mod.PROMPT_STAGES
NO_EVIDENCE_NOTICE = _tpl_mod.NO_EVIDENCE_NOTICE
RECOMMENDED_NUM_CTX = _tpl_mod.RECOMMENDED_NUM_CTX
LLMGenerator = _llm_mod.LLMGenerator
LLMError = _llm_mod.LLMError

DISCLAIMER = ("> ⚠ 本回答由检索到的公开文献自动生成，仅供专业文献检索参考，"
              "不构成临床诊疗建议；具体用药与处置须由临床医师结合患者情况判断。")

#: 出处标记，如 [S1]、[S12]
_MARKER = re.compile(r"\[(S\d+)\]")
#: 模型偶尔会把提示词里的软开关原样吐出来
_NO_THINK = re.compile(r"/no_?think\s*", re.IGNORECASE)
_THINK_TAG = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


# ============================================================================
# 一、上下文分块与筛选
# ============================================================================
def split_context_blocks(context_text: str, markers: Sequence[str],
                         separator: str = "\n\n") -> Dict[str, str]:
    """把组装好的上下文按出处头切回 {marker: 块文本}。

    不直接 `split(separator)`：证据正文本身可能含空行，按分隔符切会把一块劈成两半。
    改为**顺序锚定** —— markers 在上下文里按 S1、S2… 的顺序出现且唯一，逐个向后找
    `[S#] ` 前缀即可精确定位每块的起点。
    """
    starts: List[Tuple[str, int]] = []
    cursor = 0
    for m in markers:
        tag = f"[{m}] "
        idx = context_text.find(tag, cursor)
        if idx < 0:
            continue
        starts.append((m, idx))
        cursor = idx + len(tag)
    blocks: Dict[str, str] = {}
    for i, (m, pos) in enumerate(starts):
        end = starts[i + 1][1] - len(separator) if i + 1 < len(starts) else len(context_text)
        blocks[m] = context_text[pos:max(pos, end)].strip()
    return blocks


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def parse_evaluation(obj: Any) -> List[Dict[str, Any]]:
    """把①的输出规整成统一的评估记录列表，容忍模型返回的几种常见形状。

    见过的形状：裸数组、{"evaluations": [...]}、{"results": [...]}、单个对象、
    以及 {"S1": {...}, "S2": {...}} 这种以编号为键的字典。
    """
    rows: List[Any] = []
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        for key in ("evaluations", "results", "evidence", "items", "data"):
            if isinstance(obj.get(key), list):
                rows = obj[key]
                break
        else:
            if any(re.fullmatch(r"S\d+", str(k)) for k in obj):
                rows = [dict(v, marker=k) if isinstance(v, dict) else {"marker": k}
                        for k, v in obj.items()]
            elif "marker" in obj:
                rows = [obj]
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        marker = str(r.get("marker") or r.get("id") or r.get("source") or "").strip()
        m = _MARKER.search(marker) or re.search(r"S\d+", marker)
        if not m:
            continue
        out.append({
            "marker": m.group(0) if m.lastindex is None else m.group(1),
            "relevance": max(0, min(3, _as_int(r.get("relevance"), 0))),
            "study_type": str(r.get("study_type") or "unclear").strip().lower(),
            "directly_answers": bool(r.get("directly_answers")),
            "key_finding": str(r.get("key_finding") or "").strip(),
            "caveats": str(r.get("caveats") or "").strip(),
        })
    return out


# ============================================================================
# 二、流水线
# ============================================================================
class MedicalGenerationPipeline:
    """医学 RAG 生成流水线：检索结果 → 带出处、经审查的最终回答。

    Args:
        retriever:  阶段六的 `RetrievalPipeline`（有 `.search(query, top_k)` 即可）。
                    None 时必须由调用方传 `retrieved_docs`。
        assembler:  `ContextAssembler`，None 则按 max_context_tokens 新建
        templates:  `MedicalPromptTemplates`，None 则用默认四段
        generator:  `LLMGenerator`，None 则按 model_name/base_url 新建
        min_relevance_keep: ①评估后保留证据的相关性下限（默认 1 = 只丢"完全不相关"）
        require_citation: 后处理时校验答案里的 [S#] 是否都真实存在
    """

    def __init__(self,
                 retriever: Any = None,
                 assembler: Any = None,
                 templates: Any = None,
                 generator: Any = None,
                 max_context_tokens: int = 2800,
                 model_name: str = "qwen3:8b",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 180,
                 num_ctx: int = RECOMMENDED_NUM_CTX,
                 min_relevance_keep: int = 1,
                 require_citation: bool = True,
                 add_disclaimer: bool = True,
                 verbose: bool = True):
        self.retriever = retriever
        self.assembler = assembler or ContextAssembler(max_context_tokens=max_context_tokens,
                                                       verbose=verbose)
        self.templates = templates or MedicalPromptTemplates()
        self.generator = generator or LLMGenerator(model_name=model_name, base_url=base_url,
                                                   timeout=timeout, num_ctx=num_ctx,
                                                   verbose=verbose)
        self.min_relevance_keep = int(min_relevance_keep)
        self.require_citation = bool(require_citation)
        self.add_disclaimer = bool(add_disclaimer)
        self.verbose = verbose

    # ---------------- 小工具 ----------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _run_stage(self, key: str, values: Dict[str, Any],
                   metrics: Dict[str, Any]) -> Dict[str, Any]:
        """跑一段提示词：渲染 → 调模型 → 记录耗时与 token 用量。"""
        stage = self.templates.get(key)
        rendered = self.templates.render(key, **values)
        want_json = stage.output_format == "json"
        t0 = time.time()
        try:
            r = self.generator.generate_messages(
                rendered["messages"],
                temperature=stage.temperature,
                max_tokens=stage.max_tokens,
                json_output=want_json,
                expect="array" if key == "evidence_evaluator" else "object",
            )
            r["ok"] = True
        except (LLMError, ValueError) as e:
            r = {"ok": False, "error": str(e), "text": "", "json": None, "json_ok": False,
                 "json_error": str(e), "elapsed": round(time.time() - t0, 3),
                 "prompt_eval_count": None, "eval_count": None, "truncated": False}
        metrics["stage_times"][key] = round(time.time() - t0, 3)
        metrics["token_counts"][key] = {"prompt": r.get("prompt_eval_count"),
                                        "output": r.get("eval_count"),
                                        "truncated": bool(r.get("truncated"))}
        self._log(f"  · {key:<20} {metrics['stage_times'][key]:>6.2f}s  "
                  f"in={r.get('prompt_eval_count')} out={r.get('eval_count')}"
                  + ("" if r.get("ok") else f"  ✗ {r.get('error', '')[:80]}")
                  + ("" if not (want_json and r.get('ok')) else
                     f"  json={'ok' if r.get('json_ok') else '解析失败: ' + str(r.get('json_error'))[:60]}"))
        return r

    # ---------------- 证据评估结果 → 摘要与筛选 ----------------
    @staticmethod
    def format_evidence_summary(rows: List[Dict[str, Any]]) -> str:
        """把①的判定压成几行给②看。带上局限，让②写作时自己降调。"""
        if not rows:
            return "（本次未做证据评估）"
        lines = []
        for r in sorted(rows, key=lambda x: (-x["relevance"], x["marker"])):
            bits = [f"[{r['marker']}] 相关性{r['relevance']}/3", r["study_type"]]
            if r["directly_answers"]:
                bits.append("直接回答问题")
            if r["key_finding"]:
                bits.append(f"要点：{r['key_finding']}")
            if r["caveats"]:
                bits.append(f"局限：{r['caveats']}")
            lines.append(" · ".join(bits))
        return "\n".join(lines)

    def filter_context_by_evaluation(self, context_text: str, citations: List[Dict[str, Any]],
                                     rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """按①的相关性判定裁掉噪声证据块。**保留原有编号，不重新编号。**

        评估里没提到的编号一律保留——模型漏评一条，不该等同于判它不相关。
        """
        markers = [c["marker"] for c in citations]
        blocks = split_context_blocks(context_text, markers, self.assembler.block_separator)
        scored = {r["marker"]: r["relevance"] for r in rows}
        kept, dropped = [], []
        for m in markers:
            if m not in blocks:                       # 切不出来就保守保留原样
                continue
            if m in scored and scored[m] < self.min_relevance_keep:
                dropped.append(m)
            else:
                kept.append(m)
        if not kept:                                  # 全被判不相关：不筛，交给②如实说证据不足
            return {"context_text": context_text, "kept": markers, "dropped": [],
                    "all_dropped": True}
        text = self.assembler.block_separator.join(blocks[m] for m in kept)
        return {"context_text": text, "kept": kept, "dropped": dropped, "all_dropped": False}

    # ---------------- 后处理 ----------------
    def postprocess(self, answer: str, citations: List[Dict[str, Any]],
                    reference_list: str) -> Dict[str, Any]:
        """引用校验 + 格式美化 + 参考文献 + 免责声明。

        引用校验是这一步的重点：把答案里出现的 [S#] 与真实存在的编号对一遍，**编造的编号
        直接标出来**。这正是阶段一裸模型四类错里"编造参考文献"那一类的兜底防线。
        """
        text = _THINK_TAG.sub("", answer or "")
        text = _NO_THINK.sub("", text).strip()
        # 统一标题层级与空行，避免模型有时给 #、有时给 ###
        text = re.sub(r"^#{1,6}\s*(回答|证据要点|证据强度与一致性|局限与提示|参考文献)\s*$",
                      r"## \1", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)

        valid = {c["marker"] for c in citations}
        used = set(_MARKER.findall(text))
        fabricated = sorted(used - valid, key=lambda s: _as_int(s[1:]))
        if fabricated:
            text += ("\n\n> ⚠ 校验提示：回答中出现了证据列表里不存在的出处编号 "
                     + "、".join(f"[{m}]" for m in fabricated)
                     + "，这些标注不可信，请以下方参考文献为准。")

        if "## 参考文献" not in text and reference_list:
            text += "\n\n## 参考文献\n" + reference_list
        if self.add_disclaimer and "不构成临床诊疗建议" not in text:
            text += "\n\n" + DISCLAIMER
        return {"answer": text.strip(),
                "citations_used": sorted(used & valid, key=lambda s: _as_int(s[1:])),
                "citations_available": sorted(valid, key=lambda s: _as_int(s[1:])),
                "fabricated_markers": fabricated,
                "has_citation": bool(used & valid)}

    @staticmethod
    def _format_sources(selected_chunks: List[Any],
                        citations: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """最终结果里的可溯源来源清单：编号 → 文献标识 + 可点开的 PMC 链接。"""
        cits = citations or [c.citation(f"S{i}") for i, c in enumerate(selected_chunks, 1)]
        out = []
        for c in cits:
            pmcid = c.get("pmcid") or ""
            out.append({
                "marker": c["marker"], "pmcid": pmcid, "pmid": c.get("pmid") or "",
                "title": c.get("title") or "", "journal": c.get("journal") or "",
                "pub_year": c.get("pub_year"), "section": c.get("section") or "",
                "chunk_id": c.get("chunk_id") or "",
                "relevance_score": c.get("relevance_score"),
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else "",
            })
        return out

    # ---------------- 主入口 ----------------
    def generate(self,
                 query: str,
                 retrieved_docs: Optional[Sequence[Any]] = None,
                 top_k: int = 10,
                 evaluate: bool = True,
                 review: bool = True,
                 max_context_tokens: Optional[int] = None) -> Dict[str, Any]:
        """完整生成流程。返回结构见模块 docstring 与任务书。"""
        t_start = time.time()
        metrics: Dict[str, Any] = {"stage_times": {}, "token_counts": {}, "stage_success": {}}
        inter: Dict[str, Any] = {"evidence_evaluation": None, "draft_answer": None,
                                 "review_feedback": None}
        self._log(f"\n▶ 查询：{query}")

        # ---- 0) 检索（可选）----
        if retrieved_docs is None:
            if self.retriever is None:
                raise ValueError("没有检索器，必须传 retrieved_docs")
            t0 = time.time()
            out = self.retriever.search(query, top_k=top_k)
            retrieved_docs = out["results"]
            # 真正拿去检索的那句话（中文问题在这里被译成英文）。不记下来的话，
            # "库里真没有证据"与"问题被翻译坏了"在响应里长得一模一样，没法区分。
            eq = out.get("enhanced")
            if eq is not None:
                metrics["retrieval_query"] = getattr(eq, "core_text", None)
                metrics["translated_from"] = getattr(eq, "translated_from", None)
            metrics["stage_times"]["retrieval"] = round(time.time() - t0, 3)
            metrics["stage_success"]["retrieval"] = bool(retrieved_docs)
            self._log(f"  · {'retrieval':<20} {metrics['stage_times']['retrieval']:>6.2f}s  "
                      f"命中 {len(retrieved_docs)} 条")

        # ---- 1) 上下文组装 ----
        t0 = time.time()
        ctx = self.assembler.assemble_context(retrieved_docs, query=query,
                                              max_tokens=max_context_tokens)
        metrics["stage_times"]["context_assembly"] = round(time.time() - t0, 3)
        citations = ctx["metadata"]["citations"]
        metrics["stage_success"]["context_assembly"] = bool(ctx["selected_chunks"])
        reference_list = ContextAssembler.render_reference_list(ctx["selected_chunks"], citations)

        # 检索为空 / 全部被丢弃：不让模型自由发挥，直接走固定话术
        if not ctx["selected_chunks"]:
            self._log("  · 无可用证据，返回固定话术（不调用模型）")
            return self._assemble_result(query, NO_EVIDENCE_NOTICE, ctx, metrics, inter,
                                         t_start, no_evidence=True)

        context_text = ctx["context_text"]

        # ---- 2) 证据评估（可选）----
        eval_rows: List[Dict[str, Any]] = []
        filter_info: Dict[str, Any] = {}
        if evaluate:
            r = self._run_stage("evidence_evaluator",
                                {"question": query, "context": context_text}, metrics)
            ok = bool(r.get("ok") and r.get("json_ok"))
            metrics["stage_success"]["evidence_evaluator"] = ok
            if ok:
                eval_rows = parse_evaluation(r["json"])
                filter_info = self.filter_context_by_evaluation(context_text, citations, eval_rows)
                context_text = filter_info["context_text"]
                self._log(f"    评估 {len(eval_rows)} 条 → 保留 {len(filter_info['kept'])}，"
                          f"丢弃 {len(filter_info['dropped'])} "
                          f"{filter_info['dropped'] if filter_info['dropped'] else ''}")
            else:
                # 评估失败不阻断链路：退化成"不筛选"，②照样能基于全部证据作答
                self._log("    评估不可用，跳过筛选，用全部证据继续")
            inter["evidence_evaluation"] = {
                "parsed": eval_rows, "raw_text": r.get("text", "")[:2000],
                "json_ok": bool(r.get("json_ok")), "json_error": r.get("json_error", ""),
                "filter": filter_info,
            }
        else:
            metrics["stage_success"]["evidence_evaluator"] = None

        evidence_summary = self.format_evidence_summary(eval_rows)

        # ---- 3) 答案草稿 ----
        r2 = self._run_stage("answer_generator",
                             {"question": query, "context": context_text,
                              "evidence_summary": evidence_summary}, metrics)
        draft = (r2.get("text") or "").strip()
        metrics["stage_success"]["answer_generator"] = bool(r2.get("ok") and draft)
        inter["draft_answer"] = draft
        if not draft:
            self._log("  · 草稿为空，链路终止")
            return self._assemble_result(query, NO_EVIDENCE_NOTICE, ctx, metrics, inter,
                                         t_start, no_evidence=True)

        # ---- 4) 批判性审查（可选）----
        review_obj: Optional[Dict[str, Any]] = None
        review_text = ""
        if review:
            r3 = self._run_stage("critical_reviewer",
                                 {"question": query, "context": context_text,
                                  "draft_answer": draft}, metrics)
            ok = bool(r3.get("ok") and r3.get("json_ok") and isinstance(r3.get("json"), dict))
            metrics["stage_success"]["critical_reviewer"] = ok
            if ok:
                review_obj = r3["json"]
                review_text = json.dumps(review_obj, ensure_ascii=False, indent=2)
                n_issues = len(review_obj.get("issues") or [])
                self._log(f"    审查：verdict={review_obj.get('verdict')} "
                          f"问题 {n_issues} 条 grounded={review_obj.get('overall_grounded')}")
            else:
                self._log("    审查不可用，直接采用草稿")
            inter["review_feedback"] = {
                "parsed": review_obj, "raw_text": r3.get("text", "")[:2000],
                "json_ok": bool(r3.get("json_ok")), "json_error": r3.get("json_error", ""),
            }
        else:
            metrics["stage_success"]["critical_reviewer"] = None

        # ---- 5) 最终答案 ----
        # 有可用审查意见才跑④；没有就直接拿草稿——④缺了"审查意见"这个必填变量，
        # 空转一次既多烧一次模型，也无从据以修订。
        if review_obj is not None:
            r4 = self._run_stage("final_assembler",
                                 {"question": query, "context": context_text,
                                  "draft_answer": draft, "review": review_text,
                                  "reference_list": reference_list}, metrics)
            final = (r4.get("text") or "").strip()
            metrics["stage_success"]["final_assembler"] = bool(r4.get("ok") and final)
            if not final:
                self._log("    定稿为空，回退到草稿")
                final = draft
        else:
            metrics["stage_success"]["final_assembler"] = None
            final = draft

        # ---- 6) 后处理 ----
        t0 = time.time()
        post = self.postprocess(final, citations, reference_list)
        metrics["stage_times"]["postprocess"] = round(time.time() - t0, 3)
        metrics["stage_success"]["postprocess"] = (post["has_citation"] or not self.require_citation)
        if post["fabricated_markers"]:
            self._log(f"    ⚠ 发现编造的出处编号：{post['fabricated_markers']}")
        self._log(f"  · {'postprocess':<20} {metrics['stage_times']['postprocess']:>6.2f}s  "
                  f"引用 {len(post['citations_used'])}/{len(post['citations_available'])}")

        return self._assemble_result(query, post["answer"], ctx, metrics, inter, t_start,
                                     post=post, evaluate=evaluate, review=review)

    # ---------------- 结果组装 ----------------
    def _assemble_result(self, query: str, answer: str, ctx: Dict[str, Any],
                         metrics: Dict[str, Any], inter: Dict[str, Any], t_start: float,
                         post: Optional[Dict[str, Any]] = None, no_evidence: bool = False,
                         evaluate: bool = True, review: bool = True) -> Dict[str, Any]:
        total = round(time.time() - t_start, 3)
        tk = metrics["token_counts"]
        metrics["total_time_seconds"] = total
        metrics["llm_calls"] = len(tk)
        metrics["total_prompt_tokens"] = sum(v["prompt"] or 0 for v in tk.values())
        metrics["total_output_tokens"] = sum(v["output"] or 0 for v in tk.values())
        metrics["context_tokens"] = ctx["metadata"]["estimated_tokens"]
        metrics["answer_chars"] = len(answer)
        metrics["chain"] = {"evaluate": evaluate, "review": review, "no_evidence": no_evidence}

        result = {
            "query": query,
            "answer": answer,
            "context_metadata": ctx["metadata"],
            "generation_metrics": {
                "total_time_seconds": total,
                "stage_times": metrics["stage_times"],
                "token_counts": tk,
                "stage_success": metrics["stage_success"],
                "llm_calls": metrics["llm_calls"],
                "total_prompt_tokens": metrics["total_prompt_tokens"],
                "total_output_tokens": metrics["total_output_tokens"],
                "context_tokens": metrics["context_tokens"],
                "answer_chars": metrics["answer_chars"],
                "chain": metrics["chain"],
                # 真正送进检索层的那句话；离线（传 retrieved_docs）时为 None
                "retrieval_query": metrics.get("retrieval_query"),
                "translated_from": metrics.get("translated_from"),
            },
            "intermediate_results": inter,
            "sources": self._format_sources(ctx["selected_chunks"], ctx["metadata"]["citations"]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if post:
            result["citation_check"] = {
                "used": post["citations_used"], "available": post["citations_available"],
                "fabricated": post["fabricated_markers"], "has_citation": post["has_citation"],
            }
        self._log(f"  ✓ 共 {total:.2f}s | {metrics['llm_calls']} 次模型调用 | "
                  f"答案 {metrics['answer_chars']} 字")
        return result


# ============================================================================
# CLI 演示：离线用阶段三样例块跑完整链路
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用阶段三样例块跑一次完整四段链")
    ap.add_argument("--query", default="What does the evidence say about the role of "
                                       "inflammation in disease progression?")
    ap.add_argument("--keyword", default="inflammation", help="从样例块里挑含该词的块当证据")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--no-evaluate", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--model", default="qwen3:8b")
    args = ap.parse_args()

    if not args.demo:
        ap.print_help()
        return 0

    docs = _asm_mod.load_sample_chunks(limit=args.limit, keyword=args.keyword)
    if not docs:
        print(f"样例块里没有含 {args.keyword!r} 的内容，换个 --keyword")
        return 1
    print(f"载入样例证据 {len(docs)} 条")

    pipe = MedicalGenerationPipeline(max_context_tokens=args.budget, model_name=args.model,
                                     verbose=True)
    res = pipe.generate(args.query, retrieved_docs=docs,
                        evaluate=not args.no_evaluate, review=not args.no_review)

    print("\n" + "=" * 90)
    print(res["answer"])
    print("=" * 90)
    gm = res["generation_metrics"]
    print(f"\n耗时 {gm['total_time_seconds']}s | 模型调用 {gm['llm_calls']} 次 | "
          f"上下文 {gm['context_tokens']} tok | 答案 {gm['answer_chars']} 字")
    print(f"分阶段耗时：{gm['stage_times']}")
    print(f"阶段成功：  {gm['stage_success']}")
    print(f"引用校验：  {res.get('citation_check')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
