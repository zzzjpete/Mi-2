# -*- coding: utf-8 -*-
"""逐阶段核对「任务书 → README 对照表 → 真实代码」这条链是否闭合。

各阶段 README 里的「对照任务书」表是交付评审的核心凭据：每行写"任务书要这个 → 我这样实现 →
在这个位置"。但表是手写的，代码会改名、会挪位置，**表和代码会悄悄漂**。这个脚本把三件事
机械地查一遍，避免"看一眼觉得对"就下结论：

  A. 表里声称的**实现位置真的存在**：文件在包内，符号（类/函数/字典键）在那个文件里
  B. 表里每一行都有**明确的实现标记**（✅/⚠/❌），不允许空着
  C. 任务书原文里的**关键要求名**（类名、函数名、字段名等可机械识别的标识符）
     在对照表里被提到过 —— 抓"整条要求被漏掉"的情况

C 组必然有噪声（任务书里的示例代码片段、被刻意否掉的方案都会被标出来），所以它输出的是
**待人工确认清单**，不算失败项；A、B 两组才是硬判定。

用法：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\校验任务书对照.py
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\校验任务书对照.py --stage 6 -v
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import argparse
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = os.path.join(ROOT, "scripts")
STAGES = {
    1: "一 环境准备 + 本地 LLM 验证",
    2: "二 数据尽调 + 切块策略设计",
    3: "三 文档解析与切块",
    4: "四 向量化与索引构建",
    5: "五 查询理解与增强",
    6: "六 多路检索 + 多准则重排",
    7: "七 生成层（上下文组装 / 提示词 / LLM 流水线）",
    8: "八 答案评估 + 缓存策略 + 批量处理",
    9: "九 强约束规则开发与幻觉抑制",
}

#: 对照表所在小节的标题（各阶段措辞不完全一致，用前缀匹配）
SECTION_RE = re.compile(r"^##\s*一(?:之二)?[、.]\s*对照任务书", re.M)
#: 表格行。**两列（要求|实现）和三列（要求|实现|位置）都要认**——
#: 阶段一/三/四是"做了什么"，没有"实现在哪个类"可指，对照表就只有两列；五/六/七才有位置列。
ROW_RE = re.compile(r"^\|(?!\s*[-:]+\s*\|)([^|\n]*)\|([^|\n]*?)(?:\|([^|\n]*))?\|\s*$", re.M)
#: 位置单元格里的 `文件.py :: 符号` / `符号`
CODE_RE = re.compile(r"`([^`]+)`")
FILE_RE = re.compile(r"([\w\u4e00-\u9fff\-]+\.(?:py|ps1|sh))")
#: 任务书里可机械识别的标识符：驼峰类名、带下划线的函数名、形如 xxx_yyy 的字段
IDENT_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{4,}|[a-z][a-z0-9]+_[a-z0-9_]{2,})\b")
#: 实现标记
MARK_RE = re.compile(r"[✅⚠❌]")

#: 任务书里出现但**不该**当成"要实现的东西"的标识符（示例代码变量、被否掉的方案、通用词）
IDENT_IGNORE = {
    # python/示例代码里的通用名
    "load_dataset", "pd_timestamp", "self_count", "chunk_data", "df_raw", "chunks_df",
    "output_path", "data_split", "full_text", "base_query", "query_text", "n_results",
    "where_filter", "embedding_model", "processed_date", "original_documents",
    "output_file", "batch_size", "model_type", "chunk_size_stats", "metadata_fields",
    "index_built_at", "collection_name", "total_chunks", "embedding_dimension",
    "criteria_weights", "query_info", "top_k_vector", "top_k_keyword", "fusion_strategy",
    "retrieved_docs", "unique_chunks", "selected_chunks", "final_context",
    "context_metadata", "context_text", "estimated_tokens", "chunk_sources",
    "context_result", "generation_metrics", "intermediate_results", "total_time_seconds",
    "stage_times", "token_counts", "stage_success", "evidence_evaluation",
    "draft_answer", "review_feedback", "system_prompt", "user_prompt_template",
    "max_tokens", "model_name", "base_url", "relevance_score", "chunk_id", "doc_id",
    "chunk_index", "source_title", "token_count", "chunks_per_doc", "chunk_overlap",
    "total_chunks_retrieved", "unique_chunks_after_dedup", "chunks_selected",
    "from_pretrained", "text_embedding", "pub_date", "myocardial_infarction",
    # 模型/产品名，不是要实现的符号
    "DeepSeek", "Qwen3", "Gemma", "Llama", "Claude", "Anthropic", "Instruct", "Chroma",
    "LEANN", "LangChain", "PubMed", "Ollama", "huggingface", "chromadb", "datasets",
    "clinicalBERT", "OpenAI", "Coder", "MiMo", "Parquet", "JSONL", "ChromaDB",
    "IMRaD", "CONCLUSIONS", "METHODS", "EGFR", "Nature", "UMLS", "MeSH",
}


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_table(readme_text):
    """抽出所有「对照任务书」小节里的表格行。"""
    rows = []
    for m in SECTION_RE.finditer(readme_text):
        # 该小节到下一个 ## 标题为止
        start = m.end()
        nxt = re.search(r"^##\s", readme_text[start:], re.M)
        body = readme_text[start:start + nxt.start()] if nxt else readme_text[start:]
        for r in ROW_RE.finditer(body):
            req = (r.group(1) or "").strip()
            impl = (r.group(2) or "").strip()
            loc = (r.group(3) or "").strip()
            if not req:
                continue
            # 表头行：紧跟其后的是 |---|---| 分隔行。一个小节里可能有多张表，
            # 所以不能只按"第一行"或固定文案判，得看下一行是不是分隔符。
            after = body[r.end():].lstrip("\n")
            if re.match(r"\|\s*[-:]+\s*\|", after):
                continue
            rows.append({"req": req, "impl": impl, "loc": loc})
    return rows


def symbols_in(text):
    """一个 py 文件里定义了哪些可被引用的名字：类、函数、以及顶层字典的字符串键。"""
    names = set(re.findall(r"^\s*(?:async\s+)?def\s+(\w+)", text, re.M))
    names |= set(re.findall(r"^\s*class\s+(\w+)", text, re.M))
    names |= set(re.findall(r"^([A-Z_][A-Z0-9_]+)\s*[:=]", text, re.M))     # 常量
    names |= set(re.findall(r"^\s{4}\"(\w+)\"\s*:", text, re.M))            # 字典键（缩进4空格）
    names |= set(re.findall(r"\"(\w+)\"\s*:\s*PromptStage", text))          # 提示词阶段键
    return names


def check_stage(n, verbose=False):
    pkg = os.path.join(ROOT, f"任务{n}")
    readme_p = os.path.join(pkg, "README.md")
    task_p = os.path.join(pkg, "任务书.txt")
    out = {"stage": n, "problems": [], "notes": [], "rows": 0, "loc_ok": 0,
           "loc_checked": 0, "unmarked": 0}

    if not os.path.exists(readme_p):
        out["problems"].append("缺 README.md")
        return out
    if not os.path.exists(task_p):
        out["problems"].append("缺 任务书.txt")
        return out

    readme = read(readme_p)
    task = read(task_p)
    rows = parse_table(readme)
    out["rows"] = len(rows)
    if not rows:
        out["problems"].append("README 里找不到「对照任务书」表格")
        return out

    # ---- 收集包内脚本 ----
    script_dir = os.path.join(pkg, "脚本")
    files = {}
    for d in (script_dir, SRC):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith((".py", ".ps1", ".sh")) and fn not in files:
                    files[fn] = os.path.join(d, fn)
    sym_cache = {}

    def syms(fn):
        if fn not in sym_cache:
            sym_cache[fn] = symbols_in(read(files[fn])) if fn.endswith(".py") else set()
        return sym_cache[fn]

    # ---- A/B：逐行核对 ----
    cur_file = None
    for row in rows:
        if not MARK_RE.search(row["impl"]):
            out["unmarked"] += 1
            out["problems"].append(f"行「{row['req'][:36]}」的实现列没有 ✅/⚠/❌ 标记")
        for code in CODE_RE.findall(row["loc"]):
            fm = FILE_RE.search(code)
            if fm:
                cur_file = fm.group(1)
                out["loc_checked"] += 1
                if cur_file not in files:
                    out["problems"].append(f"位置里的文件不存在：{cur_file}"
                                           f"（行「{row['req'][:30]}」）")
                    continue
                out["loc_ok"] += 1
            # 符号部分。整格就是个文件名时不再取符号——否则 `xxx.py` 会被切成 "py" 去找
            if "::" in code:
                sym = code.split("::")[-1].strip()
            elif FILE_RE.fullmatch(code.strip()):
                continue                                        # 只写了文件名，A 组已查过
            else:
                sym = code.strip()
            sym = re.sub(r"\[.*|\(.*", "", sym).strip()        # 去掉 ["key"] / (...)
            sym = sym.split(".")[-1]                            # A.b → b
            if not re.fullmatch(r"\w+", sym) or sym in ("py", "ps1", "sh"):
                continue
            if cur_file is None or cur_file not in files:
                continue
            out["loc_checked"] += 1
            if sym in syms(cur_file):
                out["loc_ok"] += 1
            else:
                # 可能定义在同包的别的文件里（跨文件引用），放宽到全包搜索
                where = [f for f in files if f.endswith(".py") and sym in syms(f)]
                if where:
                    out["notes"].append(f"符号 {sym} 不在 {cur_file}，实际在 {where[0]}"
                                        f"（行「{row['req'][:28]}」）")
                    out["loc_ok"] += 1
                else:
                    out["problems"].append(f"位置里的符号找不到：{sym} @ {cur_file}"
                                           f"（行「{row['req'][:28]}」）")

    # ---- C：任务书里的关键标识符有没有在对照表里出现 ----
    table_blob = " ".join(r["req"] + " " + r["impl"] + " " + r["loc"] for r in rows).lower()
    body = task.split("-" * 20, 1)[-1]
    cand = {}
    for m in IDENT_RE.finditer(body):
        w = m.group(1)
        if w in IDENT_IGNORE or w.lower() in IDENT_IGNORE:
            continue
        cand[w] = cand.get(w, 0) + 1
    missing = sorted(w for w in cand if w.lower() not in table_blob)
    out["identifiers_total"] = len(cand)
    out["identifiers_missing"] = missing
    if verbose:
        out["notes"].append(f"任务书标识符 {len(cand)} 个，对照表未提及 {len(missing)} 个")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, help="只查某一阶段")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    targets = [args.stage] if args.stage else sorted(STAGES)
    all_problems, all_missing = [], {}
    print("=" * 96)
    print("任务书 → README 对照表 → 真实代码　闭合性核查")
    print("=" * 96)
    for n in targets:
        r = check_stage(n, args.verbose)
        head = f"任务{n}\\  {STAGES[n]}"
        print(f"\n{head}")
        print(f"  对照表 {r['rows']} 行 ｜ 位置引用 {r['loc_ok']}/{r['loc_checked']} 可解析"
              f" ｜ 未标记实现状态 {r['unmarked']} 行")
        if r.get("identifiers_missing") is not None:
            print(f"  任务书标识符 {r['identifiers_total']} 个，"
                  f"对照表未提及 {len(r['identifiers_missing'])} 个"
                  + (f"：{r['identifiers_missing'][:8]}" if r["identifiers_missing"] else ""))
            all_missing[n] = r["identifiers_missing"]
        for p in r["problems"]:
            print(f"    ❌ {p}")
            all_problems.append((n, p))
        if args.verbose:
            for nt in r["notes"]:
                print(f"    · {nt}")

    print("\n" + "=" * 96)
    if all_problems:
        print(f"❌ 硬判定未通过：{len(all_problems)} 处")
        for n, p in all_problems:
            print(f"  [任务{n}] {p}")
    else:
        print("✅ 硬判定全部通过：每行都有实现标记，所有声称的文件与符号都在代码里找得到")
    tot_missing = sum(len(v) for v in all_missing.values())
    print(f"\n待人工确认（C 组，必然含噪声）：任务书里 {tot_missing} 个标识符未在对照表出现。"
          f"\n  用 --stage N -v 逐阶段看；多数是示例代码变量或被明确否掉的方案。")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
