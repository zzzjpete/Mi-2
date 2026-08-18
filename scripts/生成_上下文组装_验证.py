# -*- coding: utf-8 -*-
"""第七阶段（一）· 上下文组装器 + 提示词模板 验证

每条 PASS/FAIL 都由真实数据算出并汇入总 ok，**没有任何一条是无条件打印的**（阶段五踩过
"无条件 print(✓)"的坑）。凡是需要"不会误伤"的结论（如去重不会删掉不同的正文），都在阶段三
的 1000 条真实文本块上跑，而不是只用手造的小例子。

分组：
  A 分词器      —— 与 qwen3:8b 同一套 BPE：往返一致、计数自洽、截断不超预算、降级模式保守性
  B 去重        —— Jaccard 的正确性 / 完全重复与近重复被丢 / 真实语料上不误删 / 保留高相关者
  C 排序与多样性 —— 纯相关性序、同源软惩罚与硬上限、有效分公式核对
  D 预算与截断   —— 不超预算、句/段边界截断、缩写不误判、极小预算、跳过大块继续装小块
  E 元数据与引用 —— 必需字段、计数自洽、[S#] 与入选块一一对应、参考文献列表
  F 提示词模板   —— 四段配置、变量推导与缺失报错、花括号不误伤、渲染形状、num_ctx 预算自洽

用法：
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_上下文组装_验证.py
  加 --live --bm25 E:\\rag\\data\\bm25_index_4m 时，额外跑一次真检索→组装的端到端冒烟
  （会加载 4M 向量库，约 15.8GB 内存 + 数分钟）
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import io
import json
import math
import re
import statistics
import sys
import time
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORT_PATH = os.path.join(ROOT, "report_data", "生成_上下文组装验证报告.txt")
os.makedirs(os.path.dirname(_REPORT_PATH), exist_ok=True)

# ⚠ 报告先攒在内存里，跑完再决定要不要落盘。**不能在这里 open(..., "w")**：
# 那样文件在第一条断言跑起来之前就被清空了，任何"跑完再判断该不该覆盖"的判据都来不及生效。
# 2026-08-16 正是这么把一份 65 项的 --live 报告冲成 58 项离线报告的（靠 任务7\ 包内副本救回）。
# 附带好处：中途崩溃时旧报告原样保留，而不是留下一份被截断的残报告。
_REPORT = io.StringIO()
_STDOUT = sys.stdout          # 真终端。收尾时"写了/没写"这句只给人看，不进报告正文


class _Tee:
    def __init__(self, *s): self.streams = s
    def write(self, x):
        for st in self.streams:
            try: st.write(x)
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass


sys.stdout = _Tee(sys.stdout, _REPORT)


# ---------------------------------------------------------------------------
# 防降级：弱运行不许覆盖强运行的报告
# ---------------------------------------------------------------------------
_TOTAL_RE = re.compile(r"总计\s+(\d+)\s*/\s*(\d+)\s*项通过")
_LIVE_RE = re.compile(r"实测部分\s+(\d+)\s*项")
# 旧报告里没有"实测部分 N 项"这一行（它是 2026-08-18 随本判据一起加的）。
# 退路是认这个标记：它由真检索那条路**加载成功之后**打印，跳过分支（缺阶段六脚本、
# 缺 BM25 索引）永远走不到，所以不会把"跳过了 live"误判成"有 live 证据"。
_LIVE_MARK = "[加载完成]"


def _refuse_downgrade(path, new_total, force=False, new_live=0):
    """已有报告更强时，拒绝覆盖。返回 True 表示"别写"。

    现象：2026-08-16 为验证一次与检索无关的改动，跑了本脚本的离线路径（58 项 / 6.5s），
    它把 report_data 里那份 65 项 / 88.8s 的 --live 版**静默覆盖**掉了——其中
    "G. 端到端冒烟 —— 真实检索结果 → 上下文组装"是全脚本唯一一组真库证据。
    最后靠 任务7/7.1_上下文组装与提示词工程/ 里的包内副本才捞回来。

    根因：报告是交付材料，被弱运行覆盖是纯损失且不可逆。同一个仓库里 服务_验证.py
    早有 _refuse_downgrade() 挡得住同类覆盖——**缺的不是判据写得弱，是这个脚本
    压根没有判据**。这是同一族坑的第五种形态（前四种都是"判据存在但太弱"）。

    决定：判据两条，缺一不可（与 服务_验证.py 同一套口径与措辞，故意不另造方言）：

      1. 新报告项数 >= 旧报告项数；
      2. 旧报告有 --live 实测项时，新报告也必须有。

    第 2 条不能省：离线那批喂的是固化快照，唯独 --live 那几项能证明"接上真的
    65GB 库也跑得通"。**"更多项"不等于"更强"——强弱不是一个标量**，
    只比项数这条已经被打穿过三次。

    依据数字：本脚本离线 58 项（不带 --ollama）／61 项（带 --ollama）／
    65 项（再带 --live），--live 独占 4 项。所以 58 < 65 由第 1 条挡下；
    将来"离线扩到 66 项"这种情形则由第 2 条挡下。
    """
    if force or not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return False
    m = _TOTAL_RE.search(text)
    if not m:
        return False
    old_total = int(m.group(2))
    ml = _LIVE_RE.search(text)
    # 分两种：数得出条数 / 只知道"有"。走退路时**不许编一个条数报出去**——
    # 本项目的规矩是报告里每个数字都得是真算出来的，哨兵值不能长得像实测值。
    old_live = int(ml.group(1)) if ml else (1 if _LIVE_MARK in text else 0)
    old_live_known = ml is not None

    if new_total >= old_total and not (old_live > 0 and new_live <= 0):
        return False
    print("\n" + "=" * 70)
    if old_live > 0 and new_live <= 0:
        howmany = f"{old_live} 项" if old_live_known else "一组（旧报告未记条数，按 [加载完成] 标记判定）"
        print(f"⚠ 拒绝覆盖：已有报告含 {howmany} **真实检索端到端**（--live），本次没有。")
        print(f"  本次 {new_total} 项 vs 旧的 {old_total} 项 —— **项数在这条判据里不作数**：")
        print(f"  离线组喂的是固化快照，唯独 --live 那几项证明真库也跑得通。")
        print(f"  要更新报告：加 --live --bm25 <BM25 索引目录>（需 65GB 向量库）。")
        print(f"  只想看这轮结果：--force-report。")
    else:
        print(f"⚠ 拒绝覆盖：已有报告 {old_total} 项，本次只有 {new_total} 项 —— 这是降级。")
        print(f"  用弱轮盖掉强轮等于丢掉交付证据。要这一轮的结果：--force-report。")
    print("=" * 70)
    return True


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ca = _load("shangxiawen", "生成_上下文组装.py")
_pt = _load("tishici", "生成_提示词模板.py")
_tkm = _load("fenciqi", "生成_分词器.py")
ContextAssembler, DocumentChunk = _ca.ContextAssembler, _ca.DocumentChunk
PROMPT_STAGES, MedicalPromptTemplates = _pt.PROMPT_STAGES, _pt.MedicalPromptTemplates
TokenCounter = _tkm.TokenCounter

SAMPLE_JSONL = _ca.SAMPLE_JSONL

# ----------------------------------------------------------------------------
CHECKS = []


def check(name, passed, detail=""):
    passed = bool(passed)
    CHECKS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return passed


def mk(chunk_id, text, rel, source, **meta):
    m = {"pmcid": source, "doc_id": source, "journal": meta.pop("journal", "PLoS ONE"),
         "pub_year": meta.pop("pub_year", 2021), "section": meta.pop("section", "Results"),
         "source_title": meta.pop("source_title", f"Study {source}"), "pmid": meta.pop("pmid", "")}
    m.update(meta)
    return DocumentChunk(text=text, metadata=m, relevance_score=rel, source=source, chunk_id=chunk_id)


def load_rows(limit=1000):
    rows = []
    with open(SAMPLE_JSONL, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="额外跑真检索→组装端到端（加载 4M 库）")
    ap.add_argument("--bm25", default=os.path.join(ROOT, "data", "bm25_index_4m"))
    ap.add_argument("--live-query", default="RCT evidence for pembrolizumab in NSCLC published since 2020")
    ap.add_argument("--ollama", action="store_true",
                    help="额外与 Ollama 自己的分词/显存实测对照（需 Ollama 在跑，会载入 qwen3:8b）")
    ap.add_argument("--force-report", action="store_true",
                    help="允许用更弱的一轮覆盖已有报告（默认拒绝，防止离线轮冲掉 --live 证据）")
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 96)
    print("上下文组装器 + 医学提示词模板 · 验证")
    print("=" * 96)
    rows = load_rows()
    print(f"真实语料：{SAMPLE_JSONL} 共 {len(rows)} 条文本块\n")

    # ========================================================================
    print("=" * 96)
    print("A. 分词器 —— 与 qwen3:8b 同一套 BPE（从 Ollama GGUF 重建）")
    print("=" * 96)
    t0 = time.time()
    tc = TokenCounter(mode="qwen")
    load_s = time.time() - t0
    check("Qwen3 分词器可离线加载", tc.exact and tc.mode == "qwen",
          f"来源={tc.info.get('source')} vocab={tc.info.get('vocab_size'):,} "
          f"ctx={tc.info.get('context_length')} 用时 {load_s:.2f}s")
    check("词表规模与 Qwen3 一致（151,936）", tc.info.get("vocab_size") == 151936,
          f"实际 {tc.info.get('vocab_size'):,}")

    texts = [r["text"] for r in rows]
    rt_bad = [t for t in texts if tc.decode(tc.encode(t)) != t]
    check("1000 条真实文本块 encode→decode 往返一致", len(rt_bad) == 0,
          f"不一致 {len(rt_bad)} 条")

    exact_counts = [len(tc.encode(t)) for t in texts]
    check("token 计数全为正整数", all(isinstance(n, int) and n > 0 for n in exact_counts),
          f"min={min(exact_counts)} max={max(exact_counts)} 中位={int(statistics.median(exact_counts))}")
    check("空串计为 0 token", tc.count("") == 0)
    dup_ok = all(tc.count(t + " " + t) >= tc.count(t) for t in texts[:200])
    check("计数单调（文本变长不会变少）", dup_ok, "前 200 条 text vs text+text")

    ratios = sorted(len(t) / max(1, n) for t, n in zip(texts, exact_counts))
    print(f"    实测 chars/token：min {ratios[0]:.2f} | p5 {ratios[len(ratios)//20]:.2f} | "
          f"中位 {statistics.median(ratios):.2f} | 均值 {statistics.mean(ratios):.2f} | max {ratios[-1]:.2f}")

    # 截断不超预算（精确模式）
    over = [t for t in texts[:300]
            if tc.count(tc.truncate_to_tokens(t, 64)[0]) > 64]
    check("truncate_to_tokens 结果不超上限（精确模式）", len(over) == 0, f"超限 {len(over)}/300")

    # 降级：字符启发式的保守性（实测偏高比例）
    th = TokenCounter(mode="heuristic")
    hr = [math.ceil(th.count(t) / max(1, n)) if False else th.count(t) / n
          for t, n in zip(texts, exact_counts)]
    over_rate = sum(1 for r in hr if r >= 1.0) / len(hr)
    check("降级字符启发式：≥90% 的真实块被高估（偏保守）", over_rate >= 0.90,
          f"高估比例 {over_rate:.1%}，h/e 中位 {statistics.median(hr):.2f}，最差 {min(hr):.2f}")
    over_h = [t for t in texts[:300] if th.count(th.truncate_to_tokens(t, 64)[0]) > 64]
    check("truncate_to_tokens 结果不超上限（启发式模式）", len(over_h) == 0, f"超限 {len(over_h)}/300")

    asm = ContextAssembler()
    check("组装器 estimate_tokens 与分词器一致",
          all(asm.estimate_tokens(t) == n for t, n in zip(texts[:200], exact_counts[:200])),
          "前 200 条逐条比对")

    # ========================================================================
    print("\n" + "=" * 96)
    print("B. 去重 —— Jaccard 相似性")
    print("=" * 96)
    a = texts[0]
    b = texts[1]
    check("自相似度 = 1.0", abs(asm.jaccard_similarity(a, a) - 1.0) < 1e-12,
          f"{asm.jaccard_similarity(a, a):.4f}")
    check("不同正文相似度低于去重阈值", asm.jaccard_similarity(a, b) < asm.similarity_threshold,
          f"sim={asm.jaccard_similarity(a, b):.4f} < 阈值 {asm.similarity_threshold}")

    words = a.split()
    shuffled = " ".join(words[::-1])
    sim_3g = asm.jaccard_similarity(a, shuffled)
    bow = ContextAssembler(shingle_size=1, tokenizer=tc)
    sim_bow = bow.jaccard_similarity(a, shuffled)
    check("3-gram 比词袋更能区分'同词不同序'", sim_3g < sim_bow,
          f"3-gram {sim_3g:.3f} < 词袋 {sim_bow:.3f}")

    # 完全重复 / 近重复 / 无关
    near = re.sub(r"\bthe\b", "a", a, count=3) + " Additional sentence appended here."
    sim_near = asm.jaccard_similarity(a, near)
    docs = [mk("c1", a, 0.90, "PMC1"), mk("c2", a, 0.50, "PMC2"),
            mk("c3", near, 0.80, "PMC3"), mk("c4", b, 0.70, "PMC4")]
    kept, dropped = asm.deduplicate(docs)
    kept_ids = [c.chunk_id for c in kept]
    check("完全重复被丢弃", "c2" not in kept_ids and any(d["kind"] == "exact" for d in dropped),
          f"保留 {kept_ids}，丢弃 {[d['chunk_id'] for d in dropped]}")
    check("保留的是重复簇里相关性最高的那条", "c1" in kept_ids,
          "c1(rel=0.90) 保留，c2(rel=0.50) 丢弃")
    check(f"近重复（Jaccard {sim_near:.3f} ≥ {asm.similarity_threshold}）被丢弃",
          (sim_near >= asm.similarity_threshold) == ("c3" not in kept_ids),
          f"c3 {'已丢弃' if 'c3' not in kept_ids else '被保留'}")
    check("无关正文未被误删", "c4" in kept_ids)

    real_docs = [mk(r["chunk_id"], r["text"], 0.9 - i * 0.0005, r["pmcid"]) for i, r in enumerate(rows)]
    real_chunks, too_short = asm.to_document_chunks(real_docs)
    short_texts = {c.chunk_id: c.text for c in real_docs if c.chunk_id in set(too_short)}
    check("过短的退化块被前置丢弃（正文不足 %d 字符）" % asm.min_chunk_chars,
          all(len(t.strip()) < asm.min_chunk_chars for t in short_texts.values()),
          f"丢弃 {len(too_short)} 条，样例正文 {sorted(set(short_texts.values()))[:3]}")

    kept_real, dropped_real = asm.deduplicate(real_chunks)
    norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower()))
    by_id = {c.chunk_id: c for c in real_chunks}
    false_pos = [d for d in dropped_real
                 if norm(by_id[d["chunk_id"]].text) != norm(by_id[d["duplicate_of"]].text)]
    check("真实语料上去重零误删（被丢的都是逐字重复）",
          len(false_pos) == 0,
          f"1000 条中保留 {len(kept_real)}，丢弃 {len(dropped_real)} 条（均为逐字重复），误删 {len(false_pos)} 条")

    # ========================================================================
    print("\n" + "=" * 96)
    print("C. 排序与来源多样性")
    print("=" * 96)
    same = "Randomized controlled trial of drug X in patients with disease Y. " * 6
    pool = [mk(f"A#{i}", same + f"Variant A{i} finding {i}. " * 3, 0.90 - i * 0.01, "PMC_A")
            for i in range(5)]
    pool += [mk(f"B#{i}", f"Cohort study of drug Z in disease Y, variant B{i}. " * 8,
                0.70 - i * 0.01, "PMC_B") for i in range(3)]
    pool += [mk("C#0", "Meta-analysis of drug X across 12 trials in disease Y. " * 8, 0.60, "PMC_C")]

    plain = ContextAssembler(tokenizer=tc, diversity_decay=1.0, max_per_source=99,
                             max_context_tokens=100000, similarity_threshold=1.01)
    r_plain = plain.assemble_context(pool)
    order = [c.relevance_score for c in r_plain["selected_chunks"]]
    check("关掉多样性时严格按相关性降序选取",
          all(order[i] >= order[i + 1] for i in range(len(order) - 1)) and len(order) == len(pool),
          f"{len(order)} 条，序列 {[round(x,2) for x in order[:5]]}…")

    div = ContextAssembler(tokenizer=tc, diversity_decay=0.75, max_per_source=3,
                           max_context_tokens=100000, similarity_threshold=1.01)
    r_div = div.assemble_context(pool)
    by_src = Counter(c.source for c in r_div["selected_chunks"])
    check("同源硬上限 max_per_source=3 生效", max(by_src.values()) <= 3, f"来源分布 {dict(by_src)}")
    check("多样性让低相关的其它文献进入上下文", len(by_src) == 3,
          f"覆盖 {len(by_src)} 篇文献：{dict(by_src)}")
    picked = [c.chunk_id for c in r_div["selected_chunks"]]
    idx_b0 = picked.index("B#0") if "B#0" in picked else 10 ** 9
    idx_a3 = picked.index("A#3") if "A#3" in picked else 10 ** 9
    check("同源第 4 条被排到别的文献之后（软惩罚生效）", idx_b0 < idx_a3,
          f"B#0 位次 {idx_b0} < A#3 位次 {'未入选' if idx_a3 == 10**9 else idx_a3}")

    used = Counter({"PMC_A": 2})
    expect = pool[0].relevance_score * (0.75 ** 2)
    got = div._effective_score(pool[0], used)
    check("有效分公式 = 相关性 × decay^同源已选数", abs(expect - got) < 1e-12,
          f"手算 {expect:.6f} vs 实现 {got:.6f}（误差 {abs(expect-got):.1e}）")

    # ========================================================================
    print("\n" + "=" * 96)
    print("D. token 预算与截断")
    print("=" * 96)
    real_pool = [mk(r["chunk_id"], r["text"], 0.95 - i * 0.01, r["pmcid"],
                    journal=r["journal"], pub_year=int(r["pub_year"]), section=r["section"],
                    source_title=r["source_title"], pmid=r["pmid"]) for i, r in enumerate(rows[:60])]
    budgets = [200, 500, 1200, 2800]
    rows_budget = []
    for bgt in budgets:
        a2 = ContextAssembler(tokenizer=tc, max_context_tokens=bgt)
        res = a2.assemble_context(real_pool, query="budget test")
        md = res["metadata"]
        rows_budget.append((bgt, md["estimated_tokens"], md["chunks_selected"],
                            len(md["truncated_chunks"])))
    check("各预算下上下文都不超预算", all(t <= b for b, t, _, _ in rows_budget),
          " | ".join(f"预算{b}→{t}tok/{n}块(截断{c})" for b, t, n, c in rows_budget))
    check("预算越大装的证据越多（单调不减）",
          all(rows_budget[i][2] <= rows_budget[i + 1][2] for i in range(len(rows_budget) - 1)),
          " → ".join(str(n) for _, _, n, _ in rows_budget))

    a3 = ContextAssembler(tokenizer=tc, max_context_tokens=1200)
    res3 = a3.assemble_context(real_pool, query="self-consistency")
    check("metadata.estimated_tokens 与 context_text 实测一致",
          res3["metadata"]["estimated_tokens"] == a3.estimate_tokens(res3["context_text"]),
          f"{res3['metadata']['estimated_tokens']} tok")

    # 边界截断：逐条核对 truncate_at_boundary
    bad_bound, bad_prefix, bad_window, cut_n = [], [], [], 0
    for t in texts[:200]:
        for lim in (60, 120, 240):
            body, was_cut = a3.truncate_at_boundary(t, lim)
            if not was_cut:
                continue
            cut_n += 1
            raw, _ = tc.truncate_to_tokens(t, lim)
            if not t.startswith(body[:max(1, len(body) - 1)]):
                bad_prefix.append(t[:40])
            if len(body) < len(raw.strip()) * (1 - a3.boundary_tail_ratio) - 2:
                bad_window.append((len(body), len(raw)))
            if body and not re.search(r"[.!?。！？]$", body) and len(body) >= len(raw.strip()) - 2:
                pass          # 末 10% 内找不到句末 → 保留硬截结果，允许
            elif body and not re.search(r"[.!?。！？]$", body):
                bad_bound.append(body[-40:])
    check("被截断的正文都以完整句/段结尾（或末10%内确无边界）", len(bad_bound) == 0,
          f"共触发截断 {cut_n} 次，越界 {len(bad_bound)} 次")
    check("截断结果是原文前缀（未改写内容）", len(bad_prefix) == 0, f"非前缀 {len(bad_prefix)} 次")
    check("回退查找不超出末 10% 窗口", len(bad_window) == 0,
          f"超窗 {len(bad_window)} 次（窗口比例 {a3.boundary_tail_ratio:.0%}）")

    abbr = ("We measured outcomes in three cohorts, e.g. patients with severe disease. "
            "Results were reported by Smith et al. 2019 in a multicentre setting. "
            "The primary endpoint was overall survival at 24 months. " * 3)
    hits, boundary_n = [], 0
    for lim in range(20, 160, 5):
        body, was_cut = a3.truncate_at_boundary(abbr, lim)
        if not (was_cut and body):
            continue
        raw, _ = tc.truncate_to_tokens(abbr, lim)
        if body == raw.strip():
            continue                       # 末 10% 内确无句末 → 保留硬截，不是"选了缩写点"
        boundary_n += 1
        tail = body.rstrip().lower()
        if tail.endswith("e.g.") or tail.endswith("et al.") or tail.endswith("i.e."):
            hits.append((lim, tail[-12:]))
    check("不会把 e.g. / et al. 的缩写点当句末", len(hits) == 0,
          f"{boundary_n} 次主动选边界，误切 {len(hits)} 次")
    # 反向：回退窗口里同时有真句末和更靠后的缩写点时，必须跳过缩写点、退到真句末
    mixed = ("Patients were enrolled across twelve centres and followed for two years. " * 12
             + "The primary endpoint was overall survival at 24 months. "
               "Data were pooled by Smith et al. ")
    body_m, cut_m = a3.truncate_at_boundary(mixed, tc.count(mixed) - 2)
    check("回退窗口内跳过更靠后的缩写点、退到真句末",
          cut_m and body_m.rstrip().endswith("24 months."),
          f"结尾 …{body_m.rstrip()[-28:]!r}")

    tiny = ContextAssembler(tokenizer=tc, max_context_tokens=20)
    r_tiny = tiny.assemble_context(real_pool)
    check("预算极小时安全退化（空上下文、不异常）",
          r_tiny["metadata"]["chunks_selected"] == 0 and r_tiny["context_text"] == "",
          f"入选 {r_tiny['metadata']['chunks_selected']} 块，"
          f"因预算跳过 {len(r_tiny['metadata']['skipped_by_budget'])} 块")

    big_txt = " ".join(texts[:6])                     # 明显超预算的大块
    small_txt = "Aspirin reduced recurrent stroke risk in this randomized cohort."
    mix = [mk("big", big_txt, 0.99, "PMC_BIG"), mk("small", small_txt, 0.10, "PMC_SMALL")]
    a4 = ContextAssembler(tokenizer=tc, max_context_tokens=60, min_fragment_tokens=200)
    r_mix = a4.assemble_context(mix)
    ids_mix = [c.chunk_id for c in r_mix["selected_chunks"]]
    check("大块装不下时跳过它继续装小块（而非直接停止）",
          ids_mix == ["small"] and "big" in r_mix["metadata"]["skipped_by_budget"],
          f"入选 {ids_mix}，跳过 {r_mix['metadata']['skipped_by_budget']}")

    # ========================================================================
    print("\n" + "=" * 96)
    print("E. 元数据与引用可溯源性")
    print("=" * 96)
    md3 = res3["metadata"]
    need = ["total_chunks_retrieved", "unique_chunks_after_dedup", "chunks_selected",
            "estimated_tokens", "chunk_sources"]
    check("任务书要求的 5 个元数据字段齐全", all(k in md3 for k in need),
          f"缺失 {[k for k in need if k not in md3]}")
    check("计数自洽：selected ≤ unique ≤ retrieved",
          md3["chunks_selected"] <= md3["unique_chunks_after_dedup"] <= md3["total_chunks_retrieved"],
          f"{md3['total_chunks_retrieved']} → {md3['unique_chunks_after_dedup']} → {md3['chunks_selected']}")
    check("total_chunks_retrieved 等于输入条数", md3["total_chunks_retrieved"] == len(real_pool),
          f"{md3['total_chunks_retrieved']} vs 输入 {len(real_pool)}")
    cs = md3["chunk_sources"]
    check("chunk_sources 计数与入选块自洽",
          sum(cs["by_source"].values()) == md3["chunks_selected"]
          and cs["unique_sources"] == len(cs["by_source"])
          and cs["max_from_one_source"] == (max(cs["by_source"].values()) if cs["by_source"] else 0),
          f"{cs['unique_sources']} 篇文献 / {md3['chunks_selected']} 块，最多同源 {cs['max_from_one_source']}")

    markers = re.findall(r"^\[(S\d+)\]", res3["context_text"], flags=re.M)
    want_markers = [f"S{i}" for i in range(1, md3["chunks_selected"] + 1)]
    check("[S#] 标记连续、唯一、与入选块数一致", markers == want_markers,
          f"上下文中 {len(markers)} 个标记，期望 {len(want_markers)} 个")
    cit = md3["citations"]
    ok_cit = (len(cit) == md3["chunks_selected"]
              and all(c["marker"] == f"S{i}" for i, c in enumerate(cit, 1))
              and all(c["chunk_id"] == ch.chunk_id for c, ch in zip(cit, res3["selected_chunks"])))
    check("citations 与 selected_chunks 逐条对应", ok_cit, f"{len(cit)} 条引用记录")
    meta_ok = all(c["pmcid"] == (ch.metadata or {}).get("pmcid")
                  and c["journal"] == ch.journal and c["pub_year"] == ch.year
                  for c, ch in zip(cit, res3["selected_chunks"]))
    check("引用里的 pmcid/期刊/年份与块元数据一致", meta_ok)
    hdr_ok = all(f"[{c['marker']}] {c['source']}" in res3["context_text"] for c in cit)
    check("每条证据的出处头出现在上下文里（模型可见）", hdr_ok)

    refs = ContextAssembler.render_reference_list(res3["selected_chunks"], cit)
    ref_lines = [l for l in refs.split("\n") if l.strip()]
    check("参考文献列表与引用编号一一对应",
          len(ref_lines) == len(cit) and all(l.startswith(f"[{c['marker']}]")
                                             for l, c in zip(ref_lines, cit)),
          f"{len(ref_lines)} 条")

    # ========================================================================
    print("\n" + "=" * 96)
    print("F. 医学提示词模板")
    print("=" * 96)
    T = MedicalPromptTemplates()
    keys = ["evidence_evaluator", "answer_generator", "critical_reviewer", "final_assembler"]
    check("四段提示词齐全", all(k in PROMPT_STAGES for k in keys),
          " / ".join(f"{k}={PROMPT_STAGES[k].name}" for k in keys if k in PROMPT_STAGES))
    check("温度与 max_tokens 取值合法",
          all(0.0 <= PROMPT_STAGES[k].temperature <= 2.0 and PROMPT_STAGES[k].max_tokens > 0
              for k in keys),
          " | ".join(f"{k}:T={PROMPT_STAGES[k].temperature},N={PROMPT_STAGES[k].max_tokens}" for k in keys))
    check("结构化输出的两段温度为 0（可复现、便于解析）",
          PROMPT_STAGES["evidence_evaluator"].temperature == 0.0
          and PROMPT_STAGES["critical_reviewer"].temperature == 0.0
          and PROMPT_STAGES["evidence_evaluator"].output_format == "json"
          and PROMPT_STAGES["critical_reviewer"].output_format == "json")

    want_vars = {"evidence_evaluator": ["question", "context"],
                 "answer_generator": ["question", "context", "evidence_summary"],
                 "critical_reviewer": ["question", "context", "draft_answer"],
                 "final_assembler": ["question", "context", "draft_answer", "review", "reference_list"]}
    check("模板变量自动推导正确",
          all(PROMPT_STAGES[k].required_vars == v for k, v in want_vars.items()),
          " | ".join(f"{k}:{PROMPT_STAGES[k].required_vars}" for k in keys))

    missing_raised = False
    try:
        PROMPT_STAGES["answer_generator"].format_messages(question="q", context="c")
    except KeyError:
        missing_raised = True
    check("缺少模板变量时报错（不静默留占位符）", missing_raised)

    brace_tpl = 'schema: {"marker": "S1", "relevance": 3} 问题：{question}'
    rendered_brace = _pt.render_template(brace_tpl, question="X")
    check("JSON 示例中的花括号不被误替换",
          rendered_brace == 'schema: {"marker": "S1", "relevance": 3} 问题：X', rendered_brace[:60])

    msgs = PROMPT_STAGES["answer_generator"].format_messages(
        question="阿尔茨海默病近年获批的新药有哪些？", context=res3["context_text"],
        evidence_summary="")
    shape_ok = (len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user")
    check("渲染出的 messages 形状正确", shape_ok, f"{[m['role'] for m in msgs]}")
    check("证据上下文确实进入了 user 提示词", res3["context_text"] in msgs[1]["content"],
          f"user 长度 {len(msgs[1]['content'])} 字符")
    check("未渲染的 {变量} 占位符已全部消失",
          not _pt._VAR.search(msgs[1]["content"].replace("{", "{", 1)) or
          all(v not in msgs[1]["content"] for v in ("{question}", "{context}", "{evidence_summary}")))
    check("关闭思考模式时附加 /no_think", msgs[1]["content"].rstrip().endswith("/no_think"))
    opts = PROMPT_STAGES["answer_generator"].to_ollama_options()
    check("采样参数映射到 Ollama 键名（max_tokens→num_predict）",
          opts["num_predict"] == PROMPT_STAGES["answer_generator"].max_tokens
          and opts["temperature"] == PROMPT_STAGES["answer_generator"].temperature, json.dumps(opts))

    check("链路开关可做消融（关掉评估/审查即退化为直接作答）",
          T.chain() == keys and T.chain(evaluate=False, review=False) ==
          ["answer_generator", "final_assembler"], f"{T.chain(evaluate=False, review=False)}")

    NCTX = _pt.RECOMMENDED_NUM_CTX
    default_budget = ContextAssembler().max_context_tokens
    for nc in (8192, NCTX, 16384):
        p = T.plan_budget(tc, num_ctx=nc)
        print(f"    num_ctx={nc:<6} → 证据预算 {p['recommended_context_tokens']:>5} tok"
              f"（最坏段 {p['worst_stage']}，链内输入预留 {p['chained_inputs_reserved']}，"
              f"文献列表预留 {p['reference_reserve']}）")
    plan = T.plan_budget(tc, num_ctx=NCTX)
    rec = plan["recommended_context_tokens"]
    check(f"建议 num_ctx={NCTX} 下的证据预算 ≥ 组装器默认预算",
          rec > 0 and default_budget <= rec,
          f"建议 {rec} tok（最吃上下文的一段：{plan['worst_stage']}），组装器默认 {default_budget} tok")
    check("num_ctx=8192 下证据预算仍为正（可降级运行，需调小组装器预算）",
          T.plan_budget(tc, num_ctx=8192)["recommended_context_tokens"] > 0,
          f"{T.plan_budget(tc, num_ctx=8192)['recommended_context_tokens']} tok "
          f"< 默认 {default_budget}，此时须显式传 max_context_tokens")

    # 最坏一段的真实渲染长度 + 输出上限，必须落在 num_ctx 内
    worst = PROMPT_STAGES["final_assembler"]
    ctx_full = ContextAssembler(tokenizer=tc, max_context_tokens=default_budget
                                ).assemble_context(real_pool)
    draft_stub = "草" * (PROMPT_STAGES["answer_generator"].max_tokens)      # 每个汉字≥1 token，偏保守
    review_stub = "审" * (PROMPT_STAGES["critical_reviewer"].max_tokens)
    m2 = worst.format_messages(question="Q" * 100, context=ctx_full["context_text"],
                               draft_answer=draft_stub, review=review_stub,
                               reference_list=ContextAssembler.render_reference_list(
                                   ctx_full["selected_chunks"], ctx_full["metadata"]["citations"]))
    used_tok = tc.count(m2[0]["content"]) + tc.count(m2[1]["content"])
    total = used_tok + worst.max_tokens
    check(f"最坏一段（默认预算证据 + 满载草稿/审查 + 最长输出）仍在 num_ctx={NCTX} 内",
          total <= NCTX,
          f"输入 {used_tok}（含证据 {ctx_full['metadata']['estimated_tokens']}）"
          f" + 输出上限 {worst.max_tokens} = {total} ≤ {NCTX}")

    # ========================================================================
    if args.ollama:
        print("\n" + "=" * 96)
        print("H. 与 Ollama 实测对照 —— 分词是否真的一致 / num_ctx 与显存")
        print("=" * 96)
        import requests
        URL = "http://localhost:11434"

        def gen(prompt, **opts):
            o = {"num_predict": 1, "temperature": 0}       # num_predict=0 会让 Ollama 挂住不返回
            o.update(opts)
            r = requests.post(f"{URL}/api/generate", timeout=300, json={
                "model": "qwen3:8b", "prompt": prompt, "raw": True, "stream": False, "options": o})
            r.raise_for_status()
            return r.json()

        # H1 分词一致性：raw=true 不套 chat 模板，prompt_eval_count 就是 llama.cpp 切出的 token 数。
        # 这一条是必须的——byte-level BPE 的 decode 只是拼接字节，**任何**切分都能往返一致，
        # 所以 A 组的往返一致只能证明词表没载错，不能证明切分（从而 token 数）与推理端相同。
        probes = [texts[1], texts[7], texts[42], texts[300], texts[999],
                  "\n\n".join(texts[10:14]), "Hello world",
                  "阿尔茨海默病近年获批的新药有哪些？",
                  ctx_full["context_text"][:4000]]
        pairs = [(tc.count(s), gen(s)["prompt_eval_count"]) for s in probes]
        diffs = sorted({b - a for a, b in pairs})
        check("本地重建的分词与 Ollama 实际分词逐条一致",
              diffs == [0],
              f"{len(pairs)} 个样本（{min(a for a,_ in pairs)}~{max(a for a,_ in pairs)} tok）差值集合 {diffs}")

        # H2 Ollama 的默认 num_ctx 确实只有 4096（这是文档里那条警告的依据）
        gen("ping")
        ps = requests.get(f"{URL}/api/ps", timeout=30).json()["models"][0]
        ctx_default = ps.get("context_length")
        check("Ollama 默认 num_ctx = 4096（必须显式调大，否则四段链被静默截断）",
              ctx_default == 4096, f"实测 {ctx_default}")

        # H3 建议 num_ctx 下的真实显存占用（此前只是按架构算的估计）
        nctx = _pt.RECOMMENDED_NUM_CTX
        gen("ping", num_ctx=nctx)
        ps2 = requests.get(f"{URL}/api/ps", timeout=30).json()["models"][0]
        vram = ps2.get("size_vram", 0) / 1024 ** 3
        check(f"num_ctx={nctx} 时显存占用留有余量（10GB 卡）",
              ps2.get("context_length") == nctx and 0 < vram < 9.0,
              f"实测 size_vram {vram:.2f} GB（context_length={ps2.get('context_length')}）")

    # ========================================================================
    # 记下 live 组开跑前的条数：收尾时用差值算出"这一轮真跑了几项 --live"，交给防降级判据。
    # ⚠ 不能用 args.live 当布尔标志——它在下面缺库时会被改成 False，而"本来想跑 live
    #   但跳过了"必须与"真跑了 live"分开算，否则判据会被自己骗过去（空洞断言的老形态）。
    _live_start = len(CHECKS)
    if args.live:
        print("\n" + "=" * 96)
        print("G. 端到端冒烟 —— 真实检索结果 → 上下文组装")
        print("=" * 96)
        # 这一组跨阶段：要阶段六的检索脚本 + 4M 向量库/BM25 索引。本包（任务7）按约定只带
        # 自己 import 的文件，不重复收录阶段六那一链——缺了就明确说清楚，不要甩一个 traceback。
        _need = os.path.join(_HERE, "检索_多路检索.py")
        if not os.path.exists(_need):
            print(f"  [跳过] 本目录下没有 检索_多路检索.py（阶段六脚本，见 任务6\\脚本\\）。\n"
                  f"         --live 还需要 data\\chroma_db_4m（65 GB）与 {args.bm25}，\n"
                  f"         请在完整项目的 E:\\rag\\scripts\\ 下运行本脚本。")
            args.live = False
        elif not os.path.isdir(args.bm25):
            print(f"  [跳过] BM25 索引不存在：{args.bm25}（用 检索_构建BM25索引.py 重建，约 13 min）")
            args.live = False
    if args.live:
        _mp = _load("duolu", "检索_多路检索.py")
        t0 = time.time()
        pipe = _mp.RetrievalPipeline(bm25_dir=args.bm25)
        print(f"[加载完成] 用时 {time.time()-t0:.0f}s")
        out = pipe.search(args.live_query, top_k=10)
        live_asm = ContextAssembler(tokenizer=tc, verbose=True)
        live = live_asm.assemble_context(out["results"], query=args.live_query)
        lmd = live["metadata"]
        check("真实 Candidate 可直接转成 DocumentChunk",
              lmd["total_chunks_retrieved"] == len(out["results"]) and lmd["chunks_selected"] > 0,
              f"检索 {lmd['total_chunks_retrieved']} → 入选 {lmd['chunks_selected']}")
        check("真实结果的相关性取自重排总分",
              all((c.metadata.get("_retrieval") or {}).get("rerank_score") == c.relevance_score
                  for c in live["selected_chunks"]),
              "relevance_score == rerank_score")
        check("真实上下文不超预算", lmd["estimated_tokens"] <= live_asm.max_context_tokens,
              f"{lmd['estimated_tokens']}/{live_asm.max_context_tokens} tok")
        check("真实证据来自多篇文献", lmd["chunk_sources"]["unique_sources"] >= 2,
              f"{lmd['chunk_sources']['unique_sources']} 篇：{list(lmd['chunk_sources']['by_source'])[:5]}")
        print("\n--- 真实上下文（前 1200 字符）---")
        print(live["context_text"][:1200])

    # ========================================================================
    print("\n" + "=" * 96)
    n_pass = sum(1 for _, p, _ in CHECKS if p)
    live_n = len(CHECKS) - _live_start
    print(f"总计 {n_pass}/{len(CHECKS)} 项通过    用时 {time.time()-t_start:.1f}s")
    if live_n:
        print(f"实测部分 {live_n} 项（G. 真检索 → 上下文组装，走 65GB 向量库与 BM25 索引）")
    if n_pass != len(CHECKS):
        print("未通过：")
        for name, p, detail in CHECKS:
            if not p:
                print(f"  · {name} — {detail}")
    print("=" * 96)

    # 汇总打完了才决定落不落盘。下面这几句只走真终端、不进报告正文——
    # 报告里不该出现"我拒绝写我自己"这种话。
    sys.stdout = _STDOUT
    if _refuse_downgrade(_REPORT_PATH, len(CHECKS), force=args.force_report, new_live=live_n):
        print(f"  本轮 {n_pass}/{len(CHECKS)} 项的完整输出见上方终端回滚，未落盘。")
    else:
        with open(_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(_REPORT.getvalue())
            f.write(f"报告已写入 {_REPORT_PATH}\n")   # 与历史报告逐字节同形，别删这行
        print(f"报告已写入 {_REPORT_PATH}")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
