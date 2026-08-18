# -*- coding: utf-8 -*-
"""第八阶段 · 验证：答案评估器 / 缓存 / 批量处理

**不调用模型**，7 秒内跑完。每一条 PASS/FAIL 都由实际算出的变量比对得出——
不存在任何无条件 `print("✓")`（阶段五踩过这个坑，交付结论当时没有代码支撑）。

凡是"不会误判"这类结论，尽量放到**真实产物**上跑（`report_data\\生成_流水线测试_live.jsonl`
里的真答案、`检索快照_live.json` 里的真证据），而不是只用手造的小例子。真产物缺失时
该组自动跳过并**在报告里写明跳过**，不会静默变成"通过"。

用法：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\评估_验证.py
    加 --quiet 只看汇总
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT, evidence_path

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "report_data")
REPORT_PATH = os.path.join(REPORT_DIR, "评估_验证报告.txt")
LIVE_JSONL = evidence_path("生成_流水线测试_live.jsonl")
SNAPSHOT = evidence_path("检索快照_live.json")


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("pinggu_pinggu", "评估_答案评估器.py")
gc_ = _load("shengcheng_huancun", "生成_缓存.py")
bp = _load("shengcheng_piliang", "生成_批量处理.py")


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

    def check(self, label: str, ok: bool, detail: str = ""):
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
        passed = sum(1 for _, _, ok, _ in self.rows if ok)
        return passed, len(self.rows)


# ============================================================================
# A 文本相似性（ROUGE）
# ============================================================================
def group_rouge(c: Checker, real_answers: List[str]):
    c.head("A 文本相似性 · ROUGE（后端：%s）" % ev.ROUGE_BACKEND)

    same = "lecanemab 10 mg/kg every two weeks reduced CDR-SB by 0.45 points"
    g = ev.rouge_scores(same, same)
    c.check("完全相同的文本 ROUGE-L f ≈ 1", g["rouge-l"]["f"] > 0.999999,
            f"f={g['rouge-l']['f']:.9f}（rouge 库分母含 1e-8，故非严格 1.0）")

    g0 = ev.rouge_scores("阿司匹林用于抗血小板", "quantum entanglement in photons")
    c.check("毫无重合的文本各项 ROUGE 全为 0",
            all(g0[m]["f"] == 0.0 for m in ev.ROUGE_METRICS),
            f"1/2/L f = {[round(g0[m]['f'], 4) for m in ev.ROUGE_METRICS]}")

    a = "lecanemab reduced CDR-SB by 0.45 points at 18 months"
    b = "lecanemab reduced the CDR-SB score by 0.45 at month 18 in CLARITY-AD"
    gp = ev.rouge_scores(a, b)
    c.check("部分重合的文本落在 (0,1) 之间", 0.0 < gp["rouge-l"]["f"] < 1.0,
            f"ROUGE-L f={gp['rouge-l']['f']:.4f}")
    c.check("ROUGE-1 ≥ ROUGE-2（n 越大越难重合）",
            gp["rouge-1"]["f"] >= gp["rouge-2"]["f"],
            f"{gp['rouge-1']['f']:.4f} ≥ {gp['rouge-2']['f']:.4f}")

    # —— 中文为什么必须换分词：拿数字说话 ——
    zh1 = "该药可延缓早期阿尔茨海默病患者的认知功能下降"
    zh2 = "该药能够延缓早期阿尔茨海默病的认知下降速度"
    naive_tokens = len(zh1.split())
    mixed_tokens = len(ev.mixed_tokenize(zh1))
    c.check("中文按空白分词会塌成 1 个 token（所以不能直接喂 rouge 库）",
            naive_tokens == 1 and mixed_tokens > 10,
            f"空白分词 {naive_tokens} 个 token / 混合分词 {mixed_tokens} 个")

    from rouge import Rouge as _R                      # 直接用库，做"不预分词"的对照
    naive = _R().get_scores(zh1, zh2)[0]["rouge-l"]["f"]
    fixed = ev.rouge_scores(zh1, zh2)["rouge-l"]["f"]
    c.check("不预分词时两句高度相似的中文被判为 0；预分词后能分辨",
            naive == 0.0 and fixed > 0.5,
            f"未预分词 f={naive:.4f} → 预分词后 f={fixed:.4f}")

    # —— rouge 库把 "." 当句子分隔符，而医学答案满是小数 ——
    dec = "CDR-SB 差值为 0.45 分，p<0.001。随访 18 个月。"
    naive_join = " ".join(ev.mixed_tokenize(dec))
    safe_join = ev._tokenized_for_rouge(dec)
    c.check("小数点被保护，「.」只当句子分隔符（否则 0.45 会被 rouge 库当句号劈开）",
            naive_join.count(".") == 2 and safe_join.count(".") == 1
            and "0·45" in safe_join,
            f"直接拼接含 {naive_join.count('.')} 个句点（都是小数点）→ "
            f"处理后 {safe_join.count('.')} 个（真句子边界）")

    # —— rouge 的 LCS 回溯是递归的，整段当一句会栈溢出；给它真正的句子边界就不会 ——
    long_one_sentence = " ".join(["token%d" % i for i in range(3000)])
    from rouge import Rouge as _R0
    try:
        _R0().get_scores(long_one_sentence, long_one_sentence)
        blew_up = False
    except RecursionError:
        blew_up = True
    long_real = "。".join([f"这是第{i}句测试文本，内容各不相同" for i in range(120)])
    ok_scores = ev.rouge_scores(long_real, long_real)
    c.check("超长文本：库直接喂会 RecursionError，本模块按句切分后不崩",
            blew_up and ok_scores["rouge-l"]["f"] > 0.999
            and ok_scores["backend"] == "rouge-lib",
            f"库对 3000 词单句栈溢出={blew_up}；本模块对 {len(long_real)} 字文本正常出分")

    e1 = ev.rouge_scores("", "abc")
    e2 = ev.rouge_scores("abc", "")
    c.check("空文本不抛异常且记 0（rouge 库本身会抛 ValueError）",
            e1["rouge-l"]["f"] == 0.0 and e2["rouge-l"]["f"] == 0.0,
            "两种空输入均返回 0")

    # —— 自研实现 vs rouge 库：在**真实答案**上逐对比，报告实测最大差值 ——
    if len(real_answers) >= 2:
        deltas: List[float] = []
        for i in range(len(real_answers) - 1):
            h, r = real_answers[i], real_answers[i + 1]
            lib = ev.rouge_scores(h, r, backend="rouge-lib")
            own = ev.rouge_scores(h, r, backend="builtin")
            for m in ev.ROUGE_METRICS:
                for s in ("r", "p", "f"):
                    deltas.append(abs(lib[m][s] - own[m][s]))
        mx = max(deltas)
        c.check(f"自研实现与 rouge 库在 {len(real_answers)} 份真实答案上数值一致",
                mx < 1e-9, f"{len(deltas)} 个数值逐个比对，最大差值 {mx:.2e}")
    else:
        c.skip("自研实现 vs rouge 库（真实答案对照）", f"{LIVE_JSONL} 里的答案不足 2 份")


# ============================================================================
# B 关键信息抽取与召回
# ============================================================================
def group_key_info(c: Checker):
    c.head("B 关键信息抽取与召回")
    x = ev.MedicalKeyInfoExtractor()

    got = x.extract("有效率为 12.6%，另有百分之三十的患者缓解")["percentage"]
    c.check("百分比：符号式与中文式都能抽", "12.6%" in got and len(got) == 2, f"{got}")

    got = x.extract("剂量为 10 mg/kg，每次 500mg，维生素 D 800 IU")["dosage"]
    c.check("剂量：数值+单位，含 /kg 与无空格写法",
            {"10mg/kg", "500mg", "800iu"} <= set(got), f"{got}")

    got = x.extract("随访 18 个月，每两周给药一次，疗程 4-6 周")["duration"]
    c.check("时间范围：中文量词「个」、中文数字「两」、区间都能抽",
            {"18month", "2week", "4-6week"} <= set(got), f"{got}")

    zh = x.extract("需警惕不良反应与用药风险")["safety"]
    en = x.extract("adverse events and safety risks were reported")["safety"]
    c.check("安全信息：中英同义写法归到同一 concept（否则跨语种召回恒为 0）",
            set(zh) & set(en) == {"adverse_effect", "risk"},
            f"中文 {zh} ∩ 英文 {en} = {sorted(set(zh) & set(en))}")

    # 召回公式手算核对：gt 有 3 条百分比，答案覆盖 2 条 → 2/3
    gt = "缓解率 40%、有效率 55%、不良反应发生率 12.6%"
    gen = "缓解率约 40%，不良反应发生率 12.6%"
    r = x.recall(gen, gt)
    per = r["per_category"]["percentage"]
    c.check("召回率 = overlap / gt_matches（与手算一致）",
            per["gt_matches"] == 3 and per["overlap"] == 2
            and abs(per["recall"] - 2 / 3) < 1e-12,
            f"overlap={per['overlap']} gt={per['gt_matches']} recall={per['recall']:.4f}")

    c.check("漏掉的信息被列出来（可定位是哪条丢了）", per["missed"] == ["55%"],
            f"missed={per['missed']}")

    r_empty = x.recall("任意答案", "")
    c.check("参照为空时 recall 记 None 而不是 0（没参照 ≠ 零召回）",
            r_empty["recall"] is None and r_empty["gt_matches"] == 0,
            f"recall={r_empty['recall']}")

    r_full = x.recall("剂量 10 mg/kg", "剂量 10mg/kg")
    c.check("归一化：「10 mg/kg」与「10mg/kg」视为同一条信息",
            r_full["per_category"]["dosage"]["recall"] == 1.0,
            f"recall={r_full['per_category']['dosage']['recall']}")

    same = x.recall(gt, gt)
    c.check("答案与参照完全相同时总召回 = 1.0", same["recall"] == 1.0,
            f"{same['overlap']}/{same['gt_matches']}")

    c.check("任务书点名的六类字段齐全",
            ev.TASKBOOK_CATEGORIES == ["percentage", "dosage", "duration", "safety",
                                       "recommendation", "mechanism"],
            f"{ev.TASKBOOK_CATEGORIES}")


# ============================================================================
# C 幻觉信号检测
# ============================================================================
def group_hallucination(c: Checker):
    c.head("C 幻觉信号检测")
    d = ev.HallucinationDetector()

    clean = ("现有证据显示该药在早期患者中可延缓认知下降 [S1]。"
             "证据来自 1 项 III 期随机对照试验，随访 18 个月 [S2]。")
    r = d.detect(clean)
    c.check("措辞谨慎且带出处的答案：无未缓解信号，风险 0",
            r["signals_unmitigated"] == 0 and r["risk_score"] == 0.0,
            f"信号 {r['signals_total']} 个，风险 {r['risk_score']}")

    cases = {
        "vague_citation": "研究表明该疗法有效。",
        "unqualified_proof": "该结论已被证明。",
        "absolute_percentage": "有效率达到 100%。",
        "over_absolute": "这种疗法完全安全。",
        "universal_quantifier": "所有患者都能获益。",
    }
    hit_all = []
    for name, text in cases.items():
        got = d.detect(text)["by_signal"]
        hit_all.append(name in got)
    c.check("任务书点名的四类信号 + 全称量化，逐条都能命中",
            all(hit_all), f"{list(cases)} → 命中 {sum(hit_all)}/{len(cases)}")

    cited = d.detect("研究表明该疗法可延缓病程 [S3]。")
    bare = d.detect("研究表明该疗法可延缓病程。")
    c.check("同一句话，带 [S#] 记为 mitigated、不带才计入风险",
            cited["signals_unmitigated"] == 0 and bare["signals_unmitigated"] == 1
            and cited["signals_total"] == bare["signals_total"] == 1,
            f"带出处未缓解 {cited['signals_unmitigated']} / 不带 {bare['signals_unmitigated']}")

    strict = ev.HallucinationDetector(count_cited_as_signal=True)
    c.check("严格模式下带出处的断言也计入风险（开关生效）",
            strict.detect("研究表明该疗法可延缓病程 [S3]。")["signals_unmitigated"] == 1,
            "count_cited_as_signal=True")

    pad = "补充说明这一点。" * 20                      # 同等篇幅，只改信号数量
    one = d.detect("研究表明该药有效。" + pad)
    two = d.detect("研究表明该药有效。已被证明有效。" + pad)
    four = d.detect("研究表明该药有效。已被证明完全安全，所有患者都能获益，有效率 100%。" + pad)
    c.check("信号越多风险分越高（同等篇幅，严格单调）",
            one["risk_score"] < two["risk_score"] < four["risk_score"],
            f"1 个 {one['risk_score']:.4f} < 2 个 {two['risk_score']:.4f} "
            f"< 4 个 {four['risk_score']:.4f}")
    # 在**真实答案量级**（本项目实测均 2528 字）上，四个信号仍应分得出轻重，不该顶死
    real_pad = "补充说明这一点。" * 300                # ≈2400 字，与真实答案同量级
    r1 = d.detect("研究表明该药有效。" + real_pad)
    r4 = d.detect("研究表明该药有效。已被证明完全安全，所有患者都能获益，有效率 100%。"
                  + real_pad)
    c.check("真实答案量级下不顶死在 1.0，仍能分出轻重",
            0.0 < r1["risk_score"] < r4["risk_score"] < 1.0,
            f"{len(real_pad)} 字文本：1 个信号 {r1['risk_score']:.4f} < "
            f"4 个信号 {r4['risk_score']:.4f} < 1.0")
    c.note("（信号密度极高的短文本仍会逼近 1.0——那是渐近上界，不是截断。"
           f"如上面 200 字含 4 个信号的例子实测 {four['risk_score']:.4f}）")

    longer = d.detect("研究表明该药有效。" + pad * 4)
    c.check("风险分随篇幅归一（长答案不会仅因为字多就被判高危）",
            longer["risk_score"] < one["risk_score"],
            f"同样 1 个信号，篇幅 ×4 后风险 {longer['risk_score']:.4f} < {one['risk_score']:.4f}")

    c.check("风险分恒在 [0,1] 区间",
            all(0.0 <= x["risk_score"] <= 1.0 for x in (r, one, two, four, cited, bare)),
            f"实测 {[round(x['risk_score'], 3) for x in (r, one, two, four, cited, bare)]}")

    sents = ev.split_sentences("给药方式 e.g. 静脉注射。随访 18 个月。")
    c.check("句子切分不把 e.g. 的点当句号", len(sents) == 2, f"切出 {len(sents)} 句：{sents}")


# ============================================================================
# D 可读性
# ============================================================================
def group_readability(c: Checker):
    c.head("D 可读性")
    rd = ev.ReadabilityEvaluator()

    text = "一二三四五。六七八九十。"                     # 两句各 5 字（不含句号）
    a = rd.analyze(text)
    expect = (6 + 6) / 2                                # split 后每句含句号 → 6 字符
    c.check("平均句子长度与手算一致", a["sentences"] == 2
            and abs(a["avg_sentence_length"] - expect) < 1e-9,
            f"{a['sentences']} 句，均 {a['avg_sentence_length']} 字（手算 {expect}）")

    long_text = "短句。" + "长" * 100 + "。"
    b = rd.analyze(long_text)
    c.check("长句比例按阈值算对", abs(b["long_sentence_ratio"] - 0.5) < 1e-9,
            f"2 句中 1 句超过 {b['long_sentence_threshold_chars']} 字 → "
            f"{b['long_sentence_ratio']}")

    mixed = "这是第一句中文。这是第二句中文。这是第三句中文。This sentence is in English."
    m = rd.analyze(mixed)
    c.check("中英混排比例算得出（阶段七的已知缺陷，现在可量化）",
            abs(m["mixed_language_ratio"] - 0.25) < 1e-9 and m["dominant_language"] == "zh",
            f"4 句里 1 句异种语言 → {m['mixed_language_ratio']}")

    s = rd.analyze("## 标题\n- 要点一\n- 要点二\n正文一句话。")
    c.check("能识别标题与列表结构", s["structured"] and s["headings"] == 1
            and s["list_items"] == 2,
            f"标题 {s['headings']} 个，列表 {s['list_items']} 项")

    empty = rd.analyze("")
    c.check("空文本不崩，各项为 0", empty["sentences"] == 0
            and empty["avg_sentence_length"] == 0.0, f"{empty['sentences']} 句")

    scores = [rd.analyze(t)["readability_score"] for t in (text, long_text, mixed, "")]
    c.check("可读性分恒在 [0,1] 区间", all(0.0 <= x <= 1.0 for x in scores),
            f"实测 {[round(x, 3) for x in scores]}")


# ============================================================================
# E 缓存
# ============================================================================
def group_cache(c: Checker):
    c.head("E 缓存：键 / TTL / 容量 / 温度门限 / 落盘 / 线程安全")
    cache = gc_.GenerationCache(max_entries=3, max_bytes=10 ** 9, ttl_seconds=60,
                                max_temperature=0.0)

    q, ctx = "问题 A", "证据块 X"
    k1 = cache.make_key(query=q, context=ctx, model="m", temperature=0.0)
    k2 = cache.make_key(query=q, context=ctx, model="m", temperature=0.0)
    c.check("相同查询 + 相同上下文 → 同一个键（相同输入相同输出）", k1 == k2, k1[:24] + "…")
    c.check("换了上下文 → 键不同",
            k1 != cache.make_key(query=q, context="证据块 Y", model="m", temperature=0.0))
    c.check("换了模型/温度参数 → 键不同",
            k1 != cache.make_key(query=q, context=ctx, model="m", temperature=0.3)
            and k1 != cache.make_key(query=q, context=ctx, model="n", temperature=0.0))

    ka = cache.make_key(query=q, context=ctx, a=1, b=2)
    kb = cache.make_key(query=q, context=ctx, b=2, a=1)
    c.check("键与参数书写顺序无关（canonical 序列化，不是 str(dict)）", ka == kb)

    cache.set(k1, {"answer": "A"}, temperature=0.0, meta={"elapsed": 12.5})
    c.check("写入后能命中", cache.get(k1) == {"answer": "A"})
    c.check("未写入的键返回 None", cache.get("不存在的键") is None)
    c.check("命中时累加省下的耗时", cache.stats["seconds_saved"] == 12.5,
            f"seconds_saved={cache.stats['seconds_saved']}")

    wrote = cache.set(cache.make_key(query="高温"), {"x": 1}, temperature=0.3)
    c.check("温度 0.3 > 门限 0.0 → 拒绝写入（只缓存确定性输出）",
            wrote is False and cache.stats["skipped_temperature"] == 1)
    c.check("温度 0.0 ≤ 门限 → 允许写入",
            cache.should_cache(0.0) and not cache.should_cache(0.3))

    # LRU：写满后再写新的，被淘汰的必须是**最久未使用**的那条
    lru = gc_.GenerationCache(max_entries=2, ttl_seconds=60, max_temperature=0.0)
    a, b, d = [lru.make_key(query=x) for x in "abd"]
    lru.set(a, {"v": "a"}, temperature=0.0)
    lru.set(b, {"v": "b"}, temperature=0.0)
    lru.get(a)                                   # 让 a 变成最近使用，b 成为最久未用
    lru.set(d, {"v": "d"}, temperature=0.0)
    c.check("容量上限触发 LRU 淘汰，淘汰的是最久未用的那条",
            len(lru) == 2 and (a in lru) and (d in lru) and (b not in lru),
            f"淘汰 {lru.stats['evicted_lru']} 条，剩下 a={a in lru} b={b in lru} d={d in lru}")

    small = gc_.GenerationCache(max_entries=1000, max_bytes=400, ttl_seconds=60,
                                max_temperature=0.0)
    for i in range(20):
        small.set(small.make_key(query=f"q{i}"), {"payload": "x" * 100}, temperature=0.0)
    c.check("字节上限生效，占用不超过 max_bytes",
            small.info()["bytes"] <= 400 and small.stats["evicted_bytes"] > 0,
            f"{small.info()['bytes']} ≤ 400 字节，按字节淘汰 {small.stats['evicted_bytes']} 条")

    ttl = gc_.GenerationCache(ttl_seconds=0.4, max_temperature=0.0)
    kt = ttl.make_key(query="过期测试")
    ttl.set(kt, {"v": 1}, temperature=0.0)
    before = ttl.get(kt) is not None
    time.sleep(0.5)
    after = ttl.get(kt) is None
    c.check("TTL 到期后读不到（医学知识有时效性，不能永久缓存）",
            before and after and ttl.stats["expired"] == 1,
            f"过期前命中={before}，过期后 miss={after}")

    # 落盘往返：有效条目留下、过期条目不被载入
    path = os.path.join(REPORT_DIR, "评估_验证_缓存往返.json")
    disk = gc_.GenerationCache(ttl_seconds=60, max_temperature=0.0, path=path)
    kk = disk.make_key(query="落盘测试", context="ctx")
    disk.set(kk, {"answer": "落盘的答案"}, temperature=0.0)
    disk.set(disk.make_key(query="短命"), {"v": 0}, temperature=0.0, ttl_seconds=0.3)
    disk.save()
    time.sleep(0.4)
    reload = gc_.GenerationCache(ttl_seconds=60, max_temperature=0.0)
    n_loaded = reload.load(path)
    c.check("落盘再载入：有效条目原样恢复，过期条目被丢弃",
            n_loaded == 1 and reload.get(kk) == {"answer": "落盘的答案"},
            f"载入 {n_loaded} 条（写入 2 条，其中 1 条已过期）")
    os.remove(path)

    # 线程安全：并发写不同键 + 并发读同一键
    tsafe = gc_.GenerationCache(max_entries=10_000, ttl_seconds=60, max_temperature=0.0)
    N, T = 400, 8
    errors: List[str] = []

    def worker(tid: int):
        try:
            for i in range(N):
                k = tsafe.make_key(query=f"t{tid}-{i}")
                tsafe.set(k, {"v": i}, temperature=0.0)
                tsafe.get(k)
        except Exception as e:                       # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(T)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c.check("多线程并发读写：无异常、条目数与命中数都对得上",
            not errors and len(tsafe) == N * T and tsafe.stats["hits"] == N * T
            and tsafe.stats["sets"] == N * T,
            f"{T} 线程 × {N} 次：条目 {len(tsafe)}，命中 {tsafe.stats['hits']}，"
            f"异常 {len(errors)} 个")


# ============================================================================
# F 批量处理
# ============================================================================
def group_batch(c: Checker):
    c.head("F 批量处理：顺序 / 错误隔离 / 并发度")

    def work(i: int) -> str:
        time.sleep(0.02 * ((i * 7) % 5 + 1))          # 耗时不一，故意打乱完成顺序
        if i % 5 == 3:
            raise ValueError(f"第 {i} 项故意失败")
        return f"r{i}"

    items = list(range(20))
    proc = bp.ParallelBatchProcessor(max_workers=6, kind="io")
    out = proc.run(items, work)

    c.check("输出长度与输入一致", len(out) == len(items), f"{len(out)} == {len(items)}")
    c.check("输出顺序与输入一一对应（不是按完成先后）",
            all(r["index"] == i for i, r in enumerate(out))
            and all(r["value"] == f"r{i}" for i, r in enumerate(out) if r["ok"]),
            "逐下标核对 index 与 value")

    failed = [r["index"] for r in out if not r["ok"]]
    c.check("单个任务失败不影响其他任务",
            failed == [3, 8, 13, 18] and sum(1 for r in out if r["ok"]) == 16,
            f"失败 {failed}，成功 {sum(1 for r in out if r['ok'])} 项")
    c.check("失败项保留位置并带错误信息（不塌陷、不静默）",
            out[3]["value"] is None and "第 3 项故意失败" in out[3]["error"]
            and bool(out[3]["traceback"]),
            f"out[3].error = {out[3]['error']}")

    # 换并发度，结果必须完全相同
    outs = {w: bp.ParallelBatchProcessor(max_workers=w, kind="io").run(items, work)
            for w in (1, 3, 8)}
    ref = [(r["index"], r["ok"], r["value"]) for r in outs[1]]
    same = all([(r["index"], r["ok"], r["value"]) for r in outs[w]] == ref for w in outs)
    c.check("并发度 1/3/8 得到完全相同的结果序列", same,
            "顺序与取值均一致，说明并发度不影响语义")

    w_llm, w_cpu, w_io = (bp.recommended_workers(k) for k in ("llm", "cpu", "io"))
    c.check("并发度推荐值随工作类型区分且不超核数上限",
            1 <= w_llm <= 4 and 1 <= w_cpu <= bp.CPU_COUNT and w_io >= w_cpu,
            f"CPU 逻辑核 {bp.CPU_COUNT} → llm={w_llm} cpu={w_cpu} io={w_io}")
    c.check("任务数少于并发度时按任务数收敛（不空开线程）",
            bp.recommended_workers("cpu", n_items=2) == 2)

    empty = bp.ParallelBatchProcessor(max_workers=4).run([], work)
    c.check("空输入返回空列表且不报错", empty == [], f"{empty}")

    rows = bp.benchmark(list(range(12)), lambda i: time.sleep(0.1), [1, 4], kind="io",
                        verbose=False)
    faster = rows[1]["wall_seconds"] < rows[0]["wall_seconds"]
    c.check("I/O 型任务并发确实更快（4 线程 vs 1 线程实测）", faster,
            f"1 线程 {rows[0]['wall_seconds']}s → 4 线程 {rows[1]['wall_seconds']}s "
            f"（{rows[1]['relative_to_first']}×）")
    c.note("注：本项目的 LLM 批量瓶颈在单块 GPU，不在 CPU；真实加速比见 评估_跑测试集.py 的实测。")


# ============================================================================
# G 与阶段七流水线的集成（用假生成器，不调模型）
# ============================================================================
class FakeGenerator:
    """冒充 LLMGenerator：只记录被调用了几次，返回可预测的结果。"""

    JSON_FORMAT_INSTRUCTION = "\n\n只输出 JSON。"

    def __init__(self):
        self.model_name = "fake-model"
        self.temperature = 0.0
        self.max_tokens = 100
        self.num_ctx = 12288
        self.think = False
        self.calls = 0
        self.stats = {"calls": 0}

    def generate_messages(self, messages, temperature=None, max_tokens=None,
                          json_output=False, expect="any", **kw):
        self.calls += 1
        return {"text": f"回答#{self.calls}", "elapsed": 1.0, "json": {"ok": True},
                "json_ok": True, "prompt_eval_count": 10, "eval_count": 5,
                "truncated": False, "attempts": []}

    def only_on_generator(self):
        return "转发成功"


def group_integration(c: Checker):
    c.head("G 缓存包装器与阶段七生成器的接口兼容（假生成器，不调模型）")
    fake = FakeGenerator()
    cache = gc_.GenerationCache(ttl_seconds=60, max_temperature=0.0)
    wrapped = gc_.CachedLLMGenerator(fake, cache)

    c.check("未知属性/方法原样转发给被包装的生成器",
            wrapped.model_name == "fake-model" and wrapped.only_on_generator() == "转发成功")

    # 这一条是被真跑测试集时的「写入 0 却跳过 16」抓出来的：GenerationCache 定义了 __len__，
    # 空缓存布尔值为 False，`cache or GenerationCache(...)` 会把注入的缓存悄悄换掉。
    c.check("注入的缓存对象确实被用上（空缓存不会因 __len__ 为 0 而被 `or` 换掉）",
            wrapped.cache is cache and len(cache) == 0 and bool(cache) is True,
            f"len(cache)={len(cache)} 但 bool(cache)={bool(cache)}，注入对象同一性={wrapped.cache is cache}")

    msgs = [{"role": "system", "content": "你是医学助理"},
            {"role": "user", "content": "问题 1"}]
    r1 = wrapped.generate_messages(msgs, temperature=0.0, max_tokens=100)
    r2 = wrapped.generate_messages(msgs, temperature=0.0, max_tokens=100)
    c.check("相同调用第二次命中缓存，底层生成器只被调用一次",
            fake.calls == 1 and r1["text"] == r2["text"] and r2.get("cache_hit") is True,
            f"底层调用 {fake.calls} 次，第二次 cache_hit={r2.get('cache_hit')}")
    c.check("命中时不谎报耗时（elapsed 归零，原耗时记进 cache_saved_seconds）",
            r2["elapsed"] == 0.0 and r2["cache_saved_seconds"] == 1.0,
            f"elapsed={r2['elapsed']} saved={r2['cache_saved_seconds']}")

    changed = [dict(msgs[0], content="你是严谨的医学助理"), msgs[1]]
    wrapped.generate_messages(changed, temperature=0.0, max_tokens=100)
    c.check("改了 system 提示词就不再命中（提示词进了键，杜绝读到旧答案）",
            fake.calls == 2, f"底层调用 {fake.calls} 次")

    before = fake.calls
    wrapped.generate_messages(msgs, temperature=0.3, max_tokens=100)
    wrapped.generate_messages(msgs, temperature=0.3, max_tokens=100)
    c.check("温度 0.3 的调用不走缓存（每次都真调模型）",
            fake.calls == before + 2, f"底层调用从 {before} 增到 {fake.calls}")

    class FailingGen(FakeGenerator):
        def generate_messages(self, messages, temperature=None, max_tokens=None,
                              json_output=False, expect="any", **kw):
            self.calls += 1
            return {"text": "", "elapsed": 0.5, "json": None, "json_ok": False,
                    "json_error": "解析失败", "prompt_eval_count": 1, "eval_count": 0,
                    "truncated": True, "attempts": []}

    fg = FailingGen()
    w2 = gc_.CachedLLMGenerator(fg, gc_.GenerationCache(ttl_seconds=60, max_temperature=0.0))
    w2.generate_messages(msgs, temperature=0.0)
    w2.generate_messages(msgs, temperature=0.0)
    c.check("失败结果不入缓存（否则把一次故障钉死成永久答案）",
            fg.calls == 2, f"两次调用都真的打到了底层：calls={fg.calls}")

    key = gc_.make_pipeline_key(cache, "查询", [{"chunk_id": "c1", "text": "证据一"}])
    key_same = gc_.make_pipeline_key(cache, "查询", [{"chunk_id": "c1", "text": "证据一"}])
    key_diff = gc_.make_pipeline_key(cache, "查询", [{"chunk_id": "c1", "text": "证据二"}])
    key_order = gc_.make_pipeline_key(cache, "查询",
                                      [{"chunk_id": "c2", "text": "证据二"},
                                       {"chunk_id": "c1", "text": "证据一"}])
    c.check("流水线级键：查询+检索证据一致则同键，证据内容或顺序变则换键",
            key == key_same and key != key_diff and key != key_order,
            "证据顺序会改变上下文，所以必须换键")


# ============================================================================
# 主流程
# ============================================================================
def load_real_answers(limit: int = 6) -> List[str]:
    """从阶段七的真实测试产物里取答案，供 A 组做实测对照。"""
    if not os.path.exists(LIVE_JSONL):
        return []
    out: List[str] = []
    with open(LIVE_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                a = json.loads(line).get("answer") or ""
            except json.JSONDecodeError:
                continue
            if a.strip():
                out.append(a)
            if len(out) >= limit:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="只打印汇总")
    args = ap.parse_args()

    t0 = time.time()
    c = Checker(quiet=args.quiet)
    c._out("=" * 92)
    c._out("阶段八验证 · 答案评估器 / 缓存 / 批量处理")
    c._out(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   "
           f"ROUGE 后端：{ev.ROUGE_BACKEND}   CPU 逻辑核：{bp.CPU_COUNT}")
    c._out("每条 PASS/FAIL 均由实际算出的变量比对得出，无一条是无条件打印。")

    real = load_real_answers()
    group_rouge(c, real)
    group_key_info(c)
    group_hallucination(c)
    group_readability(c)
    group_cache(c)
    group_batch(c)
    group_integration(c)

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
