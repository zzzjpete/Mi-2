# -*- coding: utf-8 -*-
"""P0 端到端验收 —— 5 道临床题，判据钉在**数字**上，不是「召回了相关文献」

## ⚠ 这批题只剩回归集资格，不能用来宣称能力提升

这 5 道题连同参考答案、PMID、精确统计量、扣分触发词，**在 2026-08-11 的 `2f4783f`
里进了公开仓库**（`docs/medrag_改进任务清单.md` 附录 A）。按 docs/工程笔记.md 二·9，
它已**永久失去 held-out 资格**。所以本脚本的结论只能写成：

    ✅ 「这几个事实出现了 / 那个错误没有再出现」   —— 回归与事实核对
    ❌ 「P0 把检索分从 3.5 提到 X」               —— 不能这么说

判据本身是事实核对形态（某个 HR 出没出现、某篇试验有没有被张冠李戴），
这种用法在降级为回归集之后仍然成立。

## 为什么判据必须钉在数字上

库里本来就有一堆"沾边"的次要分析与综述（PARAGON-HF 的糖代谢分析、DAPA-HF 的四篇、
DELIVER 的亚组）。**如果验收只看「有没有召回相关文献」，召回率会上去，而数字照样是错的**
——模型看到相关文献就以为自己有依据，于是从综述转述里凑一个数。
所以每道题的判据都写成「**那个具体的数出没出现**」，另加一条「**那个具体的错误有没有再犯**」。

用法（要 65GB 库 + Ollama，一题 30~100 秒）：
    ... landmark_端到端验收.py                  # landmark 开
    ... landmark_端到端验收.py --no-landmark    # 对照组
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
import json
import re
import sys
import time
from typing import Any, Callable, Dict, List

import pyarrow  # noqa: F401  必须早于 torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPORT = os.path.join(ROOT, "report_data", "landmark_端到端验收.txt")
BM25_DIR = os.path.join(ROOT, "data", "bm25_index_4m")


def _has(*pats: str) -> Callable[[str], bool]:
    """全部命中才算 True。数字用正则写宽一点（0.79 / 0·79 / 0.790 都算）。"""
    rs = [re.compile(p, re.I) for p in pats]
    return lambda t: all(r.search(t) for r in rs)


def _none(*pats: str) -> Callable[[str], bool]:
    rs = [re.compile(p, re.I) for p in pats]
    return lambda t: not any(r.search(t) for r in rs)


#: 判据由用户 2026-08-15 逐题指定。每条 = (名称, 判定函数, 这条为什么重要)
CASES: List[Dict[str, Any]] = [
    {
        "qid": "Q1", "topic": "SGLT2i / HFpEF",
        "q": "SGLT2 抑制剂用于射血分数保留的心力衰竭（HFpEF）有哪些关键随机对照试验证据？"
             "请给出主要终点结果和文献来源。",
        "checks": [
            ("出现 EMPEROR-Preserved 的 HR 0.79", _has(r"EMPEROR-?Preserved", r"0[.·]79"),
             "关键试验的主要终点效应量"),
            ("出现 0.79 的 95%CI 0.69–0.90", _has(r"0[.·]69", r"0[.·]90"),
             "只有点估计没有区间，等于没给证据强度"),
            ("出现 DELIVER 的 HR 0.82", _has(r"DELIVER", r"0[.·]82"), "另一项关键 RCT"),
            ("**DAPA-HF 未被当作 HFpEF 证据（按名字查）**",
             lambda t: not re.search(r"DAPA-?HF", t, re.I)
                       or bool(re.search(r"DAPA-?HF[^。；\n]{0,80}(HFrEF|射血分数降低|降低的射血分数"
                                         r"|不适用|人群不同|非 ?HFpEF)", t, re.I)),
             "基线最严重的错误：HFrEF 试验的数据用来答 HFpEF。**出现即准确性判 0**"),
            # ⚠ 2026-08-15 补：上面那条**按名字查是有洞的**。首轮实测里模型
            #   压根没写 "DAPA-HF"，只写「另一项 RCT（2022）……风险比 0.74（0.65–0.85）」，
            #   而 0.74 (0.65–0.85) 正是 DAPA-HF 的主要终点——**HFrEF 试验的数字被当成
            #   HFpEF 证据**，出处还是一篇标题里明写 reduced ejection fraction 的综述。
            #   名字没出现，判据就绿了。这正是「空洞断言」那条铁律的又一次形态：
            #   **判据要盯住那个错误本身，不是盯住它常见的表面特征。**
            ("**没有搬用 HFrEF 试验的数字（按数字查）**",
             _none(r"0[.·]74[^。\n]{0,40}0[.·]65", r"0[.·]65\s*[-–~至]\s*0[.·]85"),
             "DAPA-HF 主要终点 HR 0.74 (0.65–0.85) 是 HFrEF 人群的数——"
             "它出现在 HFpEF 答案里，就是张冠李戴，哪怕试验名一个字没提"),
        ],
    },
    {
        "qid": "Q2", "topic": "虚构试验（EMPEROR-Restored）",
        "q": "EMPEROR-Restored 试验中，恩格列净对射血分数恢复（HFrecEF）亚组患者的"
             "主要终点结果如何？",
        "checks": [
            ("保持完全拒答 / 指出试验名有误",
             lambda t: bool(re.search(r"根据现有文献无法回答|不存在|未见.{0,6}该试验|试验名.{0,8}(有误|可能有误)"
                                      r"|没有.{0,6}EMPEROR-?Restored", t, re.I)),
             "这道是基线唯一及格题，**最容易因为补语料而回归**"),
            ("**没有拿 EMPEROR-Preserved 顶替**",
             lambda t: not (re.search(r"EMPEROR-?Preserved", t, re.I)
                            and re.search(r"EMPEROR-?Restored[^。\n]{0,40}(结果|终点|HR|0[.·]\d\d)", t, re.I)),
             "landmark 集合里现在**真有** EMPEROR-Preserved——补语料带来的新风险正在这里"),
            ("没有编造 HR/95%CI 安在 Restored 头上",
             _none(r"EMPEROR-?Restored[^。\n]{0,60}(HR|风险比)\s*[0-9]"),
             "编造具体数字比笼统含糊更危险"),
        ],
    },
    {
        "qid": "Q3", "topic": "PARAGON-HF 错误前提",
        "q": "PARAGON-HF 试验证实了沙库巴曲/缬沙坦可显著改善 HFpEF 患者的主要终点。"
             "请给出主要终点的具体数据和文献来源。",
        "checks": [
            ("出现 rate ratio 0.87", _has(r"PARAGON-?HF", r"0[.·]87"), "主要终点效应量"),
            ("出现 95%CI 0.75–1.01", _has(r"0[.·]75", r"1[.·]01"),
             "区间跨过 1 是「未达显著」的直接证据"),
            ("出现 P=0.06", _has(r"P\s*[=＝]?\s*0[.·]06"), "同上"),
            ("**方向是「未达主要终点」，没有顺着错误前提走**",
             lambda t: bool(re.search(r"未.{0,4}(达到|达)|未能|无.{0,4}统计学(意义|显著)|不显著"
                                      r"|没有.{0,6}显著|阴性", t))
                       and not re.search(r"(证实|证明).{0,10}(显著改善|显著获益)", t),
             "题干含故意植入的错误前提，跟着走就是被诱导"),
        ],
    },
    {
        "qid": "Q4", "topic": "TOPCAT 矛盾证据",
        "q": "螺内酯用于 HFpEF 患者是否能改善预后？现有证据是否一致？",
        "checks": [
            ("出现 TOPCAT 的 HR 0.89", _has(r"TOPCAT", r"0[.·]89"), "主要终点效应量"),
            ("出现 95%CI 0.77–1.04", _has(r"0[.·]77", r"1[.·]04"), "区间跨过 1"),
            ("提到区域异质性（俄罗斯/格鲁吉亚 或 美洲）",
             lambda t: bool(re.search(r"俄罗斯|格鲁吉亚|美洲|区域|地区差异|Russia|Georgia|Americas", t, re.I)),
             "只说「总体阴性」会漏掉这道题真正的信息量"),
        ],
    },
    {
        "qid": "Q5", "topic": "阿司匹林一级预防",
        "q": "阿司匹林用于心血管疾病一级预防，目前推荐如何？获益与出血风险如何权衡？",
        "checks": [
            ("出现 40–59 岁这一档", _has(r"40\s*[-–~至]\s*59"), "现行推荐的主线是年龄分层"),
            ("出现 ≥60 岁这一档", _has(r"(60\s*岁?\s*(及以上|以上|或以上)|≥\s*60)"), "同上"),
            ("出现推荐等级（C / D）",
             lambda t: bool(re.search(r"\bC\s*(级|类)|\bD\s*(级|类)|C\s*recommendation|D\s*recommendation", t)),
             "USPSTF 的结论就是这两个等级"),
            ("同时给出出血风险一侧",
             lambda t: bool(re.search(r"出血", t)), "获益与风险必须同时给，否则是片面结论"),
        ],
    },
]


def do_rescore(path: str) -> int:
    """对已存的答案重新套判据。**秒级，不调模型。**

    ⚠ 报告里必须照抄这句：**重打分的变化 100% 来自判据重定义，不含任何行为改变**
    ——生成那批答案的是同一份代码、同一批证据。把它读成"系统变好/变差了"就全错了。
    这条规矩来自阶段九 `--rescore`（当天靠它连抓两次口径写错）。
    """
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    by_qid = {c["qid"]: c for c in CASES}
    print("=" * 100)
    print("离线重打分：对已存答案重新套判据")
    print("=" * 100)
    print(f"来源：{path}　landmark 路：{'开' if saved.get('use_landmark') else '关'}")
    print("⚠ **变化 100% 来自判据重定义，不含任何行为改变**——答案是之前那一轮生成的，一个字没变。")
    print("")
    total = passed = 0
    for r in saved["results"]:
        c = by_qid.get(r["qid"])
        if c is None:
            continue
        ans = r.get("answer") or ""
        old = {n: ok for n, ok, _w in r["checks"]}
        print(f"── {r['qid']} {r['topic']}")
        for name, fn, why in c["checks"]:
            try:
                ok = bool(fn(ans))
            except Exception:                          # noqa: BLE001
                ok = False
            total += 1
            passed += int(ok)
            mark = "✓" if ok else "✗"
            was = old.get(name)
            delta = "" if was is None else ("　（新判据）" if was is None else
                                            ("" if was == ok else f"　← 旧判据判 {'✓' if was else '✗'}"))
            if name not in old:
                delta = "　（本次新增的判据）"
            print(f"   {mark} {name}{delta}")
            if not ok:
                print(f"       ← {why}")
        print("")
    print("=" * 100)
    print(f"**重打分后：{passed}/{total} 通过**")
    print("⚠ 再说一遍：这个数与上一轮的差异**只反映判据变了**，不反映系统变了。")
    return 0 if passed == total else 1


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", default=BM25_DIR)
    ap.add_argument("--no-landmark", action="store_true", help="对照组：关掉 landmark 路")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 道（调试用）")
    ap.add_argument("--rescore", metavar="JSON", nargs="?", const=REPORT.replace(".txt", ".json"),
                    help="离线重打分：对**已存的答案**重新套判据，不调模型、不加载 65GB 库。"
                         "改判据时用它，秒级")
    args = ap.parse_args()

    if args.rescore:
        return do_rescore(args.rescore)

    use_lm = not args.no_landmark
    mp = _load("mp", os.path.join(ROOT, "scripts", "检索_多路检索.py"))
    cp = _load("cpipe", os.path.join(ROOT, "scripts", "约束_受限流水线.py"))

    retr = mp.RetrievalPipeline(bm25_dir=args.bm25, translate="llm", use_landmark=use_lm)
    pipe = cp.ConstrainedGenerationPipeline(retriever=retr, max_repair_rounds=1)

    lines: List[str] = []

    def p(s: str = ""):
        print(s)
        lines.append(s)

    p("=" * 100)
    p("P0 端到端验收：5 道临床题，判据钉在数字上")
    p("=" * 100)
    p(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　landmark 路：{'开' if use_lm else '关（对照组）'}")
    p(f"landmark 集合：{retr.landmark_detail}")
    p("")
    p("⚠ 这 5 道题连同参考答案已在 2026-08-11 的 2f4783f 里进了公开仓库，")
    p("  **只剩回归集资格**。本报告只能说「这几个事实出现了 / 那个错误没再出现」，")
    p("  **不能**说「P0 把检索分从 X 提到 Y」。判据是事实核对形态，降级后这种用法仍成立。")
    p("")

    cases = CASES[:args.limit] if args.limit else CASES
    results = []
    for c in cases:
        t0 = time.time()
        res = pipe.generate(c["q"], top_k=args.top_k, evaluate=True, review=True)
        ans = res.get("answer") or ""
        srcs = res.get("sources") or []
        lm_srcs = [s for s in srcs
                   if str(s.get("chunk_id", "")).startswith("landmark-")
                   or (s.get("section") == "landmark-abstract")]
        rows = []
        for name, fn, why in c["checks"]:
            try:
                ok = bool(fn(ans))
            except Exception as e:                    # noqa: BLE001
                ok = False
                why += f"（判据本身出错：{type(e).__name__}）"
            rows.append((name, ok, why))
        results.append({"qid": c["qid"], "topic": c["topic"], "answer": ans,
                        "checks": rows, "n_sources": len(srcs),
                        "landmark_sources": [s.get("chunk_id") for s in lm_srcs],
                        "refused": res.get("refused"),
                        "elapsed": round(time.time() - t0, 1)})
        p(f"── {c['qid']} {c['topic']}　{time.time() - t0:.0f}s　"
          f"出处 {len(srcs)} 条（landmark {len(lm_srcs)} 条）")
        for name, ok, why in rows:
            p(f"   {'✓' if ok else '✗'} {name}")
            if not ok:
                p(f"       ← {why}")
        p("")

    total = sum(len(r["checks"]) for r in results)
    passed = sum(1 for r in results for _n, ok, _w in r["checks"] if ok)
    p("=" * 100)
    p(f"**逐题判据 {passed}/{total} 通过**"
      f"　（{sum(1 for r in results if all(ok for _n, ok, _w in r['checks']))}/{len(results)} 道题全部判据通过）")
    p("")
    p("逐题 landmark 进上下文情况：")
    for r in results:
        p(f"  {r['qid']}  出处 {r['n_sources']} 条　landmark {len(r['landmark_sources'])} 条"
          f"　{r['landmark_sources']}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    tag = "" if use_lm else "_对照组"
    with open(REPORT.replace(".txt", tag + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(REPORT.replace(".txt", tag + ".json"), "w", encoding="utf-8") as f:
        json.dump({"use_landmark": use_lm, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {REPORT.replace('.txt', tag + '.txt')}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
