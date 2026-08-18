# -*- coding: utf-8 -*-
"""P0 · landmark 补充集合 —— 取摘要（唯一需要联网的一步）+ 建独立 collection

## 为什么要有这个集合

外部黑盒评测里，10 篇心内科 landmark（EMPEROR-Preserved / DELIVER / PARAGON-HF /
TOPCAT / DAPA-HF / EMPEROR-Reduced / USPSTF 2022 / ASPREE / ASCEND）**命中 0 篇**。
2026-08-15 按 PMID 在 `docs_catalog.db` 里逐条查证：**10 篇全部不在 4M 索引里**
（情况① 不在 oa_comm，②③ 各 0）。语料是 PMC OA 的 `oa_comm` 子集，NEJM/JAMA/Lancet
的原始 RCT 与学会指南许可证不满足 CC BY，**物理上不在库里**——
任何 rerank / prompt 优化都补不上，只能补语料。

## ⚠ 联网边界（用户 2026-08-15 明确批准，措辞照抄）

    **构建期一次性联网**（取 landmark 摘要，10 个 PMID），**运行期仍全离线**。

「全离线」是手段不是目的：这次是构建期一次性取数，取完落盘，之后建库/检索/问答全程离线，
**跑服务的人永远不需要联网**。`data/landmark/entries.json` 落盘后，
重建 collection（`--build`）不需要再联网。

发出去的内容只有 10 个 PMID，取回的是公开摘要。**没有发送任何本机数据、用户信息或查询日志。**

## ⚠ 校验规矩：只用 PMID 定位，绝不用标题反查

实测缩写撞名很严重：`TOPCAT` 撞上妇科肿瘤试验
*Trial of Optimal Personalised Care After Treatment for Gynaecological cancer*，
`ASCEND` 撞上一堆 *ascending aorta*。所以：

  · **定位只用 PMID**；
  · efetch 回来**逐条核对 PMID ↔ 期刊 / 年份 / 标题关键词**，
    对不上就**跳过并报出来**，`--fetch` 以非 0 退出；
  · **不许静默接受，也不许静默丢弃**——三项分开报，由人看着定。

## ⚠ 条目里的数字必须是原文，不能是转述

这个集合是**证据**，不是笔记。所以 `text` 存的是 **efetch 返回的摘要原文**（含结构化小标题），
一个字不改写；`n_randomized` / `effect_size` 这些结构化字段只由**确定性正则**从原文里抽，
抽不到就留 `null` **并在 `extraction` 里标明**，绝不猜、绝不调模型去"理解"。

理由是这个项目的立身之本：如果条目本身的 HR 是模型转述出来的，那它就是一个**带出处、
可溯源、但数字是编的**证据源——比没有这个集合更糟。

## 用法

    ... landmark_构建集合.py --fetch      # 唯一联网的一步：取摘要 + 校验 + 落盘
    ... landmark_构建集合.py --verify     # 离线重放校验（读已落盘的 entries.json）
    ... landmark_构建集合.py --build      # 离线：向量化并写入 collection medrag_landmark
    ... landmark_构建集合.py --show       # 离线：看每条抽到了什么、缺什么
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED_PATH = os.path.join(ROOT, "scripts", "landmark_种子清单.json")
OUT_DIR = os.path.join(ROOT, "data", "landmark")
ENTRIES_PATH = os.path.join(OUT_DIR, "entries.json")
RAW_DIR = os.path.join(OUT_DIR, "raw")
CHROMA_DIR = os.path.join(ROOT, "data", "chroma_landmark")
COLLECTION = "medrag_landmark"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"        # 与主库对齐，否则两路向量不可比

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
#: NCBI 建议带上 tool 与 email 以便联系。**email 留空是刻意的**——
#: 用户批准的是「取 10 个 PMID 的摘要」，没批准把本人邮箱发给第三方。
#: 单次小请求不带 email 符合 NCBI 的使用条款（email 是 recommended 不是 required）。
ETOOL = "medrag-landmark"
EEMAIL = ""


# ============================================================================
# 一、取（唯一联网的一步）
# ============================================================================
def _load_seed(path: str = SEED_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def efetch(pmids: List[str], timeout: int = 30) -> str:
    """一次请求取全部 PMID。**只发一次**——NCBI 无 key 时限 3 请求/秒，
    十个 id 拼一次请求既礼貌又省事。"""
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": ETOOL}
    if EEMAIL:
        params["email"] = EEMAIL
    url = EUTILS + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"{ETOOL}/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _text(node: Optional[ET.Element]) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""


def parse_pubmed_xml(xml_text: str) -> Dict[str, Dict[str, Any]]:
    """PubmedArticle → {pmid: {...}}。摘要保留结构化小标题（NEJM 的 RCT 摘要
    带 BACKGROUND/METHODS/RESULTS/CONCLUSIONS，正是我们要的那几段）。"""
    out: Dict[str, Dict[str, Any]] = {}
    root = ET.fromstring(xml_text)
    for art in root.findall(".//PubmedArticle"):
        pmid = _text(art.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue
        journal = art.find(".//Journal")
        year = _text(journal.find(".//PubDate/Year")) if journal is not None else ""
        if not year:
            medline_date = _text(journal.find(".//PubDate/MedlineDate")) if journal is not None else ""
            m = re.search(r"\b(19|20)\d{2}\b", medline_date)
            year = m.group(0) if m else ""
        pieces, labels = [], []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = (ab.get("Label") or "").strip()
            body = _text(ab)
            if not body:
                continue
            labels.append(label.upper() if label else "")
            pieces.append(f"{label}: {body}" if label else body)
        ids = {el.get("IdType"): _text(el) for el in art.findall(".//ArticleIdList/ArticleId")}
        out[pmid] = {
            "pmid": pmid,
            "title": _text(art.find(".//ArticleTitle")),
            "journal": _text(journal.find(".//ISOAbbreviation")) if journal is not None else "",
            "journal_full": _text(journal.find(".//Title")) if journal is not None else "",
            "pub_year": int(year) if year.isdigit() else None,
            "abstract": "\n\n".join(pieces),
            "abstract_labels": labels,
            "pmcid": ids.get("pmc", ""),
            "doi": ids.get("doi", ""),
            "pub_types": [_text(t) for t in art.findall(".//PublicationType")],
        }
    return out


# ============================================================================
# 二、校验（PMID ↔ 期刊 / 年份 / 标题关键词，三项分开报）
# ============================================================================
def verify(seed_entry: Dict[str, Any], fetched: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """返回逐项判定。**不替人拍板**：三项分别给 True/False/None，
    调用方决定怎么处理，报告里三项都打出来。"""
    if fetched is None:
        return {"ok": False, "reason": "efetch 没返回这个 PMID", "checks": {}}
    checks: Dict[str, Any] = {}

    want_j = (seed_entry.get("expect_journal") or "").lower()
    got_j = (fetched.get("journal") or "").lower()
    got_jf = (fetched.get("journal_full") or "").lower()
    # 期刊名有多种写法（N Engl J Med / The New England Journal of Medicine），
    # 用"任一方向包含"判定，避免因缩写风格不同误判
    checks["journal"] = bool(want_j) and (want_j in got_j or got_j in want_j
                                          or want_j in got_jf
                                          or _nejm_alias(want_j, got_j, got_jf))

    want_y = seed_entry.get("expect_year")
    got_y = fetched.get("pub_year")
    checks["year"] = (want_y is None) or (got_y is not None and abs(got_y - want_y) <= 1)

    want_t = (seed_entry.get("expect_title_contains") or "").lower()
    checks["title_kw"] = (not want_t) or (want_t in (fetched.get("title") or "").lower())

    checks["has_abstract"] = bool((fetched.get("abstract") or "").strip())

    ok = all(v for v in checks.values())
    return {"ok": ok, "checks": checks,
            "reason": "" if ok else "、".join(k for k, v in checks.items() if not v)}


def _nejm_alias(want: str, got: str, got_full: str) -> bool:
    alias = {"n engl j med": ("new england journal of medicine",)}
    for k, vs in alias.items():
        if want == k and any(v in got_full or v in got for v in vs):
            return True
    return False


# ============================================================================
# 三、结构化字段：只用确定性正则，抽不到就留 null
# ============================================================================
#: NEJM/JAMA 的效应量几乎总写在一对括号里：
#: 「(hazard ratio, 0.79; 95% confidence interval [CI], 0.69 to 0.90; P<0.001)」
#: ⚠ 第一版写成 `[^;.\n]{0,60}` 去跨越 HR 与 95%CI，结果**一条都没抽到**——
#: 那两段之间正好隔着一个分号，被字符类排除了。整段括号一起取，既简单又不会切半截。
_RE_HR = re.compile(
    r"\([^()]{0,200}?(?:hazard ratio|risk ratio|rate ratio|odds ratio|relative risk|"
    r"\bHR\b|\bRR\b|\bOR\b)[^()]{0,200}?\d+\.\d+[^()]{0,200}?\)", re.I)
_RE_N = re.compile(
    r"(?:a total of|randomly assigned|randomized|enrolled|we assigned)\D{0,40}?"
    r"([\d,]{3,9})\s*(?:patients|participants|adults|persons|subjects)", re.I)
_RE_P = re.compile(r"P\s*[<=]\s*0?\.\d+|P\s*[<=]\s*\d+\.\d+e-\d+", re.I)


def extract_fields(fetched: Dict[str, Any]) -> Dict[str, Any]:
    """从**摘要原文**里抽结构化字段。抽不到留 null 并在 `extraction` 里标明。

    ⚠ 刻意不调模型：这个集合是证据源，条目里的 HR 若是模型转述出来的，
    它就成了一个「带出处、可溯源、但数字是编的」来源——比没有这个集合更糟。
    """
    abstract = fetched.get("abstract") or ""
    sections = _split_labeled(abstract)

    n = None
    mn = _RE_N.search(sections.get("METHODS", "") or abstract)
    if mn:
        try:
            n = int(mn.group(1).replace(",", ""))
        except ValueError:
            n = None

    eff = None
    me = _RE_HR.search(sections.get("RESULTS", "") or abstract)
    if me:
        eff = me.group(0).strip()

    p = None
    mp = _RE_P.search(sections.get("RESULTS", "") or abstract)
    if mp:
        p = mp.group(0)

    got = {"n_randomized": n, "effect_size": eff, "p_value": p}
    # ⚠ 这三个字段**存的是摘要的整段原文**，名字必须如实叫 *_text。
    #   最初我把它们命名成 population / primary_endpoint / key_secondary，
    #   结果 2026-08-15 加人群标签时，`**ext` 展开**静默覆盖**了种子清单里的
    #   `population`（真正的人群标签），锚点行变成了一整段 METHODS 正文。
    #   **一个字段名承诺了它没有的语义，就迟早会有人（包括我）按名字去用它。**
    return {
        "methods_text": sections.get("METHODS") or None,      # 原文段，不改写
        "results_text": sections.get("RESULTS") or None,
        "conclusion_text": sections.get("CONCLUSIONS") or None,
        **got,
        "extraction": {k: ("regex" if v is not None else "未抽到（留空，不猜）")
                       for k, v in got.items()},
    }


def _split_labeled(abstract: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for block in abstract.split("\n\n"):
        m = re.match(r"^([A-Z][A-Z /&-]{2,30}):\s*(.+)$", block, re.S)
        if m:
            out[m.group(1).strip().upper()] = m.group(2).strip()
    return out


def evidence_level(fetched: Dict[str, Any]) -> str:
    types = " ".join(fetched.get("pub_types") or []).lower()
    if "randomized controlled trial" in types:
        return "RCT"
    if "meta-analysis" in types:
        return "meta"
    if "guideline" in types or "practice guideline" in types:
        return "guideline"
    return "other"


def build_text(entry: Dict[str, Any]) -> str:
    """供向量化的自然语言文本。**开头是标题、试验名与别名（检索锚点），主体是摘要原文**。
    摘要一个字不改写——改写等于在证据里引入一次转述。

    ⚠ **别名那一行是 2026-08-15 补的，起因是一次真实的漏检**：
    问「Does sacubitril/valsartan reduce hospitalization in HFpEF?」时，
    PARAGON-HF 的交叉编码器 rel 只有 **0.4087**，而主库里几篇标题直接写着
    `sacubitril/valsartan` / `HFpEF` 的文章拿到 **0.97~0.99**（其中一篇还是干细胞治疗综述）。
    根因是这篇论文的标题写的是 *Angiotensin–Neprilysin Inhibition*——
    **和临床医生真正会说的词一个都对不上**。

    别名只放**该试验的事实性属性**（药物通用名 / 药理类别 / 人群简写），不是关键词堆砌，
    更不是为了提分而编。⚠ 但它确实会提高 landmark 被任意 query 拉进来的概率，
    所以**改完必须重跑 golden 回归确认代价仍≈0**，只看探针会自我印证。

    ⚠ **人群写在第一行，这个位置是量出来的**（2026-08-15 的对照实验，HFpEF 题面）：

        排布                 EMPEROR-Preserved(对)   DAPA-HF(错人群)
        标题+别名（现状）           0.6117               0.3965
        别名前置                    0.8603               0.7446   ← 对的错的一起抬
        **人群前置**                0.8307               0.4399   ← 只抬对人群的

    别名前置把 EMPEROR-Preserved 抬得更高，但同时把 **HFrEF 试验 DAPA-HF 抬到 0.74**——
    而 Q1 问的是 HFpEF，DAPA-HF 的数据用来答 HFpEF 正是这套评测里最严重的那个错。
    **一个把正确答案和错误答案一起抬起来的改法不是改法。**
    人群前置则让交叉编码器先看到人群标签，对人群的抬、错人群的原地不动甚至下降。
    """
    lines = []
    if entry.get("population"):
        lines.append(f"Population: {entry['population']}")
    if entry.get("aliases"):
        lines.append("Key terms: " + "; ".join(entry["aliases"]))
    lines.append(f"{entry['trial_name']} ({entry['journal']} {entry['pub_year']}). "
                 f"{entry['title']}")
    return "\n".join(lines) + "\n\n" + (entry.get("abstract") or "")


# ============================================================================
# 四、命令
# ============================================================================
def cmd_fetch(args: argparse.Namespace) -> int:
    seed = _load_seed(args.seed)
    pmids = [e["pmid"] for e in seed["entries"]]
    print(f"种子清单 {args.seed}｜{len(pmids)} 条 PMID")
    print("⚠ 这是**本项目唯一一次对外网络请求**：向 NCBI E-utilities 取这 10 个 PMID 的公开摘要。")
    print("  发出去的只有 PMID，取回后落盘；之后建库/检索/问答全程离线。\n")

    t0 = time.time()
    try:
        xml_text = efetch(pmids, timeout=args.timeout)
    except Exception as e:                       # noqa: BLE001
        print(f"❌ efetch 失败：{type(e).__name__}: {e}")
        print("   没有落盘任何东西。检查网络/代理后重跑；离线环境可让人工导出 XML 后用 --from-xml。")
        return 2
    print(f"efetch 返回 {len(xml_text):,} 字节，用时 {time.time() - t0:.1f}s")

    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, f"efetch_{time.strftime('%Y%m%d_%H%M%S')}.xml")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(xml_text)
    print(f"原始 XML → {raw_path}（留证，出问题时能回看是取回来的就是这样，还是解析错了）")

    return _process(seed, xml_text)


def cmd_from_xml(args: argparse.Namespace) -> int:
    """离线通道：别人代取的 XML 也能进来（用户若不希望本机联网时用）。"""
    with open(args.from_xml, "r", encoding="utf-8") as f:
        xml_text = f.read()
    print(f"读本地 XML {args.from_xml}（不联网）")
    return _process(_load_seed(args.seed), xml_text)


def _process(seed: Dict[str, Any], xml_text: str) -> int:
    fetched = parse_pubmed_xml(xml_text)
    print(f"解析出 {len(fetched)} 条记录\n")

    print("=" * 104)
    print(f"{'试验':<26}{'PMID':<11}{'期刊':<8}{'年份':<6}{'标题词':<8}{'摘要':<6}实际标题")
    print("-" * 104)

    entries: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str, Dict[str, Any]]] = []
    for s in seed["entries"]:
        f = fetched.get(s["pmid"])
        v = verify(s, f)
        mark = lambda b: "✓" if b else ("✗" if b is False else "—")   # noqa: E731
        c = v["checks"]
        title = (f or {}).get("title", "")
        print(f"{s['trial_name'][:24]:<26}{s['pmid']:<11}"
              f"{mark(c.get('journal')):<8}{mark(c.get('year')):<6}"
              f"{mark(c.get('title_kw')):<8}{mark(c.get('has_abstract')):<6}{title[:44]}")
        if not v["ok"]:
            skipped.append((s["trial_name"], v["reason"], {"seed": s, "fetched": f}))
            continue
        ext = extract_fields(f)
        entry = {
            "trial_name": s["trial_name"], "topic": s.get("topic", ""),
            "pmid": f["pmid"], "pmcid": f.get("pmcid", ""), "doi": f.get("doi", ""),
            "journal": f["journal"], "pub_year": f["pub_year"], "title": f["title"],
            "evidence_level": evidence_level(f),
            "abstract": f["abstract"], "abstract_labels": f["abstract_labels"],
            "caveats": s.get("caveats", ""), "aliases": s.get("aliases") or [],
            "population": s.get("population", ""),
            **{k: v2 for k, v2 in ext.items()},
            "verified": v["checks"], "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        entry["text"] = build_text(entry)
        entries.append(entry)

    print("-" * 104)
    print(f"通过 {len(entries)} 条｜跳过 {len(skipped)} 条")
    if skipped:
        print("\n⚠ 以下条目**校验没过，已跳过，没有写入**（不静默接受）：")
        for name, reason, ctx in skipped:
            got = ctx["fetched"]
            print(f"  · {name}（PMID {ctx['seed']['pmid']}）：{reason} 对不上")
            if got:
                print(f"      期望 {ctx['seed'].get('expect_journal')} "
                      f"{ctx['seed'].get('expect_year')} 含「{ctx['seed'].get('expect_title_contains')}」")
                print(f"      实际 {got.get('journal')} {got.get('pub_year')}｜{got.get('title', '')[:70]}")
            else:
                print("      efetch 没返回这个 PMID")
        print("  处理方式：核对种子清单里的期望值是不是抄错了（期望值本来就未经证实），")
        print("            确认无误再改清单重跑——**不要为了让它通过而放宽校验**。")

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "specialty": seed.get("specialty"),
               "source": "PubMed E-utilities efetch（构建期一次性联网，运行期不需要）",
               "seed": os.path.basename(SEED_PATH),
               "n_entries": len(entries), "n_skipped": len(skipped),
               "skipped": [{"trial_name": n, "reason": r, "pmid": c["seed"]["pmid"]}
                           for n, r, c in skipped],
               "entries": entries}
    with open(ENTRIES_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n条目 → {ENTRIES_PATH}")
    print("  ⚠ 这份文件含摘要正文，**在 .gitignore 内、不进公开仓库**；"
          "种子清单（只有 PMID）才进仓库。")
    print("  ⚠ 有了它，重建 collection（--build）**不需要再联网**。")

    _report_extraction(entries)
    return 0 if not skipped else 1


def _report_extraction(entries: List[Dict[str, Any]]) -> None:
    """把「哪几条的关键数字没抽到」明着报出来——这正是 P0 验收要看的东西。"""
    if not entries:
        return
    print("\n" + "=" * 104)
    print("结构化字段抽取情况（抽不到就是 null，不猜）")
    print("=" * 104)
    print(f"{'试验':<26}{'入组人数':<12}{'效应量（原文片段）'}")
    print("-" * 104)
    miss = 0
    for e in entries:
        n = e.get("n_randomized")
        eff = (e.get("effect_size") or "")[:58]
        if n is None or not eff:
            miss += 1
        print(f"{e['trial_name'][:24]:<26}{str(n or '未抽到'):<12}{eff or '未抽到'}")
    print("-" * 104)
    print(f"{len(entries) - miss}/{len(entries)} 条同时抽到了入组人数与效应量。")

    # ---- P0 的验收判据看的是这一格：**要害的数字在不在 text 里** ----
    # 用户 2026-08-15 定的口径：验收不是「有没有召回相关文献」，而是
    # 「召回的那一块里有没有那个数」（主要终点 HR、95%CI、入组人数）。
    print("\n关键信息在不在 `text` 里（检索与引用真正用到的那段）：")
    print("⚠ 判据按条目类型分开——**指南没有 HR 是结构使然，不是缺陷**。")
    print("  拿 RCT 的判据去量指南，会得出一个「不合格」的假结论，而那只是尺子用错了。")
    print(f"{'试验':<26}{'类型':<10}{'要害信息':<10}{'明细'}")
    print("-" * 104)
    ok_n = 0
    for e in entries:
        t = e.get("text") or ""
        lvl = e.get("evidence_level", "other")
        # ⚠ 第一版漏了 "rate ratio" 与 "relative risk"，把 ASCEND 误报成「没有效应量」——
        # 而它的摘要里明明白白写着 rate ratio, 0.88。**判据写窄了会造出假阴性。**
        has_ratio = bool(re.search(
            r"(hazard ratio|risk ratio|rate ratio|odds ratio|relative risk|"
            r"\bHR\b|\bRR\b|\bOR\b)", t, re.I))
        has_ci = bool(re.search(r"95%\s*(confidence interval|CI)", t, re.I))
        has_n = bool(re.search(r"[\d,]{3,9}\s*(patients|participants|adults|persons)", t, re.I))
        if lvl == "guideline":
            # 指南的「要害」是推荐本身：等级 + 适用人群/年龄分层
            has_grade = bool(re.search(r"\b[A-D]\s+recommendation|grade\s+[A-D]\b", t, re.I))
            has_pop = bool(re.search(r"\b\d{2}\s*(to|-|–)\s*\d{2}\s*years|aged\s*\d{2}", t, re.I))
            ok = has_grade and has_pop
            detail = f"推荐等级 {'✓' if has_grade else '✗'}　年龄分层 {'✓' if has_pop else '✗'}"
        else:
            ok = has_ratio and has_ci and has_n
            detail = (f"效应量 {'✓' if has_ratio else '✗'}　95%CI {'✓' if has_ci else '✗'}"
                      f"　入组人数 {'✓' if has_n else '✗'}")
        ok_n += int(ok)
        print(f"{e['trial_name'][:24]:<26}{lvl:<10}{'✓' if ok else '✗':<10}{detail}"
              + ("" if ok else f"　← {(e.get('caveats') or '')[:34]}"))
    print("-" * 104)
    print(f"**{ok_n}/{len(entries)} 条的 text 里带着该类型的要害信息**"
          f"　← P0 验收真正要保证的是这一格")
    print("⚠ 结构化字段抽不到**不等于摘要里没有**：`text` 存的是摘要原文，检索与引用用的是它，")
    print("  结构化字段只用于展示与过滤。所以这张表比上面那张更重要。")
    if ok_n < len(entries):
        print("⚠ 没打勾的那几条**不要当成 bug 去修**：摘要里确实没有那个数（比如事后分析的 HR "
              "在正文不在摘要）。它们的 caveats 已写明，检索到时生成侧应据此声明不确定，"
              "**而不是从别处凑一个数**——那正是外部评测里「从综述转述里凑数字」的来源。")


def cmd_verify(args: argparse.Namespace) -> int:
    """离线重放：读已落盘的 entries.json，把校验与抽取情况再打一遍。"""
    if not os.path.isfile(ENTRIES_PATH):
        print(f"没有 {ENTRIES_PATH}，先跑 --fetch")
        return 1
    with open(ENTRIES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print(f"{ENTRIES_PATH}｜取于 {payload.get('created')}｜"
          f"{payload.get('n_entries')} 条通过 / {payload.get('n_skipped')} 条跳过")
    for e in payload["entries"]:
        print(f"  {e['trial_name']:<26}{e['pmid']:<11}{e['journal']} {e['pub_year']}"
              f"｜{e['evidence_level']}｜摘要 {len(e.get('abstract') or '')} 字符"
              f"｜校验 {e.get('verified')}")
    _report_extraction(payload["entries"])
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """离线：向量化并写入独立 collection。**不碰 65GB 主库**——
    单独放 data/chroma_landmark，主库在阶段四被写坏过一次，不给第二次机会。"""
    if not os.path.isfile(ENTRIES_PATH):
        print(f"没有 {ENTRIES_PATH}，先跑 --fetch（或 --from-xml）")
        return 1
    with open(ENTRIES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload["entries"]
    if not entries:
        print("条目为空，不建库")
        return 1

    import pyarrow                                   # noqa: F401  （必须早于 torch）
    import chromadb

    # ⚠ **必须复用项目自己的 BGEEmbedder，不能用 chromadb 的 SentenceTransformer 包装器。**
    # 主库是 BGEEmbedder 建的：CLS pooling + L2 归一化，文档不加前缀、查询加
    # QUERY_INSTRUCTION。检索侧 `检索_多路检索.py` 也是自己算好 query 向量再传
    # `query_embeddings=`，collection 上**不挂 embedding_function**。
    # 换一个包装器 = 换一套池化/归一化约定，两个 collection 的余弦分就不可比，
    # 而融合那一步正是拿它们直接比大小——错得很安静。
    _spec = importlib.util.spec_from_file_location(
        "xlh_jianku", os.path.join(ROOT, "scripts", "向量化_建库.py"))
    _vb = importlib.util.module_from_spec(_spec)
    sys.modules["xlh_jianku"] = _vb
    _spec.loader.exec_module(_vb)

    print(f"向量化 {len(entries)} 条（{EMBED_MODEL}，与主库同一个 BGEEmbedder）…")
    embedder = _vb.BGEEmbedder("bge-base", device="cpu")   # 十条而已，不占显存
    vecs = embedder.embed_documents([e["text"] for e in entries], batch_size=8)
    print(f"  向量 {vecs.shape}｜与主库同维度 {vecs.shape[1] == 768}")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:                                # noqa: BLE001
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"landmark-{e['pmid']}" for e in entries],
        documents=[e["text"] for e in entries],
        embeddings=[v.tolist() for v in vecs],
        metadatas=[{
            "source_type": "landmark",              # 检索侧按它给保底配额
            "trial_name": e["trial_name"], "pmid": e["pmid"],
            "pmcid": e.get("pmcid", ""), "journal": e["journal"],
            "pub_year": e["pub_year"] or 0, "evidence_level": e["evidence_level"],
            "topic": e.get("topic", ""), "title": e["title"],
            "n_randomized": e.get("n_randomized") or 0,
            "caveats": e.get("caveats", ""),
        } for e in entries],
    )
    print(f"collection `{COLLECTION}` → {CHROMA_DIR}｜{col.count()} 条")
    print("⚠ 独立目录，**没有碰 data/chroma_db_4m**。")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if not os.path.isfile(ENTRIES_PATH):
        print(f"没有 {ENTRIES_PATH}")
        return 1
    with open(ENTRIES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for e in payload["entries"]:
        print("=" * 96)
        print(f"{e['trial_name']}｜PMID {e['pmid']}｜{e['journal']} {e['pub_year']}"
              f"｜{e['evidence_level']}")
        print(f"标题：{e['title']}")
        print(f"入组：{e.get('n_randomized')}　效应量：{e.get('effect_size')}"
              f"　P：{e.get('p_value')}")
        print(f"摘要（前 300 字）：{(e.get('abstract') or '')[:300]}…")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="P0 landmark 集合：取摘要（联网一次）+ 建独立 collection")
    ap.add_argument("--seed", default=SEED_PATH)
    ap.add_argument("--fetch", action="store_true", help="唯一联网的一步：efetch 取摘要 + 校验 + 落盘")
    ap.add_argument("--from-xml", metavar="PATH", help="离线通道：用别人代取的 efetch XML")
    ap.add_argument("--verify", action="store_true", help="离线重放校验（读已落盘的 entries.json）")
    ap.add_argument("--build", action="store_true", help="离线：建 collection medrag_landmark")
    ap.add_argument("--show", action="store_true", help="离线：逐条打印")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    if args.fetch:
        return cmd_fetch(args)
    if args.from_xml:
        return cmd_from_xml(args)
    if args.verify:
        return cmd_verify(args)
    if args.build:
        return cmd_build(args)
    if args.show:
        return cmd_show(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
