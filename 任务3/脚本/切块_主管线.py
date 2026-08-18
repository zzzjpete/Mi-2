"""
切块_主管线.py — 文档解析与分割（切块）主管线
对应任务「文档解析与分割」，策略沿用《RAG数据分析与设计说明》：

  章节感知 + 递归兜底（两层），chunk_size=512 / overlap=64，用 bge-m3 分词器计长。
  - 短文：整篇 ≤512 token → 不分割，整篇作 1 块（chunk_id = doc_id）           【路径 b】
  - 长文：按顶层 <sec> 切；单章节 ≤512 整段成块，>512 用递归分割器兜底            【路径 a】
  - doc_id：优先 pmid，缺失时用 100% 完整的 pmcid 兜底
  - 每块挂元数据：chunk_id / doc_id / chunk_index / total_chunks / source_title /
    token_count（约定的核心字段）+ pmcid / pmid / journal / pub_year / section
    （报告设计的溯源 + 过滤字段）

工程：按源 tar.gz 逐包流式处理，分批写 parquet（控内存），断点续跑（产物已存在则跳过）。

用法：
  # 在现有样本包（PMC000, 3028 篇）上跑通并验证：
  python 切块_主管线.py --package data/pubmed/oa_comm_xml.PMC000xxxxxx.baseline.2026-06-18.tar.gz
  # 处理 data/pubmed 下所有 baseline 包（全量，断点续跑）：
  python 切块_主管线.py --all
  # 只取前 N 篇（快速验证）：
  python 切块_主管线.py --package <tar> --limit 200

产物：
  data/chunks/chunks_<PKG>.parquet   每包一个（列顺序即约定 schema）
  data/chunks/_manifest.json          每包处理统计（追加）
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT_PATH as ROOT

import os
import sys
import json
import time
import argparse
import tarfile
from pathlib import Path

# 硬覆盖：用户级持久化的 HF_HOME 仍指向旧路径 E:\medrag\hf-cache（项目已改名为 rag），
# 必须在 import transformers 前强制指到实际缓存目录，否则离线加载 bge-m3 会失败。
os.environ["HF_HOME"] = str(ROOT / "hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree
from transformers import AutoTokenizer

DATA = ROOT / "data" / "pubmed"
OUT_DIR = ROOT / "data" / "chunks"

CHUNK_SIZE = 512
OVERLAP = 64
MODEL_LIMIT = 8192          # bge-m3 上限，用于质量校验参考
FLUSH_EVERY = 4000         # 每处理多少篇 flush 一次 parquet row group

# parquet 列顺序：约定的核心字段在前，溯源/过滤字段在后
SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("text", pa.string()),
    ("doc_id", pa.string()),
    ("chunk_index", pa.int32()),
    ("total_chunks", pa.int32()),
    ("source_title", pa.string()),
    ("token_count", pa.int32()),
    ("section", pa.string()),
    ("pmcid", pa.string()),
    ("pmid", pa.string()),
    ("journal", pa.string()),
    ("pub_year", pa.int32()),
])

# ------------------------- 分词 / 分割器 -------------------------
_tok = None


def get_tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _tok


def count_tokens(text):
    if not text:
        return 0
    return len(get_tok().encode(text, add_special_tokens=False))


_SENT_RE = re.compile(r"(?<=[.!?;])\s+")


def sentence_split(text):
    return [p for p in _SENT_RE.split(text.strip()) if p]


def _token_window(ids, size, overlap):
    """对超长单句（>size token）按 token 滑窗兜底，窗口间恒定 overlap。"""
    tok = get_tok()
    step = size - overlap
    out, i, n = [], 0, len(ids)
    while i < n:
        out.append(tok.decode(ids[i:i + size]))
        if i + size >= n:
            break
        i += step
    return out


def smart_split(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """句子感知 + token 控长的递归兜底切分：
    - 贪心把句子打包进 ≤size token 的窗口；
    - 每个新窗口用上一窗口尾部若干句子做 ~overlap token 的重叠（保证跨块上下文连续）；
    - 单句 >size 时退回 token 滑窗。
    比 LangChain RecursiveCharacterTextSplitter 更可控：在无换行的连续正文上，后者按 ". "
    切分时 overlap 会塌缩为 0，无法满足设计要求的 overlap≈64。
    """
    tok = get_tok()
    if count_tokens(text) <= size:
        return [text]
    sents = sentence_split(text)
    if not sents:
        return [text]
    # 一次性批量分词所有句子（比逐句 encode 快很多，是全量跑的关键优化）
    enc = tok(sents, add_special_tokens=False)["input_ids"]
    units = [(s, len(ids)) for s, ids in zip(sents, enc)]

    windows = []          # 每个元素是一组 (句子, token数)
    cur, cur_len = [], 0
    i, N = 0, len(units)
    while i < N:
        s, l = units[i]
        if l > size:                       # 超长单句：先冲掉当前窗口，再滑窗切这句
            if cur:
                windows.append(cur); cur, cur_len = [], 0
            ids = tok.encode(s, add_special_tokens=False)
            for piece in _token_window(ids, size, overlap):
                windows.append([(piece, min(size, len(ids)))])
            i += 1
            continue
        if cur_len + l <= size:
            cur.append((s, l)); cur_len += l; i += 1
        else:
            windows.append(cur)
            # 用尾部句子构造重叠：累加到刚好达到/略过 overlap 的那句为止（使重叠≈overlap，
            # 而非塌到 0）；再保证给下一句留出空间（避免死循环）
            ov, ov_len = [], 0
            for s2, l2 in reversed(cur):
                ov.insert(0, (s2, l2)); ov_len += l2
                if ov_len >= overlap:
                    break
            while ov and ov_len + l > size:
                ov_len -= ov.pop(0)[1]
            cur, cur_len = ov, ov_len
    if cur:
        windows.append(cur)
    return [" ".join(s for s, _ in w) for w in windows]


# ------------------------- XML 解析（复用步骤1/4的逻辑） -------------------------
def text_of(el):
    return " ".join(el.itertext()).strip() if el is not None else ""


def first_abstract(root):
    for ab in root.findall(".//front//abstract"):
        atype = (ab.get("abstract-type") or "").lower()
        if atype in ("graphical", "teaser", "graphical-abstract"):
            continue
        t = text_of(ab)
        if t:
            return t
    return ""


def pub_year(root):
    best = None
    for pd_el in root.findall(".//front//pub-date"):
        y = pd_el.findtext("year")
        if not y:
            continue
        ptype = (pd_el.get("pub-type") or pd_el.get("date-type") or "").lower()
        try:
            yi = int(y)
        except ValueError:
            continue
        if ptype in ("ppub", "collection", "pub"):
            return yi
        if best is None:
            best = yi
    return best


def article_id(root, idtype):
    for aid in root.iter("article-id"):
        if aid.get("pub-id-type") == idtype:
            return (aid.text or "").strip()
    return ""


def journal_title(root):
    return text_of(root.find(".//journal-meta//journal-title"))


def top_sections(root):
    """顶层 <sec> 的 (title, text)；无 sec 的文章整篇正文作为一个单元。"""
    body = root.find(".//body")
    if body is None:
        return []
    secs = body.findall("sec")
    if not secs:
        t = text_of(body)
        return [("(no-section)", t)] if t else []
    out = []
    for s in secs:
        title = text_of(s.find("title"))
        out.append((title or "(untitled)", text_of(s)))
    return out


def parse_member(root, fallback_name):
    pmcid = article_id(root, "pmc")
    pmid = article_id(root, "pmid")
    # doc_id 用 pmcid 优先：pmcid 是 PMC 唯一登录号（100% 完整且全库唯一）；
    # pmid 有 ~9% 缺失、且并非唯一（勘误/更正记录会与原文共用 pmid），不适合当主键。
    doc_id = pmcid or pmid or ("FILE_" + fallback_name)
    return {
        "doc_id": doc_id,
        "title": text_of(root.find(".//title-group/article-title")),
        "abstract": first_abstract(root),
        "sections": top_sections(root),
        "pmcid": pmcid,
        "pmid": pmid,
        "journal": journal_title(root),
        "pub_year": pub_year(root),
    }


# ------------------------- 切块核心 -------------------------
def chunk_document(doc):
    """把一篇文献切成块列表（每块一个 dict）。"""
    doc_id = doc["doc_id"]
    title = doc["title"]

    # 可切内容单元：摘要（若有）+ 各顶层章节
    units = []
    if doc["abstract"]:
        units.append(("Abstract", doc["abstract"]))
    units.extend([(n, t) for n, t in doc["sections"] if t])

    joined = "\n\n".join([title] + [t for _, t in units]).strip()
    full_tok = count_tokens(joined)

    # 无任何内容（标题/摘要/正文全空，如仅含 supplementary-material 的薄记录）→ 不产块
    if full_tok == 0:
        return []

    base = {
        "doc_id": doc_id, "source_title": title,
        "pmcid": doc["pmcid"], "pmid": doc["pmid"],
        "journal": doc["journal"], "pub_year": doc["pub_year"],
    }

    # 路径 b：整篇 ≤512 → 不分割
    if full_tok <= CHUNK_SIZE:
        return [{**base, "chunk_id": doc_id, "text": joined, "chunk_index": 0,
                 "total_chunks": 1, "token_count": full_tok, "section": "(full-doc)"}]

    # 路径 a：章节感知 + 递归兜底
    pieces = []  # (section_name, text)
    for name, text in units:
        if count_tokens(text) <= CHUNK_SIZE:
            pieces.append((name, text))
        else:
            for sub in smart_split(text, CHUNK_SIZE, OVERLAP):
                sub = sub.strip()
                if sub:
                    pieces.append((name, sub))

    total = len(pieces)
    if total == 0:  # 兜底：极端情况下退回整篇
        return [{**base, "chunk_id": doc_id, "text": joined, "chunk_index": 0,
                 "total_chunks": 1, "token_count": full_tok, "section": "(full-doc)"}]

    out = []
    for i, (name, text) in enumerate(pieces):
        cid = doc_id if total == 1 else f"{doc_id}#{i}"
        out.append({**base, "chunk_id": cid, "text": text, "chunk_index": i,
                    "total_chunks": total, "token_count": count_tokens(text),
                    "section": name})
    return out


# ------------------------- 分批 parquet 写入 -------------------------
class ParquetBatcher:
    def __init__(self, path, schema):
        self.writer = pq.ParquetWriter(str(path), schema, compression="zstd")
        self.schema = schema
        self.buf = []

    def add(self, rows):
        self.buf.extend(rows)

    def flush(self):
        if not self.buf:
            return
        cols = {name: [r.get(name) for r in self.buf] for name in self.schema.names}
        table = pa.table(
            {name: pa.array(cols[name], type=self.schema.field(name).type)
             for name in self.schema.names},
            schema=self.schema,
        )
        self.writer.write_table(table)
        self.buf = []

    def close(self):
        self.flush()
        self.writer.close()


# ------------------------- 单包处理 -------------------------
def _iter_xml(targz, limit=None):
    """流式产出 (文件名, xml字节)。tar 顺序解压在主进程，切块交给 worker。"""
    cnt = 0
    with tarfile.open(targz, "r:gz") as tar:
        for m in tar:
            if not m.name.endswith(".xml"):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            fn = m.name.split("/")[-1].replace(".xml", "")
            yield fn, f.read()
            cnt += 1
            if limit and cnt >= limit:
                break


def _chunk_one(args):
    """worker：解析一篇 XML 字节 + 切块。返回块列表，失败返回 None。"""
    fn, xml = args
    try:
        root = etree.fromstring(xml)
        doc = parse_member(root, fn)
        return chunk_document(doc)
    except Exception:
        return None


def _worker_init():
    get_tok()  # 每个 worker 进程预热分词器（离线）


def process_package(targz, out_parquet, limit=None, workers=1):
    batcher = ParquetBatcher(out_parquet, SCHEMA)
    n_doc = n_fail = n_chunk = 0
    pending = []
    t0 = time.time()

    def handle(rows):
        nonlocal n_doc, n_fail, n_chunk, pending
        if rows is None:
            n_fail += 1
            return
        n_doc += 1
        n_chunk += len(rows)
        pending.extend(rows)
        if n_doc % FLUSH_EVERY == 0:
            batcher.add(pending)
            batcher.flush()
            pending = []
            rate = n_doc / (time.time() - t0)
            print(f"  {n_doc} 篇 / {n_chunk} 块  ({rate:.0f} 篇/s)", flush=True)

    if workers and workers > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=_worker_init) as pool:
            for rows in pool.imap_unordered(_chunk_one,
                                            _iter_xml(targz, limit), chunksize=16):
                handle(rows)
    else:
        for fn, xml in _iter_xml(targz, limit):
            handle(_chunk_one((fn, xml)))

    batcher.add(pending)
    batcher.close()
    dt = time.time() - t0
    return {"docs": n_doc, "parse_fail": n_fail, "chunks": n_chunk,
            "seconds": round(dt, 1), "docs_per_sec": round(n_doc / dt, 1) if dt else 0}


def pkg_tag(targz_path):
    name = Path(targz_path).name
    # oa_comm_xml.PMC001xxxxxx.baseline.2026-06-18.tar.gz -> PMC001
    for part in name.split("."):
        if part.startswith("PMC"):
            return part[:6]
    return name.replace(".tar.gz", "")


def record_manifest(entry):
    mf = OUT_DIR / "_manifest.json"
    data = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    data[entry["package"]] = entry
    mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(targz, limit=None, force=False, workers=1):
    targz = Path(targz)
    tag = pkg_tag(targz)
    out_parquet = OUT_DIR / f"chunks_{tag}.parquet"
    if out_parquet.exists() and not force:
        print(f"[skip] {tag}: 产物已存在 {out_parquet.name}")
        return None
    print(f"[run ] {tag}: {targz.name}  (workers={workers})")
    # 先写临时文件，成功后原子改名；被中断只会留下 .tmp，不会被误判为已完成
    tmp = out_parquet.with_name(out_parquet.name + ".tmp")
    stats = process_package(targz, tmp, limit=limit, workers=workers)
    tmp.replace(out_parquet)
    entry = {"package": tag, "source": targz.name,
             "output_file": str(out_parquet),
             "chunk_size": CHUNK_SIZE, "chunk_overlap": OVERLAP,
             "processed_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
             **stats,
             "chunks_per_doc": round(stats["chunks"] / stats["docs"], 2)
             if stats["docs"] else 0}
    record_manifest(entry)
    print(f"[done] {tag}: {stats['docs']} 篇 -> {stats['chunks']} 块 "
          f"({entry['chunks_per_doc']}/篇, {stats['seconds']}s)")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", help="单个 tar.gz 路径")
    ap.add_argument("--all", action="store_true",
                    help="处理 data/pubmed 下所有 baseline 包")
    ap.add_argument("--limit", type=int, default=None, help="每包最多处理篇数")
    ap.add_argument("--force", action="store_true", help="覆盖已存在产物")
    ap.add_argument("--workers", type=int, default=1, help="并行进程数（切块）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        pkgs = sorted(DATA.glob("oa_comm_xml.PMC*baseline*.tar.gz"))
        if not pkgs:
            print("data/pubmed 下没有 baseline 包"); sys.exit(1)
        print(f"待处理 {len(pkgs)} 个包")
        for p in pkgs:
            run_one(p, limit=args.limit, force=args.force, workers=args.workers)
    elif args.package:
        run_one(args.package, limit=args.limit, force=args.force, workers=args.workers)
    else:
        print("需要 --package <tar> 或 --all"); sys.exit(1)


if __name__ == "__main__":
    main()
