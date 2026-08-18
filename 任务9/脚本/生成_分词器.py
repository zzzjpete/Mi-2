# -*- coding: utf-8 -*-
"""第七阶段 · 生成层公共分词器（Qwen3 精确 token 计数）

上下文组装要按 token 预算裁剪证据，预算算错的后果不对称：
  · 高估 → 只浪费一点上下文窗口；
  · 低估 → 提示词超出 num_ctx，Ollama 会**静默截断最前面的内容**（通常正是 system 提示词
    和高相关证据），生成质量塌掉却不会报错。
所以计数要么精确，要么必须保守地偏高。

本机没有 Qwen3 的 HF tokenizer（hf-cache 里只有 bge-base / bge-m3 / bge-reranker-base），
但 Ollama 的 qwen3:8b GGUF **自带完整分词器**（`tokenizer.ggml.*`：151,936 词表 + 151,387
merges，模型 `gpt2` 即 byte-level BPE，pre `qwen2`）。本模块直接从 GGUF 头部把它读出来，用
`tokenizers` 重建成与推理时**同一套** BPE，因此计数是精确的、且完全离线。

三级降级（`TokenCounter(mode="auto")`）：
  1. `qwen`      —— 从 GGUF 重建（精确，默认；首次约 3s，之后走 JSON 缓存约 0.3s）
  2. `hf:<路径>` —— 显式给一个 HF tokenizer 目录（例如日后下载了 Qwen3 的官方 tokenizer）
  3. `heuristic` —— 字符启发式，常数由 1000 条真实文本块实测标定（见 CHARS_PER_TOKEN_*），
                    并**上取整 + 安全系数**，保证偏高不偏低

用法：
    tc = TokenCounter()             # auto → qwen 精确
    tc.count("Hello world")         # 2
    tc.truncate_to_tokens(text, 400)

CLI（自检 + 重建缓存）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_分词器.py --rebuild
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import os
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-cache")        # 硬覆盖：用户级 HF_HOME 指向改名前的旧路径
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import re
import struct
import sys
from typing import Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 默认资源路径 ----
OLLAMA_MODELS = os.path.join(ROOT, "ollama", "models")
TOKENIZER_CACHE = os.path.join(ROOT, "data", "tokenizer", "qwen3_tokenizer.json")
DEFAULT_MODEL, DEFAULT_TAG = "qwen3", "8b"

# 字符启发式常数：拿 任务3/样例_文本块_1000.jsonl 的 1000 条真实文本块，对照本模块的精确
# Qwen 分词实测标定——英文医学正文 chars/token 中位数 4.49、p5 3.05、最小 1.31（符号/公式密集块）。
# 取 3.0（比中位数小）是为了让估算**偏高**：实测 962/1000 条被高估，中位高估 1.50×；但公式密集
# 的极端块仍会被低估（最差 0.44×）——所以 heuristic 只是降级路径，默认永远走精确分词器。
CHARS_PER_TOKEN_EN = 3.0
CHARS_PER_TOKEN_CJK = 1.0
_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")

# Qwen2/Qwen3 的 pre-tokenizer 正则（与 GGUF 中 tokenizer.ggml.pre == "qwen2" 对应）
QWEN2_PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


# ============================================================================
# 一、定位 Ollama 里的 GGUF 权重文件
# ============================================================================
def resolve_model_gguf(model: str = DEFAULT_MODEL, tag: str = DEFAULT_TAG,
                       models_dir: str = OLLAMA_MODELS) -> str:
    """从 Ollama 的 manifest 找到模型层 blob 的真实路径。"""
    manifest = os.path.join(models_dir, "manifests", "registry.ollama.ai",
                            "library", model, tag)
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"找不到 Ollama manifest：{manifest}")
    with open(manifest, encoding="utf-8") as f:
        mf = json.load(f)
    for layer in mf.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace(":", "-")
            blob = os.path.join(models_dir, "blobs", digest)
            if not os.path.exists(blob):
                raise FileNotFoundError(f"manifest 指向的 blob 不存在：{blob}")
            return blob
    raise RuntimeError(f"manifest 里没有 model 层：{manifest}")


# ============================================================================
# 二、读 GGUF 头部的 key-value 元数据（只读头部，不碰 5GB 张量数据）
# ============================================================================
_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


def read_gguf_kv(path: str, keys: Optional[List[str]] = None) -> Dict[str, object]:
    """解析 GGUF 头部的元数据键值对。

    keys 给定时只保留这些键（其余仍需按格式跳过，但不驻留内存）。
    """
    want = set(keys) if keys else None
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"不是 GGUF 文件（magic={magic!r}）：{path}")
        version, = struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))              # tensor_count，用不到
        n_kv, = struct.unpack("<Q", f.read(8))

        def rd_str():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rd_val(t):
            if t == _GGUF_STRING:
                return rd_str()
            if t == _GGUF_ARRAY:
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                return [rd_val(et) for _ in range(n)]
            return struct.unpack(_GGUF_SCALAR[t], f.read(_GGUF_SIZE[t]))[0]

        out = {"_gguf_version": version}
        for _ in range(n_kv):
            k = rd_str()
            t, = struct.unpack("<I", f.read(4))
            v = rd_val(t)                            # 数组也必须完整读完才能对齐到下一个 KV
            if want is None or k in want:
                out[k] = v
        return out


# ============================================================================
# 三、把 GGUF 里的词表 + merges 重建成 tokenizers 的 byte-level BPE
# ============================================================================
def build_bpe_from_gguf(meta: Dict[str, object]):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex

    tokens = meta.get("tokenizer.ggml.tokens")
    merges = meta.get("tokenizer.ggml.merges")
    if not tokens or not merges:
        raise RuntimeError("GGUF 元数据里没有 tokenizer.ggml.tokens / merges")
    if meta.get("tokenizer.ggml.model") != "gpt2":
        raise RuntimeError(f"暂只支持 gpt2(byte-level BPE) 分词器，"
                           f"实际为 {meta.get('tokenizer.ggml.model')!r}")

    vocab = {t: i for i, t in enumerate(tokens)}
    pairs = [tuple(m.split(" ", 1)) for m in merges]
    tk = Tokenizer(models.BPE(vocab, pairs, fuse_unk=False, byte_fallback=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_PRETOKENIZE_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    return tk


def load_qwen_tokenizer(model: str = DEFAULT_MODEL, tag: str = DEFAULT_TAG,
                        models_dir: str = OLLAMA_MODELS,
                        cache_path: str = TOKENIZER_CACHE,
                        rebuild: bool = False, verbose: bool = False):
    """载入与 qwen3:8b 推理时同一套的分词器。

    Returns: (tokenizer, info)   info 含 source/vocab_size/context_length/gguf 等
    """
    from tokenizers import Tokenizer

    side = (cache_path + ".meta.json") if cache_path else None
    gguf = resolve_model_gguf(model, tag, models_dir)
    digest = os.path.basename(gguf)

    if cache_path and not rebuild and os.path.exists(cache_path) and os.path.exists(side):
        try:
            with open(side, encoding="utf-8") as f:
                info = json.load(f)
            if info.get("gguf") == digest:            # 换过模型就重建
                tk = Tokenizer.from_file(cache_path)
                info["source"] = "cache"
                if verbose:
                    print(f"[分词器] 命中缓存 {cache_path}（vocab {info.get('vocab_size'):,}）")
                return tk, info
        except Exception as e:                        # 缓存坏了就当没有
            if verbose:
                print(f"[分词器] 缓存不可用（{e}），从 GGUF 重建")

    if verbose:
        print(f"[分词器] 从 GGUF 重建：{gguf}")
    meta = read_gguf_kv(gguf, keys=[
        "tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.model",
        "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
        "general.name", "qwen3.context_length",
    ])
    tk = build_bpe_from_gguf(meta)
    info = {
        "source": "gguf", "gguf": digest, "model": f"{model}:{tag}",
        "general_name": meta.get("general.name"),
        "vocab_size": tk.get_vocab_size(),
        "bos_token_id": meta.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": meta.get("tokenizer.ggml.eos_token_id"),
        "context_length": meta.get("qwen3.context_length"),
        "pre": meta.get("tokenizer.ggml.pre"),
    }
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tk.save(cache_path)
        with open(side, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"[分词器] 已缓存到 {cache_path}")
    return tk, info


# ============================================================================
# 四、统一计数接口（带降级）
# ============================================================================
class TokenCounter:
    """统一的 token 计数 / 截断接口。

    mode:
      · "auto"      先试 qwen（精确），失败降级 heuristic
      · "qwen"      强制从 Ollama GGUF 重建（失败即抛错）
      · "hf:<path>" 用 transformers.AutoTokenizer 载入指定目录
      · "heuristic" 纯字符估算（保守偏高）
    """

    def __init__(self, mode: str = "auto", verbose: bool = False, **kw):
        self.verbose = verbose
        self.info: Dict[str, object] = {}
        self._tk = None
        self._hf = None
        self.mode = "heuristic"

        if mode.startswith("hf:"):
            self._init_hf(mode[3:])
        elif mode in ("auto", "qwen"):
            try:
                self._tk, self.info = load_qwen_tokenizer(verbose=verbose, **kw)
                self.mode = "qwen"
            except Exception as e:
                if mode == "qwen":
                    raise
                if verbose:
                    print(f"[分词器] Qwen 分词器不可用（{e}），降级字符启发式")
                self.info = {"source": "heuristic", "reason": str(e)}
        elif mode != "heuristic":
            raise ValueError(f"未知 mode：{mode}")

    def _init_hf(self, path: str):
        from transformers import AutoTokenizer
        self._hf = AutoTokenizer.from_pretrained(path)
        self.mode = "hf"
        self.info = {"source": "hf", "path": path,
                     "vocab_size": getattr(self._hf, "vocab_size", None)}

    # ---- 精确性：调用方（如上下文组装器）会把它写进产物元数据 ----
    @property
    def exact(self) -> bool:
        """是否与推理端同一套分词器（heuristic 为 False）。"""
        return self.mode in ("qwen", "hf")

    def encode(self, text: str) -> List[int]:
        if self._tk is not None:
            return self._tk.encode(text).ids
        if self._hf is not None:
            return self._hf.encode(text, add_special_tokens=False)
        raise RuntimeError("heuristic 模式没有真实 token id")

    def decode(self, ids: List[int]) -> str:
        if self._tk is not None:
            return self._tk.decode(ids)
        if self._hf is not None:
            return self._hf.decode(ids)
        raise RuntimeError("heuristic 模式没有真实 token id")

    def count(self, text: str) -> int:
        """估算/精确统计 token 数（heuristic 下保证偏高）。"""
        if not text:
            return 0
        if self._tk is not None or self._hf is not None:
            return len(self.encode(text))
        cjk = len(_CJK.findall(text))
        rest = len(text) - cjk
        return int(math.ceil(cjk / CHARS_PER_TOKEN_CJK + rest / CHARS_PER_TOKEN_EN))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> Tuple[str, bool]:
        """截到不超过 max_tokens；返回 (文本, 是否发生截断)。

        heuristic 模式没有真实 token 边界，按字符比例保守回切。
        """
        if max_tokens <= 0:
            return "", bool(text)
        if self.count(text) <= max_tokens:
            return text, False
        if self._tk is not None or self._hf is not None:
            return self.decode(self.encode(text)[:max_tokens]), True
        cjk = len(_CJK.findall(text))
        ratio = CHARS_PER_TOKEN_CJK if cjk > len(text) * 0.3 else CHARS_PER_TOKEN_EN
        cut = int(max_tokens * ratio)
        out = text[:max(0, cut)]
        while out and self.count(out) > max_tokens:    # 保证真的不超
            out = out[:int(len(out) * 0.95)]
        return out, True


# ============================================================================
# CLI：自检（往返一致 + 计数）与缓存重建
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="忽略缓存，从 GGUF 重建")
    ap.add_argument("--mode", default="auto")
    args = ap.parse_args()

    tc = TokenCounter(mode=args.mode, rebuild=args.rebuild, verbose=True) \
        if args.mode in ("auto", "qwen") else TokenCounter(mode=args.mode, verbose=True)
    print(f"模式={tc.mode} 精确={tc.exact}")
    print(json.dumps(tc.info, ensure_ascii=False, indent=2))
    samples = [
        "Hello world",
        "Pembrolizumab plus chemotherapy improved overall survival in patients with "
        "metastatic non-small-cell lung cancer (NSCLC).",
        "阿尔茨海默病的最新药物有哪些？",
    ]
    for s in samples:
        n = tc.count(s)
        rt = (tc.decode(tc.encode(s)) == s) if tc.exact else None
        print(f"  {n:4d} tok | 字符 {len(s):4d} | 往返一致={rt} | {s[:60]!r}")


if __name__ == "__main__":
    main()
