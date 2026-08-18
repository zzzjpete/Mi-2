# -*- coding: utf-8 -*-
"""第八阶段 · 生成结果缓存：键设计 / 容量上限 / TTL / 温度门限

一句话：**同样的输入不要再花 40 秒问一遍模型**。四段链单题实测 29~42 s，
评测迭代时同一批题要反复跑，缓存省下的是整段等待。

四件事都不是可选的，各自对应一个会出问题的场景：

  1. **键 = 查询 + 上下文 + 所有会改变输出的参数**的哈希。
     只哈希查询是不够的——同一个问题、换一批检索证据，答案本该不同；
     模型名、温度、max_tokens、num_ctx、**提示词模板本身**改了也一样。
     所以 `CachedLLMGenerator` 直接把**完整 messages**（含 system 提示词）纳入键материал：
     提示词一改，键自动变，不会读到旧答案。这是"改了提示词却读到缓存"这类坑的根治法。
  2. **容量双上限**（条数 + 字节数），LRU 淘汰。答案一条 2~3 KB，中间产物带上就更大；
     不限量的 dict 在长跑的评测进程里就是内存泄漏。
  3. **TTL**。医学知识有时效性：指南会更新、药会撤市。永久缓存等于把过期结论钉死。
     默认 7 天，可按用途调（做一轮评测时设几小时就够）。
  4. **温度门限**。温度 > 0 的输出本来就不是确定性的，缓存它等于**把一次抽样的结果
     固定下来**——复跑评测时看起来"稳定"，其实是假的稳定。默认只缓存 temperature ≤ 0.0；
     要缓存高温结果必须显式抬高 `max_temperature`，抬高时本模块会在 stats 里留痕。

线程安全：内部一把 `RLock`。批量处理走多线程（见 `生成_批量处理.py`），
缓存必然被并发读写，不加锁会丢计数、也可能读到半个 OrderedDict。

用法：
    import importlib.util
    spec = importlib.util.spec_from_file_location("gc", r"E:\\rag\\scripts\\生成_缓存.py")
    gc = importlib.util.module_from_spec(spec); spec.loader.exec_module(gc)

    cache = gc.GenerationCache(max_entries=200, ttl_seconds=7*86400,
                               path=r"E:\\rag\\report_data\\生成缓存.json")
    key = cache.make_key(query="…", context="…", model="qwen3:8b", temperature=0.0)
    hit = cache.get(key)
    cache.set(key, value, temperature=0.0)

    # 给流水线套一层：四段链的每一次模型调用都自动走缓存，阶段七代码一行不用改
    gen = gc.CachedLLMGenerator(LLMGenerator(...), cache)
    pipe = MedicalGenerationPipeline(generator=gen)

CLI 演示（不需要 Ollama）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_缓存.py --demo
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

#: 医学知识有时效性，默认 7 天。做一轮评测时把它调到几小时更合适。
DEFAULT_TTL_SECONDS = 7 * 24 * 3600
#: 只缓存确定性输出。0.0 = 严格；本项目四段链里只有 ①③（温度 0）会被缓存。
DEFAULT_MAX_TEMPERATURE = 0.0
DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_BYTES = 64 * 1024 * 1024          # 64 MB


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    """把任意可序列化对象压成稳定字符串：键排序，不依赖 dict 插入顺序。

    ⚠ 不能用 `str(dict)` —— 那个受插入顺序影响，同样的参数换个写法就会得到不同的键。
    """
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          default=str)
    except (TypeError, ValueError):
        return str(obj)


def _size_of(value: Any) -> int:
    """条目占多少字节（按 UTF-8 序列化长度估）。用于字节上限，不求精确到堆内存。"""
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(value).encode("utf-8"))


class CacheEntry:
    """一条缓存。`expires_at=None` 表示不过期（只有显式传 ttl_seconds=0 才会这样）。"""

    __slots__ = ("value", "created_at", "expires_at", "hits", "size_bytes", "meta")

    def __init__(self, value: Any, ttl_seconds: Optional[float],
                 meta: Optional[Dict[str, Any]] = None):
        now = time.time()
        self.value = value
        self.created_at = now
        self.expires_at = (now + ttl_seconds) if ttl_seconds else None
        self.hits = 0
        self.size_bytes = _size_of(value)
        self.meta = meta or {}

    def expired(self, now: Optional[float] = None) -> bool:
        return self.expires_at is not None and (now or time.time()) >= self.expires_at

    def to_json(self) -> Dict[str, Any]:
        return {"value": self.value, "created_at": self.created_at,
                "expires_at": self.expires_at, "hits": self.hits, "meta": self.meta}


class GenerationCache:
    """带容量上限、TTL 与温度门限的生成缓存（线程安全，可落盘）。

    Args:
        max_entries:     条数上限，超出按 LRU 淘汰
        max_bytes:       总字节上限，超出按 LRU 淘汰（两个上限同时生效，谁先到算谁）
        ttl_seconds:     存活时长；0 或 None = 不过期（**不建议**，见模块 docstring）
        max_temperature: 只缓存温度 ≤ 此值的结果
        path:            落盘路径（JSON）。给了就在 load()/save() 时读写
        namespace:       键前缀，用于隔离不同用途的缓存（如 "stage" / "pipeline"）
    """

    def __init__(self,
                 max_entries: int = DEFAULT_MAX_ENTRIES,
                 max_bytes: int = DEFAULT_MAX_BYTES,
                 ttl_seconds: Optional[float] = DEFAULT_TTL_SECONDS,
                 max_temperature: float = DEFAULT_MAX_TEMPERATURE,
                 path: Optional[str] = None,
                 namespace: str = "",
                 verbose: bool = False):
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.ttl_seconds = ttl_seconds
        self.max_temperature = float(max_temperature)
        self.path = path
        self.namespace = namespace
        self.verbose = verbose

        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "skipped_temperature": 0,
                      "evicted_lru": 0, "evicted_bytes": 0, "expired": 0,
                      "seconds_saved": 0.0}
        if path and os.path.exists(path):
            self.load()

    # ---------------- 键 ----------------
    def make_key(self, query: str, context: Any = "", **params: Any) -> str:
        """缓存键 = sha256(命名空间 + 查询 + 上下文摘要 + 其余参数)。

        上下文可能有几千 token，先单独哈希再进键材料——键材料保持短小，
        同时"上下文变了键就变"这一点不受影响。
        """
        if not isinstance(context, str):
            context = _canonical(context)
        material = _canonical({
            "ns": self.namespace,
            "query": (query or "").strip(),
            "context_sha256": _sha256(context),
            "context_len": len(context),
            "params": params,
        })
        return _sha256(material)

    # ---------------- 温度门限 ----------------
    def should_cache(self, temperature: Optional[float]) -> bool:
        """温度 > 门限的结果不缓存：那是抽样，不是确定性输出。"""
        if temperature is None:
            return True                       # 调用方没声明温度，按可缓存处理
        return float(temperature) <= self.max_temperature

    # ---------------- 读写 ----------------
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            e = self._store.get(key)
            if e is None:
                self.stats["misses"] += 1
                return None
            if e.expired():
                self._remove(key)
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None
            e.hits += 1
            self._store.move_to_end(key)      # LRU：命中即变成最近使用
            self.stats["hits"] += 1
            self.stats["seconds_saved"] += float(e.meta.get("elapsed") or 0.0)
            return e.value

    def set(self, key: str, value: Any, temperature: Optional[float] = None,
            ttl_seconds: Optional[float] = None,
            meta: Optional[Dict[str, Any]] = None) -> bool:
        """写入。**返回是否真的写了**——被温度门限拦下时返回 False。"""
        if not self.should_cache(temperature):
            with self._lock:
                self.stats["skipped_temperature"] += 1
            if self.verbose:
                print(f"[缓存] 温度 {temperature} > 门限 {self.max_temperature}，不缓存")
            return False
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        entry = CacheEntry(value, ttl, meta)
        with self._lock:
            if key in self._store:
                self._bytes -= self._store[key].size_bytes
                del self._store[key]
            self._store[key] = entry
            self._bytes += entry.size_bytes
            self.stats["sets"] += 1
            self._evict()
        return True

    def get_or_compute(self, key: str, factory: Callable[[], Any],
                       temperature: Optional[float] = None,
                       meta: Optional[Dict[str, Any]] = None) -> Tuple[Any, bool]:
        """命中就返回，否则调 factory 算一次再写入。Returns: (值, 是否命中缓存)。

        ⚠ 刻意**不在 factory 期间持锁**：一次生成要几十秒，持锁会把并发批量卡成串行。
        代价是同一个键可能被并发算两次（都写入，结果相同）。对纯函数式的生成来说这是
        划算的取舍：多花一次算力，换不阻塞。
        """
        hit = self.get(key)
        if hit is not None:
            return hit, True
        t0 = time.time()
        value = factory()
        m = dict(meta or {})
        m.setdefault("elapsed", round(time.time() - t0, 3))
        self.set(key, value, temperature=temperature, meta=m)
        return value, False

    # ---------------- 淘汰与清理 ----------------
    def _remove(self, key: str):
        e = self._store.pop(key, None)
        if e is not None:
            self._bytes -= e.size_bytes

    def _evict(self):
        """先按条数、再按字节淘汰最久未用的。调用方必须已持锁。"""
        while len(self._store) > self.max_entries:
            _, e = self._store.popitem(last=False)      # last=False = 最久未用的那头
            self._bytes -= e.size_bytes
            self.stats["evicted_lru"] += 1
        while self._bytes > self.max_bytes and self._store:
            _, e = self._store.popitem(last=False)
            self._bytes -= e.size_bytes
            self.stats["evicted_bytes"] += 1

    def _recount(self):
        self._bytes = sum(e.size_bytes for e in self._store.values())

    def purge_expired(self) -> int:
        """主动清掉过期条目，返回清掉的条数。"""
        now = time.time()
        with self._lock:
            dead = [k for k, e in self._store.items() if e.expired(now)]
            for k in dead:
                self._remove(k)
            self.stats["expired"] += len(dead)
            return len(dead)

    def clear(self):
        with self._lock:
            self._store.clear()
            self._bytes = 0

    # ---------------- 落盘 ----------------
    def save(self, path: Optional[str] = None) -> Optional[str]:
        """写 JSON。原子落盘（先写 .tmp 再 replace），避免进程被杀时留下半个文件。"""
        p = path or self.path
        if not p:
            return None
        with self._lock:
            payload = {"saved_at": time.time(), "namespace": self.namespace,
                       "ttl_seconds": self.ttl_seconds,
                       "max_temperature": self.max_temperature,
                       "stats": self.stats,
                       "entries": {k: e.to_json() for k, e in self._store.items()
                                   if not e.expired()}}
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, p)
        return p

    def load(self, path: Optional[str] = None) -> int:
        """读 JSON，**过期条目直接丢掉**（落盘期间时间照样在走）。返回载入条数。"""
        p = path or self.path
        if not p or not os.path.exists(p):
            return 0
        try:
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if self.verbose:
                print(f"[缓存] 读取 {p} 失败（{e}），当作空缓存起步")
            return 0
        now = time.time()
        n = 0
        with self._lock:
            for k, d in (payload.get("entries") or {}).items():
                exp = d.get("expires_at")
                if exp is not None and now >= exp:
                    self.stats["expired"] += 1
                    continue
                e = CacheEntry(d.get("value"), None, d.get("meta"))
                e.created_at = d.get("created_at", now)
                e.expires_at = exp
                e.hits = d.get("hits", 0)
                self._store[k] = e
                n += 1
            self._recount()
            self._evict()
        return n

    # ---------------- 观测 ----------------
    def info(self) -> Dict[str, Any]:
        with self._lock:
            total = self.stats["hits"] + self.stats["misses"]
            return {
                "entries": len(self._store), "max_entries": self.max_entries,
                "bytes": self._bytes, "max_bytes": self.max_bytes,
                "mb": round(self._bytes / 1024 ** 2, 3),
                "ttl_seconds": self.ttl_seconds, "max_temperature": self.max_temperature,
                "hit_rate": round(self.stats["hits"] / total, 4) if total else None,
                **self.stats,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __bool__(self) -> bool:
        """缓存对象本身恒为真。

        没有这一条，`len(cache) == 0` 会让空缓存在布尔上下文里变成 False，
        于是 `cache or 新建一个` 这种常见写法会**静默丢掉**调用方传进来的缓存。
        缓存"是不是空的"该用 `len(cache)` 问，不该用 `if cache` 问。
        """
        return True

    def __contains__(self, key: str) -> bool:
        with self._lock:
            e = self._store.get(key)
            return e is not None and not e.expired()


# ============================================================================
# 给 LLMGenerator 套一层缓存
# ============================================================================
class CachedLLMGenerator:
    """`LLMGenerator` 的缓存包装：接口完全一致，可直接塞进阶段七的流水线。

        pipe = MedicalGenerationPipeline(generator=CachedLLMGenerator(gen, cache))

    键材料 = **完整 messages**（含 system 提示词）+ 模型名 + 温度 + max_tokens +
    num_ctx + json 模式。把整段提示词纳进去是刻意的：提示词模板一改，键立刻变，
    不会出现"改了提示词却读到旧答案"的假象。

    未知属性一律转发给被包装的生成器（`.stats` / `.model_name` / `.runtime_info()` …），
    所以调用方基本感觉不到它的存在。
    """

    def __init__(self, generator: Any, cache: Optional[GenerationCache] = None,
                 enabled: bool = True, verbose: bool = False):
        self.generator = generator
        # ⚠ 必须写 `is not None`，**不能写 `cache or GenerationCache(...)`**：
        # GenerationCache 定义了 __len__，空缓存的布尔值是 False，`or` 会把调用方传进来的
        # 缓存**悄悄丢掉**并新建一个。表现极隐蔽——包装器照样能命中（用的是新建的那个），
        # 但调用方设的 TTL / 容量 / 温度门限全部失效，统计也永远是 0。
        # 这个 bug 是被 `评估_跑测试集.py` 的「写入 0 却跳过 16」判定抓出来的。
        self.cache = cache if cache is not None else GenerationCache(namespace="llm")
        self.enabled = bool(enabled)
        self.verbose = verbose
        self.cache_stats = {"hits": 0, "misses": 0, "skipped": 0, "seconds_saved": 0.0}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.generator, name)          # 其余接口原样转发

    def _key(self, messages: Sequence[Dict[str, str]], temperature: Optional[float],
             max_tokens: Optional[int], json_output: bool, expect: str) -> str:
        g = self.generator
        return self.cache.make_key(
            query=_canonical(list(messages)),          # 完整对话（含 system）就是"输入"
            context="",
            model=getattr(g, "model_name", "?"),
            temperature=temperature if temperature is not None
            else getattr(g, "temperature", None),
            max_tokens=max_tokens if max_tokens is not None
            else getattr(g, "max_tokens", None),
            num_ctx=getattr(g, "num_ctx", None),
            think=getattr(g, "think", None),
            json_output=bool(json_output), expect=expect)

    def generate_messages(self, messages: Sequence[Dict[str, str]],
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          json_output: bool = False,
                          expect: str = "any",
                          **kw: Any) -> Dict[str, Any]:
        eff_temp = temperature if temperature is not None else getattr(
            self.generator, "temperature", None)
        if not self.enabled or not self.cache.should_cache(eff_temp):
            self.cache_stats["skipped"] += 1
            return self.generator.generate_messages(
                messages, temperature=temperature, max_tokens=max_tokens,
                json_output=json_output, expect=expect, **kw)

        key = self._key(messages, temperature, max_tokens, json_output, expect)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_stats["hits"] += 1
            saved = float(hit.get("elapsed") or 0.0)
            self.cache_stats["seconds_saved"] += saved
            r = dict(hit)
            # 不谎报耗时：本次实际几乎不花时间，原始耗时挪到 cache_saved_seconds 留档
            r["cache_hit"] = True
            r["cache_saved_seconds"] = saved
            r["elapsed"] = 0.0
            if self.verbose:
                print(f"[缓存] 命中，省下 {saved}s")
            return r

        self.cache_stats["misses"] += 1
        r = self.generator.generate_messages(
            messages, temperature=temperature, max_tokens=max_tokens,
            json_output=json_output, expect=expect, **kw)
        # 只缓存**成功**的结果：失败结果（连接断、JSON 解析不出）缓存下来等于把故障钉死
        if r.get("text") and (not json_output or r.get("json_ok")):
            storable = {k: v for k, v in r.items() if k not in ("attempts",)}
            self.cache.set(key, storable, temperature=eff_temp,
                           meta={"elapsed": r.get("elapsed"),
                                 "model": getattr(self.generator, "model_name", "?")})
        r["cache_hit"] = False
        return r

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 json_output: bool = False, expect: str = "any", **kw: Any) -> Dict[str, Any]:
        """与 `LLMGenerator.generate` 同签名；内部转成 messages 再走缓存。"""
        jf = getattr(self.generator, "JSON_FORMAT_INSTRUCTION", None)
        if jf is None:
            jf = sys.modules[self.generator.__class__.__module__].JSON_FORMAT_INSTRUCTION
        user = prompt + (jf if json_output else "")
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
            + [{"role": "user", "content": user}]
        return self.generate_messages(messages, temperature=temperature,
                                      max_tokens=max_tokens, json_output=json_output,
                                      expect=expect, **kw)

    def info(self) -> Dict[str, Any]:
        return {**self.cache_stats, "cache": self.cache.info()}


# ============================================================================
# 流水线级缓存键：查询 + 检索到的证据
# ============================================================================
def evidence_fingerprint(docs: Sequence[Any]) -> str:
    """把一批检索结果压成稳定指纹：块 id + 正文哈希，与顺序有关（顺序变了上下文就变）。"""
    parts: List[str] = []
    for d in docs or []:
        if isinstance(d, dict):
            cid, text = d.get("chunk_id") or "", d.get("text") or ""
        else:
            cid, text = getattr(d, "chunk_id", "") or "", getattr(d, "text", "") or ""
        parts.append(f"{cid}:{_sha256(text)[:16]}")
    return "|".join(parts)


def make_pipeline_key(cache: GenerationCache, query: str, docs: Sequence[Any],
                      **params: Any) -> str:
    """流水线级缓存键 —— 任务书说的「查询 + 上下文的哈希」。

    上下文用检索结果的指纹，而不是组装后的 context_text：组装是确定性的，
    但要拿到 context_text 就得先跑一遍组装，那样缓存就省不掉组装那步了。
    """
    return cache.make_key(query=query, context=evidence_fingerprint(docs), **params)


# ============================================================================
# CLI 演示（不需要 Ollama）
# ============================================================================
def _demo() -> int:
    print("=" * 88)
    print("生成缓存演示：键设计 / TTL / 容量淘汰 / 温度门限")
    print("=" * 88)

    cache = GenerationCache(max_entries=3, ttl_seconds=2, max_temperature=0.0, verbose=True)
    q = "What is the evidence for lecanemab in Alzheimer disease?"
    ctx_a = "[S1] PMC123 · NEJM (2023) · Results\nCDR-SB difference was -0.45."
    ctx_b = "[S1] PMC999 · Lancet (2019) · Results\nSomething else entirely."

    k1 = cache.make_key(query=q, context=ctx_a, model="qwen3:8b", temperature=0.0)
    k2 = cache.make_key(query=q, context=ctx_a, model="qwen3:8b", temperature=0.0)
    k3 = cache.make_key(query=q, context=ctx_b, model="qwen3:8b", temperature=0.0)
    k4 = cache.make_key(query=q, context=ctx_a, model="qwen3:8b", temperature=0.3)
    print(f"\n① 相同查询+相同上下文 → 同一个键：{k1 == k2}   {k1[:16]}…")
    print(f"② 换了上下文        → 不同键：{k1 != k3}   {k3[:16]}…")
    print(f"③ 换了温度参数      → 不同键：{k1 != k4}   {k4[:16]}…")

    cache.set(k1, {"answer": "…带 [S1] 的回答…", "elapsed": 41.2}, temperature=0.0,
              meta={"elapsed": 41.2})
    print(f"\n④ 写入后命中：{cache.get(k1) is not None}，第二次命中计数 {cache.stats['hits']}")

    wrote = cache.set(cache.make_key(query="高温", context=""), {"a": 1}, temperature=0.3)
    print(f"⑤ 温度 0.3 > 门限 0.0 → 拒绝写入：{not wrote}"
          f"（skipped={cache.stats['skipped_temperature']}）")

    for i in range(4):                       # max_entries=3，写第 4 条会淘汰最久未用的
        cache.set(cache.make_key(query=f"q{i}", context=""), {"i": i}, temperature=0.0)
    print(f"⑥ 容量上限 3，写了 5 条后仍为 {len(cache)} 条"
          f"（LRU 淘汰 {cache.stats['evicted_lru']} 条）")

    print("⑦ TTL=2s，等 2.1s 后……", end=" ")
    time.sleep(2.1)
    n = cache.purge_expired()
    print(f"过期清理 {n} 条，剩 {len(cache)} 条")

    print(f"\n缓存状态：{json.dumps(cache.info(), ensure_ascii=False)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="离线演示（不调模型）")
    args = ap.parse_args()
    if args.demo:
        return _demo()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
