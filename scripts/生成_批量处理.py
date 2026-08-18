# -*- coding: utf-8 -*-
"""第八阶段 · 并行批量处理：ThreadPoolExecutor + 顺序保持 + 错误隔离

三条硬要求（任务书点名），实现方式各自对应一句代码级的保证：

  1. **输入输出顺序一致** —— 结果列表按下标预分配，`as_completed` 只用来报进度，
     写回一律按 `future → index` 的映射。**绝不** `results.append(...)`：
     那样顺序取决于谁先跑完，同一批任务两次跑能得到两种顺序，对不上号。
  2. **单个任务失败不影响其他** —— 每个任务在 worker 内自带 try/except，
     失败项以 `{"ok": False, "error": …}` 占位，位置仍然保留（不塌陷）。
     捕获的是 `BaseException` 之外的 `Exception`：KeyboardInterrupt 仍能中断整批。
  3. **按 CPU 核心数定并发度，避免过度竞争** —— `recommended_workers()`。

⚠ 关于"并行能不能加速本项目的生成"，先把事实摆在前面，别期待落空：

    本机是**单卡 10G + 单个 Ollama 进程**。qwen3:8b 常驻约 6.3 GB 显存，多个请求
    争的是同一份权重和同一块 GPU；Ollama 的 `OLLAMA_NUM_PARALLEL` 不显式设时会按显存
    自行决定（常见为 1），此时并发请求会在服务端排队，**并行度提上去不等于吞吐提上去**。
    所以本模块的默认 `kind="llm"` 给的是保守并发度，并提供 `benchmark()` 让人
    **用实测数字**决定该开几个，而不是照搬"CPU 核数"这个直觉。
    对**不调模型**的批量工作（评估答案、算 ROUGE、解析 JSON——阶段八新增的这些都是），
    并行是实打实有效的，用 `kind="cpu"`。

用法：
    import importlib.util
    spec = importlib.util.spec_from_file_location("bp", r"E:\\rag\\scripts\\生成_批量处理.py")
    bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)

    proc = bp.ParallelBatchProcessor(kind="cpu")          # 或 kind="llm"
    out  = proc.run(queries, lambda q: pipe.generate(q))  # 顺序与 queries 一致
    out[0]["ok"] / out[0]["value"] / out[0]["error"] / out[0]["seconds"]

CLI 演示（不需要 Ollama）：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\生成_批量处理.py --demo
    ... --benchmark        # 实测不同并发度的加速比
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CPU_COUNT = os.cpu_count() or 4

#: 各类工作的并发度上限。数字背后的理由写在 recommended_workers 的 docstring 里。
WORKER_CAPS = {"llm": 4, "cpu": CPU_COUNT, "io": 32}


def recommended_workers(kind: str = "cpu", n_items: Optional[int] = None) -> int:
    """按工作性质与 CPU 核数给出并发度。任务再多也不会超过任务数本身。

      kind="llm"  调本地大模型：瓶颈是**单块 GPU**，不是 CPU。取 min(4, 核数/4)，
                  再由 benchmark 实测校正。开太多只会让请求在 Ollama 服务端排队，
                  还会同时占住多份 KV cache 显存。
      kind="cpu"  纯计算（评估、分词、正则、ROUGE）：取核数，但**留一核**给主线程与系统，
                  否则进度输出和 Ctrl-C 都会变迟钝。
      kind="io"   读写文件/网络：不受核数约束，取 min(32, 核数×2)。
    """
    if kind == "llm":
        w = max(1, min(WORKER_CAPS["llm"], CPU_COUNT // 4))
    elif kind == "io":
        w = max(1, min(WORKER_CAPS["io"], CPU_COUNT * 2))
    else:
        w = max(1, CPU_COUNT - 1)
    if n_items:
        w = max(1, min(w, int(n_items)))
    return w


class ParallelBatchProcessor:
    """并行跑一批任务，保证顺序与错误隔离。

    Args:
        max_workers: 并发度；None 则按 kind 与 CPU 核数推荐
        kind:        "llm" / "cpu" / "io"，见 recommended_workers
        timeout:     单个任务的超时（秒）；None 不限
        verbose:     打印每条完成情况
    """

    def __init__(self, max_workers: Optional[int] = None, kind: str = "cpu",
                 timeout: Optional[float] = None, verbose: bool = False):
        self.kind = kind
        self.max_workers = int(max_workers) if max_workers else recommended_workers(kind)
        self.timeout = timeout
        self.verbose = verbose
        self.stats: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ---------------- 主入口 ----------------
    def run(self, items: Sequence[Any], worker: Callable[[Any], Any],
            progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
            ) -> List[Dict[str, Any]]:
        """并行执行 `worker(item)`。

        Returns: 与 items **等长、同序**的列表，每项
                 {"index", "ok", "value", "error", "seconds", "traceback"}
        """
        n = len(items)
        workers = max(1, min(self.max_workers, n)) if n else 1
        # 结果按下标预分配 —— 顺序保证就在这一行，不靠"谁先完成谁先 append"
        results: List[Optional[Dict[str, Any]]] = [None] * n
        t_start = time.time()
        done = 0

        def _task(idx: int, item: Any) -> Dict[str, Any]:
            t0 = time.time()
            try:
                v = worker(item)
                return {"index": idx, "ok": True, "value": v, "error": None,
                        "seconds": round(time.time() - t0, 3), "traceback": ""}
            except Exception as e:                     # noqa: BLE001 —— 隔离，别让一条毁一批
                return {"index": idx, "ok": False, "value": None, "error": f"{type(e).__name__}: {e}",
                        "seconds": round(time.time() - t0, 3),
                        "traceback": traceback.format_exc(limit=6)}

        if n:
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="batch") as pool:
                futures = {pool.submit(_task, i, it): i for i, it in enumerate(items)}
                for fut in as_completed(futures, timeout=self.timeout):
                    i = futures[fut]
                    try:
                        r = fut.result()
                    except Exception as e:             # noqa: BLE001 —— 池本身的异常
                        r = {"index": i, "ok": False, "value": None,
                             "error": f"{type(e).__name__}: {e}", "seconds": 0.0,
                             "traceback": traceback.format_exc(limit=6)}
                    results[i] = r                     # 按下标写回，不是 append
                    with self._lock:
                        done += 1
                        k = done
                    if progress:
                        progress(k, n, r)
                    elif self.verbose:
                        mark = "ok" if r["ok"] else "FAIL"
                        print(f"[批量] {k}/{n} #{i} {mark} {r['seconds']}s"
                              + ("" if r["ok"] else f" · {r['error'][:80]}"))

        wall = time.time() - t_start
        out = [r if r is not None else
               {"index": i, "ok": False, "value": None, "error": "未执行", "seconds": 0.0,
                "traceback": ""} for i, r in enumerate(results)]
        serial = sum(r["seconds"] for r in out)
        self.stats = {
            "items": n, "workers": workers, "kind": self.kind,
            "ok": sum(1 for r in out if r["ok"]),
            "failed": sum(1 for r in out if not r["ok"]),
            "wall_seconds": round(wall, 3),
            "sum_task_seconds": round(serial, 3),
            # 加速比 = 各任务耗时之和 / 墙钟时间。串行时约等于 1；被服务端排队卡住时也接近 1
            "speedup_vs_serial": round(serial / wall, 2) if wall > 0 else None,
            "cpu_count": CPU_COUNT,
        }
        return out

    # ---------------- 便捷形态 ----------------
    def map_values(self, items: Sequence[Any], worker: Callable[[Any], Any],
                   default: Any = None) -> List[Any]:
        """只要值，失败位置填 default（顺序仍与输入一致）。"""
        return [r["value"] if r["ok"] else default for r in self.run(items, worker)]


def benchmark(items: Sequence[Any], worker: Callable[[Any], Any],
              workers_list: Sequence[int], kind: str = "cpu",
              verbose: bool = True) -> List[Dict[str, Any]]:
    """同一批任务在不同并发度下各跑一遍，**用实测决定该开几个线程**。

    对本机的 LLM 批量来说，这个函数存在的意义是把"多线程一定更快"这个直觉证伪或证实，
    而不是替它辩护。
    """
    rows: List[Dict[str, Any]] = []
    base: Optional[float] = None
    for w in workers_list:
        proc = ParallelBatchProcessor(max_workers=w, kind=kind)
        out = proc.run(items, worker)
        st = dict(proc.stats)
        base = base if base is not None else st["wall_seconds"]
        st["relative_to_first"] = (round(base / st["wall_seconds"], 2)
                                   if st["wall_seconds"] > 0 else None)
        st["all_ok"] = st["failed"] == 0
        rows.append(st)
        if verbose:
            print(f"  workers={w:<3} 墙钟 {st['wall_seconds']:>7.2f}s  "
                  f"任务耗时和 {st['sum_task_seconds']:>7.2f}s  "
                  f"加速比 {st['speedup_vs_serial']}×  "
                  f"相对 workers={workers_list[0]} {st['relative_to_first']}×  "
                  f"失败 {st['failed']}")
    return rows


# ============================================================================
# CLI 演示（不需要 Ollama）
# ============================================================================
def _demo() -> int:
    print("=" * 88)
    print(f"并行批量处理演示（CPU 逻辑核 {CPU_COUNT}）")
    print("=" * 88)
    print(f"推荐并发度：llm={recommended_workers('llm')}  "
          f"cpu={recommended_workers('cpu')}  io={recommended_workers('io')}")

    # ① 顺序一致 + ② 错误隔离：第 3、7 项故意抛异常
    def work(i: int) -> str:
        time.sleep(0.05 * ((i % 3) + 1))            # 故意让耗时不一，打乱完成顺序
        if i in (3, 7):
            raise ValueError(f"第 {i} 项故意失败")
        return f"结果-{i}"

    items = list(range(10))
    proc = ParallelBatchProcessor(max_workers=4, kind="io")
    out = proc.run(items, work)

    order_ok = all(r["index"] == i for i, r in enumerate(out))
    values_ok = all(r["value"] == f"结果-{i}" for i, r in enumerate(out) if r["ok"])
    failed_idx = [r["index"] for r in out if not r["ok"]]
    ok_count = sum(1 for r in out if r["ok"])

    print(f"\n① 输出顺序与输入一致：{order_ok}")
    print(f"   值与下标一一对应：  {values_ok}")
    print(f"② 失败的是第 {failed_idx} 项，其余 {ok_count} 项照常完成（失败不扩散）")
    print(f"   失败项的错误信息：  {out[3]['error']}")
    print(f"③ 统计：{json.dumps(proc.stats, ensure_ascii=False)}")

    print("\n④ 不同并发度实测（每项 sleep 0.2s，模拟 I/O 等待）：")
    benchmark(list(range(12)), lambda i: time.sleep(0.2), [1, 2, 4, 8], kind="io")
    return 0


def _benchmark_cpu() -> int:
    """CPU 密集型对照：Python 有 GIL，纯计算的多线程加速有限，这里把它量出来。"""
    print("=" * 88)
    print("CPU 密集任务的多线程实测（GIL 影响，用实测代替猜测）")
    print("=" * 88)

    def burn(n: int) -> int:
        s = 0
        for i in range(400_000):
            s += i * i % 7
        return s

    benchmark(list(range(8)), burn, [1, 2, 4, recommended_workers("cpu")], kind="cpu")
    print("\n注：纯 Python 计算受 GIL 限制，加速比通常远小于线程数；"
          "\n    真正吃满多核要用多进程。本项目批量的瓶颈在模型推理与 I/O，故用线程池。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="顺序/错误隔离/并发度演示")
    ap.add_argument("--benchmark", action="store_true", help="CPU 密集任务的并发度实测")
    args = ap.parse_args()
    if args.benchmark:
        return _benchmark_cpu()
    if args.demo:
        return _demo()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
