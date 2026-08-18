# -*- coding: utf-8 -*-
"""第十阶段（一）· 服务化 —— 日志配置、请求 ID 与调用记录库

任务书要的是"记录每次调用的请求ID、耗时、结果状态到日志或数据库（用于后续统计）"。
这里**两个都做**，因为它们回答的不是同一个问题：

  · **日志**（滚动文本）回答"这一次到底发生了什么"——按 request_id grep，
    能看到进了哪几段、每段多久、哪一步报的错。人在排障时读的是它。
  · **SQLite**（结构化）回答"这一批总体怎么样"——成功率、p95 耗时、拒答率、token 花销。
    "用于后续统计"这句话意味着要能聚合，而聚合一份滚动文本是自找麻烦。

一个必须说清的取舍：**request_id 用 contextvar 传，不用参数一路透传**。
FastAPI 的同步路由跑在线程池里，`contextvars` 会随 `anyio.to_thread.run_sync` 一起复制过去，
所以中间件里 set 一次，日志格式化器、SQLite 写入、异常处理器都能拿到同一个值。
（这条**必须靠实测确认**而不是相信文档——验证脚本 D 组里有一条就是在线程池里断言它没丢。）

关于耗时口径：流式接口的耗时**不能**在中间件里量。中间件在响应头发出去那一刻就结束了，
而 SSE 的正文还要再流一两分钟。所以流式请求的记录由流生成器自己在收尾时写。

用法：
    lg  = _load("lg", r"E:\\rag\\scripts\\服务_日志.py")
    lg.configure_logging()                       # 幂等，可重复调用
    rid = lg.new_request_id(); lg.set_request_id(rid)
    log = lg.get_logger("qa"); log.info("...")   # 行首自动带 [rid]

    store = lg.CallLogStore()                    # 默认 E:\\rag\\data\\service\\api_calls.db
    store.record(request_id=rid, path="/api/v1/qa/ask", status="ok", elapsed_ms=1234.5, ...)
    rows, total = store.page(page=1, page_size=20, status="ok")
    store.stats(since_hours=24)

CLI：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\服务_日志.py --stats
    ... --tail 20        最近 20 条调用记录
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import argparse
import json
import logging
import logging.handlers
import os
import sqlite3
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG_DIR = os.path.join(ROOT, "logs", "service")
DB_DIR = os.path.join(ROOT, "data", "service")
DB_PATH = os.path.join(DB_DIR, "api_calls.db")

#: 当前请求的 ID。中间件 set，其余各处 get；线程池会自动继承（见模块 docstring）
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id(prefix: str = "req") -> str:
    """短而可读的 ID：`req-8f3a1c9d`。比完整 uuid 好复制，8 位十六进制在单机量级足够。"""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def set_request_id(rid: str) -> None:
    _request_id.set(rid or "-")


def get_request_id() -> str:
    return _request_id.get()


# ============================================================================
# 一、日志配置
# ============================================================================
class _RequestIdFilter(logging.Filter):
    """把 contextvar 里的 request_id 塞进每条记录，这样格式串里可以直接用 %(request_id)s。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False
_cfg_lock = threading.Lock()


def configure_logging(log_dir: str = LOG_DIR, level: str = "INFO", console: bool = True,
                      max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5,
                      force: bool = False) -> Dict[str, Any]:
    """配置根日志器。**幂等**——重复调用不会叠加 handler（uvicorn reload 时会重复 import）。

    Returns: 实际生效的配置，供 `/health` 与验证脚本核对。
    """
    global _configured
    with _cfg_lock:
        root = logging.getLogger()
        if _configured and not force:
            return {"already_configured": True, "level": logging.getLevelName(root.level),
                    "handlers": [type(h).__name__ for h in root.handlers]}
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        os.makedirs(log_dir, exist_ok=True)
        lv = getattr(logging, str(level).upper(), logging.INFO)
        root.setLevel(lv)
        fmt = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        flt = _RequestIdFilter()

        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "api.log"), maxBytes=max_bytes,
            backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(flt)
        root.addHandler(fh)

        # 错误单独再落一份：排障时先看这个文件，不用在几十兆正常日志里翻
        eh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "error.log"), maxBytes=max_bytes,
            backupCount=backup_count, encoding="utf-8")
        eh.setLevel(logging.WARNING)
        eh.setFormatter(fmt)
        eh.addFilter(flt)
        root.addHandler(eh)

        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            ch.addFilter(flt)
            root.addHandler(ch)

        # uvicorn 自带的两个 logger 默认有自己的 handler，会导致同一行打两遍
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            lg = logging.getLogger(name)
            lg.handlers = []
            lg.propagate = True

        _configured = True
        return {"already_configured": False, "level": logging.getLevelName(lv),
                "log_dir": log_dir, "files": ["api.log", "error.log"],
                "handlers": [type(h).__name__ for h in root.handlers]}


def get_logger(name: str = "medrag.api") -> logging.Logger:
    lg = logging.getLogger(name if name.startswith("medrag") else f"medrag.{name}")
    lg.addFilter(_RequestIdFilter())     # 直接对 logger 记录时也要有 request_id
    return lg


# ============================================================================
# 二、调用记录库
# ============================================================================
_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    request_id     TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    ts             REAL NOT NULL,
    method         TEXT DEFAULT '',
    path           TEXT DEFAULT '',
    mode           TEXT DEFAULT '',      -- sync / stream
    session_id     TEXT,
    query          TEXT DEFAULT '',
    top_k          INTEGER,
    status         TEXT DEFAULT '',      -- ok / error / busy
    code           INTEGER DEFAULT 0,
    http_status    INTEGER DEFAULT 200,
    elapsed_ms     REAL DEFAULT 0,
    llm_calls      INTEGER,
    prompt_tokens  INTEGER,
    output_tokens  INTEGER,
    answer_chars   INTEGER,
    sources        INTEGER,
    refused        INTEGER,              -- 0/1/NULL；NULL = 没走到生成
    compliant      INTEGER,              -- 层 D 判定；NULL = 未开启约束
    client         TEXT DEFAULT '',
    error          TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_calls_ts      ON api_calls(ts DESC);
CREATE INDEX IF NOT EXISTS idx_calls_status  ON api_calls(status);
CREATE INDEX IF NOT EXISTS idx_calls_session ON api_calls(session_id);
"""

#: 允许写入的列。白名单而不是"把 kwargs 拼进 SQL"——后者是注入与拼错列名的双重来源。
_COLUMNS = ("request_id", "created_at", "ts", "method", "path", "mode", "session_id",
            "query", "top_k", "status", "code", "http_status", "elapsed_ms", "llm_calls",
            "prompt_tokens", "output_tokens", "answer_chars", "sources", "refused",
            "compliant", "client", "error")


def _pct(values: List[float], q: float) -> float:
    """最近邻分位数。样本量小的时候插值反而制造精度幻觉，直接取最近的那个观测值。"""
    if not values:
        return 0.0
    xs = sorted(values)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return round(xs[i], 2)


class CallLogStore:
    """调用记录的 SQLite 存储。单连接 + 锁 + WAL。

    为什么不开连接池：写入量是"每次问答一条"，而一次问答要一百多秒——这里永远不会是
    瓶颈。单连接换来的是"绝不会出现半条记录"和调试时能直接用 sqlite3 命令行打开。

    ⚠ 所有写入都吞掉异常并转成日志：**记录失败绝不能让问答请求失败**。
    日志是观测手段，观测手段挂了应该报警，不该连累业务。
    """

    def __init__(self, db_path: str = DB_PATH, timeout: float = 10.0):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=timeout)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        self.log = get_logger("calllog")
        #: 写失败计数——健康检查会看它，不让存储故障静默
        self.write_failures = 0

    # ---------------- 写 ----------------
    def record(self, request_id: str, **fields: Any) -> bool:
        """写一条调用记录（同 request_id 覆盖，便于流式先占位后补全）。"""
        row: Dict[str, Any] = {"request_id": request_id,
                               "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "ts": time.time()}
        for k, v in fields.items():
            if k not in _COLUMNS:
                continue
            if isinstance(v, bool):
                v = int(v)
            row[k] = v
        if isinstance(row.get("query"), str) and len(row["query"]) > 2000:
            row["query"] = row["query"][:2000]          # 别让一条超长问题把库撑大
        if isinstance(row.get("error"), str) and len(row["error"]) > 1000:
            row["error"] = row["error"][:1000]

        cols = list(row)
        sql = (f"INSERT OR REPLACE INTO api_calls ({','.join(cols)}) "
               f"VALUES ({','.join('?' * len(cols))})")
        try:
            with self._lock:
                self._conn.execute(sql, [row[c] for c in cols])
                self._conn.commit()
            return True
        except sqlite3.Error as e:
            self.write_failures += 1
            self.log.error("调用记录写入失败（不影响本次问答）：%s", e)
            return False

    # ---------------- 读 ----------------
    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM api_calls WHERE request_id=?", (request_id,))
            r = cur.fetchone()
        return dict(r) if r else None

    def page(self, page: int = 1, page_size: int = 20, status: Optional[str] = None,
             session_id: Optional[str] = None, mode: Optional[str] = None,
             since_hours: Optional[float] = None) -> Tuple[List[Dict[str, Any]], int]:
        """按时间倒序分页。返回 (本页行, 满足条件的总数)。"""
        where, args = [], []
        if status:
            where.append("status=?"); args.append(status)
        if session_id:
            where.append("session_id=?"); args.append(session_id)
        if mode:
            where.append("mode=?"); args.append(mode)
        if since_hours:
            where.append("ts>=?"); args.append(time.time() - float(since_hours) * 3600)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        off = max(0, (max(1, int(page)) - 1) * int(page_size))
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM api_calls{clause}",
                                       args).fetchone()[0]
            # ⚠ 必须带 rowid 兜底：`ts` 是 `time.time()`，同一毫秒内写入的多条记录
            # 时间戳会完全相同，只按 ts 排序时**顺序不稳定**——分页时同一条可能重复出现
            # 或整条漏掉。验证里那条"按时间倒序"就是被这个绊倒的（三条记录写在同一瞬间）。
            rows = self._conn.execute(
                f"SELECT * FROM api_calls{clause} ORDER BY ts DESC, rowid DESC "
                f"LIMIT ? OFFSET ?", args + [int(page_size), off]).fetchall()
        return [dict(r) for r in rows], int(total)

    def stats(self, since_hours: Optional[float] = None) -> Dict[str, Any]:
        """聚合统计。分位数在 Python 侧算（SQLite 没有 percentile，这个量级也不值得上扩展）。"""
        where, args = "", []
        if since_hours:
            where, args = " WHERE ts>=?", [time.time() - float(since_hours) * 3600]
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                f"SELECT status, code, mode, elapsed_ms, refused, compliant, "
                f"prompt_tokens, output_tokens, llm_calls FROM api_calls{where}",
                args).fetchall()]

        total = len(rows)
        by_status: Dict[str, int] = {}
        by_code: Dict[str, int] = {}
        by_mode: Dict[str, int] = {}
        for r in rows:
            by_status[r["status"] or "?"] = by_status.get(r["status"] or "?", 0) + 1
            by_code[str(r["code"] or 0)] = by_code.get(str(r["code"] or 0), 0) + 1
            by_mode[r["mode"] or "?"] = by_mode.get(r["mode"] or "?", 0) + 1

        el = [float(r["elapsed_ms"] or 0) for r in rows]
        ok_rows = [r for r in rows if r["status"] == "ok"]
        refused = [r["refused"] for r in ok_rows if r["refused"] is not None]
        compl = [r["compliant"] for r in ok_rows if r["compliant"] is not None]

        return {
            "total": total,
            "by_status": by_status, "by_code": by_code, "by_mode": by_mode,
            "success_rate": round(len(ok_rows) / total, 4) if total else 0.0,
            "refusal_rate": round(sum(refused) / len(refused), 4) if refused else None,
            "compliant_rate": round(sum(compl) / len(compl), 4) if compl else None,
            "elapsed_ms": {"avg": round(sum(el) / len(el), 2) if el else 0.0,
                           "p50": _pct(el, 0.50), "p95": _pct(el, 0.95),
                           "max": round(max(el), 2) if el else 0.0},
            "tokens": {"prompt": sum(int(r["prompt_tokens"] or 0) for r in rows),
                       "output": sum(int(r["output_tokens"] or 0) for r in rows),
                       "llm_calls": sum(int(r["llm_calls"] or 0) for r in rows)},
            "window": f"最近 {since_hours} 小时" if since_hours else "全部",
        }

    def export(self, path: str, since_hours: Optional[float] = None,
               limit: int = 5000) -> int:
        """把调用记录导成人能读的表格（交付/汇报用）。返回导出条数。

        不导 `.db` 本身：交付包里放一个二进制库，收件人还得装个工具才能打开；
        而这份表格用记事本就能看，字段含义也直接写在表头说明里。
        """
        rows, total = self.page(1, int(limit), since_hours=since_hours)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        st = self.stats(since_hours)
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 118 + "\n")
            f.write("医学知识 RAG 问答服务 · 调用记录导出\n")
            f.write("=" * 118 + "\n")
            f.write(f"导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　"
                    f"来源库：{self.db_path}　"
                    f"范围：{'最近 %s 小时' % since_hours if since_hours else '全部'}　"
                    f"共 {total} 条（本文件含 {len(rows)} 条）\n\n")
            f.write("字段说明：\n"
                    "  request_id  本次请求的唯一 ID，与响应体、响应头、服务端日志行三处一致\n"
                    "  mode        sync=同步接口 / stream=流式接口\n"
                    "  status      ok=成功 / error=失败 / busy=并发闸门拦下\n"
                    "  code        业务错误码，0 为成功（码表见 GET /api/v1/errors）\n"
                    "  refused     答案里出现了固定拒答短语（**不是错误**，是守住知识库边界）\n"
                    "  compliant   阶段九层 D 的格式/引用/术语校验是否通过\n\n")
            f.write("=" * 118 + "\n")
            head = (f"{'时间':<20}{'request_id':<16}{'mode':<8}{'status':<8}{'code':>6}"
                    f"{'耗时ms':>11}{'调用':>5}{'入tok':>8}{'出tok':>7}{'字数':>6}"
                    f"{'拒答':>5}{'合规':>5}  问题\n")
            f.write(head)
            f.write("-" * 118 + "\n")
            for r in rows:
                def _b(v):
                    return "-" if v is None else ("是" if v else "否")
                f.write(f"{str(r['created_at']):<20}{str(r['request_id']):<16}"
                        f"{str(r['mode'] or ''):<8}{str(r['status'] or ''):<8}"
                        f"{r['code'] or 0:>6}{(r['elapsed_ms'] or 0):>11.1f}"
                        f"{r['llm_calls'] if r['llm_calls'] is not None else '-':>5}"
                        f"{r['prompt_tokens'] if r['prompt_tokens'] is not None else '-':>8}"
                        f"{r['output_tokens'] if r['output_tokens'] is not None else '-':>7}"
                        f"{r['answer_chars'] if r['answer_chars'] is not None else '-':>6}"
                        f"{_b(r['refused']):>5}{_b(r['compliant']):>5}  "
                        f"{(r['query'] or '')[:40]}\n")
                if r["error"]:
                    f.write(f"{'':>20}{'':>16}└─ 错误：{r['error'][:90]}\n")
            f.write("=" * 118 + "\n")
            f.write("聚合统计\n")
            f.write("=" * 118 + "\n")
            f.write(json.dumps(st, ensure_ascii=False, indent=2) + "\n")
        return len(rows)

    def purge(self, keep_days: float = 30.0) -> int:
        """清理过期记录，返回删除条数。"""
        cutoff = time.time() - keep_days * 86400
        with self._lock:
            cur = self._conn.execute("DELETE FROM api_calls WHERE ts<?", (cutoff,))
            self._conn.commit()
            return cur.rowcount

    def healthy(self) -> Tuple[bool, str]:
        """存储可用性：能不能读。写失败计数一并报出来。"""
        try:
            with self._lock:
                self._conn.execute("SELECT 1 FROM api_calls LIMIT 1").fetchone()
            if self.write_failures:
                return False, f"已有 {self.write_failures} 次写入失败"
            return True, self.db_path
        except sqlite3.Error as e:
            return False, str(e)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--stats", action="store_true", help="打印聚合统计")
    ap.add_argument("--tail", type=int, default=0, help="打印最近 N 条调用记录")
    ap.add_argument("--hours", type=float, default=None, help="只统计最近 N 小时")
    ap.add_argument("--purge-days", type=float, default=None, help="删除 N 天前的记录")
    ap.add_argument("--export", default=None, metavar="路径",
                    help="把调用记录导成人能读的表格（交付/汇报用）")
    args = ap.parse_args()

    if not (args.stats or args.tail or args.purge_days or args.export):
        ap.print_help()
        return 0
    if not os.path.exists(args.db):
        print(f"还没有调用记录库：{args.db}（服务跑过一次问答后自动生成）")
        return 1

    store = CallLogStore(args.db)
    if args.export:
        n = store.export(args.export, since_hours=args.hours)
        print(f"已导出 {n} 条调用记录 → {args.export}")
    if args.purge_days:
        print(f"已删除 {store.purge(args.purge_days)} 条 {args.purge_days} 天前的记录")
    if args.stats:
        print(json.dumps(store.stats(args.hours), ensure_ascii=False, indent=2))
    if args.tail:
        rows, total = store.page(1, args.tail, since_hours=args.hours)
        print(f"最近 {len(rows)} / 共 {total} 条")
        for r in rows:
            print(f"  {r['created_at']}  {r['request_id']}  {r['mode']:<6} "
                  f"{r['status']:<5} code={r['code']:<5} {r['elapsed_ms']:>9.1f}ms  "
                  f"{(r['query'] or '')[:40]}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
