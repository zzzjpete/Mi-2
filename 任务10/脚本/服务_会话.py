# -*- coding: utf-8 -*-
"""第十阶段（一）· 服务化 —— 会话存储与追问改写

任务书："若传入 session_id，则关联历史对话"。**"关联"要落到实处**，否则就只是把历史存进库
再原样吐出来，对答案毫无影响。在 RAG 里历史真正起作用的地方只有一个：

    追问句本身检索不到东西。

"它的副作用呢？"这句话拿去检索全库 399.8 万块英文文献，召回的是噪声；模型再怎么守约束，
基于噪声证据只能拒答。所以本模块做两件事：

  1. **存**：SQLite 两张表（会话 / 轮次），带 TTL 清理与分页。
  2. **改写**：把追问补全成能独立检索的问题，再交给流水线。

改写分三级，**默认 `llm`，失败自动降级**（降级路径必须存在：改写只是优化，
它挂了不该让问答挂掉）：

    none    不改写。历史只存不用——想量"改写到底有没有用"时的对照组。
    concat  把上一轮问题拼在前面。零成本、零依赖，但只对**检索**有效：
            模型看到的问题仍是"它的副作用呢？"，指代没解决。
    llm     让模型把追问重写成独立问题，检索与提问都用它。多一次约 2 秒的调用。

**指代检测器只是个省钱的闸门，不是判决**：它说"是追问"就花一次调用去改写，说"不是"就跳过。
所以它的假阳性代价只是白花 2 秒，假阴性才是真损失。验证脚本会在一组人工标注的例句上
把这两个数分别报出来——不写"检测准确"，而是给出它到底错在哪一侧。

⚠ 未知的 session_id 走**自动建会话**而不是报 3002：聊天类接口要求先建会话再提问是徒增一次
往返。`GET /api/v1/sessions/{id}` 那条路径上不存在就是 3002，两者不矛盾——一个是写入语义，
一个是查询语义。

用法：
    ss = _load("ss", r"E:\\rag\\scripts\\服务_会话.py")
    store = ss.SessionStore()                       # E:\\rag\\data\\service\\sessions.db
    store.append_turn(sid, "user", "……", request_id=rid)
    hist = store.history(sid, max_turns=4)          # [{'role','content',...}, ...]
    rw   = ss.FollowupRewriter(generator=gen)
    out  = rw.rewrite("它的副作用呢？", hist, mode="llm")
    out["resolved"] / out["rewritten"] / out["mode_used"] / out["note"]

CLI：
    E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\服务_会话.py --detect
    ... --list          列出会话
    ... --show <sid>    看某个会话的全部轮次
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB_PATH = os.path.join(ROOT, "data", "service", "sessions.db")

DEFAULT_TTL_DAYS = 30.0
DEFAULT_HISTORY_TURNS = 4
MAX_CONTENT_CHARS = 8000        # 单轮内容上限，防一份长答案把库撑爆


# ============================================================================
# 一、会话存储
# ============================================================================
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    ts_updated  REAL NOT NULL,
    title       TEXT DEFAULT '',
    turns       INTEGER DEFAULT 0,
    meta        TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    request_id  TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    ts          REAL NOT NULL,
    meta        TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_turns_sid ON turns(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_sess_ts   ON sessions(ts_updated DESC);
"""


def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"


class SessionStore:
    """会话与轮次的 SQLite 存储。单连接 + 锁 + WAL，理由同 `服务_日志.CallLogStore`。"""

    def __init__(self, db_path: str = DB_PATH, ttl_days: float = DEFAULT_TTL_DAYS,
                 timeout: float = 10.0):
        self.db_path = db_path
        self.ttl_days = float(ttl_days)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=timeout)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---------------- 会话 ----------------
    def create(self, session_id: Optional[str] = None, title: str = "",
               meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sid = (session_id or "").strip() or new_session_id()
        now, ts = time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at, "
                "ts_updated, title, turns, meta) VALUES (?,?,?,?,?,0,?)",
                (sid, now, now, ts, title[:200], json.dumps(meta or {}, ensure_ascii=False)))
            self._conn.commit()
        return self.get(sid) or {"session_id": sid, "created_at": now, "updated_at": now,
                                 "title": title, "turns": 0}

    def get(self, session_id: str, with_history: int = 0) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM sessions WHERE session_id=?",
                                   (session_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["meta"] = json.loads(d.get("meta") or "{}")
        d["history"] = self.history(session_id, max_turns=with_history) if with_history else []
        return d

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM sessions WHERE session_id=?",
                                      (session_id,)).fetchone() is not None

    def ensure(self, session_id: str) -> Tuple[Dict[str, Any], bool]:
        """取会话；不存在就建。返回 (会话, 是否新建)。见模块 docstring 的 ⚠ 说明。"""
        s = self.get(session_id)
        if s:
            return s, False
        return self.create(session_id), True

    def list_sessions(self, page: int = 1, page_size: int = 20
                      ) -> Tuple[List[Dict[str, Any]], int]:
        off = max(0, (max(1, int(page)) - 1) * int(page_size))
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY ts_updated DESC LIMIT ? OFFSET ?",
                (int(page_size), off)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d.get("meta") or "{}")
            out.append(d)
        return out, int(total)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ---------------- 轮次 ----------------
    def append_turn(self, session_id: str, role: str, content: str,
                    request_id: str = "", meta: Optional[Dict[str, Any]] = None) -> int:
        """追加一条消息，返回它的 turn_index（从 0 开始，user/assistant 各占一个）。"""
        assert role in ("user", "assistant"), f"role 只能是 user/assistant，收到 {role!r}"
        self.ensure(session_id)
        now, ts = time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
        text = (content or "")[:MAX_CONTENT_CHARS]
        with self._lock:
            idx = self._conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns WHERE session_id=?",
                (session_id,)).fetchone()[0]
            self._conn.execute(
                "INSERT INTO turns (session_id, turn_index, role, content, request_id, "
                "created_at, ts, meta) VALUES (?,?,?,?,?,?,?,?)",
                (session_id, idx, role, text, request_id, now, ts,
                 json.dumps(meta or {}, ensure_ascii=False)))
            self._conn.execute(
                "UPDATE sessions SET updated_at=?, ts_updated=?, turns=turns+1, "
                "title=CASE WHEN title='' AND ?='user' THEN ? ELSE title END "
                "WHERE session_id=?",
                (now, ts, role, text[:60], session_id))
            self._conn.commit()
        return int(idx)

    def history(self, session_id: str, max_turns: int = DEFAULT_HISTORY_TURNS
                ) -> List[Dict[str, Any]]:
        """最近 `max_turns` 轮（一问一答算一轮，所以取 2×N 条消息），**按时间正序**返回。

        正序很重要：改写提示词里历史必须按发生顺序读，倒序会让"上一轮"指向最早那轮。
        """
        if max_turns <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn_index, role, content, request_id, created_at FROM turns "
                "WHERE session_id=? ORDER BY turn_index DESC LIMIT ?",
                (session_id, int(max_turns) * 2)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def last_user_query(self, session_id: str) -> str:
        with self._lock:
            r = self._conn.execute(
                "SELECT content FROM turns WHERE session_id=? AND role='user' "
                "ORDER BY turn_index DESC LIMIT 1", (session_id,)).fetchone()
        return (r["content"] if r else "") or ""

    # ---------------- 维护 ----------------
    def purge_expired(self, ttl_days: Optional[float] = None) -> int:
        """删掉 TTL 之外的会话（连同轮次）。返回删除的会话数。"""
        cutoff = time.time() - (self.ttl_days if ttl_days is None else float(ttl_days)) * 86400
        with self._lock:
            sids = [r[0] for r in self._conn.execute(
                "SELECT session_id FROM sessions WHERE ts_updated<?", (cutoff,)).fetchall()]
            if sids:
                qs = ",".join("?" * len(sids))
                self._conn.execute(f"DELETE FROM turns WHERE session_id IN ({qs})", sids)
                self._conn.execute(f"DELETE FROM sessions WHERE session_id IN ({qs})", sids)
                self._conn.commit()
        return len(sids)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n_s = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            n_t = self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        return {"sessions": int(n_s), "turns": int(n_t), "db": self.db_path,
                "ttl_days": self.ttl_days}

    def healthy(self) -> Tuple[bool, str]:
        try:
            with self._lock:
                self._conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()
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
# 二、追问检测
# ============================================================================
#: 指代/承接线索。中文这些词歧义大，所以**一律带上后接名词**再匹配：
#: 光认「该」会把「应该」算进来，光认「此」会把「因此」算进来，光认「其」会命中「其中/其他」。
_CUES: Tuple[Tuple[str, str], ...] = (
    (r"它们?|他们|她们", "第三人称代词"),
    (r"该(?:药物?|疗法|治疗|方案|试验|研究|方法|技术|基因|蛋白|指标|疾病|患者|机制)", "该+名词"),
    (r"[这此](?:药|类|种|个|些|项|款|疗法|方案|试验|研究|方法|技术|机制)", "这/此+名词"),
    (r"那(?:个|些|种|款)", "那+量词"),
    (r"上述|前述|上面(?:提到|说)|前面(?:提到|说)|刚才|刚刚|之前(?:提到|说)", "回指上文"),
    (r"其(?!中|他|它|余|实|次)", "其+名词"),
    (r"呢[？?！!]?\s*$", "句末「呢」"),
    (r"^(?:那么|那|还有|另外|以及|顺便|接着|然后)", "承接词开头"),
    (r"\b(?:it|its|they|them|their|this|that|these|those)\b", "英文代词"),
    (r"\b(?:the above|the former|the latter|as mentioned)\b", "英文回指"),
)
_CUES_C = tuple((re.compile(p, re.IGNORECASE), why) for p, why in _CUES)

#: 去掉标点后短于这个长度就当追问。10 是个折中：「副作用呢」4 字、「有效率是多少」6 字
#: 都该抓到，而「什么是随机对照试验」9 字会被误判——这个假阳性只浪费一次改写调用，
#: 假阴性（漏掉真追问）才会真的检索到噪声。两侧代价不对称，所以阈值往宽了取。
SHORT_QUERY_CHARS = 10
_PUNCT = re.compile(r"[\s，。？！、；：,.?!;:\"'（）()【】\[\]…—-]+")


def detect_followup(query: str, history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """判断这句是不是需要靠历史才能理解的追问。

    Returns: {"is_followup": bool, "reason": str, "cues": [...]}
    """
    if not history:
        return {"is_followup": False, "reason": "没有历史", "cues": []}
    q = (query or "").strip()
    if not q:
        return {"is_followup": False, "reason": "空问题", "cues": []}

    hits = [why for pat, why in _CUES_C if pat.search(q)]
    if hits:
        return {"is_followup": True, "reason": f"命中线索：{'、'.join(hits)}", "cues": hits}

    bare = _PUNCT.sub("", q)
    if len(bare) <= SHORT_QUERY_CHARS:
        return {"is_followup": True, "reason": f"过短（{len(bare)} 字，阈值 {SHORT_QUERY_CHARS}）",
                "cues": ["short"]}
    return {"is_followup": False, "reason": "无指代线索且长度足够", "cues": []}


# ============================================================================
# 三、追问改写
# ============================================================================
REWRITE_SYSTEM = """你是检索查询改写器。给你一段对话历史和用户的最新一句话，
请把最新这句话改写成一个**不依赖上下文也能看懂**的完整问题。

规则：
- 只补全指代（它、该药、上述……）和省略的主语，**不要回答问题**，不要加解释。
- **补全必须落到具体名称上**：把「它 / 它们 / 这些药 / 该疗法」换成历史里那个疾病、
  药物或技术的**名字**。改写完如果句子里还剩「这些」「该」「上述」，就是没改到位。
- 改写后的问题里**必须出现上一轮问题的主题词**（那个疾病 / 药物 / 技术的名字）——
  这句话会被直接拿去检索文献库，主题词丢了就什么也检索不到。
- 只有当这句话**不含任何指代词、主语已经写明**时，才算独立完整，可以原样输出。
- 保持原来的语言（中文问题输出中文）。
- 只输出改写后的那一句话，不要引号、不要前后缀、不要换行。"""


class FollowupRewriter:
    """把追问改写成可独立检索的问题。三级策略见模块 docstring。

    Args:
        generator: 任何有 `generate(prompt, system_prompt=..., temperature=..., max_tokens=...)`
                   的对象（阶段七 `LLMGenerator` 即是）。None 时 `llm` 模式自动降级为 `concat`。
        max_tokens / temperature: 改写调用的采样参数。温度 0——改写要的是确定性。
    """

    def __init__(self, generator: Any = None, max_tokens: int = 120,
                 temperature: float = 0.0, history_turns: int = DEFAULT_HISTORY_TURNS):
        self.generator = generator
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.history_turns = int(history_turns)
        #: 用量统计，供报告与健康检查引用
        self.stats = {"calls": 0, "rewritten": 0, "skipped": 0, "fallback": 0, "failed": 0}

    # ---------------- 拼接式 ----------------
    @staticmethod
    def concat(query: str, history: Sequence[Dict[str, Any]]) -> str:
        """上一轮用户问题 + 本轮问题。

        **只对检索有效**：BM25 与向量都吃得下这种拼接（多出来的词就是主题词），
        但模型看到的问题里指代仍未解决。所以 concat 只写进 `resolved_query`，
        提问给模型时用的仍是原句——这一点在响应体里如实标了出来。
        """
        cur = (query or "").strip()
        prev = ""
        for h in reversed(list(history)):
            if h.get("role") != "user":
                continue
            text = (h.get("content") or "").strip()
            # 跳过与当前问题相同的那条：调用方若把本轮问题也塞进了 history，
            # 不跳过就会拼出"它的不良反应呢？ 它的不良反应呢？"这种自我复读的检索式
            if text and text != cur:
                prev = text
                break
        if not prev:
            return query
        return f"{prev} {query}".strip()

    # ---------------- 模型改写 ----------------
    def _format_history(self, history: Sequence[Dict[str, Any]]) -> str:
        lines = []
        for h in list(history)[-self.history_turns * 2:]:
            who = "用户" if h.get("role") == "user" else "助手"
            c = (h.get("content") or "").strip().replace("\n", " ")
            if who == "助手" and len(c) > 300:
                c = c[:300] + "…"          # 助手的答案很长，改写只需要主题不需要全文
            lines.append(f"{who}：{c}")
        return "\n".join(lines)

    def _guard(self, out: str, query: str) -> Tuple[bool, str]:
        """模型很容易把"改写"做成"回答"。这里把明显跑偏的挡掉，返回 (是否可用, 原因)。"""
        s = (out or "").strip().strip('"“”\'`').strip()
        s = s.split("\n")[0].strip()                       # 只要第一行
        if not s:
            return False, "改写输出为空"
        if len(s) > max(len(query) * 4, len(query) + 120):
            return False, f"改写过长（{len(s)} 字），疑似在回答而不是改写"
        if s.startswith("#") or "## " in s or "[S" in s:
            return False, "改写输出里出现章节标题或出处编号，疑似在回答"
        return True, s

    def rewrite(self, query: str, history: Sequence[Dict[str, Any]],
                mode: str = "llm") -> Dict[str, Any]:
        """返回 {resolved, prompt_query, rewritten, mode_used, note, detect, seconds}。

        `resolved`      —— 拿去**检索**的问题
        `prompt_query`  —— 拿去**提问**的问题（llm 模式与 resolved 相同；concat 模式是原句）
        """
        t0 = time.time()
        base = {"resolved": query, "prompt_query": query, "rewritten": False,
                "mode_used": "none", "note": "", "detect": None, "seconds": 0.0}
        mode = (mode or "none").lower()
        if mode == "none" or not history:
            base["note"] = "未启用改写" if mode == "none" else "无历史可用"
            return base

        det = detect_followup(query, history)
        base["detect"] = det
        if not det["is_followup"]:
            self.stats["skipped"] += 1
            base["note"] = f"判为独立问题，跳过改写（{det['reason']}）"
            return base

        if mode == "concat" or self.generator is None:
            merged = self.concat(query, history)
            base.update(resolved=merged, prompt_query=query,
                        rewritten=merged != query, mode_used="concat",
                        note=("拼接上一轮问题（仅用于检索）" if self.generator is not None
                              else "没有可用生成器，llm 降级为 concat"),
                        seconds=round(time.time() - t0, 3))
            if self.generator is None and mode == "llm":
                self.stats["fallback"] += 1
            self.stats["rewritten"] += int(base["rewritten"])
            return base

        # ---- llm ----
        prompt = (f"对话历史：\n{self._format_history(history)}\n\n"
                  f"最新一句：{query}\n\n改写后的独立问题：")
        try:
            self.stats["calls"] += 1
            r = self.generator.generate(prompt, system_prompt=REWRITE_SYSTEM,
                                        temperature=self.temperature,
                                        max_tokens=self.max_tokens)
            ok, val = self._guard(r.get("text", ""), query)
        except Exception as e:                    # 改写失败绝不能让问答失败
            ok, val = False, f"改写调用失败：{type(e).__name__} {str(e)[:120]}"

        if not ok:
            merged = self.concat(query, history)
            self.stats["failed"] += 1
            self.stats["fallback"] += 1
            base.update(resolved=merged, prompt_query=query, rewritten=merged != query,
                        mode_used="concat", note=f"{val}；已降级为拼接",
                        seconds=round(time.time() - t0, 3))
            return base

        self.stats["rewritten"] += int(val != query)
        base.update(resolved=val, prompt_query=val, rewritten=val != query,
                    mode_used="llm", note="模型改写", seconds=round(time.time() - t0, 3))
        return base


# ============================================================================
# 四、指代检测器的标注样例（验证脚本与 CLI 共用同一份，避免两处各写一套）
# ============================================================================
#: (问题, 是否追问)。有历史为前提。
DETECT_SAMPLES: Tuple[Tuple[str, bool], ...] = (
    ("它的副作用有哪些？", True),
    ("该药物的推荐剂量是多少？", True),
    ("上述研究的样本量分别是多少？", True),
    ("这类疗法的禁忌症呢？", True),
    ("那么长期使用的安全性怎么样？", True),
    ("有效率是多少？", True),
    ("What about its adverse events?", True),
    ("这项试验的主要终点是什么？", True),
    ("CRISPR-Cas9 的脱靶效应如何检测？", False),
    ("酶替代治疗在法布里病中的疗效证据有哪些？", False),
    ("抗淀粉样蛋白单抗治疗阿尔茨海默病的三期试验结论是什么？", False),
    ("What is the evidence for pembrolizumab in non-small cell lung cancer?", False),
    ("多发性硬化的一线治疗药物有哪些类别？", False),
    # ↓ 已知会被误判的一类：短但独立。假阳性只浪费一次改写调用，如实留在样例里
    ("什么是随机对照试验？", False),
)


def detector_report(samples: Sequence[Tuple[str, bool]] = DETECT_SAMPLES) -> Dict[str, Any]:
    """在标注样例上跑检测器，**把两类错误分开报**（它们的代价不对称）。"""
    fake_hist = [{"role": "user", "content": "帕博利珠单抗在非小细胞肺癌中的疗效如何？"},
                 {"role": "assistant", "content": "## 核心答案\n……[S1]"}]
    fp, fn, rows = [], [], []
    for q, want in samples:
        got = detect_followup(q, fake_hist)
        rows.append({"query": q, "expect": want, "got": got["is_followup"],
                     "reason": got["reason"]})
        if got["is_followup"] and not want:
            fp.append(q)
        if not got["is_followup"] and want:
            fn.append(q)
    n = len(samples)
    return {"n": n, "correct": n - len(fp) - len(fn),
            "accuracy": round((n - len(fp) - len(fn)) / n, 4) if n else 0.0,
            "false_positive": fp, "false_negative": fn, "rows": rows}


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--detect", action="store_true", help="在标注样例上跑指代检测器")
    ap.add_argument("--list", action="store_true", help="列出会话")
    ap.add_argument("--show", default=None, help="打印某个会话的全部轮次")
    ap.add_argument("--purge-days", type=float, default=None)
    args = ap.parse_args()

    if args.detect:
        rep = detector_report()
        print("=" * 84)
        print("指代/追问检测器 —— 标注样例实测（假阳性=白花一次改写调用，假阴性=检索到噪声）")
        print("=" * 84)
        for r in rep["rows"]:
            mark = "PASS" if r["expect"] == r["got"] else ("假阳性" if r["got"] else "假阴性")
            print(f"  [{mark:^5}] 期望{'追问' if r['expect'] else '独立'} "
                  f"实得{'追问' if r['got'] else '独立'}  {r['query'][:40]:<42} {r['reason']}")
        print("-" * 84)
        print(f"  {rep['correct']}/{rep['n']} 正确（accuracy={rep['accuracy']}）｜"
              f"假阳性 {len(rep['false_positive'])} 条｜假阴性 {len(rep['false_negative'])} 条")
        return 0

    if not (args.list or args.show or args.purge_days):
        ap.print_help()
        return 0
    if not os.path.exists(args.db):
        print(f"还没有会话库：{args.db}（服务带 session_id 跑过一次后自动生成）")
        return 1

    store = SessionStore(args.db)
    if args.purge_days is not None:
        print(f"已清理 {store.purge_expired(args.purge_days)} 个过期会话")
    if args.list:
        rows, total = store.list_sessions(1, 50)
        print(f"共 {total} 个会话，显示前 {len(rows)} 个：")
        for r in rows:
            print(f"  {r['session_id']}  轮次 {r['turns']:>3}  {r['updated_at']}  {r['title'][:40]}")
    if args.show:
        s = store.get(args.show, with_history=50)
        if not s:
            print(f"会话不存在：{args.show}")
            return 1
        print(f"{s['session_id']}｜创建 {s['created_at']}｜{s['turns']} 条消息")
        for h in s["history"]:
            print(f"  [{h['turn_index']:>2}] {h['role']:<9} {h['content'][:120]}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
