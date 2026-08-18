"""
多轮问答.py — 带医学 system prompt + 全程日志的交互式医学问答。

阶段一产物，**不走 RAG**。用它代替 `ollama run`：每轮问答写入 logs/chat_log.jsonl，
方便复盘与调优。可调项在文件顶部「可调区」：SYSTEM_PROMPT / temperature /
num_ctx / THINK（医学问答建议温度 0.2~0.3）。

输入：交互式键入　输出：模型回答 + logs/chat_log.jsonl（UTF-8）。

用法::

    & $py scripts\多轮问答.py

对话中的命令：/bye 退出 ｜ /sys 查看系统提示 ｜ /clear 清空多轮上下文

⚠ 路径别写进普通字符串：`E:\\rag` 里的 `\\r` 会被当成回车转义，
  这一行原本就是这么被吃掉半截的。
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import sys
import json
import time
import datetime
from pathlib import Path
import ollama

# --- Windows 控制台中文安全 ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = "qwen3:8b"

# ============ 可调区（改这里来 tune） ============
SYSTEM_PROMPT = (
    "你是一名严谨的医学知识助手，面向有医学背景的用户。回答要求："
    "1) 用简洁专业的中文，先给结论再解释；"
    "2) 分点作答，避免空话；"
    "3) 涉及药物使用通用名；"
    "4) 证据不足或存在争议时明确指出，不要臆断；"
    "5) 不得编造参考文献或数据；"
    "6) 必要时附一句：本回答仅供专业参考，不构成临床诊疗建议。"
)
OPTIONS = {
    "temperature": 0.3,   # 医学问答偏确定性 -> 调低
    "num_ctx": 8192,      # 上下文窗口，支持多轮
}
THINK = False             # qwen3 思考模式：False = 直接给答案（不输出 <think>）
# ===============================================

LOG = Path(os.path.join(ROOT, "logs", "chat_log.jsonl"))
LOG.parent.mkdir(parents=True, exist_ok=True)


def log_turn(question, answer, meta):
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "question": question,
        "answer": answer,
        **meta,
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    print(f"医学问答 [{MODEL}]  温度={OPTIONS['temperature']} 思考模式={'开' if THINK else '关'}")
    print("命令: /bye 退出 | /sys 看系统提示 | /clear 清空上下文")
    print(f"对话日志: {LOG}\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/bye", "/exit", "/quit"):
            break
        if q == "/sys":
            print("【系统提示】" + SYSTEM_PROMPT + "\n")
            continue
        if q == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("（已清空多轮上下文）\n")
            continue

        messages.append({"role": "user", "content": q})
        print("助手 > ", end="", flush=True)
        t0 = time.time()
        parts, last = [], None
        try:
            for chunk in ollama.chat(model=MODEL, messages=messages,
                                     options=OPTIONS, think=THINK, stream=True):
                tok = chunk["message"]["content"]
                parts.append(tok)
                print(tok, end="", flush=True)
                last = chunk
        except Exception as e:
            print(f"\n[调用出错] {e!r}\n")
            messages.pop()
            continue
        print("\n")

        answer = "".join(parts).strip()
        messages.append({"role": "assistant", "content": answer})
        meta = {"seconds": round(time.time() - t0, 1)}
        if last and last.get("eval_count") and last.get("eval_duration"):
            meta["tok_per_s"] = round(last["eval_count"] / (last["eval_duration"] / 1e9), 1)
        log_turn(q, answer, meta)

    print("已退出，日志保存在", LOG)


if __name__ == "__main__":
    main()
