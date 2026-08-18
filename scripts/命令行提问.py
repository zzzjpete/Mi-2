"""
命令行提问.py — 命令行直接问本地 qwen3:8b 一个问题（中文安全，走 UTF-8）。

阶段一产物，**不走 RAG**：直接打 Ollama，用来快速验模型可用、或随手问一句。
输入：命令行参数拼成的问题　输出：模型回答打到 stdout。

用法::

    & $py scripts\命令行提问.py 他汀类药物的主要副作用是什么

⚠ 路径别写进普通字符串：`E:\\rag` 里的 `\\r` 会被当成回车转义。
  这一行原本就是这么被吃掉半截的——打印出来只剩 `python E:`，而源码看着完全正常。
"""
import sys
import ollama

MODEL = "qwen3:8b"

def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "你好，请做个自我介绍。"
    resp = ollama.generate(model=MODEL, prompt=question, think=False)
    print(resp["response"].strip())

if __name__ == "__main__":
    main()
