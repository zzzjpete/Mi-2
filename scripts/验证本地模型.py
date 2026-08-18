"""
验证本地模型.py — 验证本地 qwen3:8b 在 Ollama 上能否正常推理。

阶段一的第一道闸门：环境装完先跑它，跑通了才谈后面的检索与生成。
同时验证两条调用路径：
  1) ollama 原生 python 客户端
  2) langchain_ollama (后续 RAG 框架要用的路径)
用法::

    & $py scripts\验证本地模型.py
"""
import time
import ollama

MODEL = "qwen3:8b"
QUESTION = "用一句话解释2型糖尿病中的胰岛素抵抗是什么。"


def test_native():
    print(f"=== [1/2] ollama 原生客户端测试: {MODEL} ===")
    t0 = time.time()
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION}],
        options={"temperature": 0.2},
    )
    dt = time.time() - t0
    print("问题:", QUESTION)
    print("回答:", resp["message"]["content"].strip())
    print(f"总耗时: {dt:.1f}s")
    if resp.get("eval_count") and resp.get("eval_duration"):
        tps = resp["eval_count"] / (resp["eval_duration"] / 1e9)
        print(f"生成速度: {tps:.1f} tokens/s  (eval_count={resp['eval_count']})")
    print()


def test_langchain():
    print("=== [2/2] langchain_ollama 路径测试 ===")
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=MODEL, temperature=0.2)
        out = llm.invoke("一句话说明什么是高血压。")
        print("LangChain 回答:", out.content.strip()[:300])
        print("LangChain→Ollama 链路: OK")
    except Exception as e:
        print("LangChain 测试失败:", repr(e))
    print()


if __name__ == "__main__":
    test_native()
    test_langchain()
    print("验证完成。若上面两段都有合理回答，说明本地 LLM 推理链路打通。")
