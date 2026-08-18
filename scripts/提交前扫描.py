#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提交前扫描：密钥泄漏 + AI 协作署名。

为什么单独写一个脚本，而不是每次临时拼 grep：
**临时正则连续误报了三次**——
  ① `claude` 匹配到文件名 `CLAUDE.md`
  ② `token\\s*=` 匹配到 `_TOKEN = re.compile(...)`（英文 token 的正则常量）
  ③ `AI` 大小写不敏感匹配到 `gm**ai**l.com`
每次都要人工判一遍「是不是又误报」。**久了就会有人直接跳过警报，那扫描就白做了**——
警报只有在「响了就一定有事」时才有价值。所以把判据固化下来，并逐条写清为什么这么写。

用法：
    $py = "E:\\rag\\conda\\envs\\medrag\\python.exe"
    & $py scripts\\提交前扫描.py              # 扫**已暂存**的改动（提交前用）
    & $py scripts\\提交前扫描.py --commit HEAD # 扫某条已有提交
退出码非 0 即有命中。
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT


import argparse
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# 一、密钥类 —— 只匹配**有形状的凭据**，不匹配「像是变量名」的东西
# ---------------------------------------------------------------------------
# 关键设计：一律要求「赋值符 + 引号 + 足够长的高熵值」。
# 光有 `token =` 不算——那是变量名；`token = "ghp_xxxxxxxx..."` 才算。
_ASSIGN = r"""(?:=|:)\s*['"]"""
SECRET_PATTERNS = [
    (r"(?i)\b(api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token|"
     r"password|passwd|client[_-]?secret)\b" + _ASSIGN + r"[^'\"]{8,}['\"]",
     "疑似硬编码凭据（键名 + 引号包住的长值）"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub personal access token"),
    (r"github_pat_[A-Za-z0-9_]{50,}", "GitHub fine-grained PAT"),
    (r"sk-[A-Za-z0-9]{32,}", "OpenAI 风格密钥"),
    (r"sk-ant-[A-Za-z0-9\-_]{20,}", "Anthropic 密钥"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----", "私钥文件内容"),
    (r"ssh-rsa\s+AAAA[0-9A-Za-z+/]{100,}", "SSH 公钥（长度足以确认是真钥匙）"),
    (r"(?i)\bmongodb(?:\+srv)?://[^\s:@]+:[^\s:@]+@", "含口令的 MongoDB 连接串"),
    (r"(?i)\bpostgres(?:ql)?://[^\s:@]+:[^\s:@]+@", "含口令的 Postgres 连接串"),
]

# ---------------------------------------------------------------------------
# 二、AI 协作署名 —— 只扫**提交信息与作者字段**，不扫代码内容
# ---------------------------------------------------------------------------
# 为什么不扫代码：代码里出现 "claude" 完全正常（本仓库就有 `CLAUDE.md`、
# `claude code 不用读这个文档.txt`），扫了必然误报。
# 真正要防的是**提交元数据里带上协作署名**，那只可能出现在 message / author / committer。
#
# ⚠ 所有词都加词边界或用足够长的字面量：
#   · 不能用裸 `AI`——它会匹配 gm**ai**l、m**ai**n、ret**ai**n
#   · 不能用裸 `assistant`——正常中文提交信息里不会有，但英文里可能是普通名词，
#     所以只在「与协作署名同现」的形状里匹配
#
# ⚠ 本仓库里有几个**文件名**天然含 "claude"，在提交信息里引用它们完全正常。
#   扫描前先把这些**已知良性字面量**整体抠掉，再跑下面的模式——
#   这是第四次误报的来源（提交信息写「见 CLAUDE.md 六之三」被判成 AI 署名）。
#   ⚠ 只抠**精确的文件名**，不抠裸词 "claude"：真出现无文件名修饰的 Claude 仍要报。
BENIGN_LITERALS = [
    "CLAUDE.md",
    "claude code 不用读这个文档.txt",
]

TRAILER_PATTERNS = [
    (r"(?im)^\s*co-authored-by:\s*.*(claude|anthropic|copilot|gpt|assistant)",
     "Co-Authored-By 里带 AI 署名"),
    (r"(?i)generated\s+with\s+\[?(claude|chatgpt|copilot|cursor)", "「Generated with X」"),
    (r"🤖", "机器人 emoji（常见于自动署名行）"),
    (r"(?i)\bclaude(\.ai|\s+code|\s+opus|\s+sonnet|\s+haiku)?\b", "提交信息里出现 Claude"),
    (r"(?i)\banthropic\b", "提交信息里出现 Anthropic"),
    (r"(?i)\bgithub\s+copilot\b", "提交信息里出现 Copilot"),
    (r"(?i)\bai[- ]generated\b|\bwritten by (an )?ai\b", "自陈 AI 生成"),
]


def sh(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def scan_secrets(diff_text, label):
    hits = []
    cur = "?"
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue          # 只看新增行；删掉的旧密钥不需要再报
        body = line[1:]
        for pat, why in SECRET_PATTERNS:
            if re.search(pat, body):
                hits.append((cur, why, body.strip()[:110]))
    print(f"\n■ 密钥扫描（{label}，只看新增行）")
    if not hits:
        print("  ✓ 无命中")
    for f, why, snip in hits:
        print(f"  ✗ {why}\n      {f}\n      {snip}")
    return hits


def scan_attribution(msg, an, ae, cn, ce):
    hits = []
    # 先抠掉已知良性字面量（项目自己的文件名），再判——见 BENIGN_LITERALS 的注释
    probe = msg
    n_benign = 0
    for lit in BENIGN_LITERALS:
        n_benign += probe.lower().count(lit.lower())
        probe = re.sub(re.escape(lit), " ", probe, flags=re.I)
    for pat, why in TRAILER_PATTERNS:
        m = re.search(pat, probe)
        if m:
            hits.append(("提交信息", why, m.group()[:80]))
    for name, val in (("author", f"{an} <{ae}>"), ("committer", f"{cn} <{ce}>")):
        # 作者字段只查这几个确定性字面量，避免 gmail 里的 ai 之类
        for w in ("claude", "anthropic", "copilot", "noreply@anthropic"):
            if w in val.lower():
                hits.append((name, f"作者字段含 {w}", val))
    print("\n■ AI 协作署名扫描（只扫提交信息与作者字段，不扫代码内容）")
    print(f"  author    : {an} <{ae}>")
    print(f"  committer : {cn} <{ce}>")
    if n_benign:
        print(f"  （已忽略 {n_benign} 处项目自有文件名：{'、'.join(BENIGN_LITERALS)}）")
    if not hits:
        print("  ✓ 无命中")
    for where, why, snip in hits:
        print(f"  ✗ [{where}] {why}\n      {snip}")
    return hits


def main():
    ap = argparse.ArgumentParser(description="提交前扫描：密钥 + AI 署名")
    ap.add_argument("--commit", default="", help="扫某条已有提交；留空则扫已暂存的改动")
    args = ap.parse_args()

    print("=" * 92)
    print("提交前扫描")
    print("=" * 92)

    if args.commit:
        diff = sh(["git", "show", "--format=", "--unified=0", args.commit])
        meta = sh(["git", "log", "-1", "--format=%B%x00%an%x00%ae%x00%cn%x00%ce",
                   args.commit]).split("\x00")
        label = f"提交 {args.commit}"
    else:
        diff = sh(["git", "diff", "--cached", "--unified=0"])
        meta = ["", sh(["git", "config", "user.name"]).strip(),
                sh(["git", "config", "user.email"]).strip(),
                sh(["git", "config", "user.name"]).strip(),
                sh(["git", "config", "user.email"]).strip()]
        label = "已暂存的改动"
        print("  （提交信息尚不存在，只扫作者配置与暂存内容）")

    msg, an, ae, cn, ce = (meta + [""] * 5)[:5]
    s = scan_secrets(diff, label)
    a = scan_attribution(msg, an.strip(), ae.strip(), cn.strip(), ce.strip())

    print("\n" + "=" * 92)
    total = len(s) + len(a)
    print(f"结论：{'✓ 通过，可以推送' if total == 0 else f'✗ {total} 处命中，**先处理再推**'}")
    print("=" * 92)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
