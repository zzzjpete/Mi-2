# -*- coding: utf-8 -*-
"""扫描器自测：良性引用不报，真署名照报。

⚠ 加了「已知良性字面量」白名单之后，必须验它**没有把真警报一起削掉**——
否则就是把哑弹换成了瞎子。

输入：内置用例表（提交信息文本 + 期望判定）　输出：逐条 PASS/FAIL 与退出码。

用法::

    & $py scripts\提交前扫描_自测.py
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import sys, importlib.util, io, contextlib
sys.stdout.reconfigure(encoding="utf-8")
s = importlib.util.spec_from_file_location("sc", os.path.join(ROOT, "scripts", "提交前扫描.py"))
m = importlib.util.module_from_spec(s); sys.modules["sc"] = s and m; s.loader.exec_module(m)

U, E = "Jinxi Zhang", "jinxizhang236@gmail.com"

CASES = [
    # (提交信息, 作者, 邮箱, 期望是否命中, 说明)
    ("标注评测集降级；规矩见 CLAUDE.md 六之三", U, E, False, "引用项目自有文件名 CLAUDE.md"),
    ("根目录 claude code 不用读这个文档.txt 加进 .gitignore", U, E, False, "引用那份笔记的文件名"),
    ("修复检索漂移\n\nCo-Authored-By: Claude <noreply@anthropic.com>", U, E, True, "真 trailer"),
    ("重构生成链\n\n🤖 Generated with [Claude Code]", U, E, True, "真 Generated with"),
    ("改进重排", "Claude", "noreply@anthropic.com", True, "作者字段是 AI"),
    ("接入 Anthropic 官方 SDK 做对照实验", U, E, True, "正文出现 Anthropic（宁可报）"),
    ("普通提交：修 gmail 通知与 main 分支保护", U, E, False, "gmail/main 里的 ai 不该触发"),
    ("加 _TOKEN = re.compile 常量", U, E, False, "token= 变量名不该触发"),
]

ok = True
for msg, an, ae, want, why in CASES:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hits = m.scan_attribution(msg, an, ae, an, ae)
    got = len(hits) > 0
    good = got == want
    ok = ok and good
    print(f"  {'✓' if good else '✗'} 期望{'报' if want else '不报'} 实际{'报' if got else '不报'}"
          f"  | {why}")
    if not good:
        print(f"      提交信息: {msg[:70]!r}")
        for h in hits:
            print(f"      命中: {h}")

# 密钥侧也验一遍：变量名不报、真凭据要报
#
# ⚠ 假密钥样本**必须运行时拼接，不能以字面量写进文件**——否则这个自测文件自己就成了
#   "含密钥形状字符串的已提交文件"，被自己的扫描器抓到（实测抓到过）。
#   这不只是为了让扫描过关：往仓库里塞长得像凭据的东西本身就是不良卫生，
#   以后任何第三方扫描器（GitHub secret scanning 等）也会对它报警。
_SK = "sk-" + "a" * 32                       # OpenAI 形状
_GHP = "ghp_" + "b" * 36                     # GitHub PAT 形状
SEC = [
    ("+_TOKEN = re.compile(r'[A-Za-z]+')", False, "token= 变量名"),
    ("+api_key = os.environ['X']", False, "从环境读，没有硬编码值"),
    (f"+api_key = '{_SK}'", True, "硬编码长值"),
    (f"+{_GHP}", True, "GitHub PAT"),
    ("+邮箱 jinxizhang236@gmail.com", False, "gmail 不是凭据"),
]
print()
for line, want, why in SEC:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hits = m.scan_secrets("+++ b/x.py\n" + line, "自测")
    got = len(hits) > 0
    good = got == want
    ok = ok and good
    print(f"  {'✓' if good else '✗'} 期望{'报' if want else '不报'} 实际{'报' if got else '不报'}  | {why}")

print("\n" + ("全部通过" if ok else "**有失败项**"))
sys.exit(0 if ok else 1)
