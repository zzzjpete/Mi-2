# -*- coding: utf-8 -*-
"""把 scripts\\ 下硬编码的 E:\\rag 机械改写成从 `_medrag_root.ROOT` 派生。

**目的不是"改成 Mac 路径"，是"两台机器上源码逐字相同"**——见 `_medrag_root.py`
开头那段。改成 Mac 路径等于把代码库分叉，出差回来每个碰过的文件都在路径行上冲突。

改写规则：
  1. `ROOT = r"E:\\rag"`        → 删掉，改由引导块 `from _medrag_root import ROOT` 提供
  2. `ROOT = Path(r"E:\\rag")`  → 同上，导入 `ROOT_PATH as ROOT`
     （⚠ 不能就地替换成 `Path(ROOT)`，那是自引用）
  3. 其余字符串字面量 `r"E:\\rag\\a\\b"` → `os.path.join(ROOT, "a", "b")`，裸根 → `ROOT`
  4. 引导块插在**模块 docstring 之后、其它一切之前**
     （⚠ 必须早于 `os.environ["HF_HOME"] = ...` 那条铁律行，否则 HF_HOME 拿不到 ROOT；
       引导块只 import 标准库，不会破坏「pyarrow 必须早于 torch」那条铁律）

**跳过的**：注释、三引号字符串（docstring 与用法示例）、f-string。
前两类共 47 处，都只是文字说明，改了纯属噪声；f-string 无法 literal_eval，
会被单独列出来让人处理，**不静默跳过**。

用法::

    python 迁移_路径可移植.py --dry-run      # 只报告要改什么，不写盘
    python 迁移_路径可移植.py --apply        # 真改（改前每个文件存 .bak）
    python 迁移_路径可移植.py --restore      # 从 .bak 全部还原
"""
import argparse
import ast
import io
import os
import re
import shutil
import sys
import tokenize

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _medrag_root import ROOT                                    # noqa: E402

SRC_DIR = os.path.join(ROOT, "scripts")

#: 本模块自己和根解析模块不能被改写
SELF_EXCLUDE = {"_medrag_root.py", "迁移_路径可移植.py"}

LIT_RE = re.compile(r'(?i)E:[\\/]+rag')

BOOTSTRAP = """# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import {names}
"""


# ---------------------------------------------------------------------------
# 读写：⚠ 仓库里部分脚本带 UTF-8 BOM（golden_跑测.py 就是）。
# Python 自己的导入机制认得 BOM，但 ast.parse / compile 会报
# "invalid non-printable character U+FEFF"。所以一律按 utf-8-sig 读掉 BOM，
# 写回时再按原样补上——不能顺手把别人的 BOM 抹掉，那会变成一次无关的全文件改动。
# ---------------------------------------------------------------------------
def read_source(path):
    with open(path, "rb") as f:
        raw = f.read()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    return raw.decode("utf-8-sig"), had_bom


def write_source(path, text, had_bom):
    with open(path, "w", encoding="utf-8", newline="") as f:
        if had_bom:
            f.write(chr(0xFEFF))          # 用转义写，别在源码里放一个看不见的字符
        f.write(text)


# ---------------------------------------------------------------------------
# 定位
# ---------------------------------------------------------------------------
def find_docstring_end_line(src):
    """返回模块 docstring 之后的插入行号（0-based）；没有 docstring 就返回首个非
    注释/非编码声明行。引导块插在这里。"""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        return tree.body[0].end_lineno          # 1-based 末行 → 作为 0-based 插入位
    # 没有 docstring：插在编码声明/shebang 之后
    lines = src.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("#!") or "coding" in lines[i][:40]
                              or lines[i].strip() == ""):
        i += 1
    return i


def collect_targets(path):
    """按路径读文件后交给 `collect_targets_from_text`。"""
    src, _ = read_source(path)
    return collect_targets_from_text(src)


def collect_targets_from_text(src):
    """用 tokenize 找出所有该改的字符串字面量。

    ⚠ 必须能对**任意源码文本**调用，不能只吃文件路径——改写后要拿它复查
    「还剩没剩可执行的 E:\\rag」，那段文本还没落盘。

    返回 (targets, skipped, root_assign_lines)
      targets: [(row, col_start, col_end, 原文, 新文)]  row/col 均 1-based/0-based 同 tokenize
      skipped: [(row, 原文, 原因)]
    """
    targets, skipped = [], []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        return None, [(0, "", f"无法 tokenize: {e}")], []

    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue                                   # 注释：跳过（纯文字说明）
        if tok.type != tokenize.STRING:
            continue
        text = tok.string
        if not LIT_RE.search(text):
            continue

        prefix = text[:len(text) - len(text.lstrip("rRbBuUfF"))]
        body_q = text[len(prefix):]
        if body_q.startswith('"""') or body_q.startswith("'''"):
            skipped.append((tok.start[0], text.replace("\n", "⏎")[:60], "三引号（docstring/用法示例）"))
            continue
        if "f" in prefix.lower():
            skipped.append((tok.start[0], text, "f-string，无法 literal_eval，需人工处理"))
            continue
        if tok.start[0] != tok.end[0]:
            skipped.append((tok.start[0], text[:60], "跨行字符串"))
            continue

        try:
            value = ast.literal_eval(text)
        except Exception as e:
            skipped.append((tok.start[0], text, f"literal_eval 失败: {e}"))
            continue
        if not isinstance(value, str):
            skipped.append((tok.start[0], text, "非 str 字面量"))
            continue

        m = LIT_RE.match(value)
        if not m:
            # E:\rag 出现在字符串中间而不是开头（例如一句说明文字），不改
            skipped.append((tok.start[0], text, "E:\\rag 不在字符串开头，疑似说明文字"))
            continue

        rest = value[m.end():].strip("\\/")
        parts = [p for p in re.split(r"[\\/]+", rest) if p]
        if parts:
            joined = ", ".join('"%s"' % p for p in parts)
            new = f"os.path.join(ROOT, {joined})"
        else:
            new = "ROOT"
        targets.append((tok.start[0], tok.start[1], tok.end[1], text, new))

    return targets, skipped, None


ROOT_ASSIGN_RE = re.compile(
    r'^(?P<indent>\s*)ROOT\s*=\s*(?P<wrap>Path\(\s*)?r?["\']E:[\\/]+rag["\']\s*\)?\s*$')


def rewrite_file(path, dry_run=True):
    """返回 (改动数, 说明列表, (新内容, had_bom) or None)"""
    src, had_bom = read_source(path)
    if not LIT_RE.search(src):
        return 0, [], None

    lines = src.splitlines(keepends=True)
    notes = []

    # ── 1) ROOT = ... 赋值行：整行删除，改由引导块提供 ──────────────────────
    wants_path = False
    root_assign_rows = []
    for i, line in enumerate(lines):
        m = ROOT_ASSIGN_RE.match(line.rstrip("\n"))
        if m:
            root_assign_rows.append(i)
            if m.group("wrap"):
                wants_path = True
            notes.append(f"  L{i+1}  删除 ROOT 赋值：{line.strip()}"
                         f"{'（原为 Path(...)，改导入 ROOT_PATH）' if m.group('wrap') else ''}")

    # ── 2) 其余字面量 ────────────────────────────────────────────────────────
    targets, skipped, _ = collect_targets(path)
    if targets is None:
        return 0, [f"  ⚠ {skipped[0][2]}"], None

    targets = [t for t in targets if (t[0] - 1) not in root_assign_rows]
    for row, c0, c1, old, new in sorted(targets, key=lambda t: (-t[0], -t[1])):
        line = lines[row - 1]
        lines[row - 1] = line[:c0] + new + line[c1:]
        notes.append(f"  L{row}  {old}  →  {new}")

    for row, txt, why in skipped:
        notes.append(f"  L{row}  跳过（{why}）：{txt[:70]}")

    # ── 3) 删除 ROOT 赋值行（倒序，避免行号漂移）────────────────────────────
    for i in sorted(root_assign_rows, reverse=True):
        del lines[i]

    # ── 4) 插引导块 ─────────────────────────────────────────────────────────
    body = "".join(lines)
    names = "ROOT_PATH as ROOT" if wants_path else "ROOT"
    ins = find_docstring_end_line(body)
    if ins is None:
        return 0, ["  ⚠ 语法解析失败，跳过该文件"], None
    lines = body.splitlines(keepends=True)
    block = BOOTSTRAP.format(names=names)
    lines.insert(ins, "\n" + block + "\n")
    notes.append(f"  L{ins+1}  插入引导块（from _medrag_root import {names}）")

    new_src = "".join(lines)

    # ── 5) 自检：改完必须还能编译，且不再有可执行的 E:\rag ──────────────────
    try:
        compile(new_src, path, "exec")
    except SyntaxError as e:
        return 0, [f"  ❌ 改写后语法错误，已放弃该文件：{e}"], None

    # 改完不许还剩可执行的 E:\rag —— 只允许留在注释与三引号里
    leftover, _, _ = collect_targets_from_text(new_src)
    if leftover:
        rows = ", ".join(f"L{r}" for r, *_ in leftover)
        return 0, [f"  ❌ 改写后仍有可执行的 E:\\rag（{rows}），已放弃该文件"], None

    return len(targets) + len(root_assign_rows), notes, (new_src, had_bom)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="只报告，不写盘")
    g.add_argument("--apply", action="store_true", help="真改，改前每个文件存 .bak")
    g.add_argument("--restore", action="store_true", help="从 .bak 还原全部")
    ap.add_argument("--only", help="只处理某个文件名，调试用")
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(SRC_DIR)
                   if n.endswith(".py") and n not in SELF_EXCLUDE)
    if args.only:
        names = [n for n in names if n == args.only]

    if args.restore:
        n = 0
        for name in names:
            bak = os.path.join(SRC_DIR, name + ".bak")
            if os.path.exists(bak):
                shutil.copy2(bak, os.path.join(SRC_DIR, name))   # 按字节还原
                os.remove(bak)
                n += 1
        print(f"已从 .bak 还原 {n} 个文件")
        return 0

    total_files, total_changes, failed = 0, 0, []
    for name in names:
        path = os.path.join(SRC_DIR, name)
        cnt, notes, new_src = rewrite_file(path, dry_run=args.dry_run)
        has_problem = any(n.strip().startswith(("❌", "⚠")) for n in notes)
        if cnt == 0 and not has_problem:
            continue
        print(f"\n{name}  —— {cnt} 处")
        for nline in notes:
            print(nline)
        if has_problem:
            failed.append(name)
            continue
        total_files += 1
        total_changes += cnt
        if args.apply and new_src is not None:
            text, had_bom = new_src
            # .bak 按字节复制，不走文本层——BOM、行尾、任何编码细节都原样保留
            shutil.copy2(path, path + ".bak")
            write_source(path, text, had_bom)

    print("\n" + "=" * 74)
    mode = "预演（未写盘）" if args.dry_run else "已应用（原文件存为 .bak）"
    print(f"{mode}：{total_files} 个文件 / {total_changes} 处改动")
    if failed:
        print(f"❌ {len(failed)} 个文件有问题需人工处理：{', '.join(failed)}")
        return 1
    if total_changes == 0:
        print("❌ 一处都没改到——要么已经改过，要么匹配规则失效了。这不是成功。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
