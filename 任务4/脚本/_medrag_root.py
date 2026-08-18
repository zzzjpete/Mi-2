# -*- coding: utf-8 -*-
"""项目根解析 —— 让同一份代码在 Windows 主机和 Mac 上都能跑。

**为什么不是把 `E:\\rag` 换成 Mac 路径**：换了就等于把代码库分叉。两台机器上同一行
永远不同，出差三天回来 git 每次都在这一行上冲突，而冲突的正是路径行本身——
最没价值的冲突。这里让路径自己解析出来，两边源码逐字相同，git 只看见真正的改动。

解析顺序（先到先得）：
  1. 环境变量 ``MEDRAG_ROOT`` —— 显式覆盖。交付包 `任务N/脚本/` 单独发出去时用这个
  2. 从 ``__file__`` 上溯找标志目录（同时含 ``scripts/`` 与 ``requirements.txt``）
  3. 兜底 ``E:\\rag`` —— 且**仅当它真实存在**，保证本机历史行为逐字不变

⚠ **解析不出来就抛异常，不返回一个不存在的路径。** 这是本项目反复付过学费的地方：
   一个指向不存在目录的路径不会报错——Chroma 会当场建一个空库、BM25 目录缺失只是
   把那一路静默关掉——最终表现成「库里没有证据」，和真的没有证据在响应体里长得
   一模一样。宁可在 import 阶段响亮地失败。

⚠ 本模块**只准 import 标准库**：它被 70 个脚本在 import 阶段加载，任何第三方依赖
   都会变成全项目的依赖。

用法::

    from _medrag_root import ROOT
    BM25_DIR = os.path.join(ROOT, "data", "bm25_index_4m")

    from _medrag_root import ROOT_PATH          # 要 pathlib.Path 的场合

    from _medrag_root import evidence_path     # 验证夹具：本机产物优先，回退仓库冻结副本
    SNAPSHOT = evidence_path("检索快照_live.json")
"""
import os
from pathlib import Path

__all__ = ["ROOT", "ROOT_PATH", "resolve_root", "evidence_path"]

#: 兜底路径：这台 Windows 主机的历史位置。仅当它真实存在时才会被采用。
_LEGACY_ROOT = r"E:\rag"

#: 上溯层数上限。scripts/ 在根下一层、任务N/脚本/ 在根下两层，6 层是宽裕的余量。
_MAX_WALK_UP = 6


def _is_project_root(d):
    """标志：同时含 scripts/ 目录与 requirements.txt。

    单用 scripts/ 不够——任务N 包里也可能出现同名目录；配上 requirements.txt
    才能把项目根和交付包目录区分开。
    """
    return (os.path.isdir(os.path.join(d, "scripts"))
            and os.path.isfile(os.path.join(d, "requirements.txt")))


def resolve_root(start=None):
    """返回项目根的绝对路径；解析不出来抛 RuntimeError。

    `start` 只在自测时传，正常调用不用管。
    """
    env = os.environ.get("MEDRAG_ROOT")
    if env:
        env = os.path.abspath(os.path.expanduser(env))
        if os.path.isdir(env):
            return env
        raise RuntimeError(
            f"环境变量 MEDRAG_ROOT 指向的目录不存在：{env}\n"
            f"  要么把它改对，要么删掉这个变量让它自动解析。")

    d = start if start else os.path.dirname(os.path.abspath(__file__))
    for _ in range(_MAX_WALK_UP):
        if _is_project_root(d):
            return d
        parent = os.path.dirname(d)
        if parent == d:          # 到根了
            break
        d = parent

    if os.path.isdir(_LEGACY_ROOT):
        return _LEGACY_ROOT

    raise RuntimeError(
        "解析不出项目根目录。\n"
        f"  从 {os.path.dirname(os.path.abspath(__file__))} 向上找了 {_MAX_WALK_UP} 层，"
        "没找到同时含 scripts/ 与 requirements.txt 的目录；\n"
        f"  兜底路径 {_LEGACY_ROOT} 也不存在（说明不在原来那台 Windows 上）。\n"
        "  解法：设环境变量 MEDRAG_ROOT 指向项目根，例如\n"
        "    macOS/Linux:  export MEDRAG_ROOT=~/rag\n"
        "    Windows:      $env:MEDRAG_ROOT = 'E:\\rag'")


#: 项目根（str）。绝大多数调用点用这个。
ROOT = resolve_root()

#: 项目根（pathlib.Path）。原先写 `ROOT = Path(r"E:\rag")` 的那些脚本用这个。
ROOT_PATH = Path(ROOT)


#: 仓库自带的冻结验证夹具目录。`report_data/` 整目录不进 git，clone 下来只有这一份。
_FIXTURE_DIR = ("reports", "fixtures")


def evidence_path(name):
    """验证夹具的解析：优先本机跑出来的最新产物，回退仓库自带的冻结副本。

    为什么需要两处：`report_data/` 整目录不进 git（中间产物 + 个人文件混在一起），
    而几个验证脚本坚持用**真实产物**做断言——真检索快照、真答案，而不是手编的小例子。
    于是 clone 下来的仓库里这些夹具一份都没有，实测表现是：
    约束验证 99 → 95、评估验证 66 → 65（两者都如实写明「跳过，未静默当作通过」），
    而服务验证**直接崩在解引用上**。三种表现里没有一种适合交到别人手里。

    解析顺序：

      1. ``report_data/<name>`` —— 本机跑过就用本机最新那份
      2. ``reports/fixtures/<name>`` —— 仓库里带着的冻结副本，clone 即可复现
      3. 都没有 → 仍返回第 1 条那个路径，让调用方原有的「缺文件就跳过」逻辑照常生效

    ⚠ 第 3 条**不抛异常**，与 `resolve_root()` 的处理刻意不同：项目根解析不出来会让
    整条链静默跑偏（空库、少一路召回），必须响亮失败；而夹具缺失是可降级的，
    调用方本来就有跳过分支，抛异常反而把一个可用的降级路径堵死。
    """
    fresh = os.path.join(ROOT, "report_data", name)
    if os.path.exists(fresh):
        return fresh
    frozen = os.path.join(ROOT, *_FIXTURE_DIR, name)
    if os.path.exists(frozen):
        return frozen
    return fresh


if __name__ == "__main__":
    print(f"ROOT       = {ROOT}")
    print(f"MEDRAG_ROOT= {os.environ.get('MEDRAG_ROOT') or '(未设)'}")
    print(f"标志判定    = scripts/ {'有' if os.path.isdir(os.path.join(ROOT, 'scripts')) else '无'}"
          f" / requirements.txt {'有' if os.path.isfile(os.path.join(ROOT, 'requirements.txt')) else '无'}")
    for sub in ("scripts", "data", "report_data", "hf-cache"):
        p = os.path.join(ROOT, sub)
        print(f"  {'存在' if os.path.exists(p) else '缺失'}  {p}")
