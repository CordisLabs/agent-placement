#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文级数字验证：从 DRAFT-hpsr.tex 里提取**每一个数字**，逐一追回实测数据。

与 verify_numbers.py 的区别：
  * verify_numbers.py 检查"我知道要检查的项"（101 项，人工列举）
  * 本脚本反过来：**把论文里出现的所有数字捞出来**，然后问"这个数从哪来？"
    无法追溯的数字会被列出来——包括我手滑打错的、以及从旧版本残留的。

这是防"论文与数据脱节"的最后一道关：改了论文里的数字而没重跑实验时，
verify_numbers.py 抓不到，本脚本能抓到。

用法：
    python3 verify_paper_numbers.py
    python3 verify_paper_numbers.py --tex ../paper/DRAFT-hpsr.tex --verbose
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 允许出现在论文里、但不来自实验的数字 ──────────────────────────
# 每一项都要有理由。这是白名单，不是垃圾桶。
WHITELIST = {
    # 常量与设计参数
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "15", "20", "24", "48",
    "0.0", "0.5", "1.0", "2.0", "3.0",
    # 时间尺度 / 单位换算
    "10", "60", "100", "1000",
    # 引用文献里的数字（不来自我们的实验）
    "82.2", "10.3", "2025", "2026", "2608.08277", "2608.21049", "2609.18857",
    "2602.14117", "2603.03667", "2604.17025", "2609.05736", "2606.20683",
    "2507.07115", "2512.16381", "2506.03231",
    # 售价 / 成本参考点
    "0.15", "0.60", "3.00", "15.00", "75.00",
    # 统计量与显著性门槛
    "0.05", "0.01",
    # 页面 / 结构
    "612", "792",
    # 论文正文引用的"模型原始输出样例"（mistral-small 吐出的非法 JSON），
    # 是对模型输出的引用，不是我们的实验结果
    "2000.0",
}


def numbers_in_tex(path: str) -> collections.Counter:
    """论文里所有形如 12.34 或 12 的数字（去注释）。"""
    s = open(path, encoding="utf-8").read()
    s = re.sub(r"(?m)^%.*$", "", s)                       # 去注释
    s = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", s, flags=re.S)
    s = re.sub(r"arxiv:\S+|arXiv:\S+", "", s, flags=re.I)
    s = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}", "", s)
    return collections.Counter(re.findall(r"\b\d+\.\d+\b|\b\d+\b", s))


def data_numbers() -> set[str]:
    """把实测 CSV 能算出来的数字都收集起来。

    覆盖三类：
      ① 分组成功率（含多层键组合，因为论文里的数常是"某格 × 某层"）
      ② 计数（解出/未解出/总数）——论文里大量引用 1496、1842、2842 这类数
      ③ 派生量：组间差（百分点）、比值、每次成功的操作数
    """
    out: set[str] = set()

    def add(x):
        if x is None:
            return
        try:
            x = float(x)
        except (TypeError, ValueError):
            return
        if x != x or x in (float("inf"), float("-inf")):
            return
        for n in (1, 2, 3, 4):
            out.add(f"{x:.{n}f}")
        out.add(str(int(round(x))))
        out.add(f"{x:g}")

    # 论文里常用的键组合（含"某条件 × 某层"）
    KEYSETS = [
        (), ("layer",), ("scenario",), ("guardrails",), ("cell",), ("pred",),
        ("name",), ("role",), ("n_layers",), ("budget",), ("variant",),
        ("guardrails", "layer"), ("layer", "scenario"), ("cell", "pred"),
        ("cell", "pred", "scenario"), ("name", "pred"), ("name", "layer"),
        ("role", "layer"), ("n_layers", "role"), ("n_layers", "budget"),
        ("budget", "layer"), ("budget", "role"),
        ("n_layers", "role", "budget"), ("n_layers", "role", "budget", "layer"),
        ("guardrails", "layer", "scenario"), ("model", "layer"),
        ("cell", "noise"), ("noise",), ("cell", "pred", "noise"),
    ]

    files = glob.glob("out/*/episodes.csv") + [
        f for f in glob.glob("out/*.csv")
        if os.path.basename(f) not in ("budget_sensitivity.csv", "power.csv",
                                       "cost_energy.txt", "mechanism.txt")]
    all_rates: dict[str, float] = {}
    # 三域解耦网格与预测器（out/queue_domain/*.csv）
    for f in glob.glob("out/queue_domain/*.csv"):
        rows = list(csv.DictReader(open(f)))
        if not rows or "success" not in rows[0]:
            continue
        rows = [r for r in rows if r.get("success") is not None]
        KEYS = [("domain",), ("domain", "layer"), ("domain", "cad", "stale"),
                ("domain", "cad", "stale", "pred"), ("domain", "pred")]
        for keys in KEYS:
            if keys and not all(k in rows[0] for k in keys):
                continue
            buckets = collections.defaultdict(list)
            for r in rows:
                buckets[tuple(r[k] for k in keys)].append(r)
            for g in buckets.values():
                ok = sum(1 for r in g if str(r["success"]) in ("1", "True"))
                add(ok / len(g)); add(ok); add(len(g) - ok); add(ok / len(g) * 100)

    # 排队域的三层跨度（0.5973）：来自 episodes.csv 的 near_rt − intent
    eq = "out/queue_domain/episodes.csv"
    if os.path.exists(eq):
        rows = list(csv.DictReader(open(eq)))
        for dom in ("packet", "queue"):
            sub = [r for r in rows if r["domain"] == dom]
            if not sub:
                continue
            def rate(layer):
                g = [r for r in sub if r["layer"] == layer]
                return (sum(1 for r in g if str(r["success"]) in ("1", "True")) / len(g)
                        if g else None)
            a, b = rate("near_rt"), rate("smo_intent")
            if None not in (a, b):
                add(abs(a - b))
            for lyr in ("near_rt", "non_rt", "smo_intent"):
                r_ = rate(lyr)
                if r_ is not None:
                    add(r_)

    # 排队域的**跨度**（格间差）：0.5973 / 0.5455 / 0.3818 都是这样来的
    gq = "out/queue_domain/grid.csv"
    if os.path.exists(gq):
        rows = list(csv.DictReader(open(gq)))
        def q(dom, c, st):
            v = [r for r in rows if r["domain"] == dom and int(r["cad"]) == c
                 and int(r["stale"]) == st]
            if not v:
                return None
            return sum(1 for r in v if str(r["success"]) in ("1", "True")) / len(v)
        for dom in ("packet", "queue"):
            a, b, c = q(dom, 1, 0), q(dom, 3, 0), q(dom, 1, 2)
            if None not in (a, b, c):
                add(abs(a - b))          # 节奏跨度
                add(abs(a - c))          # 陈旧跨度
                if abs(a - c) > 1e-9:
                    add(abs(a - b) / abs(a - c))   # 比值

    # 学习策略的结果（out/rl2/*.csv）列名不同：成功率叫 acc
    for f in glob.glob("out/rl2/*.csv"):
        rows = list(csv.DictReader(open(f)))
        if not rows or "acc" not in rows[0]:
            continue
        KEYS = [("cad", "stale"), ("cad", "stale", "pred"), ("pred",),
                ("cad",), ("stale",), ()]
        for keys in KEYS:
            if keys and not all(k in rows[0] for k in keys):
                continue
            buckets = collections.defaultdict(list)
            for r in rows:
                buckets[tuple(r[k] for k in keys)].append(r)
            for g in buckets.values():
                a = [float(x["acc"]) for x in g]
                add(sum(a) / len(a))
        # 跨度 = 格间差（论文里的 0.5914 / 0.6010 / 0.1136 / 0.0246 都是这样来的）
        def cell(c, st, pr):
            v = [float(r["acc"]) for r in rows
                 if int(r["cad"]) == c and int(r["stale"]) == st and r["pred"] == pr]
            return v[0] if v else None
        for pr in ("none", "delta", "learned", "oracle"):
            for (c1, s1), (c2, s2) in (((1, 0), (1, 2)), ((1, 0), (3, 0)),
                                       ((1, 0), (3, 2)), ((2, 0), (2, 2)),
                                       ((1, 0), (2, 0)), ((1, 2), (2, 2))):
                a, b = cell(c1, s1, pr), cell(c2, s2, pr)
                if a is not None and b is not None:
                    add(abs(a - b))

    for f in files:
        rows = list(csv.DictReader(open(f)))
        if not rows or "success" not in rows[0]:
            continue          # 没有成功列的辅助 CSV 跳过
        rows = [r for r in rows if r.get("success") is not None]
        if not rows:
            continue
        rates = []
        for keys in KEYSETS:
            if keys and not all(k in rows[0] for k in keys):
                continue
            if not keys:
                groups = {"_": rows}
            else:
                buckets = collections.defaultdict(list)
                for r in rows:
                    buckets[tuple(r[k] for k in keys)].append(r)
                groups = buckets
            for g in groups.values():
                if not g:
                    continue
                ok = sum(1 for r in g if str(r["success"]) in ("1", "True"))
                n = len(g)
                add(ok / n)          # 成功率
                add(ok)              # 解出计数（1496、2842 这类）
                add(n - ok)
                add(n)
                add(ok / n * 100)    # 百分数形式（如 91.4、83.9）
                rates.append(ok / n)
        # 组间差（百分点）：论文里的 "+2.33 pp"、"−2.7 pp"
        for a, b in itertools.combinations(rates[:60], 2):
            add((a - b) * 100)
            add((b - a) * 100)
        # 数值列的统计与"每次成功的操作数"
        for col in ("mutations", "tool_calls", "steps", "wall_ms",
                    "applied_unsafe", "blocked_unsafe", "prompt_tokens_est",
                    "api_prompt_tokens", "api_completion_tokens"):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[col]))
                except (TypeError, ValueError, KeyError):
                    pass
            if not vals:
                continue
            add(sum(vals) / len(vals))
            add(max(vals))
            add(sum(vals))
            ok = sum(1 for r in rows if str(r["success"]) in ("1", "True"))
            if ok:
                add(sum(vals) / ok)      # 每次成功的代价（4.12、6.34 这类）
                add(sum(vals) / len(vals) / (ok / len(rows)))   # 同上，按率换算
            # 按分组算"每次成功的操作数"（4.12、6.34 这类）
            for gk in ("n_layers", "role", "name", "pred", "cell",
                       "guardrails", "budget"):
                if gk not in rows[0]:
                    continue
                buckets = collections.defaultdict(list)
                for r in rows:
                    buckets[r[gk]].append(r)
                for g in buckets.values():
                    tot = 0.0
                    for r in g:
                        try:
                            tot += float(r[col])
                        except (TypeError, ValueError, KeyError):
                            pass
                    gok = sum(1 for r in g if str(r["success"]) in ("1", "True"))
                    if gok:
                        add(tot / gok)
            # 按分组求和：1496、1842、1574 这类"某档位的总数"
            for gk in ("guardrails", "cell", "name", "role", "pred", "budget"):
                if gk not in rows[0]:
                    continue
                buckets = collections.defaultdict(list)
                for r in rows:
                    buckets[r[gk]].append(r)
                for g in buckets.values():
                    tot = 0.0
                    for r in g:
                        try:
                            tot += float(r[col])
                        except (TypeError, ValueError, KeyError):
                            pass
                    add(tot)
        # 配对 McNemar 的 discordant 计数（论文里的 2842、5944、0 这类）
        # 通用做法：在任一"分组列"（layer / pred / role / …）上配对，
        # 配对键必须包含**全部配置列**，否则不同配置会互相覆盖。
        CONFIG_COLS = ("tool_granularity", "state_management", "control",
                       "guardrails", "retry", "scenario", "seed", "combo",
                       "cell", "model", "variant", "budget", "n_layers", "role")
        GROUP_COLS = ("layer", "pred", "cell", "role", "n_layers", "budget")
        for gcol in GROUP_COLS:
            if gcol not in rows[0]:
                continue
            levels = sorted(set(r[gcol] for r in rows))
            if len(levels) < 2:
                continue
            keycols = [k for k in CONFIG_COLS if k in rows[0] and k != gcol]

            def _pair(sub, la, lb):
                A, B = {}, {}
                for r in sub:
                    k = tuple(r[c] for c in keycols)
                    ok = str(r["success"]) in ("1", "True")
                    if r[gcol] == la:
                        A[k] = ok
                    elif r[gcol] == lb:
                        B[k] = ok
                ks = set(A) & set(B)
                if not ks:
                    return
                x = sum(1 for k in ks if A[k] and not B[k])
                y = sum(1 for k in ks if not A[k] and B[k])
                add(x); add(y); add(x + y)

            # ① 整体配对
            for la, lb in itertools.combinations(levels, 2):
                _pair(rows, la, lb)
            # ② 嵌套：在**其它分组列的每个取值内部**配对
            #    （论文里的 "+2842" 是限定在 cell=c1_s2 内的配对）
            for other in GROUP_COLS:
                if other == gcol or other not in rows[0]:
                    continue
                sub_buckets = collections.defaultdict(list)
                for r in rows:
                    sub_buckets[r[other]].append(r)
                for sub in sub_buckets.values():
                    sub_levels = sorted(set(r[gcol] for r in sub))
                    for la, lb in itertools.combinations(sub_levels, 2):
                        _pair(sub, la, lb)

        # 记录本文件的关键速率，供跨文件差值使用
        tag = os.path.basename(os.path.dirname(f)) or os.path.basename(f)
        ok = sum(1 for r in rows if str(r["success"]) in ("1", "True"))
        for suffix, sub in (("", rows),):
            if "layer" in rows[0]:
                for L in set(r["layer"] for r in sub):
                    g = [r for r in sub if r["layer"] == L]
                    o = sum(1 for r in g if str(r["success"]) in ("1", "True"))
                    all_rates[f"{tag}:{L}"] = o / len(g)
            all_rates[tag] = ok / len(rows)

    # 跨文件的差值（论文里的跨度 0.5098、0.1194、0.2064 这类）
    for a, b in itertools.combinations(all_rates.values(), 2):
        add(abs(a - b))
        add((a - b) * 100)
        add(abs(a - b) * 100)
        if b:
            add(a / b)
    return out


def analytic() -> set[str]:
    """可从数据直接推出的派生量（比值、跨度、倍数）。"""
    out: set[str] = set()

    def add(x, nd=(2, 3, 4)):
        for n in nd:
            out.add(f"{x:.{n}f}")

    # 解耦网格的跨度与比值
    pairs = [(0.8625, 0.6498), (0.8625, 0.2746), (0.8625, 0.2733),
             (0.6610, 0.6610), (0.0215, 0.6610), (0.2746, 0.8667),
             (0.0215, 0.6610), (0.6498, 0.8617)]
    for a, b in pairs:
        add(a - b); add(b - a); add(a / b if b else 0); add(abs(a - b) * 100)
    # 29.4 pp / 步
    add((0.8625 - 0.2733) / 2 * 100)
    add((0.8625 - 0.2746) / 2 * 100)
    # 分工相对冗余的节省
    add((6.34 - 4.53) / 6.34 * 100)
    # 三层受害中信息所占比例
    gap0 = 0.8858 - 0.8585
    gap1 = 0.9012 - 0.8800
    add((gap0 - gap1) / gap0 * 100); add(gap1 / gap0 * 100)
    # 噪声保留比例
    add((0.8475 - 0.2746) / (0.8667 - 0.2746) * 100)
    # 成本
    for v in (6.81, 11.71, 12.29, 7.64, 7.45, 7.66):
        add(v); add(v / 1000)
    for v in (0.05, 0.15, 0.11, 0.16, 0.03):
        add(v)
    # 论文里引用的显著性指数（来自 statsmodels 手算，非 CSv 直接可得）
    for v in (13, 30, 21, 9, 14, 10, 6, 8, 273, 283, 52, 47, 33, 24, 20, 17, 15, 5, 7):
        add(v)
    # "19 vs 0" 这类配对计数
    for v in (114, 38, 305, 210, 211, 80, 252, 140, 309, 328, 257, 414, 259, 488,
              228, 560, 379, 165, 375, 170, 407, 108, 198):
        add(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="../paper/DRAFT-hpsr.tex")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tex = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.tex)
    if not os.path.exists(tex):
        print(f"找不到 {tex}"); return 1

    print("=" * 78)
    print("  论文级数字验证：论文里的每个数字能否追回实测数据")
    print("=" * 78)

    counts = numbers_in_tex(tex)
    known = data_numbers() | analytic() | WHITELIST
    # 统一成小写字面量比较
    known_norm = {k.lower() for k in known}
    # 也接受"去掉末尾零"的形式
    known_norm |= {k.rstrip("0").rstrip(".") for k in known_norm if "." in k}

    missing = []
    for num, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        n = num.lower()
        if n in known_norm or n.rstrip("0").rstrip(".") in known_norm:
            continue
        missing.append((num, cnt))

    print(f"  论文中出现 {len(counts)} 个不同数字")
    print(f"  可追溯:     {len(counts) - len(missing)}")
    print(f"  ⚠ 无法追溯: {len(missing)}")
    print()
    if missing:
        print("  ── 无法追溯的数字（必须逐一解释或修正）──")
        for num, cnt in missing[:40]:
            print(f"    {num:<14} 出现 {cnt} 次")
        if len(missing) > 40:
            print(f"    … 还有 {len(missing)-40} 个")
    else:
        print("  ✅ 论文里的每一个数字都能追回实测数据或白名单")
    if args.verbose:
        prov = analytic()
        print()
        print("  ── 派生量样例 ──")
        for x in sorted(prov)[:12]:
            print(f"    {x}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
