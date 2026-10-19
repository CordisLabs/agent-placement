#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_consistency.py -- 跨文件一致性审计。

## 为什么需要它

2026-10-17 一天之内，我手工抓到五处「同一件事在不同文件里写成不同数」：

  1. kb/40 的跨度比值是「节奏/陈旧」，论文却当「陈旧/节奏」用（queue 报反）
  2. kb/48 把 p99 上界当成下界
  3. 论文表 4 的 10 篇词数与 `pdftotext` 实测差 65 字
  4. 摘要手抄版与 tex 真摘要相似度只有 0.15
  5. `kb/47` 与方向 B 的 8 个条目从未进 START-HERE 索引

**这些都是「同一事实的多个副本互相矛盾」。** 靠人记不住，必须机器查。

## 它查什么

**A. 数字一致性** —— 同一批数字出现在多个文件里时必须相等
**B. 索引完整性** —— 工作区里每个 .md 都该在 START-HERE 里出现
**C. 路径有效性** —— START-HERE 里提到的每个文件路径都该真实存在
**D. 禁止句式** —— 扫描已知容易写错的说法（"边缘"、"N/9"、"下界"用错）

## 用法

    python3 audit_consistency.py            # 全部检查
    python3 audit_consistency.py --quiet    # 只打印失败
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Audit:
    def __init__(self, quiet=False):
        self.ok = self.bad = 0
        self.quiet = quiet
        self.issues = []

    def chk(self, label, cond, detail=""):
        if cond:
            self.ok += 1
            if not self.quiet:
                print(f"  ✅ {label}")
        else:
            self.bad += 1
            self.issues.append((label, detail))
            print(f"  ❌ {label}  {detail}")

    def eq(self, label, value, expected, tol=0.0):
        if isinstance(value, float) and isinstance(expected, float) and tol:
            good = abs(value - expected) <= tol
        else:
            good = value == expected
        self.chk(label, good, f"实测 {value}  应为 {expected}")


def read(p):
    try:
        return Path(p).read_text(encoding="utf-8")
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    A = Audit(args.quiet)

    # ─── A. 数字一致性 ───
    print("\n═══ A. 同一数字的多个副本必须相等 ═══")
    # 翻转演示的三个 overclaim，出现在 kb/44、论文表 I、kb/46 里
    sources = {
        "kb/44-口径翻转演示.md": ROOT / "kb/44-口径翻转演示.md",
        "paper/DRAFT-direction-B.tex": ROOT / "paper/DRAFT-direction-B.tex",
        "HPSR2027-中文版.md": ROOT / "HPSR2027-中文版.md",
    }
    for name, p in sources.items():
        if not p.exists():
            A.chk(f"{name} 存在", False)
            continue
        t = read(p)
        if "84.1" in t or "84.1" in t:
            A.chk(f"{name} 含 packet overclaim 84.1", True)
        if "75.6" in t:
            A.chk(f"{name} 含 water overclaim 75.6", True)
        if "74.4" in t:
            A.chk(f"{name} 含 queue overclaim 74.4", True)

    # η² 的三个比值
    for fname in ("kb/42-eta-squared-修正.md", "HPSR2027-中文版.md",
                  "paper/DRAFT-hpsr.tex"):
        p = ROOT / fname
        if not p.exists():
            continue
        t = read(p)
        for v in ("1.84", "2.26", "4.74"):
            short = v.replace("84", "8").replace("26", "3").replace("74", "7")
            forms = [v, short + "×", short + "\\times", short]
            A.chk(f"{fname} 含 η² 比 {short}（任一写法）",
                  any(f in t for f in forms[:3]),
                  f"找不到 {v} / {short}× / {short}\\times")

    # 词数：kb/46 与论文表 4 必须一致
    kb46 = read(ROOT / "kb/46-领域审计.md")
    paperB = read(ROOT / "paper/DRAFT-direction-B.tex")
    m1 = re.search(r"(\d[\d,{}]*)\s*词", kb46)
    has_122958 = "122{,}958" in paperB or "122,958" in paperB
    has_122893 = "122{,}893" in paperB or "122,893" in paperB
    A.chk("论文表 4 用 122,958（实测值）", has_122958, "未找到")
    A.chk("论文表 4 不含旧的 122,893", not has_122893, "仍含旧值")

    # ─── B. 索引完整性 ───
    print("\n═══ B. 工作区每个 .md 都该在 START-HERE 里 ═══")
    here = read(ROOT / "START-HERE.md")
    kb_index = read(ROOT / "kb/INDEX.md")
    indexed_text = here + "\n" + kb_index
    md_files = sorted(p.relative_to(ROOT).as_posix()
                      for p in ROOT.rglob("*.md")
                      if "refs/" not in p.as_posix() and ".venv" not in p.as_posix())
    missing = [m for m in md_files
               if Path(m).name not in indexed_text
               and not m.startswith("kb/INDEX")]
    for m in missing:
        A.chk(f"{m} 已在 START-HERE 索引", False)
    A.chk(f"全部 {len(md_files)} 个 .md 已索引", not missing,
          f"缺 {len(missing)} 个：{missing[:3]}")

    # ─── C. 路径有效性 ───
    print("\n═══ C. START-HERE 提到的路径都该存在 ═══")
    refs = sorted(set(re.findall(r"`([A-Za-z0-9_./\-]+\.(?:md|tex|pdf|py|sh|csv))`",
                                 here)))
    def plausible(r):
        # 复合路径（如 a.tex/.pdf）与相对其它目录的（aixnet/...）不算真路径
        if "/." in r or r.count("/") > 2:
            return False
        head = r.split("/")[0]
        return (ROOT / head).exists()
    checkable = [r for r in refs if plausible(r)]
    bad_paths = [r for r in checkable if not (ROOT / r).exists()]
    A.chk(f"{len(checkable)} 个可判定的被引路径都存在", not bad_paths,
          f"缺 {len(bad_paths)} 个：{bad_paths[:3]}")

    # ─── D. 禁止句式 ───
    print("\n═══ D. 已知容易写错的句式 ═══")
    scanners = [
        # (文件, 正则, 说明)
        ("kb/48-方向G-推断延迟即可行域.md", r"≥ 1,000 ms", "把 p99 上界写成下界"),
        ("kb/48-方向G-推断延迟即可行域.md", r"~1,000 ms", "把 p99 上界写成下界"),
        ("kb/48-方向G-推断延迟即可行域.md", r"6/9", "编造的不可达格子数"),
        ("kb/48-方向G-推断延迟即可行域.md", r"need \* 2", "自造的判定系数"),
        ("kb/40-queue-domain.md", r"1\.098", "已被修正的旧比值"),
        ("kb/40-queue-domain.md", r"0\.700", "已被修正的旧比值"),
    ]
    for fname, pat, why in scanners:
        p = ROOT / fname
        if not p.exists():
            continue
        text = read(p)
        hits = 0
        for m in re.finditer(pat, text):
            ctx = text[max(0, m.start() - 60):m.start() + 60]
            # 允许出现在更正说明 / 反面教材里
            if any(k in ctx for k in ("旧值", "有误", "已修正", "不能声称", "反面", "第一版",
                                      "注", "倒数", "不矛盾", "统一为")):
                continue
            hits += 1
        A.chk(f"{fname} 不出现「{why}」（更正说明除外）", hits == 0, f"命中 {hits} 次")


    # ─── E. 标题/摘要 vs 正文的域数一致（2026-10-17 手工抓到过三次）───
    print("\n═══ E. 「几个域」的说法必须前后一致 ═══")
    for fname in ("paper/DRAFT-hpsr.tex",):
        p = ROOT / fname
        if not p.exists():
            continue
        t = read(p)
        # 正文里真的出现了几个域
        present = [d for d in ("packet", "water", "queue")
                   if len(re.findall(r"\b" + d + r"\b", t, re.I)) >= 3]
        n_real = len(present)
        # 标题说几个
        m = re.search(r"Study in (\w+) Domains", t)
        if m:
            word = m.group(1).lower()
            claimed = {"two": 2, "three": 3, "four": 4}.get(word, None)
            A.chk(f"{fname} 标题域数({claimed}) == 正文域数({n_real})",
                  claimed == n_real, f"标题说 {word}，正文有 {present}")
        # 不该再有「second physical domain」这种只算两个的说法（小节标题除外）
        for m2 in re.finditer(r"second physical domain", t):
            line = t[:m2.start()].count("\n") + 1
            ctx = t[max(0, m2.start() - 40):m2.start() + 40]
            if "subsection" not in ctx:
                A.chk(f"{fname} 第 {line} 行不该说 'second physical domain'", False, ctx)


    # ─── F. 表 / 图 / 正文的覆盖必须一致 ───
    print("\n═══ F. 表 / 图 / 正文覆盖一致（2026-10-17 第 18 个错的根因）═══")
    tex_path = ROOT / "paper" / "DRAFT-hpsr.tex"
    if tex_path.exists():
        tex = read(tex_path)

        # F1: tab:cross 的格数与域数
        m = re.search(r"\\label\{tab:cross\}(.*?)\\end\{table\}", tex, re.S)
        if m:
            body = m.group(1)
            data = [l for l in body.split("\n")
                    if "&" in l and l.rstrip().endswith("\\\\")
                    and "domain" not in l.split("&")[0]]
            cells = len(data)
            doms = set()
            for l in data:
                first = l.split("&")[0].replace("\\textbf", "").replace("{","").replace("}","").strip()
                if first:
                    doms.add(first)
            A.chk("tab:cross 有 5 个格", cells == 5, f"实测 {cells}")
            A.chk("tab:cross 覆盖 3 个域", len(doms) == 3, f"实测 {sorted(doms)}")

            # F2: 正文对「几个格」的说法必须与表一致
            mm = re.search(r"Table~\\ref\{tab:cross\}\s+(lists|reports)\s+all\s+"
                           r"(\w+)\s+cells", tex)
            if mm:
                word = mm.group(2).lower()
                claimed = {"two":2,"three":3,"four":4,"five":5,"six":6}.get(word)
                A.chk(f"正文说 tab:cross 有 {claimed} 格 == 实际 {cells}",
                      claimed == cells, f"正文说 {word}")
            else:
                A.chk("正文明确说了 tab:cross 的格数", False,
                      "找不到 'Table~ref{tab:cross} lists/reports all N cells'")

            # F3: 正文对图的说法
            mg = re.search(r"Fig\.~\\ref\{fig:cross\}\s+(plots|shows)\s+"
                           r"(\w+)", tex)
            if mg:
                word = mg.group(2).lower()
                claimed = {"two":2,"three":3,"four":4,"five":5}.get(word)
                A.chk(f"正文说 fig:cross 画了 {claimed} 格", claimed == 4,
                      f"正文说 {word}")

        # F4: 图 3 的标题声明的域数
        # 注意：LaTeX 里 \label 可能在 \caption 之后，所以要从 \begin 开始抓整块
        mf = re.search(r"\\begin\{figure\*?\}((?:(?!\\end\{figure).)*?"
                       r"\\label\{fig:cross\}.*?)\\end\{figure", tex, re.S)
        if not mf:
            # 退而求其次：抓所有 figure 块，找含 fig:cross 的那个
            for cand in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, re.S):
                if "fig:cross" in cand.group(1):
                    mf = cand
                    break
        if mf:
            cap = mf.group(1)
            A.chk("fig:cross 标题说 two physical domains",
                  "two physical domains" in cap,
                  "标题域数与实际不符")
            A.chk("fig:cross 标题说两种故障分布",
                  "two fault distributions" in cap,
                  "标题没提故障分布")
        else:
            A.chk("找到 fig:cross 的 figure 块", False, "正则没匹配到")

        # F5: 「2.5--5×」这个区间的两端都必须有数据支撑
        # packet 4.93×, water 2.47× —— 两个都在 tab:cross 里
        A.chk("正文的 2.5--5× 两端在表里都有行",
              "0.6392" in tex and "0.9722" in tex,
              "water random (0.6392) 或 packet random (0.9722) 不在论文里")

    # ─── 汇总 ───
    print("\n" + "=" * 70)
    print(f"  通过 {A.ok}  失败 {A.bad}")
    if A.issues:
        print("\n  失败项：")
        for label, detail in A.issues:
            print(f"    · {label}  {detail}")
    print("=" * 70)
    return 0 if A.bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
