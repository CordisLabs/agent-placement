#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨文件一致性检查：同一个事实在多个副本里是否一致。

## 为什么需要它

2026-10-17 那天发现：**tex 摘要改过 3 遍，但投递包里的摘要是旧的**。
EDAS 会存下用户粘贴的摘要文字，而它与 PDF 不是同一个文件 ——
**审稿人看到的可能是过期的那个**。

同类问题反复出现：
  · 标题写「两域」而正文写「三域」（第 22 轮）
  · 结论说「可互换」而正文说「不可」（第 25 轮）
  · 投递包指纹指向旧 PDF（多次）

**每个副本都是一次手动同步的机会，也就是一次忘记的机会。**

本脚本做机器能做的部分：
  1. 摘要：tex / 投递包 / 中文版 三处的关键数字与短语是否都在
  2. 指纹：投递包里的 sha256 / md5 / 大小是否指向当前 PDF
  3. 域数：标题、摘要、正文、表题说的是几个域
  4. 关键词：作废的短语是否还有残留

**它抓不到语义错误**（那需要通读），但能抓住「副本漂移」。

用法：
    python3 crosscheck_copies.py           # 全部检查
    python3 crosscheck_copies.py --quiet   # 只报失败
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TEX = os.path.join(ROOT, "paper", "DRAFT-hpsr.tex")
PDF = os.path.join(ROOT, "paper", "DRAFT-hpsr.pdf")
PKG = os.path.join(ROOT, "HPSR2027-投递包.md")
ZH = os.path.join(ROOT, "HPSR2027-中文版.md")

FAILS: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(f"{name}: {detail}" if detail else name)


def read(path: str) -> str:
    """读文本文件。二进制文件（PDF）返回空串 —— 指纹检查会单独按字节读。"""
    if not os.path.exists(path):
        return ""
    if path.lower().endswith(".pdf"):
        return ""
    return io.open(path, encoding="utf-8", errors="replace").read()


def detex(s: str) -> str:
    """把 LaTeX 摘要化成可粘贴的纯文本（与投递包的提取逻辑一致）。"""
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    for cmd in ("code", "emph", "textbf", "text", "textsuperscript"):
        s = re.sub(r"\\" + cmd + r"\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("{,}", ","), s)
    s = s.replace("\\times", "×").replace("\\rightarrow", "→")
    s = s.replace("\\%", "%").replace("~", " ").replace("\\ ", " ")
    s = re.sub(r"\\\\", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> int:
    quiet = "--quiet" in sys.argv
    tex = read(TEX)
    pdf = read(PDF)
    pkg = read(PKG)
    zh = read(ZH)

    if quiet:
        pass
    else:
        print("═" * 70)
        print("  跨文件一致性检查")
        print("═" * 70)

    # ── 1. 摘要三处是否同步 ────────────────────────────────────────────
    m_tex = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    check("tex 有摘要", m_tex is not None)
    abs_tex = detex(m_tex.group(1)) if m_tex else ""

    m_pkg = re.search(r"## 三、摘要（可直接粘贴）\n\n(.*?)\n\n---", pkg, re.S)
    check("投递包有摘要段", m_pkg is not None)
    abs_pkg = m_pkg.group(1).strip() if m_pkg else ""

    check("投递包摘要无 LaTeX 残留", len(re.findall(r"[\\${}]", abs_pkg)) == 0,
          f"{len(re.findall(r'[\\${}]', abs_pkg))} 个残留记号")
    check("投递包摘要无 '---'", "---" not in abs_pkg)

    if abs_tex and abs_pkg:
        # 用数字集合比对：摘要里的数字必须两边都有
        nums_tex = set(re.findall(r"\d[\d,.]*\d|\d", abs_tex))
        nums_pkg = set(re.findall(r"\d[\d,.]*\d|\d", abs_pkg))
        missing = nums_tex - nums_pkg
        check("摘要数字齐全", not missing, f"投递包缺 {sorted(missing)[:6]}")
        ratio = len(nums_tex & nums_pkg) / len(nums_tex) if nums_tex else 1.0
        check("摘要数字覆盖 ≥95%", ratio >= 0.95, f"{ratio:.0%}")
        if not quiet:
            print(f"    tex 摘要 {len(abs_tex)} 字符 · 投递包 {len(abs_pkg)} 字符 · "
                  f"数字覆盖 {ratio:.0%}")

    # ── 2. 指纹是否指向当前 PDF ───────────────────────────────────────
    if os.path.exists(PDF):
        data = io.open(PDF, "rb").read()
        sha = hashlib.sha256(data).hexdigest()
        md5 = hashlib.md5(data).hexdigest()
        size = len(data)
        check("投递包含当前 sha256", sha[:32] in pkg, f"应为 {sha[:16]}…")
        check("投递包含当前 md5", md5 in pkg, f"应为 {md5[:16]}…")
        check("投递包含当前大小", f"{size:,}" in pkg or str(size) in pkg, f"应为 {size:,}")
        if not quiet:
            print(f"    PDF {size:,} 字节 · sha {sha[:16]}… · md5 {md5[:16]}…")

    # ── 3. 域数：各处说法是否一致 ────────────────────────────────────
    # 正文与表题必须承认三个域；标题与摘要不得声称两个域
    check("标题未声称两域",
          not re.search(r"[Tt]wo\s+[Dd]omains", tex), "标题/正文还有 'Two Domains'")
    check("正文承认三域",
          ("three domains" in tex) or ("three further" in tex) or
          ("3 domains" in tex), "正文没有承认三个域")
    for phrase in ("both domains",):
        # 'both domains' 只在指 packet+water 的故障字典时合法；出现次数不应增加
        n = tex.count(phrase)
        check(f"'{phrase}' 出现次数 ≤3", n <= 3, f"{n} 次，需要人工确认每一次的语境")

    # ── 4. 已作废的短语不得残留 ──────────────────────────────────────
    # 这些是 2026-10-17 通读中发现并改掉的措辞
    RETIRED = [
        ("two of the seven", "phi4 不可达的旧结论（实为 one）"),
        ("capability amplifies placement", "尺度 artifact 的过度解读"),
        ("almost exactly interchangeable", "旧的「可互换」框架"),
    ]
    for phrase, why in RETIRED:
        check(f"已作废短语不残留: '{phrase}'", phrase not in tex, why)

    # ── 5. 中文版是否同步了关键结论 ────────────────────────────────
    if zh:
        check("中文版不是「跨两个域」", "跨两个域" not in zh,
              "中文版副标题还写两个域")
        check("中文版不是「几乎完全可互换」", "几乎完全可互换" not in zh,
              "中文版还在用旧框架")
        check("中文版不以 oracle 为上界", "作为上界" not in zh,
              "中文版还说 oracle 是上界")
        check("中文版无 182 自指计数", "182/182" not in zh and "$182$" not in zh,
              "中文版还有 182 这个会失效的计数")


    # ── 6. START-HERE 里的路径是否真实存在 ────────────────────────────
    # 路径可能相对项目根、demo/、paper/ 等几个基点；逐个试。
    start_here = read(os.path.join(ROOT, "START-HERE.md"))
    if start_here:
        BASES = [ROOT, os.path.join(ROOT, "demo"), os.path.join(ROOT, "paper"),
                 os.path.join(ROOT, "demo", "out"), os.path.join(ROOT, "refs", "lit"),
                 os.path.join(ROOT, "refs", "journal-lit"),
                 os.path.join(ROOT, "demo", "out", "figs_B"),
                 os.path.join(ROOT, "demo", "out", "figs_ext"),
                 os.path.join(ROOT, "demo", "aixnet"),
                 os.path.join(ROOT, "demo", "aixnet", "domains")]
        documented = set(re.findall(
            r"`([A-Za-z0-9_./\-]+\.(?:md|py|tex|pdf|bib|sh|json|html|csv))`", start_here))
        # 拆开 "paper/X.tex/.pdf" 这种粘连写法
        for r in list(documented):
            mm = re.match(r"^(.*\.(?:tex|md))((?:/\.[A-Za-z]+)+)$", r)
            if not mm:
                continue
            documented.discard(r)          # ← 删掉粘连的原路径
            base_ = mm.group(1)
            for ext in mm.group(2).split("/")[1:]:
                documented.add(base_.rsplit(".", 1)[0] + "." + ext.lstrip("."))
        missing = []
        for rel in documented:
            if any(os.path.exists(os.path.join(b, rel)) for b in BASES):
                continue
            # 含通配的（a/b*.csv）单独放行
            if "*" in rel:
                continue
            missing.append(rel)
        check("START-HERE 提到的路径都存在", not missing,
              f"缺 {len(missing)}/{len(documented)}: {missing[:6]}")
        if not quiet:
            print(f"    START-HERE 引用 {len(documented)} 个文件路径 · 缺失 {len(missing)}")

    # ── 7. START-HERE 里的 PDF 指纹是否过期 ─────────────────────────
    if start_here and os.path.exists(PDF):
        sha = hashlib.sha256(io.open(PDF, "rb").read()).hexdigest()
        # 只接受「完整 sha」或「长度 >=12 的前缀」
        hashes = re.findall(r"\b[0-9a-f]{12,64}\b", start_here)
        hit = any(sha.startswith(h) or h.startswith(sha[:12]) and len(h) >= 12
                  for h in hashes)
        check("START-HERE 记录当前 HPSR sha256", hit,
              "START-HERE 里没有指向当前 PDF 的 sha256 —— 可能过期")

    # ── 报告 ──────────────────────────────────────────────────────────
    print()
    print("─" * 70)
    if FAILS:
        print(f"  ❌ 通过 {CHECKS - len(FAILS)}  失败 {len(FAILS)}")
        print()
        for f in FAILS:
            print(f"     · {f}")
        return 1
    print(f"  ✅ 通过 {CHECKS}  失败 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
