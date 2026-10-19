#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""综合文档数字核对器。

把 HPSR2027-结果综合.md 与 kb/16–28 里引用的**每一个头条数字**
重新从实测 CSV 算一遍，逐项比对。

动机：这个项目跑了几十个实验、改了多次代码，
**数字之间很容易出现不一致**（本项目历史上已发生过两次：
walld-clock 因重跑而漂移、kb/15 的 no-op 汇总口径写错）。
在动笔写论文前，必须有一次全量核对。

用法：
    python3 verify_numbers.py            # 全部核对
    python3 verify_numbers.py --verbose  # 显示每一项
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, BAD = [], []


def check(label: str, got, want, tol=1e-4) -> None:
    good = abs(got - want) <= tol
    (OK if good else BAD).append((label, got, want))
    return good


def mean_success(path: str, layer: str | None = None) -> float:
    rows = list(csv.DictReader(open(path)))
    if layer:
        rows = [r for r in rows if r["layer"] == layer]
    if not rows:
        return float("nan")
    return sum(1 for r in rows if str(r["success"]) in ("1", "True")) / len(rows)


def counts(path: str, layer: str | None = None):
    rows = list(csv.DictReader(open(path)))
    if layer:
        rows = [r for r in rows if r["layer"] == layer]
    return sum(1 for r in rows if str(r["success"]) in ("1", "True")), len(rows)


def mc(path: str, la: str, lb: str, keycols):
    rows = list(csv.DictReader(open(path)))
    A, B = {}, {}
    for r in rows:
        k = tuple(r.get(c) for c in keycols)
        if r["layer"] == la:
            A[k] = str(r["success"]) in ("1", "True")
        if r["layer"] == lb:
            B[k] = str(r["success"]) in ("1", "True")
    ks = set(A) & set(B)
    b = sum(1 for k in ks if A[k] and not B[k])
    c = sum(1 for k in ks if not A[k] and B[k])
    n = b + c
    p = (min(sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2, 1.0)
         if n else 1.0)
    return len(ks), b, c, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # ── 1. 主论文三层（reps=20，完整 48 配置分解）──
    for layer, want in (("near_rt", 0.8625), ("non_rt", 0.6610), ("smo_intent", 0.2733)):
        got = mean_success("out/episodes.csv", layer)
        check(f"主论文 {layer}", got, want)

    # ── 2. 解耦网格（budget 4–6）──
    P = "out/decoupled20/episodes.csv"
    for cell, want in (("c1_s0", 0.8625), ("c1_s1", 0.6498), ("c1_s2", 0.2746),
                       ("c2_s0", 0.6610), ("c2_s1", 0.6610), ("c2_s2", 0.0215),
                       ("c3_s0", 0.2733), ("c3_s1", 0.2733), ("c3_s2", 0.2733)):
        check(f"解耦网格 {cell}", mean_success(P, cell), want)

    # ── 3. budget=20 解耦网格 ──
    B = "out/decoupled_budget20/episodes.csv"
    for cell, want in (("c1_s0", 0.9954), ("c1_s2", 0.9898), ("c2_s2", 0.8842),
                       ("c3_s0", 0.8960)):
        check(f"budget20 网格 {cell}", mean_success(B, cell), want)

    # ── 4. 跨家族（六模型）──
    fam = [
        ("qwen2.5:7b", "llm_layer_qwen2-5-7b_scoped3_*.csv", 0.453, 0.180),
        ("qwen2.5:14b", "llm_layer_qwen2-5-14b_scoped3_*.csv", 0.200, 0.067),
        ("llama3.1:8b", "llm_layer_*llama3-1-8b_scoped3_*.csv", 0.267, 0.067),
        ("gemma2:9b", "llm_layer_*gemma2-9b_scoped3_*.csv", 0.160, 0.033),
        ("phi4:14b", "llm_layer_*phi4-14b_scoped3_*.csv", 0.4667, 0.1533),
        ("mistral:7b", "llm_layer_*mistral-7b_scoped3_*.csv", 0.000, 0.000),
    ]
    for name, pat, wr, ws in fam:
        files = glob.glob("out/" + pat)
        if not files:
            BAD.append((f"跨家族 {name}: 文件缺失", float("nan"), wr)); continue
        rows = []
        for f in files:
            rows += list(csv.DictReader(open(f)))
        b = collections.defaultdict(list)
        for r in rows:
            b[r["layer"]].append(str(r["success"]) in ("1", "True"))
        check(f"跨家族 {name} near_rt", st.mean(b["near_rt"]), wr, 2e-3)
        check(f"跨家族 {name} 意图层", st.mean(b["smo_intent"]), ws, 2e-3)

    # ── 5. 27B（三层分散在两个文件）──
    rows = []
    for f in glob.glob("out/llm_arm_scoped3.csv") + glob.glob("out/llm_layer_qwen3-8-27b*.csv"):
        rows += list(csv.DictReader(open(f)))
    b = collections.defaultdict(list)
    for r in rows:
        b[r["layer"]].append(str(r["success"]) in ("1", "True"))
    check("27B near_rt", st.mean(b["near_rt"]), 0.9333, 2e-3)
    check("27B 意图层", st.mean(b["smo_intent"]), 0.2667, 2e-3)

    # ── 6. 随机故障流 ──
    check("随机流 near_rt", mean_success("out/random_faults/episodes.csv", "near_rt"), 0.9722)
    check("随机流 意图层", mean_success("out/random_faults/episodes.csv", "smo_intent"), 0.8528)
    n, bb, cc, _ = mc("out/random_faults/episodes.csv", "near_rt", "smo_intent",
                      ("scenario", "tool_granularity", "state_management",
                       "control", "guardrails", "retry", "seed"))
    check("随机流 near vs 意图 仅后者", float(cc), 0.0)

    # ── 7. 水网 ──
    check("水网 near_rt", mean_success("out/water/episodes.csv", "near_rt"), 0.7565)
    check("水网 意图层", mean_success("out/water/episodes.csv", "smo_intent"), 0.2467)
    check("水网随机 near_rt", mean_success("out/water_random_faults/episodes.csv", "near_rt"), 0.6392)
    check("水网随机 意图层",
          mean_success("out/water_random_faults/episodes.csv", "smo_intent"), 0.4328)

    # ── 8. 护栏前沿端点（必须等于主论文的 1496 / 1842）──
    g = "out/guardrail_frontier/episodes.csv"
    if os.path.exists(g):
        rows = list(csv.DictReader(open(g)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[r["guardrails"]]
            d[0] += int(r["applied_unsafe"] or 0)
            d[1] += int(r["blocked_unsafe"] or 0)
        check("护栏 G0 放行（=主论文 1496）", float(agg["G0_none"][0]), 1496.0)
        check("护栏 G6 拦截（=主论文 1842）", float(agg["G6_approval"][1]), 1842.0)
        check("护栏 G5 放行（应为 0）", float(agg["G5_noprod"][0]), 0.0)

    # ── 9. 多层 ──
    M = "out/multilayer/episodes.csv"
    if os.path.exists(M):
        rows = list(csv.DictReader(open(M)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[r["name"]]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for name, want in (("L_near_only", 0.8625), ("L_near_non", 0.8858),
                           ("L_near_smo", 0.8842), ("L_all_three", 0.8585),
                           ("L_near_non_fresh", 0.9004), ("L_near_smo_fresh", 0.8894)):
            check(f"多层 {name}", agg[name][0] / agg[name][1], want)

    # ── 10. 预算探测 ──
    if os.path.exists("out/llm_budget.csv"):
        rows = list(csv.DictReader(open("out/llm_budget.csv")))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[(int(r["budget"]), r["layer"])]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for bu, layer, want in ((4, "non_rt", 0), (6, "non_rt", 18), (10, "smo_intent", 28),
                                (20, "near_rt", 30), (20, "smo_intent", 30)):
            check(f"预算{bu} {layer} 解出数", float(agg[(bu, layer)][0]), float(want))

    # ── 11. 预测器（最强结果）──
    PPT = "out/predictor/episodes.csv"
    if os.path.exists(PPT):
        rows = list(csv.DictReader(open(PPT)))
        agg = collections.defaultdict(lambda: [0, 0])
        sets = collections.defaultdict(set)
        for r in rows:
            d = agg[(r["cell"], r["pred"])]; d[1] += 1
            ok = str(r["success"]) in ("1", "True")
            d[0] += ok
            if ok:
                sets[(r["cell"], r["pred"])].add((r["scenario"], r["seed"], r["combo"]))
        for cell, pred, want in (("c1_s2", "none", 0.2746), ("c1_s2", "delta", 0.8667),
                                 ("c2_s2", "none", 0.0215), ("c2_s2", "delta", 0.6610),
                                 ("c3_s2", "delta", 0.2733), ("c1_s0", "delta", 0.8625),
                                 ("c1_s1", "delta", 0.8617), ("c1_s0", "none", 0.8625)):
            a, n = agg[(cell, pred)]
            check(f"预测器 {cell}/{pred}", a / n, want)
        # 关键：预测器解出的是严格超集——"仅 none 解出" 必须为 0
        for cell in ("c1_s2", "c2_s2"):
            only_none = len(sets[(cell, "none")] - sets[(cell, "delta")])
            check(f"预测器 {cell}: 仅 none 解出", float(only_none), 0.0)

    # ── 12. 多层 × 预测器 ──
    MP = "out/multilayer_pred/episodes.csv"
    if os.path.exists(MP):
        rows = list(csv.DictReader(open(MP)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[(r["name"], r["pred"])]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for name, pred, want in (("L_near_only", "none", 0.8625),
                                 ("L_near_only", "delta", 0.8625),
                                 ("L_near_non", "none", 0.8858),
                                 ("L_near_non", "delta", 0.9012),
                                 ("L_all_three", "none", 0.8585),
                                 ("L_all_three", "delta", 0.8800)):
            a, n = agg[(name, pred)]
            check(f"多层×预测器 {name}/{pred}", a / n, want)

    # ── 13. 预测噪声敏感性 ──
    PN = "out/predictor_noise/episodes.csv"
    if os.path.exists(PN):
        rows = list(csv.DictReader(open(PN)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[(r["cell"], float(r["noise"]))]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for cell, noise, want in (("c1_s2", 0.5, 0.8475), ("c2_s2", 0.5, 0.6610),
                                  ("c1_s2", 0.1, 0.8667), ("c1_s2", 0.0, 0.8667)):
            a, n = agg[(cell, noise)]
            check(f"噪声 {cell}±{int(noise*100)}%", a / n, want, 1e-3)

    # ── 14. 预测器在真实 LLM 上（含恒等检查）──
    PL = "out/predictor_llm/episodes.csv"
    if os.path.exists(PL):
        rows = list(csv.DictReader(open(PL)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[(r["cell"], r["pred"])]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for cell, pred, want in (("c1_s0", "none", 27), ("c1_s0", "delta", 27),
                                 ("c1_s1", "none", 0), ("c1_s1", "delta", 18),
                                 ("c1_s2", "none", 0), ("c1_s2", "delta", 18),
                                 ("c2_s2", "delta", 0), ("c3_s0", "delta", 0)):
            check(f"LLM {cell}/{pred} 解出数", float(agg[(cell, pred)][0]), float(want))
        # 恒等性质：staleness=0 时 delta 必须与 none 逐位一致
        A = {(r["scenario"], r["seed"]): str(r["success"]) in ("1", "True")
             for r in rows if r["cell"] == "c1_s0" and r["pred"] == "none"}
        B = {(r["scenario"], r["seed"]): str(r["success"]) in ("1", "True")
             for r in rows if r["cell"] == "c1_s0" and r["pred"] == "delta"}
        diff = sum(1 for k in set(A) & set(B) if A[k] != B[k])
        check("LLM c1_s0 恒等（delta==none 逐回合）", float(diff), 0.0)

    # ── 15. mock 侧恒等性质（布尔 bug 的回归测试）──
    if os.path.exists(PPT):
        rows = list(csv.DictReader(open(PPT)))
        A = {(r["scenario"], r["seed"], r["combo"]): str(r["success"]) in ("1", "True")
             for r in rows if r["cell"] == "c1_s0" and r["pred"] == "none"}
        B = {(r["scenario"], r["seed"], r["combo"]): str(r["success"]) in ("1", "True")
             for r in rows if r["cell"] == "c1_s0" and r["pred"] == "delta"}
        diff = sum(1 for k in set(A) & set(B) if A[k] != B[k])
        check("mock c1_s0 恒等（staleness=0 → 外推=恒等）", float(diff), 0.0)
        # 外推函数本身的类型检查
        from aixnet.harness import _extrapolate
        obs = {"t": 3, "links": [{"id": "L0", "up": True, "capacity": 100.0}],
               "alarms": [{"id": "A1", "acked": False}]}
        out = _extrapolate(obs, obs, steps=0)
        check("_extrapolate 保布尔 up", 1.0 if out["links"][0]["up"] is True else 0.0, 1.0)
        check("_extrapolate 保布尔 acked", 1.0 if out["alarms"][0]["acked"] is False else 0.0, 1.0)

    # ── 16. 真实 LLM 上的多层（负面结果）──
    MLL = "out/multilayer_llm/episodes.csv"
    if os.path.exists(MLL):
        rows = list(csv.DictReader(open(MLL)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[(r["name"], r["pred"])]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for name, want in (("L_near_only", 28), ("L_near_non", 28), ("L_all_three", 28)):
            check(f"LLM 多层 {name}/none 解出数", float(agg[(name, "none")][0]), float(want))
        # 技术故障必须为零
        errs = sum(1 for r in rows if r.get("err") not in (None, "", "None"))
        check("LLM 多层：技术故障数", float(errs), 0.0)
        check("LLM 多层：总回合数", float(len(rows)), 300.0)

    # ── 17. 27B 多层（天花板效应）──
    M27 = "out/multilayer_27b/episodes.csv"
    if os.path.exists(M27):
        rows = list(csv.DictReader(open(M27)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[r["name"]]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for name, want in (("L_near_only", 49), ("L_near_non", 50), ("L_all_three", 50)):
            check(f"27B 多层 {name} 解出数", float(agg[name][0]), float(want))
        errs = sum(1 for r in rows if r.get("err") not in (None, "", "None"))
        check("27B 多层：技术故障数", float(errs), 0.0)
        check("27B 多层：总回合数", float(len(rows)), 150.0)

    # ── 18. 多层角色分工 ──
    MR = "out/multilayer_roles/episodes.csv"
    if os.path.exists(MR):
        rows = list(csv.DictReader(open(MR)))
        agg = collections.defaultdict(lambda: [0, 0, 0.0])
        for r in rows:
            d = agg[(r["layers"], r["role"])]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
            d[2] += float(r.get("mutations") or 0)
        for layers, role, want in (("L_near_only", "R_redundant", 0.8625),
                                   ("L_near_non", "R_redundant", 0.8858),
                                   ("L_near_non", "R_policy_ctrl", 0.8385),
                                   ("L_all_three", "R_redundant", 0.8585),
                                   ("L_all_three", "R_policy_all", 0.8783),
                                   ("L_all_three", "R_sense_ctrl", 0.8644)):
            a, n, _m = agg[(layers, role)]
            check(f"角色 {layers}/{role}", a / n, want)
        # 关键：三层分工的"每次成功操作数"必须显著低于三层冗余
        def ops_per_solved(k):
            a, n, m = agg[k]
            return m / a
        check("三层冗余 每次成功操作数", ops_per_solved(("L_all_three", "R_redundant")), 6.34, 0.02)
        check("三层分工 每次成功操作数", ops_per_solved(("L_all_three", "R_policy_all")), 4.53, 0.02)

    # ── 19. 层数（1→7）与角色分工 ──
    NL = "out/nlayer/episodes.csv"
    if os.path.exists(NL):
        rows = list(csv.DictReader(open(NL)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[(int(r["n_layers"]), r["role"])]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for n, role, want in ((1, "redundant", 0.8625), (2, "redundant", 0.8858),
                              (3, "differentiated", 0.8783), (4, "differentiated", 0.8744),
                              (5, "differentiated", 0.8575), (7, "redundant", 0.7933),
                              (7, "differentiated", 0.8438)):
            a, m = agg[(n, role)]
            check(f"层数 {n}/{role}", a / m, want)

    # ── 20. 层数 × 预算（机制检验）──
    NB = "out/nlayer_budget/episodes.csv"
    if os.path.exists(NB):
        rows = list(csv.DictReader(open(NB)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = agg[(int(r["budget"]), int(r["n_layers"]), r["role"])]; d[1] += 1
            d[0] += str(r["success"]) in ("1", "True")
        for b, n, role, want in ((4, 1, "redundant", 0.8452), (4, 7, "redundant", 0.7788),
                                 (4, 7, "differentiated", 0.8250),
                                 (10, 7, "redundant", 0.9715),
                                 (20, 7, "redundant", 0.9996),
                                 (20, 3, "redundant", 0.9998)):
            a, m = agg[(b, n, role)]
            check(f"预算{b}/层数{n}/{role}", a / m, want)
        # 核心断言：7 层冗余相对单层的损失随预算单调收窄
        # （这是整篇论文"多层的代价是预算性的"那条结论的实证）
        def loss(b):
            a1, m1 = agg[(b, 1, "redundant")]
            a7, m7 = agg[(b, 7, "redundant")]
            return (a7 / m7) - (a1 / m1)
        check("7 层损失 @budget4",  loss(4),  -0.0664, 2e-3)
        check("7 层损失 @budget10", loss(10), -0.0120, 2e-3)
        check("7 层损失 @budget20", loss(20), +0.0042, 2e-3)

    # ── 输出 ──
    print("=" * 78)
    print("  综合文档数字核对")
    print("=" * 78)
    print(f"  ✅ 通过 {len(OK)} 项")
    print(f"  ❌ 不符 {len(BAD)} 项")
    print()
    if args.verbose:
        for lab, got, want in OK:
            print(f"    ✅ {lab:<38} {got:.6f}  (文档 {want})")
    if BAD:
        print("  ── 不符明细 ──")
        for lab, got, want in BAD:
            print(f"    ❌ {lab:<38} 实测 {got:.6f}  文档 {want}")
    else:
        print("  ✅ 全部一致——可以据此动笔")
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
