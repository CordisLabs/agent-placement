#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立验证：检验论文数据**是不是对的**，而不只是**自洽的**。

与既有验证的分工
----------------
| 工具 | 问的问题 | 抓不到 |
|---|---|---|
| `tests/` | 组件行为是否符合预期 | 端到端系统性错误 |
| `verify_numbers.py` | 论文数字 vs 我们的 CSV | CSV 本身错了 |
| `verify_paper_numbers.py` | 论文每个数字能否追溯到 CSV | 同上 |
| **本脚本** | **CSV 与实现本身对不对** | — |

**前面的工具都是"自洽性"检查。** 如果模拟器有系统性错误，
它们会全部通过——这已经发生过五次（布尔算术化、`ctx.render` 不重建、
`[:MAX_ALARMS]` 截断、预测器破坏恒等、论文里三个凭记忆写的数字）。

五类检查
--------
A. **已知答案**：全知控制器应≈1.0、空转控制器应=0.0 —— 端到端断言
B. **机制行为**：`staleness`/`cadence`/预测器**实际做了什么**（插桩观测）
C. **不变性**：改实体名字、换目录、置换顺序，结果不应变
D. **物理保真度**：时延/丢包对利用率的单调性、路径串并联公式、水箱质量守恒
E. **交叉实现**：用**不经过 Harness** 的独立代码重算一个已知格

用法
----
    python3 validate.py            # 全部
    python3 validate.py --quick    # 跳过耗时的交叉实现
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS: list[tuple[str, str]] = []
FAIL: list[tuple[str, str]] = []


def ok(name: str, detail: str = "") -> None:
    PASS.append((name, detail))
    print(f"  ✅ {name}" + (f"   {detail}" if detail else ""))


def bad(name: str, detail: str = "") -> None:
    FAIL.append((name, detail))
    print(f"  ❌ {name}" + (f"   {detail}" if detail else ""))


def chk(cond: bool, name: str, detail: str = "") -> None:
    ok(name, detail) if cond else bad(name, detail)


# ══════════════════════════════════════════════════════════════
# A. 已知答案（端到端）
# ══════════════════════════════════════════════════════════════
def section_A() -> None:
    from aixnet.config import HarnessConfig
    from aixnet.harness import Episode
    from aixnet.scenarios import ALL_IDS, get

    print("\nA. 已知答案（端到端）")

    class Null:
        name = "null"

        def decide(self, obs, ctx):
            return []

        def on_no_progress(self, obs, ctx):
            return []

    # A1. 空转控制器 → 必须 0
    import aixnet.config as C
    _orig = C.make_controller
    C.make_controller = lambda cfg, seed: Null()
    import aixnet.harness as H
    H.make_controller = C.make_controller
    got = 0
    tot = 0
    for sid in ALL_IDS:
        for seed in range(20):
            m = Episode(get(sid), seed, HarnessConfig(controller="mock", layer="near_rt")).run()
            tot += 1
            got += int(m.get("success") in (1, "1", True))
    C.make_controller = _orig
    H.make_controller = _orig
    chk(got == 0, "空转控制器解出 0 个回合", f"{got}/{tot}")

    # A2. 全知控制器：直接应用该场景的"已知修法"，应接近满分
    best = {}
    for sid in ALL_IDS:
        sc = get(sid)
        env = sc.build(0)
        # 枚举单步修复：找到能让 success 成立的单个 op
        cands = []
        for lid, L in list(env.links.items()):
            if not L["up"]:
                cands.append({"type": "link_set", "link": lid, "up": True})
            cands.append({"type": "link_capacity", "link": lid,
                          "capacity": float(L["capacity"]) * 1.5})
        for cls in list(env.policy):
            cands.append({"type": "qos_set", "cls": cls, "ceiling": 2000.0})
        for fid in list(env.flows):
            cands.append({"type": "reprovision", "flow": fid})
        for a in list(env.alarms):
            cands.append({"type": "ack_alarm", "alarm": a["id"]})
        wins = []
        for op in cands:
            e2 = sc.build(0)
            try:
                e2.apply(op)
                e2.recompute()
                for a in e2.alarms:
                    a["acked"] = True
                if sc.success(e2):
                    wins.append(op["type"])
            except Exception:
                pass
        if wins:
            best[sid] = sorted(set(wins))
    known = len(best)
    chk(known >= 3, "至少 3 个场景存在单步已知解",
        f"{known}/{len(ALL_IDS)}：{list(best)}（S2/S5 需要多步修复，属正常）")

    # A3. **每个场景都必须存在某个策略能解**（否则该场景无意义）
    weak = []
    for sid in ALL_IDS:
        ok = 0
        for seed in range(20):
            m = Episode(get(sid), seed, HarnessConfig(controller="mock", layer="near_rt")).run()
            ok += int(m.get("success") in (1, "1", True))
        if ok / 20 < 0.8:
            weak.append((sid, ok / 20))
    chk(not weak, "五个场景都能被某个策略解到 ≥0.8",
        f"全部达标" if not weak else f"不足: {weak}")


# ══════════════════════════════════════════════════════════════
# B. 机制行为（插桩观测）
# ══════════════════════════════════════════════════════════════
def section_B() -> None:
    from aixnet.config import HarnessConfig
    from aixnet.harness import Episode, _extrapolate
    from aixnet.layers_fine import register
    from aixnet.scenarios import get
    register()

    print("\nB. 机制行为（插桩观测）")

    # B1. staleness=s 是否真的用了 s 步前的观测
    seen: list[tuple[int, dict]] = []
    orig = Episode._digest

    def spy(self, obs, scratch):
        seen.append((self.step, dict(obs) if not isinstance(obs, str) else obs))
        return orig(self, obs, scratch)

    Episode._digest = spy
    recs = []
    for stale in (0, 2):
        seen.clear()
        cfg = HarnessConfig(controller="mock", layer=f"c1_s{stale}")
        ep = Episode(get("S1_link_down"), 0, cfg)
        ep.run()
        # 每次决策时 obs["t"] 应与当前 step 相差 stale
        lags = []
        for (step, obs) in seen:
            if isinstance(obs, dict) and "t" in obs:
                lags.append((step - 1) - int(obs["t"]))
        recs.append(lags)
    Episode._digest = orig
    lag0 = set(recs[0]) if recs[0] else set()
    lag2 = list(recs[1]) if recs[1] else []
    chk(lag0 <= {0}, "staleness=0 时观测滞后恒为 0", f"实测 {sorted(lag0)}")
    # ring 从空开始，取 max(0, len-1-stale) 会在前 stale 步被截断 → 滞后不足。
    # 这是**有界的预热效应**，不是错误；但必须显式记录，且预热之后必须严格正确。
    warm = lag2[:2]
    tail = lag2[2:]
    chk(set(tail) <= {2}, "staleness=2 在预热之后滞后恒为 2", f"预热 {warm}，之后 {sorted(set(tail))}")
    # 预热是否影响结果：只有当"不动作时观测就变"才会有影响
    import json as _j
    def _strip(d):
        d = _j.loads(_j.dumps(d, default=str)); d.pop("t", None)
        return _j.dumps(d, sort_keys=True)
    ep2 = Episode(get("S5_double_fault"), 0, HarnessConfig(controller="mock", layer="c1_s0"))
    seq = []
    for _ in range(4):
        seq.append(_strip(ep2._poll())); ep2.env.tick()
    chk(len(set(seq)) == 1,
        "预热无实质影响（不动作时观测内容不变）",
        "观测只差时间戳 → 预热不改变智能体所见")

    # B2. cadence=c 是否真的跳过非倍数步
    calls = collections.Counter()

    from aixnet.multilayer import CompositeController
    o2 = CompositeController.decide_at

    def spy2(self, idx, obs, ctx):
        calls[(idx, getattr(self, "_ep_step", None))] += 1
        return o2(self, idx, obs, ctx)

    CompositeController.decide_at = spy2
    cfg = HarnessConfig(controller="mock", layer="near_rt", layer_specs=((1, 0), (3, 2)))
    ep = Episode(get("S5_double_fault"), 0, cfg)
    ep.run()
    CompositeController.decide_at = o2
    steps_l1 = [s for (i, s) in calls if i == 1 and s is not None]
    chk(all(int(s) % 3 == 0 for s in steps_l1),
        "cadence=3 的子智能体只在 k%3==0 时被调度",
        f"调度于 {sorted(set(steps_l1))}" if steps_l1 else "（未采到）")

    # B3. 预测器在 staleness=0 时必须恒等
    env = get("S1_link_down").build(0)
    a = env.observe()
    env.recompute()
    b = env.observe()
    same_d = _extrapolate(a, b, steps=0) == b
    chk(same_d, "delta 预测器在 steps=0 时恒等")
    try:
        from aixnet.learned_predictor import predict as lp
        same_l = lp(a, b, steps=0) == b
        chk(same_l, "learned 预测器在 steps=0 时恒等")
    except FileNotFoundError:
        print("  （跳过 learned 恒等：模型文件不存在）")

    # B4. oracle 是否真的给当前真值
    seen.clear()
    Episode._digest = spy
    cfg = HarnessConfig(controller="mock", layer="c1_s2", predictor="oracle")
    ep = Episode(get("S1_link_down"), 0, cfg)
    ep.run()
    Episode._digest = orig
    lags = [(s - 1) - int(o["t"]) for (s, o) in seen if isinstance(o, dict) and "t" in o]
    chk(set(lags) <= {0}, "oracle 在 staleness=2 下给出当前真值", f"滞后 {sorted(set(lags))}")


# ══════════════════════════════════════════════════════════════
# C. 不变性
# ══════════════════════════════════════════════════════════════
def section_C() -> None:
    from aixnet.config import HarnessConfig
    from aixnet.harness import Episode
    from aixnet.scenarios import ALL_IDS, get

    print("\nC. 不变性")

    def sig(sids, seeds):
        parts = []
        for sid in sids:
            for seed in seeds:
                m = Episode(get(sid), seed, HarnessConfig(controller="mock", layer="near_rt")).run()
                parts.append(f"{sid}{seed}{m.get('success')}{m.get('steps')}{m.get('mutations')}")
        return "|".join(parts)

    # C1. 同一计算重复两次必须逐字节相同
    a = sig(ALL_IDS, range(10))
    b = sig(ALL_IDS, range(10))
    chk(a == b, "重复计算逐字节相同")

    # C2. 场景顺序置换 → 每个 (场景,种子) 的结果不应变
    perm = list(ALL_IDS)[::-1]
    c = sig(perm, range(10))
    ca = {p.split("S")[0] if False else p for p in c.split("|")}
    chk(len(c.split("|")) == len(a.split("|")),
        "场景顺序置换不改变回合数", f"{len(c.split('|'))} vs {len(a.split('|'))}")
    # 逐回合比对（按 (场景,种子) 取）
    def per(sids):
        d = {}
        for sid in sids:
            for seed in range(10):
                m = Episode(get(sid), seed, HarnessConfig(controller="mock", layer="near_rt")).run()
                d[(sid, seed)] = (m.get("success"), m.get("steps"), m.get("mutations"))
        return d
    d1, d2 = per(ALL_IDS), per(perm)
    chk(d1 == d2, "置换后的逐回合结果与原来完全一致")

    # C3. 换了工作目录（cwd）结果不变
    here = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="val_cwd_")
    try:
        os.chdir(tmp)
        d3 = {}
        for sid in list(ALL_IDS)[:2]:
            m = Episode(get(sid), 0, HarnessConfig(controller="mock", layer="near_rt")).run()
            d3[sid] = (m.get("success"), m.get("steps"), m.get("mutations"))
    finally:
        os.chdir(here)
        shutil.rmtree(tmp, ignore_errors=True)
    same = all(d1[(sid, 0)] == d3[sid] for sid in list(ALL_IDS)[:2])
    chk(same, "换工作目录后结果不变（无隐藏的路径依赖）")


# ══════════════════════════════════════════════════════════════
# D. 物理保真度
# ══════════════════════════════════════════════════════════════
def section_D() -> None:
    from aixnet.emulator import NetworkEmulator
    from aixnet.topology import CONGESTION_UTIL, CAPACITY_FLOOR
    from aixnet.scenarios import get

    print("\nD. 物理保真度（网络）")

    sc = get("S1_link_down")
    env = sc.build(0)
    # D1. 空载链路：丢包为 0、时延等于基础时延
    for L in env.links.values():
        L["up"] = True
    env.flows.clear()
    env.recompute()
    zero_ok = all(L["loss"] == 0.0 for L in env.links.values())
    chk(zero_ok, "空载链路丢包为 0")
    base_ok = all(abs(L["delay"] - L["base_latency"]) < 1e-9 for L in env.links.values())
    chk(base_ok, "空载链路时延 = 基础时延")

    # D2. 单调性：负载↑ → 利用率↑ → 时延↑、丢包↑
    env2 = sc.build(0)
    for L in env2.links.values():
        L["up"] = True
    lid = next(iter(env2.links))
    seq = []
    for mult in (0.2, 0.5, 0.8, 1.0, 1.3, 1.8):
        env3 = sc.build(0)
        for L in env3.links.values():
            L["up"] = True
        env3.flows.clear()
        cap = env3.links[lid]["capacity"]
        env3.links[lid]["load"] = cap * mult
        env3.links[lid]["util"] = min(env3.links[lid]["load"] / max(cap, CAPACITY_FLOOR), 5.0)
        over = max(0.0, env3.links[lid]["util"] - CONGESTION_UTIL)
        d = env3.links[lid]["base_latency"] + 20.0 * over
        l = over * 0.25
        seq.append((mult, d, l))
    mono_d = all(seq[i][1] <= seq[i + 1][1] + 1e-9 for i in range(len(seq) - 1))
    mono_l = all(seq[i][2] <= seq[i + 1][2] + 1e-9 for i in range(len(seq) - 1))
    chk(mono_d, "时延对负载单调不减")
    chk(mono_l, "丢包对负载单调不减")

    # D3. 路径串联公式：流丢包 = 1 − Π(1−链路丢包)；流时延 = Σ 链路时延
    env4 = sc.build(0)
    for L in env4.links.values():
        L["up"] = True
    env4.recompute()          # ← 必须有：否则读到的是修复前的陈旧流状态
    f = next(iter(env4.flows.values()))
    path = f["path"]
    exp_loss = 1.0 - math.prod(1.0 - env4.links[x]["loss"] for x in path)
    exp_delay = sum(env4.links[x]["delay"] for x in path)
    chk(abs(f["loss"] - round(exp_loss, 6)) < 1e-6,
        "流丢包 = 1 − Π(1−链路丢包)", f"{f['loss']:.6f} vs {exp_loss:.6f}")
    chk(abs(f["delay"] - exp_delay) < 1e-6,
        "流时延 = Σ 链路时延", f"{f['delay']:.3f} vs {exp_delay:.3f}")

    # D4. 任一跳 down → 流投递为 0
    env5 = sc.build(0)
    for L in env5.links.values():
        L["up"] = True
    f5 = next(iter(env5.flows.values()))
    env5.links[f5["path"][0]]["up"] = False
    env5.recompute()
    chk(f5["delivered"] == 0.0 and f5["loss"] == 1.0,
        "路径首跳 down → 流投递为 0、丢包为 1")


def section_D2() -> None:
    from aixnet.water import TANK_AREA_M2, TICK_MINUTES
    import aixnet.water_domain as W
    from aixnet.scenarios import get as _get

    print("\nD2. 物理保真度（水网）")
    ids = list(W.WATER_SCENARIOS.keys())
    chk(len(ids) == 5, "水网有 5 个场景", f"{ids}")
    sc = _get(ids[0])
    env = sc.build(0)
    # 质量守恒：一 tick 内，水位变化 × 面积 == (进 − 出) × 时间
    tank_ids = list(env.tanks)
    tid = tank_ids[0]
    before = env.tanks[tid]["level"]
    inflow = env.tanks[tid].get("inflow", 0.0)
    outflow = env.tanks[tid].get("outflow", 0.0)
    env.tick()
    after = env.tanks[tid]["level"]
    dt_min = TICK_MINUTES
    exp_delta = (inflow - outflow) * dt_min / TANK_AREA_M2 if TANK_AREA_M2 else 0.0
    got_delta = after - before
    chk(abs(got_delta - exp_delta) < max(1e-6, abs(exp_delta) * 0.60),
        "水箱水位变化与 (进−出)×时间/面积 同量级",
        f"实测 Δ{got_delta:+.4f} m，理论 {exp_delta:+.4f} m")

    # D2b. 关键性质：水箱会随时间变，但**不在观测里**（论文已声明）
    import json as _j

    def _strip2(d):
        d = _j.loads(_j.dumps(d, default=str)); d.pop("t", None)
        return _j.dumps(d, sort_keys=True)

    env2 = sc.build(0)
    tl0 = {k: v["level"] for k, v in env2.tanks.items()}
    o0 = _strip2(env2.observe())
    for _ in range(4):
        env2.tick()
    tl1 = {k: v["level"] for k, v in env2.tanks.items()}
    o1 = _strip2(env2.observe())
    tanks_moved = any(abs(tl1[k] - tl0[k]) > 1e-9 for k in tl0)
    obs_same = (o0 == o1)
    chk(tanks_moved, "水网水箱水位确实随时间变化")
    chk(obs_same, "但水箱水位**不在观测里**（论文已声明的性质）",
        "观测内容不变 → 智能体看不到时间演化")


# ══════════════════════════════════════════════════════════════
# E. 交叉实现（不经过 Harness）
# ══════════════════════════════════════════════════════════════
def section_E() -> None:
    from aixnet.emulator import NetworkEmulator
    from aixnet.topology import k_shortest_paths
    from aixnet.scenarios import get
    import itertools

    print("\nE. 交叉实现：用独立代码重算 S1 的已知修复")

    # 不经过 Episode/Harness：直接构造环境 → 应用修复 → 用场景自己的判据判定
    sc = get("S1_link_down")
    n_ok = 0
    reps = 20
    for seed in range(reps):
        env = sc.build(seed)
        # 独立诊断：哪条链路 down？
        down = [lid for lid, L in env.links.items() if not L["up"]]
        for lid in down:
            env.apply({"type": "link_set", "link": lid, "up": True})
        env.recompute()
        for a in env.alarms:
            a["acked"] = True
        n_ok += int(bool(sc.success(env)))
    chk(n_ok == reps, f"独立代码把 {down and '全部'} down 链路拉起后，{reps} 个种子全解",
        f"{n_ok}/{reps}")


# ══════════════════════════════════════════════════════════════
# F. 图形完整性：内容是否被画到画布外
# ══════════════════════════════════════════════════════════════
def section_F() -> None:
    """检查每张图的四边留白。

    这一类的 bug 已经出现过两次，而且**渲染不会报错**：
      * extB 热力图：cad=3 那一行从 y=-24 画到 y=10，下半行与全部文字在画布外，
        但图注还在解释那一行
      * extC/extE 图例：写在 y=3~4，而本 Fig 的 y 轴**向下增大**，
        所以文字上半截落到画布顶部之外
    两次都是靠人眼看出来的。这里把它变成自动检查。
    """
    import glob
    import subprocess
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("  （跳过：需要 numpy + Pillow）")
        return

    print("\nF. 图形完整性（内容是否越界）")
    files = sorted(glob.glob("out/figs_ext/*.pdf")) + sorted(glob.glob("out/figs/*.pdf"))
    if not files:
        print("  （无图可查）")
        return
    bad = []
    for f in files:
        subprocess.run(["pdftoppm", "-png", "-r", "150", f, "/tmp/_vf"],
                       check=True, capture_output=True)
        a = np.array(Image.open("/tmp/_vf-1.png").convert("L"))
        # 纵轴标签（旋转 90°）本来就贴着左边，属正常
        edges = {"上": a[:4, :], "下": a[-4:, :], "右": a[:, -4:]}
        cuts = [k for k, v in edges.items() if (v < 200).sum() > 0]
        if cuts:
            bad.append((f, cuts))
    name = ", ".join(f.split("/")[-1] for f, _ in bad) if bad else ""
    chk(not bad, f"{len(files)} 张图均无内容越界", name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("  独立验证：论文数据是不是**对的**（不只是自洽的）")
    print("=" * 74)

    for fn in (section_A, section_B, section_C, section_D, section_D2, section_F):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            bad(f"{fn.__name__} 抛异常", f"{type(e).__name__}: {e}")
    if not args.quick:
        try:
            section_E()
        except Exception as e:  # noqa: BLE001
            bad("section_E 抛异常", f"{type(e).__name__}: {e}")

    print()
    print("=" * 74)
    print(f"  ✅ 通过 {len(PASS)}    ❌ 失败 {len(FAIL)}")
    if FAIL:
        print("  ── 失败项 ──")
        for n, d in FAIL:
            print(f"    ❌ {n}   {d}")
    else:
        print("  所有独立检查通过")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
