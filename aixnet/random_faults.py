#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随机故障流：替代 5 个脚本化场景，回应"故障集存在手册式解法"的批评。

动机（来自主论文 Limitations）：
  "The five faults are scripted with fixed budgets (4/3/3/4/6) and admit a
   runbook-shaped solution, which is why the non-LLM baselines win."

设计：
  * 每个"流"由 1–3 个故障组成，类型/目标/参数**随机采样**（按固定权重），
    用固定种子保证可复现。
  * **可解性判据**：链路下线的组合必须让每个业务流**仍有可用路径**
    （用 topology.k_shortest_paths 在"仅剩在线链路"的子图上判定）。
    不可解的流直接重新采样，绝不产出。
  * 步数预算随故障数缩放：`2 + 2 × 故障数`
    （对照脚本场景：1 个故障→4 步，2 个→6 步，与 S1/S5 一致）。
  * 故障全部注入在 t=0，与脚本场景一致（保证 build() 是可回放的快照）。

用法：
    from aixnet.random_faults import generate_set
    scen = generate_set(30, seed=20260930)      # 30 条随机流
"""
from __future__ import annotations

import random

from .scenarios import Scenario
from .topology import LINK_SPECS, FLOW_SPECS, k_shortest_paths

LINKS = [s[0] for s in LINK_SPECS]
LINK_ENDS = {s[0]: (s[1], s[2]) for s in LINK_SPECS}
CLASSES = ["gold", "silver", "bronze"]

# (类型, 权重, 说明)
FAULT_KINDS = [
    ("fault_linkset",    40, "生产链路下线"),
    ("fault_congestion", 30, "链路容量骤降"),
    ("fault_misconfig",  20, "策略上限被误设"),
    ("fault_alarm_storm", 10, "告警风暴（噪声）"),
]

# 各故障类型的参数采样空间
CONGESTION_FACTORS = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
MISCONFIG_CEILINGS = [40.0, 50.0, 80.0, 100.0, 150.0, 200.0]
STORM_COUNTS = (10, 60)


def _live_links(down: set[str]) -> dict:
    """拓扑字典，但不含被下线的链路（供路径搜索用）。"""
    out = {}
    for lid, (a, b), cap, delay, loss, _extra in [
            (s[0], (s[1], s[2]), s[3], s[4], s[5], s[6]) for s in LINK_SPECS]:
        if lid in down:
            continue
        out[lid] = {"id": lid, "a": a, "b": b, "capacity": cap, "delay": delay,
                    "loss": loss}
    return out


def _flow_has_path(src: str, dst: str, down: set[str]) -> bool:
    """在"仅剩在线链路"的子图上，src→dst 是否仍有路径。"""
    links = _live_links(down)
    if not links:
        return False
    try:
        return len(k_shortest_paths(links, src, dst, k=1)) > 0
    except Exception:
        return False


def all_flows_connected(down: set[str]) -> bool:
    """所有业务流是否都还有路径——可解性的必要条件。"""
    for _fid, src, dst, _cls, _demand, _delay in FLOW_SPECS:
        if not _flow_has_path(src, dst, down):
            return False
    return True


def _actually_breaks(faults: list, env_factory=None) -> bool:
    """故障必须真的破坏成功判据——否则任务一開始就是"已完成"，测不到任何东西。

    判据：注入后 default_success(env) 为 False（SLA 不满足 或 有未确认严重告警）。
    env_factory 允许在**别的物理域**上做同一个检验（跨域对称性要求）。
    """
    from .emulator import NetworkEmulator
    from .scenarios import default_success
    try:
        env = (env_factory or NetworkEmulator)(0)
        for _tick, f in faults:
            env.inject(f)
        env.refresh()
        return not default_success(env)
    except Exception:
        return False


def generate(stream_id: str, seed: int, max_faults: int = 3,
             require_break: bool = True, env_factory=None,
             prefix: str = "R") -> Scenario:
    """生成一条随机故障流。同 (stream_id, seed) 必得同一结果。

    require_break=True 时，会重采样直到故障真的破坏成功判据（最多 60 次）。
    """
    r = random.Random(seed)
    n_faults = r.choices([1, 2, 3], weights=[50, 35, 15])[0]
    n_faults = min(n_faults, max_faults)

    kinds = [k for k, _w, _d in FAULT_KINDS]
    weights = [w for _k, w, _d in FAULT_KINDS]

    faults: list[tuple] = []
    down: set[str] = set()
    used_cls: set[str] = set()
    labels: list[str] = []

    guard = 0
    while len(faults) < n_faults and guard < 200:
        guard += 1
        kind = r.choices(kinds, weights=weights)[0]

        if kind == "fault_linkset":
            cand = [l for l in LINKS if l not in down]
            if not cand:
                continue
            link = r.choice(cand)
            # 可解性判据：加上这条下线后，所有流仍须连通
            if not all_flows_connected(down | {link}):
                continue
            down.add(link)
            faults.append((0, {"type": "fault_linkset", "link": link, "up": False}))
            labels.append(f"{link} down")

        elif kind == "fault_congestion":
            link = r.choice(LINKS)
            if link in down:
                continue
            factor = r.choice(CONGESTION_FACTORS)
            faults.append((0, {"type": "fault_congestion", "link": link,
                               "factor": factor}))
            labels.append(f"{link} -{int((1-factor)*100)}% capacity")

        elif kind == "fault_misconfig":
            cand = [c for c in CLASSES if c not in used_cls]
            if not cand:
                continue
            cls = r.choice(cand)
            ceiling = r.choice(MISCONFIG_CEILINGS)
            used_cls.add(cls)
            faults.append((0, {"type": "fault_misconfig", "cls": cls,
                               "ceiling": ceiling}))
            labels.append(f"{cls} ceiling={ceiling:.0f}")

        elif kind == "fault_alarm_storm":
            count = r.randint(*STORM_COUNTS)
            faults.append((0, {"type": "fault_alarm_storm", "count": count}))
            labels.append(f"{count} noise alarms")

    if not faults:                       # 极端兜底：至少给一个可解的链路下线
        for link in LINKS:
            if all_flows_connected({link}):
                faults.append((0, {"type": "fault_linkset", "link": link, "up": False}))
                labels.append(f"{link} down")
                break

    # 有效性门：故障必须真的破坏成功判据，否则重采样（最多 60 次）
    if require_break:
        for _ in range(60):
            if _actually_breaks(faults, env_factory):
                break
            r2 = random.Random(seed * 31 + len(faults) * 7 + _)
            extra_seed = seed + (_ + 1) * 7919
            alt = generate(stream_id, extra_seed, max_faults=max_faults,
                           require_break=False, env_factory=env_factory)
            faults, labels = list(alt.faults), [alt.title[22:]]
        # 兜底：找一个确实会破坏的链路下线
        if not _actually_breaks(faults, env_factory):
            for link in LINKS:
                cand = [(0, {"type": "fault_linkset", "link": link, "up": False})]
                if all_flows_connected({link}) and _actually_breaks(cand, env_factory):
                    faults, labels = cand, [f"{link} down"]
                    break

    budget = 2 + 2 * len(faults)         # 1 故障→4 步，2→6，3→8（对齐 S1/S5）
    return Scenario(env_factory=env_factory,
        sid=stream_id,
        title="random fault stream: " + " + ".join(labels),
        thesis="Sampled from the fault distribution; solvability enforced "
               "(every flow keeps a path). No runbook-shaped shortcut is assumed.",
        max_steps=budget,
        faults=faults,
    )


def generate_set(n: int, seed: int = 0, prefix: str = "R",
                 env_factory=None) -> list[Scenario]:
    """生成 n 条互不相同的随机流（不同种子）。

    env_factory 用于跨域：传 WaterEmulator 即得**同一套生成器**产出的水网随机流，
    从而两域的实验设计完全对称。
    """
    out = []
    for i in range(n):
        out.append(generate(f"{prefix}{i:02d}", seed + i * 104729,
                            env_factory=env_factory, prefix=prefix))
    return out


def register(scenarios: list[Scenario]) -> dict:
    """把生成的流注册进全局 SCENARIOS，让现有 runner 能按 id 取用。"""
    from .scenarios import SCENARIOS
    for sc in scenarios:
        SCENARIOS[sc.sid] = sc
    return SCENARIOS


__all__ = ["generate", "generate_set", "register", "all_flows_connected",
           "FAULT_KINDS", "LINKS", "CLASSES"]
