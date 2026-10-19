#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解耦网格：把「行动节奏」与「遥测新鲜度」分开。

## 为什么需要

主论文的三层定义里，cadence 与 staleness **完全共线**：

    near_rt     cadence=1  staleness=0
    non_rt      cadence=2  staleness=1      ← cadence = staleness + 1
    smo_intent  cadence=3  staleness=2      ← 永远差 1

这意味着**主论文无法区分**下面两件事对成功率各自贡献多少：
    (a) 智能体**能多频繁地行动**（cadence）
    (b) 它决策所依据的**信息有多新鲜**（staleness）

两者被绑在同一条对角线上，任何观测到的效应都是二者之和。

## 本模块做什么

把格点铺开成完整的 **3×3 = 9 格**：

              staleness=0   staleness=1   staleness=2
    cadence=1   c1_s0 *       c1_s1         c1_s2
    cadence=2   c2_s0         c2_s1 *       c2_s2
    cadence=3   c3_s0         c3_s1         c3_s2 *

（* = 主论文已有的三点，落在对角线上）

这样就能分别读出：
    · **只变新鲜度**（cadence 固定为 1）：c1_s0 → c1_s1 → c1_s2
    · **只变节奏**（staleness 固定为 0）：c1_s0 → c2_s0 → c3_s0
    · 以及对角线与非对角线格点的比较

## 用法

    from aixnet.layers_fine import FINE_LAYERS, register
    register()                       # 把 9 格注入全局 LAYERS
    run_sweep(scenarios=ALL_IDS, configs=..., ...)
"""
from __future__ import annotations

from .config import LAYERS

# 主论文原有的三点（在 3×3 网格的对角线上）
DIAGONAL = {"c1_s0": "near_rt", "c2_s1": "non_rt", "c3_s2": "smo_intent"}

# 完整 3×3 网格
FINE_LAYERS: dict[str, dict] = {}
for _c in (1, 2, 3):
    for _s in (0, 1, 2):
        _name = f"c{_c}_s{_s}"
        FINE_LAYERS[_name] = {
            "cadence": _c,
            "staleness": _s,
            "label": f"cadence={_c}, staleness={_s}",
        }

# 便于阅读的分组
ROWS_CADENCE = {                     # 固定 staleness=0，只变节奏
    "cadence_sweep": ["c1_s0", "c2_s0", "c3_s0"],
}
COLS_STALENESS = {                   # 固定 cadence=1，只变新鲜度
    "staleness_sweep": ["c1_s0", "c1_s1", "c1_s2"],
}


def register() -> dict:
    """把 9 格注入全局 LAYERS（幂等）。返回更新后的 LAYERS。"""
    for name, spec in FINE_LAYERS.items():
        LAYERS[name] = dict(spec)
    return LAYERS


def is_registered() -> bool:
    return all(k in LAYERS for k in FINE_LAYERS)


__all__ = ["FINE_LAYERS", "DIAGONAL", "ROWS_CADENCE", "COLS_STALENESS",
           "register", "is_registered"]


def fine_configs(controller: str = "mock") -> list:
    """构造覆盖 9 格的完整因子配置。

    runner.FACTORS["layer"] 只含原三层（硬编码），所以解耦实验必须显式传 configs。
    其余 5 个因子与原实验完全一致（2×2×2×2×3 = 48），只是 layer 换成 9 格。
    """
    import itertools
    from .config import HarnessConfig

    axes = [
        ("tool_granularity", ["fine", "coarse"]),
        ("state_management", ["scratchpad", "stateless"]),
        ("control", ["reactive", "planner"]),
        ("guardrails", ["lenient", "strict"]),
        ("retry", ["none", "retry", "retry_rollback"]),
    ]
    out = []
    for layer in FINE_LAYERS:
        for combo in itertools.product(*(vals for _k, vals in axes)):
            kw = dict(zip([k for k, _v in axes], combo))
            out.append(HarnessConfig(controller=controller, layer=layer, **kw))
    return out


__all__ = ["FINE_LAYERS", "DIAGONAL", "ROWS_CADENCE", "COLS_STALENESS",
           "register", "is_registered", "fine_configs"]


# ── 可扩到 N 层的对角梯子 ──────────────────────────────────────────
# 层 i = (cadence=i+1, staleness=i)，即原三层 (1,0)(2,1)(3,2) 的自然延伸。
# ⚠️ 回合预算只有 3–6 步，所以 cadence > 6 的层整回合只会动作一次
#    —— 这是真实约束，不是 bug，但要看清楚。
DIAGONAL_LADDER = [(i + 1, i) for i in range(7)]     # (1,0) … (7,6)


def ladder_specs(n: int) -> tuple:
    """取前 n 层的 (cadence, staleness)。"""
    return tuple(DIAGONAL_LADDER[:n])


def register_ladder(n_max: int = 7) -> dict:
    """把梯子上的层注册进 LAYERS，供 HarnessConfig(layer=...) 使用。"""
    for i, (c, st) in enumerate(DIAGONAL_LADDER[:n_max]):
        LAYERS[f"d{i}_c{c}"] = {"cadence": c, "staleness": st,
                                "label": f"diagonal layer {i} (cad={c}, stale={st})"}
    return LAYERS


__all__ += ["DIAGONAL_LADDER", "ladder_specs", "register_ladder"]
