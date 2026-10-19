#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""水网的 5 个故障场景：与分组网络的 S1–S5 结构一一对应。

跨领域验证的关键：**只有物理域变了**。
故障字典、步数预算、成功判据、工具集、护栏、控制器、层定义全部相同。

| 分组网络 | 水网 | 故障字典 |
|---|---|---|
| S1 生产链路 L1 下线 | W1 L1 主管关阀，r2/r3 供水中断 | `fault_linkset` |
| S2 L2 容量降 60% | W2 L2 泵组出力降 60% | `fault_congestion` |
| S3 gold 速率上限误设 | W3 gold 供水上限误设 | `fault_misconfig` |
| S4 L5 下线 + 40 告警 | W4 L5 关阀 + 40 条传感器噪声 | 组合 |
| S5 L1 下线 + gold 上限 | W5 L1 关阀 + gold 供水上限 | 复合 |

成功判据（与网络侧同构）：**所有分区压力达标且供水达标，且无未确认严重告警**。
"""
from __future__ import annotations

from .scenarios import SCENARIOS, Scenario, default_success, failure_reason
from .water import WaterEmulator

WATER_SCENARIOS: dict[str, Scenario] = {}


def _add(sc: Scenario) -> Scenario:
    SCENARIOS[sc.sid] = sc
    WATER_SCENARIOS[sc.sid] = sc
    return sc


_add(Scenario(
    sid="W1_trunk_closed",
    title="Water: trunk main L1 closed, r2/r3 supply cut",
    thesis="Closing the trunk isolates the downstream zones. The tank at r0 keeps "
           "draining while the loop is open, so the fix must beat the reserve "
           "level - and supply is restored only after the tank refills.",
    max_steps=4,
    faults=[(0, {"type": "fault_linkset", "link": "L1", "up": False})],
    env_factory=WaterEmulator,
))

_add(Scenario(
    sid="W2_pump_degraded",
    title="Water: pump group on L2 delivers 40% of rated flow",
    thesis="Single-op fix (restore pump capacity). The cheapest scenario, so it "
           "measures harness overhead rather than diagnosis.",
    max_steps=3,
    faults=[(0, {"type": "fault_congestion", "link": "L2", "factor": 0.4})],
    env_factory=WaterEmulator,
))

_add(Scenario(
    sid="W3_supply_cap_misset",
    title="Water: gold-class supply cap mis-set to 50 L/s (demand 320)",
    thesis="A policy fault with no pipe-level signature: the cap has to be read "
           "from policy state, not inferred from pipe telemetry.",
    max_steps=3,
    faults=[(0, {"type": "fault_misconfig", "cls": "gold", "ceiling": 50.0})],
    env_factory=WaterEmulator,
))

_add(Scenario(
    sid="W4_valve_plus_sensor_storm",
    title="Water: L5 valve closed under a 40-alarm sensor storm",
    thesis="Signal diluted by 40 noise alarms. The fix is one op, so the cost of "
           "context management shows up almost purely in prompt tokens.",
    max_steps=4,
    faults=[(0, {"type": "fault_linkset", "link": "L5", "up": False}),
            (0, {"type": "fault_alarm_storm", "count": 40})],
    env_factory=WaterEmulator,
))

_add(Scenario(
    sid="W5_compound",
    title="Water: L1 closed AND gold cap 50 (compound fault)",
    thesis="Compound fault: two independent fixes. Rewards planning; punishes "
           "one-op-per-step loops and missed root causes.",
    max_steps=6,
    faults=[(0, {"type": "fault_linkset", "link": "L1", "up": False}),
            (0, {"type": "fault_misconfig", "cls": "gold", "ceiling": 50.0})],
    env_factory=WaterEmulator,
))

ALL_WATER_IDS = list(WATER_SCENARIOS.keys())

__all__ = ["WATER_SCENARIOS", "ALL_WATER_IDS", "WaterEmulator"]
