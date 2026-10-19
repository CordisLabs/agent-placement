#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排队/服务网络域的七个场景。

Q1–Q5 镜像 S1–S5 的故障结构（**可比**）；Q6–Q7 用**前两域不存在**的故障类型，
以证明本域确实不同。

| 排队域 | 镜像 | 故障 | 可比性 |
|---|---|---|---|
| `Q1_server_down` | S1 | 一台服务器停服 | 同构 |
| `Q2_capacity_drop` | S2 | 处理能力降 60% | 同构 |
| `Q3_cap_misconfig` | S3 | gold 准入上限误设 | 同构 |
| `Q4_down_plus_storm` | S4 | 停服 + 40 条噪声告警 | 同构 |
| `Q5_compound` | S5 | 停服 + 准入上限误设 | 同构 |
| `Q6_arrival_burst` | — | **到达突发** | **前两域没有** |
| `Q7_buffer_shrink` | — | **缓冲压小** | **前两域没有** |
"""
from __future__ import annotations

from .queueing import QueueEmulator
from .scenarios import SCENARIOS, Scenario

QUEUE_SCENARIOS: dict[str, Scenario] = {}


def _add(sc: Scenario) -> Scenario:
    SCENARIOS[sc.sid] = sc
    QUEUE_SCENARIOS[sc.sid] = sc
    return sc


_add(Scenario(
    sid="Q1_server_down",
    title="Queue: server s1 stops serving",
    thesis="s1 is the primary route for two streams; with it down their backlog "
           "grows without bound. Fixing the server is necessary but the queue "
           "takes time to drain, so recovery is not immediate.",
    max_steps=4,
    faults=[(0, {"type": "fault_linkset", "link": "s1", "up": False})],
    env_factory=QueueEmulator,
))

_add(Scenario(
    sid="Q2_capacity_drop",
    title="Queue: server s2 processes at 40% rate",
    thesis="Throughput falls below arrival rate, so the queue integrates upward. "
           "The single-op fix restores service rate; the backlog then drains.",
    max_steps=3,
    faults=[(0, {"type": "fault_backlog", "server": "s2", "q": 600.0}),
            (0, {"type": "fault_server_slow", "server": "s2", "factor": 0.4})],
    env_factory=QueueEmulator,
))

_add(Scenario(
    sid="Q3_cap_misconfig",
    title="Queue: gold admission cap set far too low",
    thesis="Admitted rate is throttled, so gold streams under-deliver despite "
           "ample server capacity. Single-op fix on the policy.",
    max_steps=3,
    faults=[(0, {"type": "fault_misconfig", "cls": "gold", "ceiling": 60.0})],
    env_factory=QueueEmulator,
))

_add(Scenario(
    sid="Q4_down_plus_storm",
    title="Queue: server s3 down plus 40 noise alarms",
    thesis="Diagnosis must survive an alarm storm that buries the one signal "
           "that matters.",
    max_steps=4,
    faults=[(0, {"type": "fault_linkset", "link": "s3", "up": False}),
            (0, {"type": "fault_alarm_storm", "count": 40, "target": "s2"})],
    env_factory=QueueEmulator,
))

_add(Scenario(
    sid="Q5_compound",
    title="Queue: s1 down and gold cap misconfigured",
    thesis="Two independent faults; a single-op policy cannot clear both.",
    max_steps=6,
    faults=[(0, {"type": "fault_linkset", "link": "s1", "up": False}),
            (0, {"type": "fault_misconfig", "cls": "gold", "ceiling": 60.0})],
    env_factory=QueueEmulator,
))

# ── 以下两类**前两域不存在** ─────────────────────────────────
_add(Scenario(
    sid="Q6_arrival_burst",
    title="Queue: arrival rate on J1 surges 3.5x",
    thesis="No server is down and no policy is wrong: the load changed. "
           "Backlog grows even though every component is healthy. "
           "This fault type does not exist in the packet or water domain.",
    max_steps=5,
    faults=[(0, {"type": "fault_backlog", "server": "s2", "q": 900.0}),
            (0, {"type": "fault_arrival_burst", "stream": "J1", "mult": 3.5})],
    env_factory=QueueEmulator,
))

_add(Scenario(
    sid="Q7_buffer_shrink",
    title="Queue: server s2 buffer shrunk to 35%",
    thesis="Capacity is untouched; only the buffer is smaller. Overflow drops "
           "appear even at moderate backlog. "
           "This fault type does not exist in the packet or water domain.",
    max_steps=4,
    faults=[(0, {"type": "fault_backlog", "server": "s2", "q": 800.0}),
            (0, {"type": "fault_buffer_shrink", "server": "s2", "factor": 0.30})],
    env_factory=QueueEmulator,
))

ALL_QUEUE_IDS = list(QUEUE_SCENARIOS.keys())

__all__ = ["QUEUE_SCENARIOS", "ALL_QUEUE_IDS"]
