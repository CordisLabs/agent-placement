"""Scripted fault scenarios with task-level success definitions.

A scenario = deterministic fault timeline + a success predicate evaluated on the
emulator state. Success is an OPERATIONAL outcome (all flows meet SLA and every
critical alarm is cleared), never a string match on the agent's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .emulator import NetworkEmulator

CRITICAL = "critical"


def _critical_alarms_open(env: NetworkEmulator) -> list[str]:
    return [a["id"] for a in env.alarms if a["severity"] == CRITICAL and not a["acked"]]


def default_success(env: NetworkEmulator) -> bool:
    return bool(env.kpi()["sla_ok"]) and not _critical_alarms_open(env)


def failure_reason(env: NetworkEmulator) -> str:
    bad = [f["id"] for f in env.flows.values() if not f["sla_ok"]]
    open_alarms = _critical_alarms_open(env)
    if bad and open_alarms:
        return f"flows_not_ok={bad} critical_alarms_open={open_alarms}"
    if bad:
        return f"flows_not_ok={bad}"
    if open_alarms:
        return f"critical_alarms_open={open_alarms}"
    return "ok"


@dataclass
class Scenario:
    sid: str
    title: str
    thesis: str
    max_steps: int  # operational MTTR budget in harness steps
    faults: list = field(default_factory=list)  # [(tick, fault_dict), ...]
    success: object = default_success
    # 双胞胎工厂：默认是分组网络。跨领域验证时传入别的域（如 WaterEmulator），
    # harness 其余部分完全不变——这正是"控制器固定、只变域"的落点。
    env_factory: object = None

    def build(self, seed: int) -> NetworkEmulator:
        """Initial twin state: t=0 faults injected, KPIs recomputed and the
        resulting alarms raised - so build() is a complete, replayable snapshot."""
        env = (self.env_factory or NetworkEmulator)(seed)
        for tick, fault in self.faults:
            if tick == 0:
                env.inject(fault)
        env.refresh()
        return env

    def timed_faults(self) -> list:
        return [(t, f) for t, f in self.faults if t > 0]

    def render_success(self, env: NetworkEmulator) -> str:
        return failure_reason(env)


SCENARIOS: dict[str, Scenario] = {}


def _add(sc: Scenario) -> Scenario:
    SCENARIOS[sc.sid] = sc
    return sc


_add(Scenario(
    sid="S1_link_down",
    title="Production span L1 (r1-r2) goes down",
    thesis="Restoring the production span is change-controlled (a dry_run rehearsal "
           "is required first); the alternative is a two-op detour. The retry policy "
           "decides whether the rehearsal costs an extra step.",
    max_steps=4,
    faults=[(0, {"type": "fault_linkset", "link": "L1", "up": False})],
))

_add(Scenario(
    sid="S2_congestion",
    title="L2 (r2-r3) loses 60% of capacity",
    thesis="Single-op fix (add capacity). Measures pure harness overhead: steps, "
           "tool calls, prompt tokens when the task itself is easy.",
    max_steps=3,
    faults=[(0, {"type": "fault_congestion", "link": "L2", "factor": 0.4})],
))

_add(Scenario(
    sid="S3_policy_misconfig",
    title="Gold class rate ceiling mis-set to 50 Mbps (demand 320)",
    thesis="A policy fault with no link-level signature: the class ceiling has to be "
           "read from policy state, not inferred from link KPIs.",
    max_steps=3,
    faults=[(0, {"type": "fault_misconfig", "cls": "gold", "ceiling": 50.0})],
))

_add(Scenario(
    sid="S4_alarm_storm",
    title="L5 (r5-r3) down under a 40-alarm storm",
    thesis="Signal diluted by 40 noise alarms. The fix is one op, so the cost of "
           "context management shows up almost purely in prompt tokens.",
    max_steps=4,
    faults=[(0, {"type": "fault_linkset", "link": "L5", "up": False}),
            (0, {"type": "fault_alarm_storm", "count": 40})],
))

_add(Scenario(
    sid="S5_double_fault",
    title="L1 down AND gold ceiling 50 (compound fault)",
    thesis="Compound fault: two independent fixes (or three ops on the detour route). "
           "Rewards planning; punishes 1-op-per-step loops and missed root causes.",
    max_steps=6,
    faults=[(0, {"type": "fault_linkset", "link": "L1", "up": False}),
            (0, {"type": "fault_misconfig", "cls": "gold", "ceiling": 50.0})],
))

ALL_IDS = list(SCENARIOS.keys())


def get(sid: str) -> Scenario:
    return SCENARIOS[sid]
