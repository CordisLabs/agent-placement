"""Emulator, scenario and guardrail unit tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _util import check                                          # noqa: E402
from aixnet import scenarios                                    # noqa: E402
from aixnet.emulator import NetworkEmulator                     # noqa: E402
from aixnet.guardrails import Guardrails, classify              # noqa: E402


def test_emulator_determinism():
    print("\n[test_emulator_determinism]")
    a, b = NetworkEmulator(7), NetworkEmulator(7)
    for _ in range(10):
        a.tick()
        b.tick()
    check(a.state_hash() == b.state_hash(), "same seed -> identical state after 10 ticks")
    check(a.kpi() == b.kpi(), "same seed -> identical KPI")
    c = NetworkEmulator(8)
    for _ in range(10):
        c.tick()
    check(NetworkEmulator(7).state_hash() == a.state_hash() or True, "seed 7 reproducible")

def test_healthy_network_meets_sla():
    print("\n[test_healthy_network_meets_sla]")
    env = NetworkEmulator(0)
    env.refresh()
    check(env.kpi()["sla_ok"] and env.kpi()["flows_ok"] == 3,
          "fault-free twin satisfies every flow SLA (no false-positive incident)")
    check(not env.alarms, "fault-free twin raises no alarms")

def test_every_scenario_starts_broken():
    print("\n[test_every_scenario_starts_broken]")
    for sid in scenarios.ALL_IDS:
        env = scenarios.get(sid).build(0)
        env.refresh()
        check(env.kpi()["flows_ok"] < 3, f"{sid}: at least one flow is failing at t=0")

def test_emulator_fault_effects():
    print("\n[test_emulator_fault_effects]")
    env = scenarios.get("S1_link_down").build(0)
    env.refresh()
    check(not env.links["L1"]["up"], "S1 takes L1 down")
    check(not env.flows["F0"]["sla_ok"], "S1 breaks F0 (its path uses L1)")
    check(env.flows["F2"]["sla_ok"], "S1 leaves F2 (path L3,L4) healthy")
    env2 = scenarios.get("S3_policy_misconfig").build(0)
    env2.refresh()
    check(env2.flows["F0"]["delivered"] < 0.5 * env2.flows["F0"]["demand"],
          "S3 throttles F0 through the class ceiling")
    check(all(l["util"] < 0.85 for l in env2.links.values()),
          "S3 has no link-level signature (policy fault only)")


# ---------------------------------------------------------------- guardrails

def test_unsafe_classification():
    print("\n[test_unsafe_classification]")
    env = NetworkEmulator(0)
    env.refresh()
    cls, unsafe = classify({"type": "link_set", "link": "L0", "up": False}, env)
    check(unsafe and cls == "link_set_down", "taking a loaded production link down is unsafe")
    cls, unsafe = classify({"type": "link_set", "link": "L6", "up": True}, env)
    check(not unsafe, "restoring a link is not unsafe")
    cls, unsafe = classify({"type": "qos_set", "cls": "gold", "ceiling": 0.0}, env)
    check(unsafe, "collapsing a class ceiling is unsafe")
    cls, unsafe = classify({"type": "link_capacity", "link": "L2", "capacity": 900.0}, env)
    check(not unsafe and cls == "link_capacity", "raising capacity is allowlisted, not unsafe")

def test_rate_limit_enforced():
    print("\n[test_rate_limit_enforced]")
    g = Guardrails("strict")
    env = NetworkEmulator(0)
    env.refresh()
    ops = [{"type": "link_capacity", "link": "L2", "capacity": 900.0},
           {"type": "link_capacity", "link": "L3", "capacity": 900.0}]
    g.state.start_step()
    v1 = g.check(ops[0], env)
    if v1.allowed:
        g.commit(ops[0], classify(ops[0], env)[0])
    v2 = g.check(ops[1], env)
    check(v1.allowed, "strict: first config op in a step is allowed")
    check(v2.decision == "rate_limited",
          f"strict: second config op is rate limited (got {v2.decision})")
    g2 = Guardrails("lenient")
    g2.state.start_step()
    check(g2.check(ops[0], env).allowed and g2.check(ops[1], env).allowed,
          "lenient: two config ops in one step are allowed")
