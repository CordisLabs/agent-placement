"""End-to-end harness tests: guardrails, atomicity, retry, rollback, determinism, tracing and replay."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _util import check                                          # noqa: E402
from aixnet import scenarios                                    # noqa: E402
from aixnet.harness import Episode, HarnessConfig               # noqa: E402
from aixnet.scenarios import get as get_scenario, ALL_IDS       # noqa: E402
from aixnet.trace import read_trace, replay                     # noqa: E402

class _Scripted:
    """A controller that emits a fixed intent list - used to test the harness."""

    def __init__(self, intents):
        self.intents = intents
        self.calls = 0

    def decide(self, obs, ctx):
        self.calls += 1
        return list(self.intents)


def _episode(sid, seed, cfg, intents=None, trace=None):
    ep = Episode(scenarios.get(sid), seed, cfg, trace_path=trace)
    if intents is not None:
        ep.controller = _Scripted(intents)
        ep.controller.__class__.__name__ = "Scripted"
    return ep


# ---------------------------------------------------------------- emulator

def test_guardrails_block_unsafe_end_to_end():
    print("\n[test_guardrails_block_unsafe_end_to_end]")
    bad = [{"kind": "op", "op": {"type": "link_set", "link": "L0", "up": False}}]
    for mode, expect_up in (("lenient", False), ("strict", True)):
        ep = _episode("S2_congestion", 0, HarnessConfig(guardrails=mode, retry="none"), bad)
        m = ep.run()
        check(ep.env.links["L0"]["up"] is expect_up,
              f"{mode}: L0 admin state is {'up' if expect_up else 'down'} after the attempt")
        check(m["blocked_unsafe"] >= (1 if mode == "strict" else 0),
              f"{mode}: blocked_unsafe={m['blocked_unsafe']}")
        check(m["applied_unsafe"] == (0 if mode == "strict" else m["mutations"] - 0),
              f"{mode}: applied_unsafe={m['applied_unsafe']} out of "
              f"{m['mutations']} committed ops")

def test_coarse_batch_is_atomic_and_fine_is_not():
    print("\n[test_coarse_batch_is_atomic_and_fine_is_not]")
    batch = [{"kind": "op", "op": {"type": "qos_set", "cls": "gold", "ceiling": 1000.0}},
             {"kind": "op", "op": {"type": "link_set", "link": "L0", "up": False}}]
    ep = _episode("S3_policy_misconfig", 0,
                  HarnessConfig(tool_granularity="coarse", guardrails="strict",
                                retry="none"), batch)
    m = ep.run()
    check(ep.env.policy["gold"]["ceiling"] == 50.0,
          "coarse: an allowed change in a batch with a rejected change is NOT applied")
    check(m["atomic_rejections"] >= 1, "coarse: the atomic rejection is counted")
    ep2 = _episode("S3_policy_misconfig", 0,
                   HarnessConfig(tool_granularity="fine", guardrails="strict",
                                 retry="none"), batch)
    ep2.run()
    check(ep2.env.policy["gold"]["ceiling"] == 1000.0,
          "fine: the same batch applies the allowed change and drops the rejected one")

def test_retry_rollback_undoes_regression():
    print("\n[test_retry_rollback_undoes_regression]")
    hurt = [{"kind": "op", "op": {"type": "link_set", "link": "L4", "up": False}}]
    ep = _episode("S1_link_down", 0,
                  HarnessConfig(guardrails="lenient", retry="retry_rollback"), hurt)
    m = ep.run()
    check(ep.env.links["L4"]["up"] is True, "retry_rollback: regressed config is restored")
    check(m["rollbacks"] >= 1, f"retry_rollback: rollback counted ({m['rollbacks']})")
    ep2 = _episode("S1_link_down", 0,
                   HarnessConfig(guardrails="lenient", retry="none"), hurt)
    m2 = ep2.run()
    check(ep2.env.links["L4"]["up"] is False, "without rollback the harmful change persists")
    check(m2["rollbacks"] == 0, "without rollback the counter stays 0")

def test_retry_rehearses_within_the_step():
    print("\n[test_retry_rehearses_within_the_step]")
    sc = scenarios.get("S1_link_down")
    a = [Episode(sc, s, HarnessConfig(guardrails="strict", retry="none")).run()
         for s in range(12)]
    b = [Episode(sc, s, HarnessConfig(guardrails="strict", retry="retry")).run()
         for s in range(12)]
    ma = sum(x["steps"] for x in a) / len(a)
    mb = sum(x["steps"] for x in b) / len(b)
    sa = sum(x["success"] for x in a) / 12
    sb = sum(x["success"] for x in b) / 12
    check(ma > mb, f"strict+no_retry pays an extra rehearsal step "
                   f"({ma:.2f} > {mb:.2f} mean steps, n=12)")
    check(sb >= sa, f"and does not resolve fewer episodes (retry={sb:.2f} >= "
                    f"no_retry={sa:.2f} success rate, n=12)")


# ---------------------------------------------------------------- determinism

def test_episode_determinism():
    print("\n[test_episode_determinism]")
    cfg = HarnessConfig()
    for sid in scenarios.ALL_IDS:
        a = Episode(scenarios.get(sid), 3, cfg).run()
        b = Episode(scenarios.get(sid), 3, cfg).run()
        ka = {k: v for k, v in a.items() if k != "wall_ms"}
        kb = {k: v for k, v in b.items() if k != "wall_ms"}
        check(ka == kb, f"{sid}: identical metrics for identical (seed, config)")
        check(a["final_state_hash"] == b["final_state_hash"], f"{sid}: identical state hash")

def test_trace_schema_and_replay():
    print("\n[test_trace_schema_and_replay]")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        m = Episode(scenarios.get("S5_double_fault"), 5, HarnessConfig(),
                    trace_path=path).run()
        tr = read_trace(path)
        check(tr["header"]["rec"] == "episode_start", "trace has an episode_start header")
        for field in ("episode_id", "scenario", "seed", "config", "success_def",
                      "max_steps", "initial_state_hash", "faults"):
            check(field in tr["header"], f"header documents '{field}'")
        check(len(tr["steps"]) == m["steps"], "one step record per harness step")
        for field in ("step", "t", "tool_calls", "verdicts", "ops_applied", "kpi",
                      "state_hash"):
            check(field in tr["steps"][0], f"step record documents '{field}'")
        check("metrics" in tr["end"] and "op_timeline" in tr["end"],
              "episode_end carries metrics and the op timeline")
        res = replay(path)
        check(res["ok"], f"replay reproduces the final state hash ({res.get('error')})")
        check(res["replayed_success"] == bool(m["success"]),
              "replay reproduces the success verdict")

def test_replay_after_mutation_is_detected():
    print("\n[test_replay_after_mutation_is_detected]")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        Episode(scenarios.get("S1_link_down"), 2, HarnessConfig(), trace_path=path).run()
        lines = open(path, encoding="utf-8").read().splitlines()
        mut = []
        for ln in lines:
            rec = json.loads(ln)
            if rec.get("rec") == "episode_end":
                rec["final_state_hash"] = "deadbeefdeadbeef"
            mut.append(json.dumps(rec))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(mut) + "\n")
        check(not replay(path)["ok"], "a tampered trace fails replay verification")


# ---------------------------------------------------------------- tasks/controllers
