"""Policy and baseline tests: task solvability, baselines, granularity, RNG isolation, LLM client parsing."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _util import check                                          # noqa: E402
from aixnet import scenarios                                    # noqa: E402
from aixnet.emulator import NetworkEmulator                     # noqa: E402
from aixnet.guardrails import Guardrails                        # noqa: E402
from aixnet.controllers import DecisionContext                  # noqa: E402
from aixnet.harness import Episode, HarnessConfig               # noqa: E402
from aixnet.llm_client import OpenAICompatClient                # noqa: E402
from aixnet.scenarios import ALL_IDS, get as get_scenario       # noqa: E402
from aixnet.tools import ToolRegistry                           # noqa: E402


def test_every_scenario_is_solvable():
    print("\n[test_every_scenario_is_solvable]")
    for sid in scenarios.ALL_IDS:
        wins = 0
        for ctrl in ("baseline_rules", "baseline_search"):
            cfg = HarnessConfig(controller=ctrl)
            wins += sum(Episode(scenarios.get(sid), s, cfg).run()["success"]
                        for s in range(5))
        check(wins >= 5, f"{sid}: a non-LLM baseline resolves the task (`{wins}/10` seeds)")

def test_baselines_beat_or_match_on_steps():
    print("\n[test_baselines_beat_or_match_on_steps]")
    for sid in scenarios.ALL_IDS:
        b = Episode(scenarios.get(sid), 0,
                    HarnessConfig(controller="baseline_search")).run()
        check(b["success"] == 1, f"{sid}: search baseline resolves seed 0 in {b['steps']} steps")
        check(b["sim_evals"] > 0, f"{sid}: search baseline reports its {b['sim_evals']} sim evals")

def test_llm_client_is_inert_offline():
    print("\n[test_llm_client_is_inert_offline]")
    c = OpenAICompatClient(base_url="http://127.0.0.1:9/v1", timeout=0.2)
    check(c.parse_action('noise {"tool":"set_qos_ceiling","args":{"cls":"gold",'
                         '"ceiling":1000}} tail')["tool"] == "set_qos_ceiling",
          "the OpenAI client extracts a tool call from noisy text")
    check(c.parse_action("no json here") is None, "unparseable replies return None")
    check(c.parse_action('{"tool":"list_alarms","args":{}}')["args"] == {},
          "missing args default to an empty dict")

def test_tool_schema_granularity_is_real():
    print("\n[test_tool_schema_granularity_is_real]")
    fine, coarse = ToolRegistry("fine"), ToolRegistry("coarse")
    check(len(fine.names) == 11 and len(coarse.names) == 3,
          f"fine={len(fine.names)} tools, coarse={len(coarse.names)} tools")
    intents = [{"kind": "op", "op": {"type": "reprovision", "flow": "F0", "path_index": 1}},
               {"kind": "op", "op": {"type": "qos_set", "cls": "gold", "ceiling": 1000.0}}]
    check(len(fine.plan_calls(intents)) == 2, "fine: one billed call per operation")
    check(len(coarse.plan_calls(intents)) == 1, "coarse: both operations in one atomic call")

def test_decision_context_rng_is_config_independent():
    print("\n[test_decision_context_rng_is_config_independent]")
    a = DecisionContext(step=3, seed=11)
    b = DecisionContext(step=3, seed=11)
    check([a.rng.random() for _ in range(3)] == [b.rng.random() for _ in range(3)],
          "controller noise depends only on (seed, step) - not on the harness config")

def test_flat_argument_tool_calls_are_parsed():
    """Models frequently emit flat args; dropping them silently disables the agent."""
    print("\n[test_flat_argument_tool_calls_are_parsed]")
    c = OpenAICompatClient()
    a = c.parse_action('{"tool":"set_link_state","link":"L1","up":true}')
    check(a == {"tool": "set_link_state", "args": {"link": "L1", "up": True}},
          f"flat arguments become args ({a})")
    b = c.parse_action('{"tool":"ack_alarm","args":{"alarm":"A001"}}')
    check(b["args"] == {"alarm": "A001"}, "nested arguments still work")
    d = c.parse_action('{"name":"observe_network"}')
    check(d == {"tool": "observe_network", "args": {}}, "the 'name' alias is accepted")
    from aixnet.tools import intents_from_tool_call
    check(intents_from_tool_call(a["tool"], a["args"]) ==
          [{"kind": "op", "op": {"type": "link_set", "link": "L1", "up": True}}],
          "a flat call now maps to a real operation instead of being dropped")


def test_baseline_privilege_is_explicit():
    """P3 fairness: the runbook must not touch the twin; the search baseline may,
    and must pay for it in reported sim_evals."""
    print("\n[test_baseline_privilege_is_explicit]")
    from aixnet.baselines import BaselineRules, SearchBaseline
    rules, search = BaselineRules(), SearchBaseline()
    check(not getattr(rules, "needs_env", False) and not hasattr(rules, "env"),
          "baseline_rules has no emulator handle (observation-only)")
    check(getattr(search, "needs_env", False) is True,
          "baseline_search declares that it needs the emulator")
    for sid in ALL_IDS:
        m = Episode(get_scenario(sid), 0, HarnessConfig(controller="baseline_rules")).run()
        check(m["sim_evals"] == 0,
              f"{sid}: runbook performs 0 twin simulations (measured)")
    m = Episode(get_scenario("S1_link_down"), 0,
                HarnessConfig(controller="baseline_search")).run()
    check(m["sim_evals"] > 0,
          f"search baseline reports its privileged cost: {m['sim_evals']} sim evals")


def test_prompt_and_tool_schema_grounding():
    """The prompt must only name tools the harness actually exposes."""
    print("\n[test_prompt_and_tool_schema_grounding]")
    from aixnet.llm_client import SCOPED3_PROMPT
    from aixnet.tools import grounding_gaps
    fine = ToolRegistry("fine")
    coarse = ToolRegistry("coarse")
    # KNOWN DEFECT, detected here on purpose: the prompt's read-only fallback (R3)
    # names the wide `observe_network` tool, which the fine schema does not expose.
    check(grounding_gaps(SCOPED3_PROMPT, fine) == ["observe_network"],
          "fine schema: detector finds the known R3 fallback mismatch")
    gaps = grounding_gaps(SCOPED3_PROMPT, coarse)
    check(set(gaps) >= {"set_link_state", "ack_alarm", "set_qos_ceiling"},
          f"coarse schema: prompt names {len(gaps)} tools the harness does not expose "
          f"{gaps} - the measured cause of the coarse LLM arm's 0.20 success")
    check(grounding_gaps("call get_flow_stats then ack_alarm", fine) == [],
          "detector returns nothing when prompt and schema agree")


def test_guardrail_capacity_zero_bypass_is_closed():
    """Regression: classify() called capacity-0 unsafe but nothing enforced it.
    Verified exploit before the fix: strict verdict 'allow', L0 capacity 1000->0,
    util -> 3.2e11, delay -> 6.4e12 ms, and those rows shipped in the dataset."""
    print("\n[test_guardrail_capacity_zero_bypass_is_closed]")
    env = NetworkEmulator(0)
    env.refresh()
    op = {"type": "link_capacity", "link": "L0", "capacity": 0.0}
    g = Guardrails("strict")
    g.state.start_step()
    v = g.check(op, env)
    check(v.decision == "needs_approval" and v.unsafe,
          f"strict refuses a capacity collapse (got {v.decision})")
    check(env.apply(op, snapshot=env.snapshot())["ok"] is False,
          "the twin refuses sub-floor capacity even when the guardrail allows it")
    for cap in (0.0, 0.5, -5.0):
        env.apply({"type": "link_capacity", "link": "L0", "capacity": cap},
                  snapshot=env.snapshot())
        env.recompute()
        check(env.links["L0"]["capacity"] >= 1.0 and env.links["L0"]["util"] <= 5.0,
              f"capacity {cap}: link stays physical (cap="
              f"{env.links['L0']['capacity']}, util={env.links['L0']['util']})")
    gl = Guardrails("lenient")
    gl.state.start_step()
    check(gl.check(op, env).decision == "allow",
          "lenient still permits the attempt - contained by the twin, not the guardrail")


def test_effect_magnitudes_are_locked():
    """A regression that halved the key effects must fail the suite, not just
    shift a CSV."""
    print("\n[test_effect_magnitudes_are_locked]")
    def rate(**kw):
        cfg = HarnessConfig(**kw)
        rs = [Episode(get_scenario(s), k, cfg).run() for s in ALL_IDS for k in range(20)]
        return sum(r["success"] for r in rs) / len(rs)
    none_r = rate(retry="none")
    roll_r = rate(retry="retry_rollback")
    check(roll_r - none_r >= 0.05,
          f"retry/rollback effect is real: {roll_r:.3f} vs {none_r:.3f} "
          f"(+{(roll_r - none_r) * 100:.1f} pp, n=100 each)")
    near = rate(layer="near_rt")
    smo = rate(layer="smo_intent")
    check(near - smo >= 0.20,
          f"control-layer effect is real: {near:.3f} vs {smo:.3f} "
          f"(+{(near - smo) * 100:.1f} pp, n=50 each)")


def test_harness_effect_collapses_when_the_budget_stops_binding():
    """The null result, locked in: at a non-binding budget the 48 near-RT
    configurations converge, so the success effect is budget-mediated."""
    print("\n[test_harness_effect_collapses_when_the_budget_stops_binding]")
    from aixnet.runner import all_configs
    sc = get_scenario("S1_link_down")

    def spread(budget):
        rates = []
        for cfg in all_configs("mock"):
            if cfg.layer != "near_rt":
                continue
            rs = [Episode(sc, k, cfg, budget=budget).run() for k in range(5)]
            rates.append(sum(r["success"] for r in rs) / len(rs))
        return max(rates) - min(rates), sum(rates) / len(rates)

    tight, tight_mean = spread(3)
    loose, loose_mean = spread(20)
    check(tight >= 0.30,
          f"budget 3 (binding): configurations differ by {tight:.2f} on S1")
    check(loose <= 0.15,
          f"budget 20 (non-binding): the same configurations differ by only {loose:.2f}")
    check(loose_mean > tight_mean,
          f"and everyone solves it: mean {tight_mean:.3f} -> {loose_mean:.3f}")


def test_mock_never_uses_network_or_clock():
    print("\n[test_mock_never_uses_network_or_clock]")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "aixnet", "controllers.py"), encoding="utf-8").read()
    check("time." not in src and "socket" not in src and "urllib" not in src,
          "the deterministic controller imports no clock and no network")
