#!/usr/bin/env python3
"""aixnet-2026 demo entry point. Python 3.10+, standard library only.

  python3 run_demo.py demo                     # one narrated episode per scenario
  python3 run_demo.py sweep  --reps 20         # full factorial sweep + CSVs
  python3 run_demo.py focused --reps 100       # pre-registered contrast set, tight CIs
  python3 run_demo.py replay out/traces/X.jsonl
  python3 run_demo.py replay-all out/traces
  python3 run_demo.py llm-probe                # honest reachability check (offline: fails)
  python3 run_demo.py selfcheck                # runs the whole test suite
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aixnet.config import LAYERS                            # noqa: E402
from aixnet.harness import HarnessConfig, Episode          # noqa: E402
from aixnet.runner import (baselines, run_budget_sensitivity, run_sweep,  # noqa: E402
                           summarise, write_csv)
from aixnet.scenarios import ALL_IDS, get as get_scenario  # noqa: E402
from aixnet.trace import replay                            # noqa: E402

FOCUSED = [
    ("H1_reference", HarnessConfig()),
    ("H2_no_guardrails", HarnessConfig(guardrails="lenient", retry="none")),
    ("H3_coarse_atomic", HarnessConfig(tool_granularity="coarse", control="planner")),
    ("H4_stateless", HarnessConfig(state_management="stateless")),
    ("H5_planner", HarnessConfig(control="planner")),
    ("H6_no_retry", HarnessConfig(retry="none")),
    ("H7_lenient_rb", HarnessConfig(guardrails="lenient", retry="retry_rollback")),
    ("H8_worst_case", HarnessConfig(tool_granularity="coarse",
                                    state_management="stateless", control="planner",
                                    guardrails="lenient", retry="none")),
]


def cmd_demo(args) -> int:
    print("aixnet-2026 :: harness-vs-model reliability demo (deterministic controller)")
    print("=" * 78)
    cfg = HarnessConfig(controller=args.controller)
    for sid in ALL_IDS:
        sc = get_scenario(sid)
        trace = os.path.join(args.out, "traces", f"demo-{sid}.jsonl")
        m = Episode(sc, args.seed, cfg, trace_path=trace).run()
        print(f"\n[{sid}] {sc.title}")
        print(textwrap.fill(sc.thesis, 78, initial_indent="  thesis     : ",
                            subsequent_indent=" " * 15))
        print(f"  budget     : {sc.max_steps} steps   seed={args.seed}")
        print(f"  outcome    : {'RESOLVED' if m['success'] else 'UNRESOLVED'} "
              f"in {m['steps']} steps / {m['tool_calls']} tool calls")
        print(f"  guardrails : {m['blocked_total']} blocked "
              f"({m['blocked_unsafe']} unsafe), {m['applied_unsafe']} unsafe applied")
        print(f"  retry      : {m['retries']} retries, {m['rollbacks']} rollbacks")
        print(f"  tokens     : {m['prompt_tokens_est']} (estimated, chars/4)")
        print(f"  final      : flows_ok={m['flows_ok_final']}/{m['flows_total']} "
              f"open_critical_alarms={m['unacked_critical_final']} "
              f"state={m['final_state_hash']}")
        if not m["success"]:
            print(f"  reason     : {m['failure_reason']}")
        print(f"  trace      : {trace}")
    print("\nReplay any trace with:  python3 run_demo.py replay <trace.jsonl>")
    return 0


def cmd_sweep(args) -> int:
    res = run_sweep(reps=args.reps, out_dir=args.out, controller=args.controller)
    print(json.dumps(res, indent=2))
    res2 = baselines(reps=args.reps, out_dir=args.out)
    print(json.dumps(res2, indent=2))
    rows = _read_csv(res["episodes_csv"])
    head = summarise(rows)
    _print_table(head, only=args.scenario)
    return 0


def cmd_focused(args) -> int:
    rows: list[dict] = []
    trace_dir = os.path.join(args.out, "traces")
    for sid in (args.scenario or ALL_IDS):
        sc = get_scenario(sid)
        for name, cfg in FOCUSED:
            for seed in range(args.reps):
                t = (os.path.join(trace_dir, f"{name}-{sid}-s{seed}.jsonl")
                     if seed == 0 else None)
                m = Episode(sc, seed, cfg, trace_path=t).run()
                m["variant"] = name
                rows.append(m)
        for ctrl in ("baseline_rules", "baseline_search"):
            for seed in range(args.reps):
                m = Episode(sc, seed, HarnessConfig(controller=ctrl)).run()
                m["variant"] = ctrl
                rows.append(m)
    path = os.path.join(args.out, "focused.csv")
    write_csv(rows, path)
    _print_table(summarise(rows), key="variant")
    print(f"\nwrote {path}  ({len(rows)} episodes, {args.reps} reps per cell)")
    return 0


def cmd_replay(args) -> int:
    res = replay(args.trace, verbose=args.verbose)
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


def cmd_replay_all(args) -> int:
    paths = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    if not paths:
        print(f"FAIL: no .jsonl traces found under {args.dir} - nothing was verified")
        return 1
    bad = 0
    for p in paths:
        res = replay(p)
        if not res["ok"]:
            bad += 1
            print(f"FAIL {os.path.basename(p)}: {res.get('error') or res}")
    print(f"replayed {len(paths)} traces, {len(paths) - bad} reproduced exactly, {bad} failed")
    return 0 if bad == 0 else 1


def cmd_llm_probe(args) -> int:
    from aixnet.llm_client import OpenAICompatClient
    out = OpenAICompatClient(base_url=args.base_url, model=args.model).probe()
    print(json.dumps(out, indent=2))
    if not out["reachable"]:
        print("\nNo LLM endpoint reachable. That is expected offline: every number in "
              "the study comes from --controller mock / baseline_*, and the "
              "LLM code path stays untested.")
    return 0


def cmd_llm_arm(args) -> int:
    """Run the model-driven arm (needs a reachable endpoint; the offline core of
    the study does not depend on it). --layer places the agent at one control
    layer; 'all' runs the three layers in sequence, each to its own CSV."""
    from aixnet.llm_arm import run_arms
    for layer in (list(LAYERS) if args.layer == "all" else [args.layer]):
        res = run_arms(reps=args.reps, out_dir=args.out, scenarios=args.scenario,
                       prompt=args.prompt, layer=layer, controller=args.controller)
        print(f"\nwrote {res['csv']} ({res['episodes']} episodes, prompt={args.prompt},"
              f" layer={layer}, sha1={res['prompt_sha1']})")
        for r in sorted(res["summary"], key=lambda x: (x["scenario"], x["config"])):
            print(f"  {r['scenario']:20s} {r['config'][:38]:38s} n={r['n']:3d} "
                  f"succ={r['success_rate']:.2f} steps={r['steps_mean']:.2f}")
    return 0


def cmd_demo_live(args) -> int:
    """One-command live demo: recomputes the study's factorial (~10 s, offline),
    prints it, then the shipped LLM table. Never crashes; always exits 0."""
    from aixnet.demo_live import main as live_main
    return live_main(seeds=args.seeds, out_dir=args.out)


def cmd_selfcheck(args) -> int:
    import tests.test_all as t
    return t.main()


NUMERIC = ("success", "steps", "tool_calls", "mutations", "blocked_total",
           "blocked_unsafe", "applied_unsafe", "atomic_rejections", "retries",
           "rollbacks", "prompt_chars", "prompt_tokens_est", "wall_ms",
           "flows_ok_final", "unacked_critical_final", "sim_evals")
FACTOR_COLS = ("tool_granularity", "state_management", "control", "guardrails",
               "retry", "layer", "scenario")


def cmd_analyze(args) -> int:
    """Marginal effect of every harness factor, computed from an episodes CSV."""
    rows = _read_csv(args.csv)
    cols = [c for c in FACTOR_COLS if c in rows[0]]
    print(f"source: {args.csv}   episodes: {len(rows)}")
    for col in cols:
        levels = sorted({r[col] for r in rows})
        print(f"\n--- {col} " + "-" * (60 - len(col)))
        print(f"{'level':16s} {'n':>5s} {'succ':>6s} {'steps':>7s} {'calls':>7s} "
              f"{'tok':>7s} {'blkU':>6s} {'appU':>6s} {'rb':>5s} {'sim':>7s}")
        for lv in levels:
            g = [r for r in rows if r[col] == lv]
            m = lambda k: sum(r[k] for r in g) / len(g)  # noqa: E731
            print(f"{lv[:16]:16s} {len(g):5d} {m('success'):6.3f} {m('steps'):7.2f} "
                  f"{m('tool_calls'):7.1f} {m('prompt_tokens_est'):7.0f} "
                  f"{m('blocked_unsafe'):6.3f} {m('applied_unsafe'):6.3f} "
                  f"{m('rollbacks'):5.2f} {m('sim_evals'):7.1f}")
    if args.contrast:
        import collections
        by = collections.defaultdict(list)
        for r in rows:
            by[r["scenario"]].append(r)
        print("\n--- best vs worst configuration per scenario " + "-" * 22)
        for sid, g in sorted(by.items()):
            cfgs = collections.defaultdict(list)
            for r in g:
                cfgs[(r["tool_granularity"], r["state_management"], r["control"],
                      r["guardrails"], r["retry"])].append(r)
            rank = sorted(cfgs.items(),
                          key=lambda kv: (-sum(x["success"] for x in kv[1]) / len(kv[1]),
                                          sum(x["steps"] for x in kv[1]) / len(kv[1])))
            best, worst = rank[0], rank[-1]
            print(f"{sid:20s} best {best[0]} succ={sum(x['success'] for x in best[1])/len(best[1]):.2f} "
                  f"steps={sum(x['steps'] for x in best[1])/len(best[1]):.2f} | "
                  f"worst {worst[0]} succ={sum(x['success'] for x in worst[1])/len(worst[1]):.2f} "
                  f"steps={sum(x['steps'] for x in worst[1])/len(worst[1]):.2f}")
    return 0


def cmd_budget(args) -> int:
    res = run_budget_sensitivity(budgets=tuple(args.budgets), reps=args.reps,
                                 out_dir=args.out, scenarios=args.scenario)
    print(f"\nwrote {res['csv']} and out/budget_spread.csv ({res['episodes']} episodes)\n")
    print("success spread across harness configurations, by step budget:")
    print(f"{'budget':>7s} {'scenario':20s} {'n_cfg':>6s} {'worst':>7s} {'best':>7s} "
          f"{'spread':>7s} {'mean':>7s}")
    for r in res["spread"]:
        print(f"{r['budget']:7d} {r['scenario']:20s} {r['n_configs']:6d} {r['worst']:7.3f} "
              f"{r['best']:7.3f} {r['spread']:7.3f} {r['mean']:7.3f}")
    return 0


def _read_csv(path: str) -> list[dict]:
    import csv
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in NUMERIC:
            if k in r:
                r[k] = int(float(r[k]))
    return rows


def _print_table(rows: list[dict], only: str | None = None, key: str = "config") -> None:
    if key == "variant":
        rows = [r for r in rows if r["controller"] in ("mock",)]
    print(f"\n{'variant/scenario':34s} {'n':>4s} {'succ':>6s} {'steps':>14s} "
          f"{'calls':>12s} {'blkU':>6s} {'appU':>6s} {'tok':>8s}")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: (x["scenario"], x.get("variant") or "")):
        if only and r["scenario"] != only:
            continue
        name = (r.get("variant") or r["config"]) if key == "variant" else r["config"]
        print(f"{r['scenario'][:14] + ' ' + (name or '')[:19]:34s} {r['n']:4d} "
              f"{r['success_rate']:6.2f} "
              f"{r['steps_mean']:6.2f}+-{r['steps_sd']:5.2f} "
              f"{r['tool_calls_mean']:6.1f}+-{r['tool_calls_sd']:4.1f} "
              f"{r['blocked_unsafe_mean']:6.2f} {r['applied_unsafe_mean']:6.2f} "
              f"{r['prompt_tokens_est_mean']:8.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="out")
    p = sub.add_parser("demo", parents=[common]); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--controller", default="mock"); p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("sweep", parents=[common]); p.add_argument("--reps", type=int, default=20)
    p.add_argument("--controller", default="mock"); p.add_argument("--scenario", default=None)
    p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("focused", parents=[common]); p.add_argument("--reps", type=int, default=100)
    p.add_argument("--scenario", action="append"); p.set_defaults(fn=cmd_focused)
    p = sub.add_parser("replay"); p.add_argument("trace"); p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("replay-all"); p.add_argument("--dir", default="out/traces")
    p.set_defaults(fn=cmd_replay_all)
    p = sub.add_parser("llm-probe"); p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_llm_probe)
    p = sub.add_parser("selfcheck"); p.set_defaults(fn=cmd_selfcheck)
    p = sub.add_parser("demo-live", parents=[common])
    p.add_argument("--seeds", type=int, default=20); p.set_defaults(fn=cmd_demo_live)
    p = sub.add_parser("llm-arm", parents=[common])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--scenario", action="append")
    p.add_argument("--prompt", default="terse", choices=["terse", "scoped", "scoped2", "scoped3"])
    p.add_argument("--layer", default="near_rt", choices=[*LAYERS, "all"])
    p.add_argument("--controller", default="llm", choices=["llm", "mock"])
    p.set_defaults(fn=cmd_llm_arm)
    p = sub.add_parser("budget", parents=[common])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--budgets", type=int, nargs="+", default=[4, 6, 10, 20])
    p.add_argument("--scenario", action="append"); p.set_defaults(fn=cmd_budget)
    p = sub.add_parser("analyze", parents=[common])
    p.add_argument("--csv", default="out/episodes.csv")
    p.add_argument("--contrast", action="store_true"); p.set_defaults(fn=cmd_analyze)
    args = ap.parse_args()
    if getattr(args, "out", None):
        os.makedirs(args.out, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
