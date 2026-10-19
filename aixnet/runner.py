"""Experiment runner: sweeps harness configurations x scenarios x seeds and
writes one CSV row per episode plus an aggregated summary CSV.

    python3 run_demo.py sweep --reps 20 --out out

Everything is deterministic: episode (scenario, seed, config) -> identical row
except for the wall-clock column.
"""

from __future__ import annotations

import csv
import itertools
import os
import statistics
import sys
import time
from dataclasses import replace

from .harness import CONTROLLERS, HarnessConfig, Episode
from .scenarios import ALL_IDS, get as get_scenario

FACTORS = {
    "tool_granularity": ["fine", "coarse"],
    "state_management": ["scratchpad", "stateless"],
    "control": ["reactive", "planner"],
    "guardrails": ["lenient", "strict"],
    "retry": ["none", "retry", "retry_rollback"],
    "layer": ["near_rt", "non_rt", "smo_intent"],
}

METRIC_COLS = ["success", "steps", "tool_calls", "mutations", "blocked_total",
               "blocked_unsafe", "applied_unsafe", "atomic_rejections", "retries",
               "rollbacks", "prompt_chars", "prompt_tokens_est", "wall_ms",
               "flows_ok_final", "unacked_critical_final", "sim_evals"]


def all_configs(controller: str = "mock") -> list[HarnessConfig]:
    keys = list(FACTORS)
    out = []
    for combo in itertools.product(*(FACTORS[k] for k in keys)):
        out.append(HarnessConfig(controller=controller, **dict(zip(keys, combo))))
    return out


def run_cell(scenario_id: str, cfg: HarnessConfig, reps: int,
             trace_dir: str | None = None, first_trace: bool = True) -> list[dict]:
    sc = get_scenario(scenario_id)
    rows = []
    for seed in range(reps):
        tpath = None
        if trace_dir and (seed == 0 or not first_trace):
            tpath = os.path.join(trace_dir, f"{scenario_id}-{cfg.cid()}-s{seed}.jsonl"
                                 .replace("|", "_").replace("/", "_"))
        rows.append(Episode(sc, seed, cfg, trace_path=tpath).run())
    return rows


def write_csv(rows: list[dict], path: str, cols: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# Rows read back from a CSV are strings; rows from a fresh run are numbers.
# summarise() must work on both (resume path vs normal path).
_NUMERIC_COLS = ("success", "steps", "tool_calls", "prompt_tokens_est", "wall_ms",
                 "blocked_total", "blocked_unsafe", "applied_unsafe", "retries",
                 "rollbacks", "atomic_rejections", "mutations", "sim_evals")


def _num(rows):
    """Coerce known-numeric columns to float, in place. Tolerates '' / None."""
    for r in rows:
        for c in _NUMERIC_COLS:
            if c in r:
                v = r[c]
                if isinstance(v, str):
                    try:
                        r[c] = float(v)
                    except ValueError:
                        r[c] = 0.0
                elif v is None:
                    r[c] = 0.0
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["controller"], r["scenario"], r["tool_granularity"],
               r["state_management"], r["control"], r["guardrails"], r["retry"],
               r.get("variant", ""))
        groups.setdefault(key, []).append(r)
    out = []
    _num(rows)
    for key, rs in sorted(groups.items()):
        n = len(rs)
        rec = dict(zip(("controller", "scenario", "tool_granularity",
                        "state_management", "control", "guardrails", "retry",
                        "variant"), key))
        rec["config"] = (f"{key[2][:2]}|{key[3][:4]}|{key[4][:4]}|{key[5][:4]}|{key[6]}")
        if rec["variant"]:
            rec["config"] = f"{rec['variant']} [{rec['config']}]"
        rec["n"] = n
        rec["success_rate"] = round(sum(r["success"] for r in rs) / n, 4)
        for col in ("steps", "tool_calls", "prompt_tokens_est", "wall_ms",
                    "blocked_unsafe", "applied_unsafe", "retries", "rollbacks",
                    "atomic_rejections", "mutations", "sim_evals"):
            vals = [r[col] for r in rs]
            rec[f"{col}_mean"] = round(statistics.mean(vals), 4)
            rec[f"{col}_sd"] = round(statistics.pstdev(vals), 4)
        solved = [r["steps"] for r in rs if r["success"]]
        rec["steps_solved_mean"] = round(statistics.mean(solved), 4) if solved else ""
        out.append(rec)
    return out


def run_sweep(reps: int = 20, out_dir: str = "out", controller: str = "mock",
              scenarios: list[str] | None = None, configs: list[HarnessConfig] | None = None,
              facts: dict | None = None) -> dict:
    t0 = time.time()
    scenarios = scenarios or ALL_IDS
    if configs is None:
        if facts:
            saved = {k: FACTORS[k] for k in FACTORS}
            FACTORS.update({k: v for k, v in facts.items() if k in FACTORS})
            configs = all_configs(controller)
            FACTORS.update(saved)
        else:
            configs = all_configs(controller)
    trace_dir = os.path.join(out_dir, "traces")
    rows: list[dict] = []
    for sid in scenarios:
        for cfg in configs:
            rows.extend(run_cell(sid, cfg, reps, trace_dir=trace_dir))
    episodes_csv = os.path.join(out_dir, "episodes.csv")
    summary_csv = os.path.join(out_dir, "summary.csv")
    write_csv(rows, episodes_csv)
    write_csv(summarise(rows), summary_csv)
    return {"episodes": len(rows), "cells": len(scenarios) * len(configs),
            "reps": reps, "episodes_csv": episodes_csv, "summary_csv": summary_csv,
            "seconds": round(time.time() - t0, 2)}


def run_budget_sensitivity(budgets=(4, 6, 10, 20), reps: int = 10,
                           out_dir: str = "out",
                           scenarios: list[str] | None = None) -> dict:
    """Effect size as a function of the step budget.

    Answers directly: is the harness effect real, or an artefact of the budget?
    """
    scenarios = scenarios or ALL_IDS
    configs = all_configs("mock")
    rows: list[dict] = []
    for budget in budgets:
        for sid in scenarios:
            for cfg in configs:
                for seed in range(reps):
                    sc = get_scenario(sid)
                    m = Episode(sc, seed, cfg, budget=budget).run()
                    m["budget"] = budget
                    rows.append(m)
    path = os.path.join(out_dir, "budget_sensitivity.csv")
    write_csv(rows, path)
    spread: list[dict] = []
    for budget in budgets:
        for sid in scenarios:
            cell = [r for r in rows if r["budget"] == budget and r["scenario"] == sid]
            per_cfg: dict[tuple, list] = {}
            for r in cell:
                key = (r["tool_granularity"], r["state_management"], r["control"],
                       r["guardrails"], r["retry"], r["layer"])
                per_cfg.setdefault(key, []).append(r["success"])
            rates = [sum(v) / len(v) for v in per_cfg.values()]
            spread.append({"budget": budget, "scenario": sid, "n_configs": len(rates),
                           "best": round(max(rates), 4), "worst": round(min(rates), 4),
                           "spread": round(max(rates) - min(rates), 4),
                           "mean": round(statistics.mean(rates), 4)})
    write_csv(spread, os.path.join(out_dir, "budget_spread.csv"))
    return {"episodes": len(rows), "csv": path, "spread": spread}


def baselines(reps: int = 20, out_dir: str = "out",
              scenarios: list[str] | None = None) -> dict:
    scenarios = scenarios or ALL_IDS
    rows: list[dict] = []
    for sid in scenarios:
        for ctrl in ("baseline_rules", "baseline_search"):
            cfg = HarnessConfig(controller=ctrl)
            rows.extend(run_cell(sid, cfg, reps))
    path = os.path.join(out_dir, "baselines.csv")
    write_csv(rows, path)
    write_csv(summarise(rows), os.path.join(out_dir, "baselines_summary.csv"))
    return {"episodes": len(rows), "baselines_csv": path}


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    reps = 20
    out_dir = "out"
    if "--reps" in argv:
        reps = int(argv[argv.index("--reps") + 1])
    if "--out" in argv:
        out_dir = argv[argv.index("--out") + 1]
    res = run_sweep(reps=reps, out_dir=out_dir)
    print(json_dumps(res))
    res2 = baselines(reps=reps, out_dir=out_dir)
    print(json_dumps(res2))
    return 0


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
