"""Step-by-step JSONL tracing + deterministic replay.

Log schema (one JSON object per line; schema_version 1)
-------------------------------------------------------
record "episode_start":
  v, rec, episode_id, scenario, seed, config{}, success_def, max_steps,
  initial_state_hash, faults[]
record "step" (one per harness step):
  v, rec, step, t, tool_calls[], reads[], ops_requested[],
  verdicts[{op, decision, reason, unsafe}], ops_applied[], ops_blocked[],
  retries, rollbacks, prompt_chars, kpi{}, flows_ok, state_hash
record "step_result":
  v, rec, step, success, reason, replayed_state_hash
record "episode_end":
  v, rec, success, steps, metrics{}, final_state_hash, op_timeline[]

Replay rebuilds the twin from (scenario, seed), re-applies only the recorded
`ops_applied` in order, and asserts the reconstructed final state hash matches.
"""

from __future__ import annotations

import json
import os

SCHEMA_VERSION = 1


class TraceWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")
        self.n = 0

    def write(self, rec: dict) -> None:
        rec = {"v": SCHEMA_VERSION, **rec}
        self.fh.write(json.dumps(rec, sort_keys=True) + "\n")
        self.n += 1

    def close(self) -> None:
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_trace(path: str) -> dict:
    header, steps, end = None, [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("rec") == "episode_start":
                header = rec
            elif rec.get("rec") == "step":
                steps.append(rec)
            elif rec.get("rec") == "episode_end":
                end = rec
    if header is None or end is None:
        raise ValueError(f"{path}: not a complete aixnet trace")
    return {"header": header, "steps": steps, "end": end}


def replay(path: str, verbose: bool = False) -> dict:
    """Re-apply a recorded episode to a fresh twin; verify the final state hash."""
    from .scenarios import get as get_scenario

    tr = read_trace(path)
    sc = get_scenario(tr["header"]["scenario"])
    env = sc.build(tr["header"]["seed"])
    timed = {t: f for t, f in sc.timed_faults()}
    applied = 0
    for rec in tr["steps"]:
        if env.t != rec["t"]:
            return {"ok": False, "episode_id": tr["header"]["episode_id"],
                    "error": f"clock desync: env.t={env.t} trace.t={rec['t']}"}
        for op in rec.get("ops_applied", []):
            res = env.apply(op, snapshot=env.snapshot())
            if not res["ok"]:
                return {"ok": False, "episode_id": tr["header"]["episode_id"],
                        "step": rec["step"], "error": f"replay apply failed: {res['detail']}"}
            applied += 1
        env.tick()
        if env.t in timed:
            env.inject(timed[env.t])
    final = env.state_hash()
    expected = tr["end"]["final_state_hash"]
    ok = final == expected
    out = {"ok": ok, "episode_id": tr["header"]["episode_id"],
           "steps": len(tr["steps"]), "ops_reapplied": applied,
           "replayed_state_hash": final, "recorded_state_hash": expected,
           "recorded_success": tr["end"]["success"],
           "replayed_success": bool(env.kpi()["sla_ok"])}
    if verbose:
        print(json.dumps(out, indent=2))
    return out
