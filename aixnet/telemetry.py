"""Telemetry and configuration views of the twin (read-only projections).

Kept apart from the simulation core so that what an agent can SEE is an explicit,
reviewable harness concern rather than an incidental property of the simulator.
"""

from __future__ import annotations

import hashlib
import json


def link_view(l: dict) -> dict:
    return {"id": l["id"], "endpoints": [l["a"], l["b"]], "up": l["up"],
            "capacity": l["capacity"], "weight": l["weight"],
            "production": l["production"], "util": round(l["util"], 4),
            "load_mbps": round(l["load"], 2),
            "delay_ms": round(l["delay"], 2), "loss": round(l["loss"], 4)}


def flow_view(env, f: dict, paths: bool = False) -> dict:
    view = {"id": f["id"], "cls": f["cls"], "src": f["src"], "dst": f["dst"],
            "demand_mbps": f["demand"], "delivered_mbps": round(f["delivered"], 2),
            "delay_ms": round(f["delay"], 2), "loss": round(f["loss"], 4),
            "sla_ok": f["sla_ok"], "path": list(f["path"] or [])}
    if paths:
        view["candidate_paths"] = env.candidates[f["id"]]
    return view


def observe(env, include_paths: bool = False) -> dict:
    """Full raw observation. Candidate paths cost a separate query unless the
    caller uses the wide `observe_network` tool (include_paths=True)."""
    return {
        "t": env.t,
        "links": [link_view(l) for l in env.links.values()],
        "flows": [flow_view(env, f, paths=include_paths) for f in env.flows.values()],
        "alarms": [dict(a) for a in env.alarms if not a["acked"]],
        "policy": {k: dict(v) for k, v in env.policy.items()},
        "klass": "full",
    }


def kpi(env) -> dict:
    flows = list(env.flows.values())
    return {
        "t": env.t,
        "flows_ok": sum(1 for f in flows if f["sla_ok"]),
        "flows_total": len(flows),
        "sla_ok": all(f["sla_ok"] for f in flows),
        "total_delivered": round(sum(f["delivered"] for f in flows), 2),
        "max_util": round(max(l["util"] for l in env.links.values()), 4),
        "unacked_alarms": sum(1 for a in env.alarms if not a["acked"]),
    }


def config_fingerprint(env) -> str:
    cfg = {
        "links": {k: [v["up"], v["capacity"], v["weight"]]
                  for k, v in sorted(env.links.items())},
        "policy": {k: [v["ceiling"], v["min_share"]]
                   for k, v in sorted(env.policy.items())},
        "paths": {k: v["path"] for k, v in sorted(env.flows.items())},
    }
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))[:4096]


def state_hash(env) -> str:
    payload = json.dumps([config_fingerprint(env), kpi(env)],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
