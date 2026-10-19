"""Non-LLM baselines for the comparative study the venue asks for.

BaselineRules    - a fixed operational runbook: deterministic, safe, no search.
                   It is the "what a well-written automation script does" arm.
SearchBaseline   - 1-step lookahead local search scored against a CLONE of the
                   twin. Deliberately privileged (it may fork and evaluate the
                   emulator, which no tool-using agent can), and it reports its
                   `sim_evals` cost explicitly so the comparison stays honest.
"""

from __future__ import annotations

from .reasoning import (CONGESTION_UTIL, DEFAULT_CEILING, best_alt, clean_alt,
                        required_capacity)

SETTLE_TICKS = 2


class SearchBaseline:
    name = "baseline_search"
    needs_env = True

    def __init__(self, max_candidates: int = 40):
        self.env = None
        self.max_candidates = max_candidates
        self.sim_evals = 0
        self.plan: list[dict] = []

    def bind(self, env) -> None:
        self.env = env
        self.sim_evals = 0

    # ---- scoring ----
    def _score_candidate(self, op: dict) -> tuple | None:
        sim = self.env.clone()
        res = sim.apply(op, snapshot=sim.snapshot())
        if not res["ok"]:
            return None
        for _ in range(SETTLE_TICKS):
            sim.tick()
        self.sim_evals += 1
        return self._score(sim) + (-1,)

    @staticmethod
    def _score(env) -> tuple:
        k = env.kpi()
        crit = sum(1 for a in env.alarms
                   if a["severity"] == "critical" and not a["acked"])
        return (k["flows_ok"], -crit, 0, 0)

    def _candidates(self, obs: dict) -> list[dict]:
        out: list[dict] = []
        links = {l["id"]: l for l in obs["links"]}
        failing = [f for f in obs["flows"] if not f["sla_ok"]]
        # privileged: the optimizer reads candidate paths straight from the twin
        paths = obs.get("paths") or {f["id"]: self.env.candidate_paths(f["id"])
                                     for f in obs["flows"]}
        for f in failing:
            for i, path in enumerate(paths.get(f["id"]) or []):
                if any(p in links and not links[p]["up"] for p in path):
                    continue
                if f["path"] != path:
                    out.append({"type": "reprovision", "flow": f["id"], "path_index": i})
            for p in f["path"]:
                l = links.get(p)
                if l is None:
                    continue
                if not l["up"]:
                    out.append({"type": "link_set", "link": p, "up": True})
                elif l["util"] >= CONGESTION_UTIL:
                    out.append({"type": "link_capacity", "link": p,
                                "capacity": max(l["capacity"] * 2.0,
                                                required_capacity(obs, p))})
            cls = f["cls"]
            out.append({"type": "qos_set", "cls": cls, "ceiling": DEFAULT_CEILING[cls]})
        for l in obs["links"]:
            if not l["up"]:
                out.append({"type": "link_set", "link": l["id"], "up": True})
        seen, uniq = set(), []
        for op in out:
            key = tuple(sorted(op.items()))
            if key not in seen:
                seen.add(key)
                uniq.append(op)
        return uniq[: self.max_candidates]

    def decide(self, obs: dict, ctx) -> list[dict]:
        failing = [f for f in obs["flows"] if not f["sla_ok"]]
        crit = [a for a in obs["alarms"] if a["severity"] == "critical" and not a["acked"]]
        if not failing:
            return [{"kind": "op", "op": {"type": "ack_alarm", "alarm": a["id"]}}
                    for a in crit[:6]]
        assert self.env is not None, "SearchBaseline requires bind(env)"
        best, best_score = None, self._score(self.env)
        for op in self._candidates(obs):
            sc = self._score_candidate(op)
            if sc is not None and sc > best_score:
                best, best_score = op, sc
        if best is None:
            if crit:
                return [{"kind": "op",
                         "op": {"type": "ack_alarm", "alarm": crit[0]["id"]}}]
            cand = self._candidates(obs)
            return [{"kind": "op", "op": cand[0]}] if cand else []
        return [{"kind": "op", "op": best}]


class BaselineRules:
    """Non-LLM runbook baseline: fixed, ordered, safe rules. No randomness.

    1. a link on a failing flow's path is down -> reprovision onto a candidate
       path that avoids it AND has headroom; if none is clean, restore the link,
       rehearsing first when the change-control guardrail demands it;
    2. a congested link on a failing path -> same, else raise the link capacity;
    3. failing flow with healthy links -> restore its class ceiling;
    4. all flows healthy -> acknowledge critical alarms.
    """

    name = "baseline_rules"

    def decide(self, obs: dict, ctx) -> list[dict]:
        # honour change control: if the last verdict asked for a rehearsal, do it
        for v in getattr(ctx, "last_verdicts", []):
            if v["decision"] == "needs_dry_run":
                return [{"kind": "op", "op": dict(v["op"], dry_run=True)}]
        failing = [f for f in obs["flows"] if not f["sla_ok"]]
        crit = [a for a in obs["alarms"] if a["severity"] == "critical" and not a["acked"]]
        if not failing:
            return [{"kind": "op", "op": {"type": "ack_alarm", "alarm": a["id"]}}
                    for a in crit[:6]]
        links = {l["id"]: l for l in obs["links"]}
        paths = obs.get("paths") or {}
        for f in failing:
            down = [p for p in f["path"] if p in links and not links[p]["up"]]
            if down:
                if not paths.get(f["id"]):
                    return [{"kind": "read", "what": "paths", "flow": f["id"]}]
                idx = clean_alt(paths[f["id"]], f, obs, avoid=down[0])
                if idx is not None:
                    return [{"kind": "op", "op": {"type": "reprovision",
                                                  "flow": f["id"], "path_index": idx}}]
                return [{"kind": "op", "op": {"type": "link_set", "link": down[0],
                                              "up": True}}]
        for f in failing:
            cong = [p for p in f["path"]
                    if p in links and links[p]["util"] >= CONGESTION_UTIL]
            if cong:
                if not paths.get(f["id"]):
                    return [{"kind": "read", "what": "paths", "flow": f["id"]}]
                idx = clean_alt(paths[f["id"]], f, obs, avoid=cong[0])
                if idx is not None:
                    return [{"kind": "op", "op": {"type": "reprovision",
                                                  "flow": f["id"], "path_index": idx}}]
                l = links[cong[0]]
                cap = max(l["capacity"] * 2.0, required_capacity(obs, l["id"]))
                return [{"kind": "op", "op": {"type": "link_capacity", "link": l["id"],
                                              "capacity": cap}}]
        cls = failing[0]["cls"]
        return [{"kind": "op", "op": {"type": "qos_set", "cls": cls,
                                      "ceiling": DEFAULT_CEILING[cls]}}]


__all__ = ["BaselineRules", "SearchBaseline", "best_alt", "clean_alt"]
