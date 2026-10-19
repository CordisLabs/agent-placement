"""Context assembly: what the harness puts in front of the policy each step.

This is the state-management factor made concrete. `full` renders the whole world;
`scratchpad` renders a relevance-filtered digest plus a running scratchpad. Both
policies below are explicit so the compaction rule can be reviewed and changed.

The prompt is rendered - and its size measured - even when the deterministic
controller answers, because the size of the context a harness builds is a real
property of the harness.
"""

from __future__ import annotations

import json

from .emulator import CONGESTION_UTIL
from .llm_client import SYSTEM_PROMPT, build_messages
from .config import TOKEN_DIVISOR


class ContextBuilder:
    """Mixin: observation polling, digest compaction, prompt accounting."""

    # ---------------- observation ----------------
    def _bill(self, tool: str) -> None:
        self.tool_calls += 1
        self.calls_log.append(tool)

    def _poll(self) -> dict:
        if self.registry.granularity == "coarse":
            self._bill("observe_network")
            return self.env.observe(include_paths=True)
        for t in ("get_link_stats", "get_flow_stats", "list_alarms"):
            self._bill(t)
        return self.env.observe(include_paths=False)

    def _digest(self, raw: dict, scratch: bool) -> dict:
        paths = {}
        for f in raw["flows"]:
            if "candidate_paths" in f:           # free with the wide observe tool
                self.path_cache[f["id"]] = f["candidate_paths"]
            if f["id"] in self.path_cache:
                paths[f["id"]] = self.path_cache[f["id"]]
        if scratch:
            # compaction rule: keep everything DECISION-RELEVANT and drop bulk.
            # abnormal links, plus every link on a failing flow's current or
            # candidate path (needed to evaluate alternatives), failing flows
            # only, all critical alarms, the 8 most recent others.
            relevant: set = set()
            for f in raw["flows"]:
                if f["sla_ok"]:
                    continue
                relevant.update(f["path"])
                for cand in paths.get(f["id"], []):
                    relevant.update(cand)
            alarms = [a for a in raw["alarms"] if a["severity"] == "critical"]
            rest = [a for a in raw["alarms"] if a["severity"] != "critical"][-8:]
            return {"klass": "scratchpad", "t": raw["t"],
                    "links": [l for l in raw["links"] if (not l["up"])
                              or l["util"] >= CONGESTION_UTIL or l["id"] in relevant],
                    "flows": [f for f in raw["flows"] if not f["sla_ok"]],
                    "alarms": alarms + rest, "paths": paths,
                    "policy": raw["policy"],
                    "scratchpad": self.history[-3:] +
                                  [f"refused={self._refused_summary()}"]}
        return {"klass": "stateless", "t": raw["t"], "links": raw["links"],
                "flows": raw["flows"], "alarms": raw["alarms"], "paths": paths,
                "policy": raw["policy"]}

    def _refused_summary(self) -> list[str]:
        return [f"{v['decision']}:{v['op'].get('type')}:"
                f"{v['op'].get('link') or v['op'].get('cls')}"
                for v in self.last_verdicts_prev if v["decision"] != "allow"][-3:]

    def _render_prompt(self, digest: dict, max_steps: int) -> int:
        msgs = build_messages(SYSTEM_PROMPT, json.dumps(digest),
                              self.registry.schema_json(), self.history[-3:],
                              max_steps)
        return sum(len(m["content"]) for m in msgs)
