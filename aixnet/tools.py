"""Tool registry with swappable schema granularity.

fine   : 11 narrow tools, one effect each. Precise, partial application
         (one rejected change does not roll back its siblings), many round trips.
coarse : 3 wide tools. `apply_change` is ATOMIC (all-or-nothing), which mirrors
         real NMS/controller batch APIs: one rejected change voids the whole call.

The registry also exposes the JSON schema text that the harness puts in the
prompt, so schema granularity has a measurable token cost.
"""

from __future__ import annotations

import json

FINE_TOOLS = [
    ("get_link_stats", "read", {}, "Per-link admin state, capacity, utilisation, delay, loss."),
    ("get_flow_stats", "read", {}, "Per-flow demand, delivered rate, delay, SLA verdict, path."),
    ("list_alarms", "read", {}, "Unacknowledged alarms (id, kind, target, severity)."),
    ("list_paths", "read", {"flow": "str"}, "Alternative provisioned paths for one flow."),
    ("set_link_state", "write", {"link": "str", "up": "bool"},
     "Admin up/down a link. Down on a production link is service affecting."),
    ("set_link_capacity", "write", {"link": "str", "capacity": "float"},
     "Change provisioned capacity (Mbps) of a link."),
    ("set_qos_ceiling", "write", {"cls": "str", "ceiling": "float"},
     "Set the rate ceiling (Mbps) of a traffic class."),
    ("reprovision_flow", "write", {"flow": "str", "path_index": "int"},
     "Move a flow onto one of its candidate paths (index from list_paths)."),
    ("ack_alarm", "write", {"alarm": "str"}, "Acknowledge one alarm."),
    ("snapshot_config", "control", {}, "Capture last-known-good config for rollback."),
    ("restore_snapshot", "control", {}, "Restore the last captured config snapshot."),
]

COARSE_TOOLS = [
    ("observe_network", "read", {},
     "One-shot full state dump: links, flows, alarms, candidate paths."),
    ("apply_change", "write", {"changes": "list[op]"},
     "ATOMIC batch of config changes. If any change is rejected the whole batch is void."),
    ("restore_snapshot", "control", {}, "Restore the last captured config snapshot."),
]

# intent op type -> fine tool name
OP_TO_TOOL = {
    "link_set": "set_link_state",
    "link_capacity": "set_link_capacity",
    "qos_set": "set_qos_ceiling",
    "reprovision": "reprovision_flow",
    "ack_alarm": "ack_alarm",
    "restore_snapshot": "restore_snapshot",
}

READ_TO_TOOL = {
    "links": "get_link_stats",
    "flows": "get_flow_stats",
    "alarms": "list_alarms",
    "paths": "list_paths",
}

TOOL_TO_OP = {
    "set_link_state": lambda a: {"type": "link_set", "link": a["link"], "up": a["up"]},
    "set_link_capacity": lambda a: {"type": "link_capacity", "link": a["link"],
                                    "capacity": a["capacity"]},
    "set_qos_ceiling": lambda a: {"type": "qos_set", "cls": a["cls"],
                                  "ceiling": a["ceiling"]},
    "reprovision_flow": lambda a: {"type": "reprovision", "flow": a["flow"],
                                   "path_index": a["path_index"]},
    "ack_alarm": lambda a: {"type": "ack_alarm", "alarm": a["alarm"]},
}

READ_TOOL_INTENT = {
    "get_link_stats": {"kind": "read", "what": "links"},
    "get_flow_stats": {"kind": "read", "what": "flows"},
    "list_alarms": {"kind": "read", "what": "alarms"},
    "observe_network": {"kind": "read", "what": "full"},
}


def intents_from_tool_call(tool: str, args: dict, granularity: str = "fine") -> list[dict]:
    """Map a model-emitted tool call onto harness intents (used by the LLM path)."""
    if tool in READ_TOOL_INTENT:
        return [dict(READ_TOOL_INTENT[tool])]
    if tool == "list_paths":
        return [{"kind": "read", "what": "paths", "flow": args.get("flow")}]
    if tool == "apply_change":
        changes = args.get("changes") or []
        return [{"kind": "op", "op": c} for c in changes if isinstance(c, dict)]
    if tool == "snapshot_config":
        return []
    if tool in TOOL_TO_OP:
        try:
            return [{"kind": "op", "op": TOOL_TO_OP[tool](args)}]
        except (KeyError, TypeError):
            return []
    return []


class ToolRegistry:
    def __init__(self, granularity: str = "fine"):
        if granularity not in ("fine", "coarse"):
            raise ValueError(f"unknown granularity {granularity!r}")
        self.granularity = granularity
        self.tools = COARSE_TOOLS if granularity == "coarse" else FINE_TOOLS

    @property
    def names(self) -> list[str]:
        return [t[0] for t in self.tools]

    def schema_json(self) -> str:
        return json.dumps([
            {"name": n, "class": c, "params": p, "desc": d}
            for n, c, p, d in self.tools
        ], separators=(",", ":"))

    def schema_tokens_est(self) -> int:
        return len(self.schema_json()) // 4

    def plan_calls(self, intents: list[dict]) -> list[dict]:
        """Translate controller intents into concrete tool calls for this schema.

        Returns [{'tool': name, 'args': {...}, 'ops': [op, ...]}, ...].
        Each entry is ONE billed tool call.
        """
        calls: list[dict] = []
        if self.granularity == "coarse":
            reads = [i for i in intents if i["kind"] == "read"]
            ops = [i["op"] for i in intents if i["kind"] == "op"]
            if reads:
                calls.append({"tool": "observe_network", "args": {},
                              "ops": [], "read": "full", "include_paths": True})
            if ops:
                if any(o.get("type") == "restore_snapshot" for o in ops):
                    calls.append({"tool": "restore_snapshot", "args": {},
                                  "ops": [o for o in ops if o.get("type") == "restore_snapshot"],
                                  "atomic": True})
                    ops = [o for o in ops if o.get("type") != "restore_snapshot"]
                if ops:
                    calls.append({"tool": "apply_change", "args": {"changes": ops},
                                  "ops": ops, "atomic": True})
            return calls

        for intent in intents:
            if intent["kind"] == "read":
                what = intent["what"]
                if what not in READ_TO_TOOL:
                    continue
                tool = READ_TO_TOOL[what]
                args = {"flow": intent["flow"]} if what == "paths" else {}
                calls.append({"tool": tool, "args": args, "ops": [], "read": what,
                              "flow": intent.get("flow")})
            else:
                op = intent["op"]
                tool = OP_TO_TOOL[op["type"]]
                calls.append({"tool": tool, "args": dict(op), "ops": [op],
                              "atomic": False})
        return calls


def grounding_gaps(prompt: str, registry: "ToolRegistry") -> list[str]:
    """Tool names the prompt tells the policy to call but the schema does not offer.

    A harness that renders a prompt and a tool schema which disagree is a
    grounding defect: the policy is instructed to call tools it cannot see.
    Returns the offending names (empty list == the contract is consistent).
    """
    named = set()
    for line in prompt.splitlines():
        for tok in line.replace('"', " ").replace(",", " ").replace("{", " ").split():
            tok = tok.strip(":,.()[]{}<>")
            if tok in TOOL_TO_OP or tok in READ_TOOL_INTENT:
                named.add(tok)
    return sorted(named - set(registry.names))
