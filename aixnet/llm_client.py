"""Pluggable OpenAI-compatible chat client (UNEXERCISED offline).

This is the model plug point. It is implemented, importable and unit-testable at
the message-construction level, but no LLM was reachable in the build
environment, so NO result in the report comes from this code path.

The harness calls `build_messages()` for every configuration - which is how the
prompt-token metric is produced even when the mock controller answers - and only
calls `complete()` when --controller llm is explicitly selected.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

SCOPED_PROMPT = (
    "You are a network operations agent. You are given the network state as JSON and "
    "must resolve the incident by calling exactly ONE tool per turn.\n"
    "Success requires BOTH: (a) every flow has \"sla_ok\": true, and (b) no alarm with "
    "severity \"critical\" is left unacknowledged.\n"
    "Procedure:\n"
    "1. Look at flows with \"sla_ok\": false and the links listed in their \"path\".\n"
    "2. If a link on a failing flow's path has \"up\": false, restore it with "
    "set_link_state {\"link\":\"L1\",\"up\":true}, or move the flow with "
    "reprovision_flow {\"flow\":\"F0\",\"path_index\":N} where N indexes a path from "
    "\"paths\" whose links are all up and have util below 0.85.\n"
    "3. If the links are up but one has util >= 0.85, call set_link_capacity "
    "{\"link\":\"L2\",\"capacity\":<twice the current capacity>} or move the flow.\n"
    "4. If every link on a failing flow's path is healthy and its delivered_mbps is far "
    "below demand_mbps, the class ceiling is too low: call set_qos_ceiling "
    "{\"cls\":\"gold\",\"ceiling\":1000}.\n"
    "5. Once every flow has \"sla_ok\": true, acknowledge each critical alarm with "
    "ack_alarm {\"alarm\":\"A001\"}.\n"
    "Never call a read-only tool when a mutating tool would make progress. Never repeat "
    "an action that just failed. Reply with exactly one JSON object: "
    '{"tool": "<name>", "args": {...}}'
)

SCOPED2_PROMPT = (
    "You are a network operations agent. Each turn you receive the network state as "
    "JSON and must reply with exactly ONE tool call as JSON.\n"
    "Decide with this priority rule, evaluated top to bottom, EVERY turn:\n"
    "R1. If any flow has \"sla_ok\": false, repair it and do nothing else:\n"
    "    - a link in that flow's \"path\" with \"up\": false  -> set_link_state "
    "{\"link\":<that link>,\"up\":true}\n"
    "    - a link in that flow's \"path\" with \"util\" >= 0.85 -> set_link_capacity "
    "{\"link\":<that link>,\"capacity\":<2x its current capacity>}\n"
    "    - all links on its path are up and below 0.85 util and its delivered_mbps is "
    "far below demand_mbps -> set_qos_ceiling {\"cls\":<its cls>,\"ceiling\":1000}\n"
    "R2. Otherwise, if any alarm has \"severity\": \"critical\", call ack_alarm "
    "{\"alarm\":<the id of the FIRST such alarm>}. One alarm per turn.\n"
    "R3. Otherwise reply with observe_network (or get_flow_stats).\n"
    "Never repeat the action you took on the previous turn if the state JSON already "
    "shows the result you asked for. Reply with one JSON object and no prose."
)

SCOPED3_PROMPT = (
    "You are a network operations agent. Each turn you receive the network state as "
    "JSON and must reply with exactly ONE tool call, in EXACTLY this shape:\n"
    '{"tool": "<tool_name>", "args": {<arguments>}}\n'
    "The arguments MUST be nested under \"args\".\n"
    "Success requires BOTH: (a) every flow has \"sla_ok\": true, and (b) no alarm with "
    "\"severity\": \"critical\" is left with \"acked\": false.\n"
    "Priority rule, re-evaluated from the CURRENT JSON on every turn:\n"
    "R1. If any flow has \"sla_ok\": false, repair it and do nothing else:\n"
    "    - a link in that flow's \"path\" with \"up\": false -> "
    '{"tool":"set_link_state","args":{"link":"<that link>","up":true}}\n'
    "    - else a link in its \"path\" with \"util\" >= 0.85 -> "
    '{"tool":"set_link_capacity","args":{"link":"<that link>","capacity":<2x current>}}\n'
    "    - else if every link on its path is up and below 0.85 util and its "
    "delivered_mbps is far below demand_mbps -> "
    '{"tool":"set_qos_ceiling","args":{"cls":"<its cls>","ceiling":1000}}\n'
    "R2. If EVERY flow has \"sla_ok\": true and at least one alarm still has "
    "\"acked\": false with \"severity\": \"critical\", your ONLY valid action is to "
    "acknowledge ONE of them: "
    '{"tool":"ack_alarm","args":{"alarm":"<id of the first such alarm>"}}\n'
    "    Repeat R2 on the following turns until no unacknowledged critical alarm "
    "remains. Do NOT repeat a repair once every flow shows \"sla_ok\": true.\n"
    "R3. Only if R1 and R2 do not apply: "
    '{"tool":"observe_network","args":{}}\n'
    "Reply with one JSON object and no prose."
)

SYSTEM_PROMPT = (
    "You are a network operations agent. You must resolve the incident by calling "
    "the provided tools. Success = every flow meets its SLA and every critical alarm "
    "is acknowledged within {max_steps} steps. Reply with ONE json object: "
    '{"tool": "<name>", "args": {...}} or {"tool":"apply_change","args":'
    '{"changes":[{"type":"reprovision","flow":"F0","path_index":1}]}}. '
    "No prose."
)


class OpenAICompatClient:
    """Minimal /chat/completions client. Standard library only."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or os.environ.get("AIXNET_LLM_BASE_URL")
                         or "http://127.0.0.1:11434/v1").rstrip("/")
        self.model = model or os.environ.get("AIXNET_LLM_MODEL") or "gpt-4o-mini"
        self.api_key = api_key or os.environ.get("AIXNET_LLM_API_KEY") or ""
        self.timeout = timeout or float(os.environ.get("AIXNET_LLM_TIMEOUT", "30"))
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, messages: list[dict], temperature: float = 0.0) -> str:
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": temperature}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key or 'none'}"})
        self.calls += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        usage = data.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def parse_action(text: str) -> dict | None:
        """Extract one tool call from a model reply. Returns None if unusable."""
        if not text:
            return None
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
                        tool = obj.get("tool") or obj.get("name")
                        args = obj.get("args")
                        if not isinstance(args, dict):
                            # many models emit FLAT arguments:
                            # {"tool":"set_link_state","link":"L1","up":true}
                            args = {k: v for k, v in obj.items()
                                    if k not in ("tool", "name", "args")}
                        return {"tool": tool, "args": args}
                    start = None
        return None

    def probe(self) -> dict:
        """Honest reachability probe; never raises."""
        try:
            txt = self.complete([{"role": "user", "content": "Reply with {}"}])
            return {"reachable": True, "base_url": self.base_url, "model": self.model,
                    "sample": txt[:80]}
        except (urllib.error.URLError, OSError, KeyError, ValueError,
                json.JSONDecodeError) as exc:
            return {"reachable": False, "base_url": self.base_url, "model": self.model,
                    "error": f"{type(exc).__name__}: {exc}"}


def build_messages(system: str, obs_json: str, tool_schema: str,
                   history: list[str], max_steps: int) -> list[dict]:
    """The exact message payload the harness would send. Also the token meter."""
    return [
        # plain substitution: prompts contain literal JSON braces
        {"role": "system", "content": system.replace("{max_steps}", str(max_steps))
         + "\nTOOLS: " + tool_schema},
        {"role": "user", "content": "OBSERVATION:\n" + obs_json
         + "\nRECENT_ACTIONS:\n" + ("\n".join(history) if history else "(none)")},
    ]


class LLMController:
    """Thin policy adapter over an OpenAI-compatible client.

    NOT EXERCISED IN THIS BUILD: no LLM endpoint was reachable. It exists so the
    model plug point is real code rather than a promise; every number in the
    report comes from the mock / baseline controllers.
    """

    name = "llm"
    needs_env = False

    PROMPTS = {"terse": SYSTEM_PROMPT, "scoped": SCOPED_PROMPT,
               "scoped2": SCOPED2_PROMPT, "scoped3": SCOPED3_PROMPT}

    def __init__(self, client: OpenAICompatClient, granularity: str = "fine",
                 prompt_style: str | None = None):
        self.client = client
        self.granularity = granularity
        self.prompt_style = prompt_style or os.environ.get("AIXNET_LLM_PROMPT", "terse")
        self.last_error: str | None = None
        self.parse_failures = 0

    def decide_at(self, idx: int, obs: dict, ctx) -> list[dict]:
        """多层路径的入口。

        存在的理由：harness 的多层循环只在控制器**有 decide_at** 时，
        才会用该层自己的（陈旧的）digest 重建 ctx.render。
        没有它，LLM 会一直读到基础那份渲染——**陈旧视野静默失效**。
        这个 bug 曾让一次 oracle 消融实验整体无效。
        """
        return self.decide(obs, ctx)

    def decide(self, obs: dict, ctx) -> list[dict]:
        from .tools import intents_from_tool_call
        system, obs_json, schema, history, max_steps = ctx.render
        system = self.PROMPTS.get(self.prompt_style, system)
        msgs = build_messages(system, obs_json, schema, history, max_steps)
        try:
            text = self.client.complete(msgs, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 - a dead endpoint must not kill the run
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        action = self.client.parse_action(text)
        if not action:
            self.parse_failures += 1
            return []
        return intents_from_tool_call(action["tool"], action["args"], self.granularity)
