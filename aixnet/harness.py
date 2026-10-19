"""Agent harness: loop, guardrail enforcement, retry/rollback, tracing, metrics.

Factors (full semantics in config.py / README):
  tool_granularity  fine (11 narrow tools, partial application) | coarse (3 wide
                    tools, ATOMIC apply_change, paths bundled for free)
  state_management  scratchpad (running scratchpad is re-rendered) | stateless
  control           reactive (1 op/step) | planner (ordered multi-op plan)
  guardrails        lenient | strict          (see guardrails.py)
  retry             none | retry | retry_rollback  (see the README table)
  layer             near_rt | non_rt | smo_intent  (cadence + telemetry staleness)

Context assembly lives in context.py; configuration in config.py.
"""

from __future__ import annotations

import json
import time

from . import tools as toolsmod
from .config import (CONTROLLERS, LAYERS, TOKEN_DIVISOR, HarnessConfig,
                     make_controller)
from .context import ContextBuilder
from .controllers import DecisionContext
from .guardrails import Guardrails, VerdictView, classify
from .llm_client import SYSTEM_PROMPT
from .trace import TraceWriter

__all__ = ["HarnessConfig", "Episode", "CONTROLLERS", "make_controller",
           "TOKEN_DIVISOR"]



def _learned_predict(prev: dict, cur: dict, steps: int = 1) -> dict:
    """学习型外推器（`predictor="learned"`）。

    模型不可用时**退回线性外推**而不是抛异常——训练产物是可选的，
    缺了不该让整批实验崩掉。
    """
    try:
        from .learned_predictor import predict as _lp
        return _lp(prev, cur, steps)
    except Exception:
        return _extrapolate(prev, cur, steps)


def _extrapolate(prev: dict, cur: dict, steps: int = 1) -> dict:
    """对观测做一步线性外推：v ≈ cur + (cur - prev) * steps。

    只外推数值型字段（int/float），其余原样保留。用于测试"预测能否买回陈旧代价"。
    """
    # 布尔值**不参与算术**：True + 0 会变成整数 1，
    # 对 mock 控制器无碍（真值判定），但会让 LLM 看到的提示词从
    # "up": true 变成 "up": 1 —— 语义改变，曾导致一次错误结论。
    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    out = {}
    for k, cv in cur.items():
        pv = prev.get(k)
        if num(cv) and num(pv):
            out[k] = cv + (cv - pv) * steps
        elif isinstance(cv, list) and isinstance(pv, list) and len(cv) == len(pv):
            items = []
            for a, b in zip(pv, cv):
                if isinstance(a, dict) and isinstance(b, dict):
                    items.append({kk: (vv + (vv - a[kk]) * steps
                                       if num(vv) and num(a.get(kk)) else vv)
                                  for kk, vv in b.items()})
                else:
                    items.append(b)
            out[k] = items
        else:
            out[k] = cv
    return out


def _add_noise(obs: dict, eps: float, rng) -> dict:
    """给观测的数值字段加相对噪声：v → v * (1 + eps * u)，u ~ U(-1,1)。

    布尔值（up 之类）**不加噪**——它们是状态而非测量。
    """
    def jitter(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return v * (1.0 + eps * rng.uniform(-1.0, 1.0))

    out = {}
    for k, v in obs.items():
        if isinstance(v, list):
            items = []
            for it in v:
                if isinstance(it, dict):
                    items.append({kk: jitter(vv) for kk, vv in it.items()})
                else:
                    items.append(jitter(it) if isinstance(it, (int, float))
                                 and not isinstance(it, bool) else it)
            out[k] = items
        else:
            out[k] = jitter(v)
    return out


class Episode(ContextBuilder):
    """One incident episode: build the twin, drive the policy, trace, score."""

    def __init__(self, scenario, seed: int, cfg: HarnessConfig,
                 trace_path: str | None = None, budget: int | None = None):
        self.sc = scenario
        self.budget = budget
        self.seed = seed
        self.cfg = cfg
        self.env = scenario.build(seed)
        self.registry = toolsmod.ToolRegistry(cfg.tool_granularity)
        self.guards = Guardrails(cfg.guardrails)
        self.controller = make_controller(cfg, seed)
        if getattr(self.controller, "needs_env", False):
            self.controller.bind(self.env)
        self.snapshot = self.env.snapshot()
        self.writer = TraceWriter(trace_path) if trace_path else None
        self.step = self.tool_calls = 0
        self.mutations = 0
        self.retries = self.rollbacks = 0
        self.prompt_chars = 0
        self.calls_log: list[str] = []
        self.op_timeline: list[dict] = []
        self.step_applied: list[dict] = []
        self.rejections = 0
        self.history: list[str] = []
        self.last_verdicts: list[dict] = []
        self.last_verdicts_prev: list[dict] = []
        self.last_progress = 0
        self.path_cache: dict[str, list] = {}
        self.obs_ring: list[dict] = []
        self.timed = {t: f for t, f in self.sc.timed_faults()}
        self.episode_id = f"{scenario.sid}-s{seed}-{cfg.cid()}-{cfg.controller}"

    # ---------------- guardrail + apply ----------------
    def _try_op(self, op: dict, dry: bool = False) -> dict:
        v = self.guards.check(op, self.env)
        if dry:
            self.guards.rehearse(op)
            return {"op": op, "decision": "allow", "reason": "dry-run rehearsal",
                    "unsafe": v.unsafe, "dry_run": True}
        return {"op": op, "decision": v.decision, "reason": v.reason,
                "unsafe": v.unsafe}

    def _repair(self, op: dict) -> dict | None:
        if op.get("type") == "reprovision":
            cands = self.env.candidates.get(op.get("flow"), [])
            if cands:
                return dict(op, path_index=len(cands) - 1)
        if op.get("type") == "qos_set" and op.get("cls") not in self.env.policy:
            return dict(op, cls="gold")
        return None

    def _commit_op(self, op: dict) -> None:
        """Apply one allowlisted op; record only EFFECTIVE ops (replayable)."""
        op_class, unsafe = classify(op, self.env)
        if op.get("type") == "restore_snapshot":
            pre = self.env.snapshot()
            self.env.restore(self.snapshot)
            rolled = self.env.diff_ops(pre)
            self.op_timeline.extend(rolled)
            self.step_applied.extend(rolled)
            self.rollbacks += 1
            return
        res = self.env.apply(op, snapshot=self.snapshot)
        if not res["ok"]:
            if self.cfg.retry in ("retry", "retry_rollback"):
                repaired = self._repair(op)
                if repaired is not None and self.env.apply(repaired, snapshot=self.snapshot)["ok"]:
                    self.retries += 1
                    self._record(repaired)
                    return
            self.rejections += 1          # the twin refused a non-physical operation
            return
        self._record(op)

    def _record(self, op: dict) -> None:
        op_class, unsafe = classify(op, self.env)
        self.guards.commit(op, op_class)
        self.guards.note_applied(unsafe)
        self.mutations += 1
        self.op_timeline.append(op)
        self.step_applied.append(op)

    def _execute(self, call: dict) -> None:
        self._bill(call["tool"])
        if call["tool"] in ("get_link_stats", "get_flow_stats", "list_alarms",
                            "observe_network", "list_paths"):
            if call.get("read") == "paths" and call.get("flow"):
                self.path_cache[call["flow"]] = self.env.candidate_paths(call["flow"])
            return
        ops = call.get("ops") or []
        verdicts: list[dict] = []
        for op in ops:
            v = self._try_op(op, dry=bool(op.get("dry_run")))
            if v["decision"] == "needs_dry_run" and self.cfg.retry in (
                    "retry", "retry_rollback"):
                self._bill("dry_run:" + call["tool"])
                self._try_op(op, dry=True)
                v = self._try_op(op)
            verdicts.append(v)
        if call.get("atomic") and any(v["decision"] != "allow" for v in verdicts):
            # atomic batch semantics: one rejected change voids the whole call
            for v in verdicts:
                if v["decision"] != "allow":
                    self.guards.note_blocked(v["op"], VerdictView(v))
            self.rejections += 1
            self.last_verdicts = verdicts
            return
        for v in verdicts:
            if v["decision"] != "allow":
                self.guards.note_blocked(v["op"], VerdictView(v))
                continue
            if v.get("dry_run"):
                continue
            self._commit_op(v["op"])
        self.last_verdicts = verdicts

    # ---------------- main loop ----------------
    def _trace_start(self) -> None:
        """Header record; documents the layer manipulations actually applied."""
        sc = self.sc
        self.writer.write({
            "rec": "episode_start", "episode_id": self.episode_id,
            "scenario": sc.sid, "seed": self.seed, "config": self.cfg.as_dict(),
            "controller": self.cfg.controller, "layer": self.cfg.layer,
            "layer_cadence": self.cfg.cadence, "layer_staleness": self.cfg.staleness,
            "success_def": "all flows meet SLA and no critical alarm is open",
            "max_steps": self.budget or sc.max_steps,
            "initial_state_hash": self.env.state_hash(),
            "faults": [{"tick": t, "fault": f} for t, f in sc.faults],
            "tool_schema": self.registry.names,
        })

    def run(self) -> dict:
        t0 = time.perf_counter()
        sc = self.sc
        if self.writer:
            self._trace_start()
        success = False
        for k in range(self.budget or sc.max_steps):
            self.step = k + 1
            self.guards.state.start_step()
            self.last_verdicts = []
            self.step_applied = []
            self.calls_log = []
            t_before = self.env.t
            before_ok = self.env.kpi()["flows_ok"]
            before_set = {f["id"] for f in self.env.flows.values() if f["sla_ok"]}
            scratch = self.cfg.state_management == "scratchpad"
            fresh = self._poll()
            self.obs_ring.append(fresh)
            # telemetry freshness: the policy decides on the view from `staleness`
            # steps ago (0 = instantaneous, 2 = aged SMO aggregate)
            raw = self.obs_ring[max(0, len(self.obs_ring) - 1 - self.cfg.staleness)]
            # 遥测预测器：oracle 用当前真值替代陈旧值；delta 用最近两帧线性外推。
            # 目的是把"慢层的代价是信息陈旧"与"是行动机会少"分开。
            _pred = getattr(self.cfg, "predictor", "none")
            if _pred != "none" and len(self.obs_ring) >= 2:
                if _pred == "oracle":
                    raw = self.obs_ring[-1]
                elif _pred in ("delta", "learned"):
                    prev, cur = self.obs_ring[-2], self.obs_ring[-1]
                    _st = max(0, self.cfg.staleness - 0)
                    raw = (_learned_predict(prev, cur, _st) if _pred == "learned"
                           else _extrapolate(prev, cur, _st))
                    _noise = float(getattr(self.cfg, "predictor_noise", 0.0) or 0.0)
                    if _noise > 0:
                        import random as _rnd
                        raw = _add_noise(raw, _noise, _rnd.Random(self.seed * 7919 + k))
            digest = self._digest(raw, scratch)
            obs_for_policy = (digest if self.cfg.controller == "mock"
                              else dict(raw, paths=dict(self.path_cache)))
            self.prompt_chars += self._render_prompt(digest, sc.max_steps)
            ctx = DecisionContext(step=k, seed=self.seed,
                                  last_verdicts=self.last_verdicts_prev,
                                  last_ops=[v["op"] for v in self.last_verdicts_prev],
                                  last_progress=self.last_progress,
                                  rollback=self.rollbacks > 0,
                                  history_available=scratch)
            ctx.render = (SYSTEM_PROMPT, json.dumps(digest),
                          self.registry.schema_json(), self.history[-3:], sc.max_steps)
            # decision cadence: a slower control loop gets fewer chances to act
            specs = getattr(self.cfg, "layer_specs", None)
            if specs:
                # 多层路径：每组 (cadence, staleness) 一个子智能体，各按自己的节奏行动
                for _idx, (_cad, _stale) in enumerate(specs):
                    if k % _cad != 0:
                        continue
                    _raw = self.obs_ring[max(0, len(self.obs_ring) - 1 - _stale)]
                    # 每层各自应用预测器（用**该层自己的** staleness 外推）
                    _p2 = getattr(self.cfg, "predictor", "none")
                    if _p2 != "none" and len(self.obs_ring) >= 2:
                        if _p2 == "oracle":
                            _raw = self.obs_ring[-1]
                        elif _p2 in ("delta", "learned"):
                            _raw = ((_learned_predict if _p2 == "learned"
                                     else _extrapolate)(
                                self.obs_ring[-2], self.obs_ring[-1],
                                steps=max(0, _stale)))
                            _nz = float(getattr(self.cfg, "predictor_noise", 0.0) or 0.0)
                            if _nz > 0:
                                import random as _rnd2
                                _raw = _add_noise(_raw, _nz,
                                                  _rnd2.Random(self.seed * 104729 + k * 31 + _idx))
                    _dig = self._digest(_raw, scratch)
                    _obs = (_dig if self.cfg.controller == "mock"
                            else dict(_raw, paths=dict(self.path_cache)))
                    # 每个子智能体必须看到**自己的**视野：LLM 控制器是从
                    # ctx.render 取提示词的，若沿用单层那份，所有子智能体
                    # 都会拿到基础观测而不是各自的陈旧视野。
                    _fn = getattr(self.controller, "decide_at", None)
                    if _fn and hasattr(ctx, "render"):
                        _prev_render = ctx.render
                        ctx.render = (SYSTEM_PROMPT, json.dumps(_dig),
                                      self.registry.schema_json(),
                                      self.history[-3:], sc.max_steps)
                        _ints = _fn(_idx, _obs, ctx) or []
                        ctx.render = _prev_render
                    else:
                        _ints = (_fn(_idx, _obs, ctx) if _fn
                                 else self.controller.decide(_obs, ctx)) or []
                    _roles = getattr(self.cfg, "layer_op_roles", None)
                    _allow = _roles[_idx] if _roles else None
                    for _call in self.registry.plan_calls(_ints):
                        if _allow is not None:
                            _ops = [o for o in (_call.get("ops") or [])
                                    if o.get("type") in _allow]
                            if not _ops:
                                continue      # 该层无权做这类改动，连计费都不发生
                            _call = dict(_call, ops=_ops)
                        self._execute(_call)
            elif k % self.cfg.cadence == 0:
                intents = self.controller.decide(obs_for_policy, ctx) or []
                for call in self.registry.plan_calls(intents):
                    self._execute(call)

            for v in self.last_verdicts:
                tgt = (v["op"].get("link") or v["op"].get("cls")
                       or v["op"].get("flow") or "")
                self.history.append(
                    f"step{k + 1} {v['decision']} {v['op'].get('type')} {tgt}"
                    f" :: {v['reason'][:60]}")

            self.env.recompute()          # measure the effect of this step, now
            after_ok = self.env.kpi()["flows_ok"]
            self.last_progress = after_ok - before_ok

            # --- retry policy: one bounded in-step alternative action ---
            if (self.cfg.retry in ("retry", "retry_rollback")
                    and self.step_applied and after_ok <= before_ok
                    and hasattr(self.controller, "on_no_progress")):
                ctx2 = DecisionContext(step=k, seed=self.seed + 1000,
                                       last_verdicts=self.last_verdicts,
                                       last_ops=list(self.step_applied),
                                       last_progress=self.last_progress,
                                       history_available=scratch)
                alts = self.controller.on_no_progress(obs_for_policy, ctx2) or []
                if alts:
                    self.retries += 1
                    for call in self.registry.plan_calls(alts):
                        self._execute(call)
                    self.env.recompute()
                    after_ok = self.env.kpi()["flows_ok"]
                    self.last_progress = after_ok - before_ok

            after_set = {f["id"] for f in self.env.flows.values() if f["sla_ok"]}
            regressed = before_set - after_set
            if self.cfg.retry == "retry_rollback" and regressed:
                pre = self.env.snapshot()
                self.env.restore(self.snapshot)
                rolled = self.env.diff_ops(pre)
                self.op_timeline.extend(rolled)
                self.step_applied.extend(rolled)
                self.rollbacks += 1
                self.last_progress = 0
            else:
                self.snapshot = self.env.snapshot()
            self.env.tick()
            if self.env.t in self.timed:
                self.env.inject(self.timed[self.env.t])

            if self.writer:
                self.writer.write({
                    "rec": "step", "step": k + 1, "t": t_before,
                    "tool_calls": list(self.calls_log),
                    "ops_requested": [v["op"] for v in self.last_verdicts],
                    "verdicts": [{"op": v["op"], "decision": v["decision"],
                                  "reason": v["reason"], "unsafe": v["unsafe"]}
                                 for v in self.last_verdicts],
                    "ops_applied": list(self.step_applied),
                    "kpi": self.env.kpi(), "prompt_chars": self.prompt_chars,
                    "state_hash": self.env.state_hash(),
                })
            self.last_verdicts_prev = list(self.last_verdicts)
            if sc.success(self.env):
                success = True
                break
        wall_ms = (time.perf_counter() - t0) * 1000.0
        kpi = self.env.kpi()
        metrics = {
            "episode_id": self.episode_id, "scenario": sc.sid, "seed": self.seed,
            **self.cfg.as_dict(),
            "success": int(success), "steps": self.step,
            "tool_calls": self.tool_calls, "mutations": self.mutations,
            "blocked_total": self.guards.blocked_total,
            "blocked_unsafe": self.guards.blocked_unsafe,
            "applied_unsafe": self.guards.state.applied_unsafe,
            "atomic_rejections": self.rejections,
            "retries": self.retries, "rollbacks": self.rollbacks,
            "prompt_chars": self.prompt_chars,
            "prompt_tokens_est": self.prompt_chars // TOKEN_DIVISOR,
            "wall_ms": round(wall_ms, 3),
            "flows_ok_final": kpi["flows_ok"], "flows_total": kpi["flows_total"],
            "unacked_critical_final": sum(
                1 for a in self.env.alarms
                if a["severity"] == "critical" and not a["acked"]),
            "sim_evals": getattr(self.controller, "sim_evals", 0),
            "failure_reason": "ok" if success else sc.render_success(self.env),
            "final_state_hash": self.env.state_hash(),
        }
        if self.writer:
            self.writer.write({"rec": "episode_end", "episode_id": self.episode_id,
                               "success": success, "steps": self.step, "metrics": metrics,
                               "op_timeline": self.op_timeline,
                               "final_state_hash": self.env.state_hash()})
            self.writer.close()
            self.writer = None
        return metrics

