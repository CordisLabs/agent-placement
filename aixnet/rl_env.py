#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 harness 包成一个可逐步驱动的 RL 环境。

为什么不改 `Episode.run()`：那是一个被 70 万回合实验验证过的单体循环，
动它有破坏既有结果的风险。这里**复用它的内部件**（`_execute` / `plan_calls` /
`obs_ring` / `sc.success`）另起一个逐步驱动，两条路径互不干扰。

设计要点
--------
* **状态**：定长向量，含每条链路的 (up, util, capacity, delay, load, weight)、
  每条流的 (sla_ok, 交付率, 丢包, 时延, 类别 one-hot)、告警聚合、策略上限。
* **动作**：结构化离散——"对第 i 个对象做第 j 种改动"。
  无效动作（对象不存在/越权）由 mask 屏蔽，不消耗步数。
* **陈旧**：与主实验一致——决策看的是 `ring[len-1-staleness]`。
* **节奏**：`k % cadence != 0` 的步跳过（与主实验一致）。

用法
----
    env = RLEnv(get("S1_link_down"), seed=0, spec=(1, 0))
    s = env.reset()
    s, r, done, info = env.step(action_index)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aixnet.config import HarnessConfig          # noqa: E402
from aixnet.harness import Episode               # noqa: E402

# ── 动作空间定义 ──────────────────────────────────────────────
# 每类"对象"上可做的改动。顺序固定，索引即动作编号。
LINK_ACTS = ("up", "down", "cap_up", "cap_down")
FLOW_ACTS = ("reprovision",)
CLASS_ACTS = ("qos_up", "qos_down")
ALARM_ACTS = ("ack",)

MAX_LINKS, MAX_FLOWS, MAX_CLASSES, MAX_ALARMS = 12, 6, 3, 12


def action_table():
    """返回 [(kind, obj_index, act), ...]，长度即动作数。"""
    t = []
    for i in range(MAX_LINKS):
        for a in LINK_ACTS:
            t.append(("link", i, a))
    for i in range(MAX_FLOWS):
        for a in FLOW_ACTS:
            t.append(("flow", i, a))
    for i in range(MAX_CLASSES):
        for a in CLASS_ACTS:
            t.append(("class", i, a))
    for i in range(MAX_ALARMS):
        for a in ALARM_ACTS:
            t.append(("alarm", i, a))
    t.append(("noop", 0, "noop"))
    return t


ACTIONS = action_table()
N_ACTIONS = len(ACTIONS)
CLASS_NAMES = ("gold", "silver", "bronze")


class RLEnv:
    """逐步驱动的网络运维环境。"""

    def __init__(self, scenario, seed: int = 0, spec=(1, 0), predictor: str = "none",
                 guardrails: str = "strict", control: str = "reactive",
                 granularity: str = "fine", state_mgmt: str = "scratchpad",
                 retry: str = "retry_rollback", budget: int | None = None,
                 reward_shaping: bool = True):
        self.sc = scenario
        self.seed = seed
        self.spec = (int(spec[0]), int(spec[1]))
        self.cfg = HarnessConfig(controller="rl", layer="near_rt",
                                 guardrails=guardrails, control=control,
                                 tool_granularity=granularity,
                                 state_management=state_mgmt, retry=retry,
                                 predictor=predictor)
        self.budget = budget or scenario.max_steps
        self.reward_shaping = reward_shaping
        self.ep: Episode | None = None
        self.k = 0
        self.done = True
        self._prev_bad = 0.0

    # ── 生命周期 ──────────────────────────────────────────────
    def reset(self):
        self.ep = Episode(self.sc, self.seed, self.cfg, budget=self.budget)
        self.k = 0
        self.done = False
        self.ep.obs_ring.append(self.ep.env.observe())
        self._prev_bad = self._badness()
        return self.state()

    def step(self, action: int):
        assert not self.done, "episode finished; call reset()"
        ep = self.ep
        cad, stale = self.spec
        acted = False
        unsafe = 0

        if self.k % cad == 0:                       # 节奏：这一步是否允许动手
            op = self._decode(action)
            if op is not None:
                intents = [{"kind": "op", "op": op}]
                calls = ep.registry.plan_calls(intents)
                for c in calls:
                    before = ep.mutations
                    ep._execute(c)
                    if ep.mutations > before:
                        acted = True
                unsafe = sum(1 for v in ep.last_verdicts if v.get("unsafe"))

        # 推进时钟：追加一帧新观测
        ep.env.tick() if hasattr(ep.env, "tick") else None
        ep.obs_ring.append(ep.env.observe())
        self.k += 1

        # 成功 / 超时
        success = bool(self.sc.success(ep.env))
        if success or self.k >= self.budget:
            self.done = True

        # 奖励
        bad = self._badness()
        r = 0.0
        if self.reward_shaping:
            r += (self._prev_bad - bad)              # 每减少一点"坏"，+1
        self._prev_bad = bad
        if success:
            r += 5.0
        r -= 0.05 * unsafe                           # 不安全操作轻微惩罚
        r -= 0.01                                    # 每步微小成本

        info = {"success": success, "acted": acted, "unsafe": unsafe,
                "t": self.k, "badness": bad}
        return (self.state() if not self.done else np.zeros(self.state_dim, np.float32),
                r, self.done, info)

    # ── 观测编码 ──────────────────────────────────────────────
    @property
    def state_dim(self) -> int:
        return (MAX_LINKS * 6) + (MAX_FLOWS * 7) + 6 + MAX_ALARMS + 4

    def state(self):
        ep = self.ep
        cad, stale = self.spec
        ring = ep.obs_ring
        idx = max(0, len(ring) - 1 - stale)          # 与主实验同一条陈旧规则
        obs = ring[idx] if ring else ep.env.observe()
        return encode(obs)

    # ── 动作解码 ──────────────────────────────────────────────
    def _decode(self, action: int):
        kind, oi, act = ACTIONS[action]
        obs = self.ep.obs_ring[-1]
        if kind == "noop":
            return None
        if kind == "link":
            links = obs["links"]
            if oi >= len(links):
                return None
            L = links[oi]
            if act == "up":
                return {"type": "link_set", "link": L["id"], "up": True}
            if act == "down":
                return {"type": "link_set", "link": L["id"], "up": False}
            cap = float(L.get("capacity") or 1000.0)
            if act == "cap_up":
                return {"type": "link_capacity", "link": L["id"], "capacity": cap * 1.6}
            return {"type": "link_capacity", "link": L["id"], "capacity": max(50.0, cap * 0.5)}
        if kind == "flow":
            flows = obs["flows"]
            if oi >= len(flows):
                return None
            return {"type": "reprovision", "flow": flows[oi]["id"]}
        if kind == "class":
            cls = CLASS_NAMES[oi]
            pol = obs.get("policy") or {}
            entry = pol.get(cls)
            cur = float((entry.get("ceiling") if isinstance(entry, dict) else entry) or 500.0)
            new = cur * 1.5 if act == "qos_up" else max(50.0, cur * 0.6)
            return {"type": "qos_set", "cls": cls, "ceiling": new}
        if kind == "alarm":
            alarms = obs["alarms"]
            if oi >= len(alarms):
                return None
            return {"type": "ack_alarm", "alarm": alarms[oi]["id"]}
        return None

    def action_mask(self):
        """哪些动作在当前观测下是有效的（对象存在）。"""
        m = np.zeros(N_ACTIONS, np.float32)
        obs = self.ep.obs_ring[-1] if self.ep.obs_ring else self.ep.env.observe()
        nl, nf, na = len(obs["links"]), len(obs["flows"]), len(obs["alarms"])
        for i, (kind, oi, _a) in enumerate(ACTIONS):
            ok = (kind == "noop"
                  or (kind == "link" and oi < nl)
                  or (kind == "flow" and oi < nf)
                  or (kind == "class" and oi < MAX_CLASSES)
                  or (kind == "alarm" and oi < na))
            m[i] = 1.0 if ok else 0.0
        return m

    def _badness(self) -> float:
        """越小越好：未达标流数 + 未确认的严重告警数。"""
        env = self.ep.env
        bad = sum(1 for f in env.flows.values() if not f["sla_ok"])
        alarms = getattr(env, "alarms", []) or []
        if isinstance(alarms, dict):
            alarms = list(alarms.values())
        crit = sum(1 for a in alarms
                   if isinstance(a, dict) and a.get("severity") == "critical"
                   and not a.get("acked"))
        return float(bad) + float(crit)


def encode(obs: dict) -> np.ndarray:
    """定长状态编码。所有量都做了尺度归一，便于网络学习。"""
    v = np.zeros((MAX_LINKS * 6) + (MAX_FLOWS * 7) + 6 + MAX_ALARMS + 4, np.float32)
    p = 0
    for L in obs.get("links", [])[:MAX_LINKS]:
        v[p:p + 6] = (1.0 if L.get("up") else 0.0,
                      float(L.get("util") or 0.0),
                      min(float(L.get("capacity") or 0.0) / 2000.0, 2.0),
                      min(float(L.get("delay_ms") or 0.0) / 100.0, 3.0),
                      min(float(L.get("load_mbps") or 0.0) / 1000.0, 3.0),
                      float(L.get("weight") or 0) / 3.0)
        p += 6
    p = MAX_LINKS * 6
    for f in obs.get("flows", [])[:MAX_FLOWS]:
        dem = max(float(f.get("demand_mbps") or 1.0), 1.0)
        cls = CLASS_NAMES.index(f["cls"]) if f.get("cls") in CLASS_NAMES else 0
        onehot = [0.0, 0.0, 0.0]
        onehot[cls] = 1.0
        v[p:p + 7] = (1.0 if f.get("sla_ok") else 0.0,
                      min(float(f.get("delivered_mbps") or 0.0) / dem, 2.0),
                      min(float(f.get("loss") or 0.0), 1.0),
                      min(float(f.get("delay_ms") or 0.0) / 100.0, 3.0),
                      *onehot)
        p += 7
    p = MAX_LINKS * 6 + MAX_FLOWS * 7
    pol = obs.get("policy") or {}
    for i, c in enumerate(CLASS_NAMES):
        entry = pol.get(c)
        ceil = entry.get("ceiling") if isinstance(entry, dict) else entry
        v[p + i] = min(float(ceil or 0.0) / 1000.0, 2.0)
    v[p + 3] = min(float(obs.get("t") or 0) / 20.0, 2.0)
    v[p + 4] = len(obs.get("alarms", [])) / 6.0
    v[p + 5] = 0.0
    p += 6
    for j, a in enumerate(obs.get("alarms", [])[:MAX_ALARMS]):
        v[p + j] = (2.0 if a.get("severity") == "critical" else 1.0) * (
            0.5 if a.get("acked") else 1.0)
    return v
