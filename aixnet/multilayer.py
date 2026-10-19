#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多层智能体：同一回合里，多个控制层各自以**自己的节奏与新鲜度**行动。

## 为什么做这个

主论文问的是"智能体该放在**哪一层**"。但真实部署从不只放一个：
near-RT 一个 xApp + non-RT 一个 rApp 是常态。**自然的问题是：分层协作是更好还是更糟？**

`kb/20` 的解耦网格已经给了一个预警信号：

    c2_s2 = 0.0215    ← 全场最差，比更慢的 c3_s2（0.2733）还差 25 pp

机制是**"视图早于自己的上一次动作 → 撤销自己的修复"**。
**多层部署正是这个陷阱的放大版**：慢层智能体的陈旧视图会去"修复"
快层刚刚做好的事情。

## 假设（可证伪）

    H_interfere：加入慢层智能体会**降低**成功率（因为它会撤销快层的成果）
    H_help     ：加入慢层智能体提高成功率（多一个视角总是好的）

**两者的分界应当在于**：慢层的陈旧视野是否**早于快层的最近一次动作**。

## 实现

`CompositeController` 持有 N 个子控制器，第 i 个对应第 i 组 (cadence, staleness)。
harness 在每一步遍历各组：

    for i, (cad, stale) in enumerate(specs):
        if k % cad != 0:  continue          # 这一层这步不动手
        obs_i = 按 stale 取观测
        intents = agents[i].decide(obs_i, ctx)
        应用

**当 `layer_specs` 未设置时，harness 走原路径，行为一字不变。**
"""
from __future__ import annotations

from .config import LAYERS, HarnessConfig


class CompositeController:
    """把 N 个同类型控制器打包，每个对应一组 (cadence, staleness)。

    对外暴露与普通控制器相同的 `decide(obs, ctx)`（用第 0 个，兼容单层调用路径），
    另外提供 `decide_at(i, obs, ctx)` 供 harness 的多层路径使用。
    """

    def __init__(self, kind: str, control: str, seed: int, n: int,
                 batch: int | None = None):
        self.kind = kind
        self.agents = []
        for i in range(n):
            self.agents.append(_make_one(kind, control, seed, i, batch))
        self._active = 0

    # --- 单层兼容接口 ---
    def decide(self, obs, ctx):
        return self.agents[self._active].decide(obs, ctx)

    def on_no_progress(self, obs, ctx):
        fn = getattr(self.agents[self._active], "on_no_progress", None)
        return fn(obs, ctx) if fn else []

    # --- 多层接口 ---
    def decide_at(self, idx: int, obs, ctx):
        if idx >= len(self.agents):
            return []
        self._active = idx
        return self.agents[idx].decide(obs, ctx) or []

    def on_no_progress_at(self, idx: int, obs, ctx):
        if idx >= len(self.agents):
            return []
        fn = getattr(self.agents[idx], "on_no_progress", None)
        return fn(obs, ctx) if fn else []

    def bind(self, env) -> None:
        for a in self.agents:
            if hasattr(a, "bind"):
                a.bind(env)


def _make_one(kind: str, control: str, seed: int, i: int, batch: int | None):
    """构造第 i 个子控制器。子控制器之间 seed 不同，避免完全同步。"""
    if kind == "mock":
        from .controllers import MockController
        kw = {"control": control, "seed": seed + i * 101}
        if batch is not None:
            kw["batch"] = batch
        try:
            return MockController(**kw)
        except TypeError:
            kw.pop("batch", None)
            return MockController(**kw)
    if kind == "baseline_rules":
        from .baselines import BaselineRules
        return BaselineRules()
    if kind == "baseline_search":
        from .baselines import SearchBaseline
        return SearchBaseline()
    if kind == "llm":
        # 每个子智能体一个独立的 LLMController（各自持有 client 与 granularity）
        from .llm_client import LLMController, OpenAICompatClient
        c = LLMController(OpenAICompatClient(), "fine")
        if hasattr(c, "bind"):
            pass
        return c
    raise ValueError(f"unknown controller kind {kind!r}")


# ── 预置的多层配置 ────────────────────────────────────────────────
# 每组是 (名称, [(cadence, staleness), ...])，全部沿用 LAYERS 里的定义
MULTILAYER_SETS = {
    "L_near_only":  [("near_rt",)],
    "L_non_only":   [("non_rt",)],
    "L_smo_only":   [("smo_intent",)],
    "L_near_non":   [("near_rt",), ("non_rt",)],
    "L_near_smo":   [("near_rt",), ("smo_intent",)],
    "L_all_three":  [("near_rt",), ("non_rt",), ("smo_intent",)],
    # 消融：把慢层的**新鲜度**单独拿掉（保留其节奏），检验"是不是陈旧造成的"
    "L_near_non_fresh": [("near_rt",), ("_c2_s0",)],
    "L_near_smo_fresh": [("near_rt",), ("_c3_s0",)],
}

# ── 角色分工：每层允许的操作类别 ────────────────────────────────
# 模拟真实 O-RAN 的授权边界：rApp（非实时）能设策略但不能直接改物理设备；
# xApp（近实时）能动链路/流，但策略不归它管。
ALL_OPS = frozenset({"link_set", "link_capacity", "qos_set", "reprovision",
                     "ack_alarm", "restore_snapshot"})
POLICY_ROLE = frozenset({"qos_set", "ack_alarm"})            # 只能设策略 + 确认告警
CONTROL_ROLE = frozenset({"link_set", "link_capacity",
                          "reprovision", "ack_alarm"})       # 只能动物理 + 确认告警
SENSE_ROLE = frozenset({"ack_alarm"})                        # 只能"看见"（确认=登记）

# 角色方案：名字 → 每层允许的类别
#
# ⚠️ 索引顺序跟 layer_specs 一致 = (快层, 慢层, ...)。
#    L_near_non 的 specs 是 ((1,0),(2,1)) → 索引 0 是 near_rt（快），索引 1 是 non_rt（慢）。
#    曾把顺序写反，冒烟测试立刻暴露（快层只被允许确认告警 → 整回合失败）。
_ROLE_RECIPES = {
    "R_redundant":   None,           # 现有：各层做同样的事（对照）
    "R_policy_ctrl": CONTROL_ROLE,   # 第 0 层（最快）动物理
    "R_sense_ctrl":  ALL_OPS,        # 第 0 层全权
    "R_policy_all":  ALL_OPS,        # 第 0 层全权
}
# 慢层（索引 ≥ 1）的角色：按方案不同
_SLOW_ROLE = {
    "R_redundant":   None,
    "R_policy_ctrl": POLICY_ROLE,    # 慢层只能设策略
    "R_sense_ctrl":  SENSE_ROLE,     # 慢层只能登记告警
    "R_policy_all":  POLICY_ROLE,    # 慢层只能设策略
}


def roles_for(name: str, n_layers: int) -> tuple | None:
    """按层数生成角色元组。索引 0 = 最快层。"""
    if _ROLE_RECIPES[name] is None:
        return None
    fast, slow = _ROLE_RECIPES[name], _SLOW_ROLE[name]
    return (fast,) + (slow,) * (n_layers - 1)


class _RoleSets(dict):
    """向后兼容：ROLE_SETS[名字] 在 2 层时仍返回 2 元组。"""
    def __getitem__(self, k):
        return roles_for(k, 2)


ROLE_SETS = _RoleSets({k: None for k in _ROLE_RECIPES})

# 消融用的合成层定义（cadence, staleness）
SYNTHETIC = {"_c2_s0": (2, 0), "_c3_s0": (3, 0)}


def specs_of(name: str) -> tuple[tuple[int, int], ...]:
    """把配置名解析成 ((cadence, staleness), ...)。"""
    out = []
    for grp in MULTILAYER_SETS[name]:
        for layer in grp:
            if layer in SYNTHETIC:
                out.append(SYNTHETIC[layer])
            else:
                spec = LAYERS[layer]
                out.append((spec["cadence"], spec["staleness"]))
    return tuple(out)


def make_config(name: str, controller: str = "mock", **kw) -> HarnessConfig:
    """按多层配置名构造 HarnessConfig（layer 字段留作标签用）。"""
    first = MULTILAYER_SETS[name][0][0]
    label = first if first in LAYERS else "near_rt"
    return HarnessConfig(controller=controller, layer=label,
                         layer_specs=specs_of(name), **kw)


__all__ = ["CompositeController", "MULTILAYER_SETS", "specs_of", "make_config",
           "SYNTHETIC", "ROLE_SETS", "roles_for", "ALL_OPS", "POLICY_ROLE",
           "CONTROL_ROLE", "SENSE_ROLE"]
