#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方案 B：把动作空间改成"从候选集里选"。

为什么推翻 v1
-------------
v1（`rl_env.py`）的动作空间是 73 个**固定模板**动作：

    链路: up / down / cap×1.6 / cap×0.5
    流:   reprovision
    类:   qos×1.5 / qos×0.6
    告警: ack（一次一个）

实测结果（n=300，同一协议）：

    随机          0.000
    手写贪心      0.200     ← 这套空间里的**最优**
    训练后的 RL   0.200     ← 与手写最优完全相同
    mock 控制器   0.940

分场景看根因：

    场景                   mock    贪心/RL
    S1_link_down          0.950    1.000    ← 空间能表达
    S2_congestion         0.983    0.000    ← 容量要"算出的目标值"，不是 ×1.6
    S3_policy_misconfig   0.850    0.000    ← QoS 同理
    S4_alarm_storm        0.950    0.000    ← 需要批量确认
    S5_double_fault       0.933    0.000

**RL 并没有学失败——它已经达到了那套空间的天花板（0.20）。
瓶颈是动作空间只能表达 5 个场景中 1 个的修法。**

v2 的设计
---------
* **候选集**由当前观测生成（用 `reasoning.fix_all` 的语义，值都是**算出来的**），
  并补上批量确认、备选路径等——保证是 mock 行为的**超集**。
* **动作 = 从候选中选一个**（结构化动作空间的标准做法）。
* **候选从智能体自己的视野生成**——视野陈旧时，候选本身就是错的。
  **这正是放置问题难的地方，也是 RL 该学的东西。**
* 策略网络是一个**打分器**：`score(state, candidate)`，对候选做 softmax。
  候选数可变，所以用打分而不是固定输出维度。

用法
----
    from aixnet.rl_env2 import CandEnv, N_MAX_CAND
"""
from __future__ import annotations

import copy
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aixnet.config import HarnessConfig     # noqa: E402
from aixnet.harness import Episode          # noqa: E402

MAX_LINKS, MAX_FLOWS, MAX_CLASSES, MAX_ALARMS = 12, 6, 3, 12
N_MAX_CAND = 16                  # 候选数上限（不足补 noop）
CLASS_NAMES = ("gold", "silver", "bronze")

# 候选类型 one-hot
OP_TYPES = ("link_set_up", "link_set_down", "link_capacity", "qos_set",
            "reprovision", "ack_all", "ack_one", "noop")
N_OPTYPE = len(OP_TYPES)
CAND_FEAT = N_OPTYPE + 4         # 类型 one-hot + 4 个目标/建议特征


# ── 状态编码（与 v1 相同，便于对比）──────────────────────────
def encode_state(obs: dict) -> np.ndarray:
    v = np.zeros(MAX_LINKS * 6 + MAX_FLOWS * 7 + 6 + MAX_ALARMS + 4, np.float32)
    for i, L in enumerate(obs.get("links", [])[:MAX_LINKS]):
        v[i*6:i*6+6] = (1.0 if L.get("up") else 0.0,
                        float(L.get("util") or 0.0),
                        min(float(L.get("capacity") or 0.0)/2000.0, 2.0),
                        min(float(L.get("delay_ms") or 0.0)/100.0, 3.0),
                        min(float(L.get("load_mbps") or 0.0)/1000.0, 3.0),
                        float(L.get("weight") or 0)/3.0)
    p = MAX_LINKS * 6
    for F in obs.get("flows", [])[:MAX_FLOWS]:
        dem = max(float(F.get("demand_mbps") or 1.0), 1.0)
        cls = CLASS_NAMES.index(F["cls"]) if F.get("cls") in CLASS_NAMES else 0
        oh = [0.0]*3; oh[cls] = 1.0
        v[p:p+7] = (1.0 if F.get("sla_ok") else 0.0,
                    min(float(F.get("delivered_mbps") or 0.0)/dem, 2.0),
                    min(float(F.get("loss") or 0.0), 1.0),
                    min(float(F.get("delay_ms") or 0.0)/100.0, 3.0), *oh)
        p += 7
    p = MAX_LINKS*6 + MAX_FLOWS*7
    pol = obs.get("policy") or {}
    for i, c in enumerate(CLASS_NAMES):
        e = pol.get(c)
        v[p+i] = min(float((e.get("ceiling") if isinstance(e, dict) else e) or 0)/1000.0, 2.0)
    v[p+3] = min(float(obs.get("t") or 0)/20.0, 2.0)
    v[p+4] = len(obs.get("alarms", []))/6.0
    p += 6
    # 特征编码必须定长 → 这里截断是对的。
    # 但要优先取 critical：否则告警风暴下前 12 个全是噪声，特征里看不到关键告警。
    al = obs.get("alarms", [])
    al = ([A for A in al if A.get("severity") == "critical"] + 
          [A for A in al if A.get("severity") != "critical"])[:MAX_ALARMS]
    for j, A in enumerate(al):
        v[p+j] = (2.0 if A.get("severity") == "critical" else 1.0) * (
            0.5 if A.get("acked") else 1.0)
    return v


STATE_DIM = MAX_LINKS*6 + MAX_FLOWS*7 + 6 + MAX_ALARMS + 4


# ── 候选生成 ──────────────────────────────────────────────────
def candidates(obs: dict) -> list[dict]:
    """从**给定观测**生成候选操作。值是算出来的，不是固定倍数。

    ⚠️ 关键：调用方传入的可能是**陈旧的**观测。候选会因此是错的——
    这正是放置问题难的地方，也正是要测的东西。
    """
    cands: list[dict] = []
    links = obs.get("links", [])[:MAX_LINKS]
    flows = obs.get("flows", [])[:MAX_FLOWS]
    # ⚠️ 这里**不能截断**。S4_alarm_storm 会在一回合里产生 40+ 个告警，
    #    其中 critical 的排在最后；初版用 [:MAX_ALARMS] 截断，
    #    结果两个 critical 被切掉，ack_all 候选从未出现，S4 必然失败。
    #    截断只属于定长**特征编码**，不属于候选生成。
    alarms = obs.get("alarms", [])
    pol = obs.get("policy") or {}

    # ① 批量确认（mock 的告警行为就是这个）
    crit = [A for A in alarms if A.get("severity") == "critical" and not A.get("acked")]
    if crit:
        cands.append({"type": "ack_all", "_n": len(crit), "_rec": 1})

    # ② 每条 down 的链路 → 拉起
    for L in links:
        if not L.get("up"):
            cands.append({"type": "link_set", "link": L["id"], "up": True})

    # ③ 每条链路的容量 → **算出的目标值**（负载 × 1.25，留 25% 余量）
    for L in links:
        load = float(L.get("load_mbps") or 0.0)
        cap = float(L.get("capacity") or 0.0)
        if load > 0 and cap > 0:
            target = max(load * 1.25, cap)
            if target > cap * 1.02:
                cands.append({"type": "link_capacity", "link": L["id"],
                              "capacity": round(target, 1)})

    # ④ 每个类别的上限 → 算出的目标值（该类所有流的需求之和）
    need = {c: 0.0 for c in CLASS_NAMES}
    for F in flows:
        if F.get("cls") in need:
            need[F["cls"]] += float(F.get("demand_mbps") or 0.0)
    for c in CLASS_NAMES:
        k = need[c]
        e = pol.get(c)
        cur = float((e.get("ceiling") if isinstance(e, dict) else e) or 0.0)
        if k > 0 and k > cur * 1.02:
            cands.append({"type": "qos_set", "cls": c, "ceiling": round(k, 1)})

    # ⑤ 每条不达标的流 → 重路由（多候选路径）
    for F in flows:
        if not F.get("sla_ok"):
            cands.append({"type": "reprovision", "flow": F["id"]})

    # ⑥ 兜底：模仿 fix_all 的诊断结果（若可用）
    try:
        from aixnet import reasoning
        from aixnet.controllers import DecisionContext
        ctx = DecisionContext(step=0, seed=0)
        ctx.rng = random.Random(0)
        failing = [f for f in flows if not f.get("sla_ok")]
        roots = reasoning.diagnose_all(obs, failing, ctx.rng) if failing else []
        if roots:
            for o in reasoning.fix_all(roots, obs, failing, ctx, lambda x: True):
                o = dict(o); o["_rec"] = 1
                if o not in cands:
                    cands.append(o)
    except Exception:
        pass

    # 去重：键忽略内部标记，但**标记要保留到留下的那一份上**。
    # 初版直接丢弃后来的重复项，结果 fix_all 的 `_rec` 标记被通用候选挤掉，
    # "总选推荐"策略因此永远找不到推荐项、退化成一味 noop。
    seen: dict = {}
    uniq: list[dict] = []
    for c in cands:
        k = tuple(sorted((kk, str(vv)) for kk, vv in c.items() if not kk.startswith("_")))
        if k in seen:
            if c.get("_rec"):
                uniq[seen[k]]["_rec"] = 1          # 把推荐标记合并到已保留的那份
            continue
        seen[k] = len(uniq)
        uniq.append(dict(c))
    return uniq[:N_MAX_CAND - 1]


def cand_features(obs: dict, cands: list[dict]) -> np.ndarray:
    """每个候选的特征向量。"""
    F = np.zeros((N_MAX_CAND, CAND_FEAT), np.float32)
    links = {L["id"]: (i, L) for i, L in enumerate(obs.get("links", [])[:MAX_LINKS])}
    flows = {f["id"]: (i, f) for i, f in enumerate(obs.get("flows", [])[:MAX_FLOWS])}
    _al = obs.get("alarms", [])
    alarms = ([A for A in _al if A.get("severity") == "critical"] +
              [A for A in _al if A.get("severity") != "critical"])[:MAX_ALARMS]

    for j, c in enumerate(cands[:N_MAX_CAND]):
        t = c.get("type")
        if t == "ack_all":
            F[j, 0] = 0; F[j, 5] = 1.0; F[j, -3] = min(c.get("_n", 0)/6.0, 1.0)
        elif t == "link_set":
            if c.get("up"):
                F[j, 0] = 1.0
            else:
                F[j, 1] = 1.0
            li = links.get(c.get("link"), (0, {}))[0]
            F[j, -4] = li / MAX_LINKS
        elif t == "link_capacity":
            F[j, 2] = 1.0
            li = links.get(c.get("link"), (0, {}))[0]
            F[j, -4] = li / MAX_LINKS
        elif t == "qos_set":
            F[j, 3] = 1.0
            F[j, -4] = CLASS_NAMES.index(c["cls"])/3.0 if c.get("cls") in CLASS_NAMES else 0
        elif t == "reprovision":
            F[j, 4] = 1.0
            fi = flows.get(c.get("flow"), (0, {}))[0]
            F[j, -4] = fi / MAX_FLOWS
        else:
            F[j, 7] = 1.0            # noop
        F[j, -1] = 1.0 if c.get("_rec") else 0.0
    # 末位候选固定为 noop
    F[len(cands[:N_MAX_CAND]), 7] = 1.0
    return F


# ── 环境 ──────────────────────────────────────────────────────
class CandEnv:
    """动作 = 从候选集中选一个。候选由**当前（可能陈旧的）观测**生成。"""

    def __init__(self, scenario, seed: int = 0, spec=(1, 0), predictor: str = "none",
                 guardrails: str = "strict", control: str = "reactive",
                 budget: int | None = None):
        self.sc = scenario
        self.seed = seed
        self.spec = (int(spec[0]), int(spec[1]))
        self.predictor = predictor
        self.cfg = HarnessConfig(controller="rl", layer="near_rt",
                                 guardrails=guardrails, control=control,
                                 predictor=predictor)
        self.budget = budget or scenario.max_steps
        self.ep: Episode | None = None
        self.k = 0
        self.done = True
        self._prev_bad = 0.0
        self._cands: list[dict] = []

    # ── 观测（含陈旧与预测器）────────────────────────────────
    def _view(self) -> dict:
        """智能体实际看到的观测：陈旧 + 预测器。**这是 v1 漏掉的一环。**"""
        ring = self.ep.obs_ring
        _c, stale = self.spec
        raw = ring[max(0, len(ring) - 1 - stale)]
        if self.predictor != "none" and len(ring) >= 2:
            if self.predictor == "oracle":
                raw = ring[-1]
            elif self.predictor == "delta":
                from aixnet.harness import _extrapolate
                raw = _extrapolate(ring[-2], ring[-1], steps=max(0, stale))
            elif self.predictor == "learned":
                from aixnet.harness import _learned_predict
                raw = _learned_predict(ring[-2], ring[-1], steps=max(0, stale))
        return raw

    def reset(self):
        self.ep = Episode(self.sc, self.seed, self.cfg, budget=self.budget)
        self.k = 0
        self.done = False
        self.ep.obs_ring.append(self.ep.env.observe())
        self._prev_bad = self._badness()
        self._refresh()
        return self.state()

    def _refresh(self):
        obs = self._view()
        self._cands = candidates(obs) + [{"type": "noop"}]

    def state(self):
        return encode_state(self._view())

    def cand_feats(self):
        return cand_features(self._view(), self._cands)

    @property
    def n_cand(self) -> int:
        return min(len(self._cands), N_MAX_CAND)

    # ── 一步 ──────────────────────────────────────────────────
    def step(self, choice: int):
        assert not self.done
        ep = self.ep
        cad, _stale = self.spec
        applied = 0
        unsafe = 0
        if self.k % cad == 0:
            c = self._cands[choice] if choice < len(self._cands) else {"type": "noop"}
            if c.get("type") != "noop":
                op = {k: v for k, v in c.items() if not k.startswith("_")}
                if op.get("type") == "ack_all":
                    # ⚠️ 这里**同样不能截断**。初版照抄了特征编码的 [:MAX_ALARMS]，
                    #    于是告警风暴里排在最后的 critical 被切掉，
                    #    ack_all 变成"选中了但什么都不做"（applied=0）。
                    #    同一个截断 bug 在这个文件里出现过三处，全部只应出现在特征编码里。
                    obs = self._view()
                    crit = [A for A in obs.get("alarms", [])
                            if A.get("severity") == "critical" and not A.get("acked")]
                    ints = [{"kind": "op", "op": {"type": "ack_alarm", "alarm": A["id"]}}
                            for A in crit[:6]]
                else:
                    ints = [{"kind": "op", "op": op}]
                for call in ep.registry.plan_calls(ints):
                    before = ep.mutations
                    ep._execute(call)
                    if ep.mutations > before:
                        applied += 1
                unsafe = sum(1 for v in ep.last_verdicts if v.get("unsafe"))

        ep.env.recompute()
        ep.obs_ring.append(ep.env.observe())
        self.k += 1

        success = bool(self.sc.success(ep.env))
        if success or self.k >= self.budget:
            self.done = True

        bad = self._badness()
        r = (self._prev_bad - bad) * 1.0
        self._prev_bad = bad
        if success:
            r += 5.0
        r -= 0.05 * unsafe
        r -= 0.01

        info = {"success": success, "badness": bad, "t": self.k, "applied": applied}
        self._refresh()
        return (self.state() if not self.done else np.zeros(STATE_DIM, np.float32),
                r, self.done, info)

    def _badness(self) -> float:
        env = self.ep.env
        bad = sum(1 for f in env.flows.values() if not f["sla_ok"])
        al = getattr(env, "alarms", []) or []
        if isinstance(al, dict):
            al = list(al.values())
        crit = sum(1 for a in al if isinstance(a, dict)
                   and a.get("severity") == "critical" and not a.get("acked"))
        return float(bad + crit)
