#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""水网数字孪生：跨领域验证用的第二物理域。

## 为什么用"继承 + 换物理"这种写法

跨领域验证要回答的是：**控制层位置效应是分组网络的特例，还是任何
"感知—决策—执行"回路的普遍性质？**

要让这个比较有意义，**智能体一侧必须完全相同**：同一套工具、同一套护栏、
同一套控制器、同一种层定义（cadence + staleness）。唯一该变的是**物理**。

所以本类继承 `NetworkEmulator`，**保留 links / flows / policy / alarms 的全部
数据结构与接口**，只覆盖三个方法把它变成水力：

    recompute()        用液压方程替代带宽/时延方程
    tick()             水池水位**随时间积分**（这是与分组网络的关键差别）
    _generate_alarms() 水网特有的告警种类

智能体看到的字段名不变（link/flow/demand/delivered/sla_ok），
但**底下跑的是水，不是包**。

## 关键物理差别：**记忆**

* 分组网络**无记忆**：智能体不动，状态就不变。
* 水网**有记忆**：水池水位会自行积分演化。供水中断时水位持续下降，
  即使随后修复管路，也要**等水位回升**才能恢复压力。

**可证伪的预测**：既然状态在"不作为"时也会变化，
**陈旧遥测在水网里应当比在分组网络里更昂贵**（层效应更强，尤其 staleness 轴）。

## 字段映射（智能体侧不变，物理侧重解释）

| 网络字段 | 水网含义 | 单位 |
|---|---|---|
| `links[l].up` | 阀门/管段通断 | bool |
| `links[l].capacity` | 管段输水能力 | L/s |
| `links[l].base_latency` | 水在管内的滞留时间 | min |
| `links[l].util` | 管段负荷率 | — |
| `flows[f].demand` | 分区需水量 | L/s |
| `flows[f].delivered` | 实际供水量 | L/s |
| `flows[f].sla_delay` | **最低压力要求**（重解释） | m |
| `flows[f].delay` | **实际压力**（重解释） | m |
| `policy[cls].ceiling` | 该类最大供水 | L/s |

## 新增状态

`self.tanks[node]` —— 每个节点的水池：level / area / elevation / min / max
"""
from __future__ import annotations

import copy
import hashlib
import json

from .emulator import NetworkEmulator
from .topology import NODES, TICK_SECONDS

# 水池参数（每个节点一座）。
# 标定目标：断水后水位应在"一个回合（4~8 步）"内降到告警线，
# 这样"记忆"才落在实验可观测的时间尺度上。
TANK_AREA_M2 = 120.0          # 池底面积
TANK_MIN_LEVEL = 5.0          # 保底水位（低于此即告警）
TANK_MAX_LEVEL = 14.0         # 溢流水位
TANK_INIT_LEVEL = 9.0         # 初始水位
ELEVATION = {n: 6.0 + 2.0 * i for i, n in enumerate(NODES)}   # 各节点标高（m）

# 水力参数（由健康态必须全通过反推标定，见 kb/21）
FRICTION_K = 0.50             # 沿程损失系数：h_f = K * (Q/100)^2 * 管段数
PUMP_HEAD = 40.0              # 泵站扬程（m）
PUMP_INFLOW = 260.0           # 每个在用节点的泵站补水量（L/s）
TICK_MINUTES = 5.0            # **一步 = 5 分钟**（水网控制回路的时间尺度，非秒级）
# 由此：净流量 -230 L/s 时，水位每步约降 0.575 m → 6 步降约 3.5 m，逼近告警线

# 压力 SLA：每个分区的最低压力要求（m），复用到 flows[f]['sla_delay']
MIN_PRESSURE = {"gold": 22.0, "silver": 18.0, "bronze": 14.0}


class WaterEmulator(NetworkEmulator):
    """配水网孪生。接口与 NetworkEmulator 完全一致。"""

    def __init__(self, seed: int = 0):
        # 必须先建水池：父类 __init__ 末尾会调用被覆盖的 recompute()，它要用到 tanks
        self.tanks = {
            n: {"id": n, "level": TANK_INIT_LEVEL, "area": TANK_AREA_M2,
                "elevation": ELEVATION[n], "min": TANK_MIN_LEVEL,
                "max": TANK_MAX_LEVEL, "inflow": 0.0, "outflow": 0.0}
            for n in NODES
        }
        super().__init__(seed)
        # 把 SLA 字段重解释为最低压力要求
        for f in self.flows.values():
            f["sla_delay"] = MIN_PRESSURE[f["cls"]]
        self.recompute()

    # ── 水力核心 ────────────────────────────────────────────────
    def recompute(self) -> None:
        """由当前管网与水池状态计算供水、压力与 SLA。"""
        for l in self.links.values():
            l["load"] = 0.0

        eff: dict[str, float] = {}
        for fid, f in self.flows.items():
            ceiling = self.policy[f["cls"]]["ceiling"]
            eff[fid] = min(f["demand"], ceiling)
            for lid in f["path"] or []:
                if not self.links[lid]["up"]:
                    break                      # 断管：供水到此为止
                self.links[lid]["load"] += eff[fid]

        # 管段负荷与摩阻损失（Q^2 非线性）
        for l in self.links.values():
            cap = max(l["capacity"], 1.0)
            l["util"] = min(l["load"] / cap, 5.0)
            over = max(0.0, l["util"] - 0.85)
            l["loss"] = min(0.5, over * 2.0) if l["up"] else 1.0
            # 滞留时间随负荷上升
            l["delay"] = min(l["base_latency"] + 20.0 * over, 60000.0)

        # 每个分区的供水与压力
        for fid, f in self.flows.items():
            path = f["path"] or []
            down = any(not self.links[lid]["up"] for lid in path)
            src = f["src"]
            tank = self.tanks[src]

            if down or not path:
                f["delivered"], f["delay"], f["loss"] = 0.0, 0.0, 1.0
            else:
                loss = 1.0
                for lid in path:
                    loss *= (1.0 - self.links[lid]["loss"])
                if DELIVERY_MODEL == "hard_cap":
                    # 水网原版：供水量受最小管段容量的**硬上限**约束
                    q = min([self.links[lid]["capacity"] for lid in path] + [eff[fid]])
                else:
                    # 消融版：与分组网络同构——容量只通过"利用率→损失"起作用
                    q = eff[fid]
                f["delivered"] = round(q * loss, 3)
                f["loss"] = round(1.0 - loss, 6)

                # 压力 = 泵扬程 + 水源池位 - 标高差 - 沿程损失(Q^2)
                # 摩阻：默认按 Q²（非线性）。消融时用 PRESSURE_EXPONENT=1（线性），
                # K 同步重标定，使**健康态压力一致**，从而只改非线性、不改难度基准。
                h_fric = FRICTION_K * (f["delivered"] / 100.0) ** PRESSURE_EXPONENT * len(path)
                elev_gain = tank["elevation"] - ELEVATION[f["dst"]]
                f["delay"] = round(PUMP_HEAD + tank["level"] + elev_gain - h_fric, 3)

            # SLA：压力达标 **且** 供水量达标
            f["sla_ok"] = bool(
                f["delivered"] >= 0.9 * f["demand"]
                and f["delay"] >= f["sla_delay"]
            )

    def tick(self) -> None:
        """推进一个时间步（= TICK_MINUTES 分钟），并**积分水池水位**。

        这是与分组网络最本质的差别：**不作为 ≠ 状态不变**。

        水量平衡（三种工况，由构造保证健康态稳定）：
            供水正常   inflow = demand          → 净流量 0，水位稳定
            供水中断   inflow = 0               → 净流量 −demand，水位下降
            修复后低水位 inflow = demand × 1.5    → 净流量 +demand/2，缓慢回升
        因此"修好管路"不等于"立刻恢复"，**要考虑补水时间**——
        这就是陈旧遥测在水网里代价更高的原因。
        """
        for tank in self.tanks.values():
            tank["inflow"] = 0.0
            tank["outflow"] = 0.0

        for nid, tank in self.tanks.items():
            incident = [l for l in self.links.values() if l["a"] == nid or l["b"] == nid]
            supplied = bool(incident) and all(l["up"] for l in incident)
            demand_here = sum(f["demand"] for f in self.flows.values() if f["src"] == nid)
            if demand_here <= 0:
                continue
            tank["outflow"] = demand_here
            if supplied:
                # 低水位时泵站加大出力补水
                tank["inflow"] = demand_here * (1.5 if tank["level"] < TANK_INIT_LEVEL else 1.0)

        for tank in self.tanks.values():
            net = tank["inflow"] - tank["outflow"]          # L/s
            d_level = net * (TICK_MINUTES * 60.0) / (tank["area"] * 1000.0)
            tank["level"] = max(0.0, min(tank["max"] + 3.0, tank["level"] + d_level))

        self.t += 1
        self.recompute()
        self._generate_alarms()
        self.kpi_history.append(self.kpi())

    # ── 告警 ────────────────────────────────────────────────────
    def _generate_alarms(self) -> None:
        for l in self.links.values():
            if not l["up"]:
                self._raise_alarm("PIPE_CLOSED", l["id"], "critical",
                                  f"pipe {l['id']} valve closed")
            elif l["util"] >= 0.85:
                self._raise_alarm("PIPE_OVERLOAD", l["id"], "major",
                                  f"util {l['util']:.2f}")
        for tid, tank in self.tanks.items():
            if tank["level"] < tank["min"]:
                self._raise_alarm("TANK_LOW", tid, "critical",
                                  f"level {tank['level']:.2f} m below reserve")
            elif tank["level"] > tank["max"]:
                self._raise_alarm("TANK_OVERFLOW", tid, "critical",
                                  f"level {tank['level']:.2f} m above overflow")
        for f in self.flows.values():
            if not f["sla_ok"]:
                if f["delay"] < f["sla_delay"] and f["delivered"] > 0:
                    self._raise_alarm("PRESSURE_LOW", f["id"], "critical",
                                      f"pressure {f['delay']:.1f} < {f['sla_delay']:.0f} m")
                else:
                    self._raise_alarm("SUPPLY_SHORT", f["id"], "critical",
                                      f"delivered {f['delivered']:.0f}/{f['demand']:.0f} L/s")
        for cls, pol in self.policy.items():
            demand = sum(f["demand"] for f in self.flows.values() if f["cls"] == cls)
            if demand > 0 and pol["ceiling"] < demand:
                self._raise_alarm("SUPPLY_CAP", cls, "major",
                                  f"cap {pol['ceiling']:.0f} < demand {demand:.0f} L/s")

    # ── 故障注入（沿用网络侧的种类名，语义换到水网）──────────────
    def inject(self, fault: dict) -> None:
        kind = fault["type"]
        if kind == "fault_linkset":
            self.links[fault["link"]]["up"] = bool(fault["up"])
        elif kind == "fault_congestion":
            self.links[fault["link"]]["capacity"] *= float(fault["factor"])
        elif kind == "fault_misconfig":
            self.policy[fault["cls"]]["ceiling"] = float(fault["ceiling"])
        elif kind == "fault_alarm_storm":
            import random as _r
            r = _r.Random(self.seed * 7919 + self.t)
            for _ in range(int(fault["count"])):
                lid = r.choice([l["id"] for l in self.links.values()])
                self.alarm_seq += 1
                self.alarms.append({
                    "id": f"A{self.alarm_seq:03d}", "kind": "SENSOR_NOISE",
                    "target": lid, "severity": r.choice(["major", "minor", "warning"]),
                    "detail": "sensor noise (duplicate suppressed upstream)",
                    "t": self.t, "acked": False})
        else:
            raise ValueError(f"unknown fault {kind!r}")
        self.recompute()

    # ── 状态哈希：必须把水位算进去，否则回放验证不成立 ──────────
    def state_hash(self) -> str:
        payload = json.dumps([
            self.config_fingerprint(),
            self.kpi(),
            {k: round(v["level"], 6) for k, v in sorted(self.tanks.items())},
        ], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ── 快照 / 恢复 / 克隆：把水池一起带上 ──────────────────────
    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["tanks"] = copy.deepcopy(self.tanks)
        return snap

    def restore(self, snap: dict) -> None:
        super().restore(snap)
        if "tanks" in snap:
            self.tanks = copy.deepcopy(snap["tanks"])

    def clone(self) -> "WaterEmulator":
        new = WaterEmulator.__new__(WaterEmulator)
        new.__dict__ = copy.deepcopy({k: v for k, v in self.__dict__.items()
                                      if k != "rng"})
        new.rng = __import__("random").Random(self.seed)
        return new

    # ── 便于人读 ────────────────────────────────────────────────
    def tank_summary(self) -> str:
        return " ".join(f"{k}={v['level']:.1f}m" for k, v in self.tanks.items())


__all__ = ["WaterEmulator", "MIN_PRESSURE", "ELEVATION", "TANK_INIT_LEVEL"]


# ── 剂量-反应实验用：可调记忆时间尺度 ─────────────────────────────
# 水池面积决定"水位变化多快" = 环境的记忆时间尺度。
# 面积越小 → 排空越快 → 状态在智能体不知情时变化越多 → 陈旧遥测代价应越大。
# 必须在构建 Episode **之前**调用（WaterEmulator.__init__ 会读取该模块常量）。
# 供水模型开关（消融用）：hard_cap = 水网原版；utilization_loss = 与分组网络同构
DELIVERY_MODEL = "hard_cap"


# 压力对供水的指数（消融用）：2.0 = 水网原版（Q²，非线性）；1.0 = 线性
PRESSURE_EXPONENT = 2.0
# 线性模式下重标定的 K，使健康态 F0 的摩阻与原版相同（15.36 m）
FRICTION_K_LINEAR = 1.6


def set_pressure_exponent(exp: float) -> float:
    """切换压力-流量关系的指数，并重标定 K 以保持健康态压力不变。"""
    global PRESSURE_EXPONENT, FRICTION_K
    old = PRESSURE_EXPONENT
    PRESSURE_EXPONENT = float(exp)
    FRICTION_K = FRICTION_K_LINEAR if exp == 1.0 else 0.50
    return old


def set_delivery_model(model: str) -> str:
    """切换供水模型。'hard_cap'（默认，水网原版）或 'utilization_loss'（消融）。"""
    global DELIVERY_MODEL
    old = DELIVERY_MODEL
    DELIVERY_MODEL = model
    return old


def set_tank_area(area_m2: float) -> float:
    """设定水池面积，返回旧值。用于扫描记忆时间尺度。"""
    global TANK_AREA_M2
    old = TANK_AREA_M2
    TANK_AREA_M2 = float(area_m2)
    return old


def get_tank_area() -> float:
    return TANK_AREA_M2


def drain_rate_per_step(area_m2: float | None = None,
                        demand: float = 480.0) -> float:
    """断水时每步水位下降（m/步）。demand 取 r0 的本地需求（F0+F2 = 480 L/s）。"""
    a = area_m2 if area_m2 is not None else TANK_AREA_M2
    return demand * (TICK_MINUTES * 60.0) / (a * 1000.0)


def memory_timescale_steps(area_m2: float | None = None,
                           demand: float = 480.0) -> float:
    """从标称水位降到告警线需要几步——记忆时间尺度的直接度量。"""
    return (TANK_INIT_LEVEL - TANK_MIN_LEVEL) / drain_rate_per_step(area_m2, demand)
