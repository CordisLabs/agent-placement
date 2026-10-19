#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三个域：排队/服务网络孪生。

为什么还要一个域
----------------
`kb/39` 的独立验证发现：**水网孪生与包网络共享的东西太多了**——

| 项 | 包网络 | 水网 |
|---|---|---|
| 链路/服务器数 | 8 | 8 |
| 流/类别数 | 3 / 3 | 3 / 3 |
| 容量范围 | (300, 1000) | (300, 1000) |
| 需求范围 | (160, 320) | (160, 320) |
| 观测 schema | — | **完全相同** |
| 故障种类 | 链路/容量/策略 | **同样三种** |
| 不动作时状态变化 | 否 | 否（水箱不在观测里） |

**所以"跨物理域"这个词用大了，实际只是换了状态演化内核。**

本域刻意在三个方面都与前两者不同：

1. **规模不同**：5 台服务器 / 4 个作业类 / 4 条作业流（不是 8/3/3）
2. **状态**：服务器积压量**随时间积分**，而且**在观测里可见**
   （`util` = 积压/缓冲上限，`delay_ms` = 排队时延）
   —— 前两域"不作为 = 状态不变"，这里**不作为 = 积压增长**
3. **故障类型**：新增 **到达突发**（需求瞬时放大）与 **缓冲溢出**
   —— 前两域没有这两类

模型
----
* 服务器 $i$：处理能力 $\\mu_i$（作业/秒），缓冲上限 $B_i$
* 作业流 $f$：到达率 $\\lambda_f$，经路径分配到服务器
* 每 tick：$q_i \\leftarrow \\max(0, q_i + (\\lambda_i - \\mu_i) \\cdot \\Delta t)$
* 排队时延（Little 定律）：$d_i = q_i / \\mu_i$
* 溢出：$q_i > B_i$ 时丢弃，丢包率 $= (q_i - B_i)/q_i$
"""
from __future__ import annotations

import random

from .emulator import NetworkEmulator

# ── 拓扑：刻意与前两域规模不同 ────────────────────────────────
Q_NODES = ["s0", "s1", "s2", "s3", "s4"]           # 5 台服务器
Q_CLASSES = ("platinum", "gold", "silver", "bronze")  # 4 个类别
Q_SERVERS = ["s1", "s2", "s3", "s4"]               # 可被分配的服务器
Q_STREAMS = ["J0", "J1", "J2", "J3"]               # 4 条作业流

Q_TICK_SECONDS = 1.0
# s0 是**备用容量池**：平时空闲，故障时可以接管 —— 这让"重路由"成为有效修复
Q_CAPACITY = {"s0": 1000.0, "s1": 900.0, "s2": 700.0, "s3": 500.0, "s4": 350.0}
Q_BUFFER = {"s0": 4000.0, "s1": 3000.0, "s2": 2400.0,
            "s3": 1800.0, "s4": 1200.0}
Q_DEMAND = {"J0": 300.0, "J1": 260.0, "J2": 220.0, "J3": 180.0}
Q_BASE_SERVICE_MS = 8.0
# 准入上限刻意**高于**任何单台服务器容量，以免上限掩盖容量约束
Q_POLICY = {"platinum": 2000.0, "gold": 2000.0, "silver": 2000.0, "bronze": 2000.0}
# 各类别的排队时延上限（SLA）
Q_SLA_MS = {"platinum": 120.0, "gold": 200.0, "silver": 320.0, "bronze": 500.0}
# 每条流一个主服务器 + 一个备用（s0）。重路由 = 切到备用。
Q_PATHS = {
    "J0": [["s1"], ["s0"]],
    "J1": [["s2"], ["s0"]],
    "J2": [["s3"], ["s0"]],
    "J3": [["s4"], ["s0"]],
}


class QueueEmulator(NetworkEmulator):
    """排队/服务网络孪生。接口与 `NetworkEmulator` 完全一致。

    与包网络/水网的关键差别：**积压量随时间积分，而且在观测里可见。**
    """

    def __init__(self, seed: int = 0):
        # 父类 __init__ 末尾会调用被覆盖的 recompute()，**那时拓扑还是包网络的**
        # （links 是 L0..L7、flows 是 F0..F2）。用这个标志让 recompute() 在
        # 拓扑替换完成之前保持沉默，否则会 KeyError 到 L2 之类的包网络实体。
        self._queue_ready = False
        # 必须先建队列：父类 __init__ 末尾会调用被覆盖的 recompute()
        self.queues = {
            n: {"id": n, "q": 0.0, "mu": Q_CAPACITY.get(n, 0.0),
                "buffer": Q_BUFFER.get(n, 1000.0), "arrivals": 0.0,
                "served": 0.0, "dropped": 0.0}
            for n in Q_NODES
        }
        super().__init__(seed)
        # 重新解释链路为服务器
        for lid, l in list(self.links.items()):
            if lid not in self.queues:
                del self.links[lid]
        for nid, s in self.queues.items():
            if nid not in self.links:
                self.links[nid] = {
                    "id": nid, "a": nid, "b": nid, "endpoints": [nid, nid],
                    "up": True, "capacity": s["mu"], "weight": 1,
                    "production": nid == "s0", "util": 0.0, "load": 0.0,
                    "load_mbps": 0.0, "loss": 0.0, "delay": Q_BASE_SERVICE_MS,
                    "base_latency": Q_BASE_SERVICE_MS,
                }
        # 重新解释流为作业流
        for fid in list(self.flows):
            if fid not in Q_STREAMS:
                del self.flows[fid]
        for i, sid in enumerate(Q_STREAMS):
            cls = Q_CLASSES[i % len(Q_CLASSES)]
            self.flows[sid] = {
                "id": sid, "cls": cls, "src": Q_PATHS[sid][0][0],
                "dst": Q_PATHS[sid][0][-1], "demand": Q_DEMAND[sid],
                "path": self._pick_path(sid, 0), "sla_delay": Q_SLA_MS[cls],
                "delivered": 0.0, "delay": 0.0, "loss": 0.0, "sla_ok": False,
            }
        self.policy = {c: {"ceiling": Q_POLICY[c], "min_share": 0.0}
                       for c in Q_CLASSES}
        self.candidates = {sid: [list(p) for p in Q_PATHS[sid]] for sid in Q_STREAMS}
        self._queue_ready = True
        self.recompute()

    @staticmethod
    def _pick_path(sid: str, idx: int):
        paths = Q_PATHS[sid]
        return list(paths[idx % len(paths)])

    def candidate_paths(self, flow_id: str) -> list:
        return [list(p) for p in Q_PATHS.get(flow_id, [])]

    # ── 排队核心 ──────────────────────────────────────────────
    def recompute(self) -> None:
        """由当前积压与容量计算时延、丢弃与 SLA。

        ⚠️ 这里**读的是队列的既有积压**（由 `tick()` 积分得到），
        所以 `recompute()` 不是纯函数——这是与包网络最本质的差别。
        """
        if not getattr(self, "_queue_ready", False):
            return                      # 拓扑尚未替换（父类 __init__ 阶段）
        # 到达率：受类别的准入上限约束
        for s in self.queues.values():
            s["arrivals"] = 0.0
        for fid, f in self.flows.items():
            ceiling = self.policy[f["cls"]]["ceiling"]
            eff = min(f["demand"], ceiling)
            f["_eff"] = eff
            path = f["path"] or []
            if not path or any(not self.links[n]["up"] for n in path):
                continue
            # 分配到路径末端那台服务器（简单分摊）
            self.queues[path[-1]]["arrivals"] += eff

        for nid, s in self.queues.items():
            l = self.links[nid]
            if not l["up"]:
                l["util"] = 1.0
                l["loss"] = 1.0
                l["delay"] = 1e6
                s["served"] = 0.0
                s["dropped"] = s["arrivals"]
                continue
            # ⚠️ 服务率必须从 link["capacity"] 读，而不是 queues[].mu：
            #    否则 `link_capacity` 操作改了 links 却不影响排队，故障无法用该 op 修复。
            mu = max(float(l["capacity"]), 1.0)
            s["mu"] = mu
            # 利用率 = 到达/容量（可超过 1，表示积压增长）
            l["util"] = min(s["arrivals"] / mu, 5.0)
            l["load_mbps"] = s["arrivals"]
            # 缓冲占用率 —— **这一项让积压对智能体可见**
            occ = s["q"] / max(s["buffer"], 1.0)
            if s["q"] > s["buffer"]:
                over = s["q"] - s["buffer"]
                l["loss"] = min(1.0, (over + s["arrivals"]) / max(s["arrivals"] + over, 1e-9))
                s["dropped"] = s["arrivals"] * l["loss"]
            else:
                l["loss"] = 0.0
                s["dropped"] = 0.0
            # Little 定律：排队时延 = 积压 / 处理率
            l["delay"] = Q_BASE_SERVICE_MS + 1000.0 * s["q"] / mu
            s["served"] = min(mu, s["arrivals"] + s["q"])

        for fid, f in self.flows.items():
            path = f["path"] or []
            if not path or any(not self.links[n]["up"] for n in path):
                f["delivered"], f["delay"], f["loss"] = 0.0, 0.0, 1.0
            else:
                surv = 1.0
                delay = 0.0
                for n in path:
                    surv *= (1.0 - self.links[n]["loss"])
                    delay += self.links[n]["delay"]
                f["loss"] = round(1.0 - surv, 6)
                f["delay"] = delay
                f["delivered"] = f.get("_eff", f["demand"]) * surv
            f["sla_ok"] = bool(
                f["delivered"] >= 0.9 * f["demand"] and f["delay"] <= f["sla_delay"]
            )

    def tick(self) -> None:
        """推进一个时间步，并**积分服务器积压**。

        这是本域与前两域最本质的差别：
        **不作为 ≠ 状态不变——积压会持续增长，而且智能体看得见。**
        """
        dt = Q_TICK_SECONDS
        for nid, s in self.queues.items():
            l = self.links[nid]
            if not l["up"]:
                s["q"] = min(s["buffer"] * 3.0, s["q"] + s["arrivals"] * dt)
                continue
            mu = max(float(l["capacity"]), 1.0)
            net = (s["arrivals"] - mu) * dt
            s["q"] = max(0.0, min(s["buffer"] * 3.0, s["q"] + net))

        self.t += 1
        self.recompute()
        self._generate_alarms()
        self.kpi_history.append(self.kpi())

    def _generate_alarms(self) -> None:
        for nid, s in self.queues.items():
            l = self.links[nid]
            if not l["up"]:
                self._raise_alarm("SERVER_DOWN", nid, "critical",
                                  f"server {nid} not serving")
            elif s["q"] > s["buffer"]:
                self._raise_alarm("BUFFER_OVERFLOW", nid, "critical",
                                  f"q {s['q']:.0f} > buffer {s['buffer']:.0f}")
            elif l["util"] >= 1.0:
                self._raise_alarm("OVERSATURATED", nid, "major",
                                  f"arrival/capacity {l['util']:.2f}")
        for fid, f in self.flows.items():
            if not f["sla_ok"]:
                self._raise_alarm("SLA_VIOLATION", fid, "critical",
                                  f"delivered {f['delivered']:.0f}/{f['demand']:.0f} "
                                  f"delay {f['delay']:.0f}ms")

    # ── 故障注入：含前两域没有的两类 ──────────────────────────
    def inject(self, fault: dict) -> None:
        kind = fault.get("type")
        if kind == "fault_arrival_burst":
            # **新故障类型**：到达率瞬时放大（前两域没有）
            sid = fault.get("stream")
            mult = float(fault.get("mult", 2.0))
            if sid in self.flows:
                self.flows[sid]["demand"] *= mult
            for s in self.queues.values():
                s["arrivals"] *= 1.0
            self.recompute()
            self._generate_alarms()
            return
        if kind == "fault_backlog":
            # **新故障类型**：给某台服务器一个非零初始积压。
            #    没有它，"到达突发"在 t=0 时队列还是空的、SLA 仍满足，
            #    harness 第一步就判成功结束——故障根本没机会发作。
            nid = fault.get("server")
            if nid in self.queues:
                self.queues[nid]["q"] = float(fault.get("q", 0.0))
            self.recompute()
            self._generate_alarms()
            return
        if kind == "fault_buffer_shrink":
            # **新故障类型**：缓冲上限被压小 → 溢出风险
            nid = fault.get("server")
            if nid in self.queues:
                self.queues[nid]["buffer"] *= float(fault.get("factor", 0.4))
            self.recompute()
            self._generate_alarms()
            return
        if kind == "fault_server_slow":
            nid = fault.get("server")
            if nid in self.queues:
                self.queues[nid]["mu"] *= float(fault.get("factor", 0.4))
                self.links[nid]["capacity"] = self.queues[nid]["mu"]
            self.recompute()
            self._generate_alarms()
            return
        # 其余交给父类（link_set / link_capacity / qos_set / reprovision / ack）
        super().inject(fault)

    def queue_summary(self) -> str:
        return " ".join(f"{n}:{s['q']:.0f}/{s['buffer']:.0f}"
                        for n, s in sorted(self.queues.items()))

    def state_hash(self) -> str:
        import hashlib
        import json
        payload = json.dumps(
            [self.config_fingerprint(),
             {n: round(s["q"], 3) for n, s in sorted(self.queues.items())},
             self.kpi()], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def clone(self) -> "QueueEmulator":
        import copy
        e = QueueEmulator(self.seed)
        e.links = copy.deepcopy(self.links)
        e.flows = copy.deepcopy(self.flows)
        e.policy = copy.deepcopy(self.policy)
        e.queues = copy.deepcopy(self.queues)
        e.alarms = copy.deepcopy(self.alarms)
        e.t = self.t
        return e
