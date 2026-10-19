"""Deterministic network digital twin (stdlib only, no external services).

Emulated network functions
--------------------------
1. Topology + traffic-engineered path provisioning (explicit flow->path maps,
   administrative reprovisioning; there is NO automatic failure re-routing,
   which is what makes the operator task non-trivial).
2. QoS / policy management (per-class rate ceiling + delay SLA).
3. Fault injector (link state, capacity reduction, policy misconfig, alarm storm)
   driven by a scenario timeline, deterministic per (scenario, seed).
4. KPI / telemetry store with alarm generation and history.

Everything is a pure function of (seed, injected faults, applied operations).
No wall-clock, no I/O, no randomness outside the seeded RNG.
"""

from __future__ import annotations

import copy
import random

from . import telemetry
from .topology import (CAPACITY_FLOOR, CLASSES, CONGESTION_UTIL, DEFAULT_POLICY,
                       DELAY_CEILING, FLOW_SPECS, LINK_SPECS, LOSS_SLOPE, MAX_LOSS,
                       NODES, TICK_SECONDS, UTIL_CEILING, k_shortest_paths)

__all__ = ["NetworkEmulator", "NODES", "CONGESTION_UTIL", "DEFAULT_POLICY",
           "CLASSES", "TICK_SECONDS"]


class NetworkEmulator:
    """The digital twin. All mutation goes through apply(); reads via kpi()/observe()."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.t = 0
        self.links = {
            lid: {
                "id": lid, "a": a, "b": b, "capacity": cap, "base_latency": lat,
                "weight": w, "production": prod, "up": True, "util": 0.0,
                "delay": lat, "loss": 0.0, "load": 0.0,
            }
            for lid, a, b, cap, lat, w, prod in LINK_SPECS
        }
        self.policy = copy.deepcopy(DEFAULT_POLICY)
        self.flows = {
            fid: {"id": fid, "src": s, "dst": d, "cls": c, "demand": dem,
                  "sla_delay": sla, "delivered": dem, "delay": 0.0, "loss": 0.0,
                  "sla_ok": True, "path": None}
            for fid, s, d, c, dem, sla in FLOW_SPECS
        }
        self.candidates = {fid: k_shortest_paths(self.links, f["src"], f["dst"], 4)
                           for fid, f in self.flows.items()}
        for fid, f in self.flows.items():
            f["path"] = self.candidates[fid][0]
        self.alarms: list[dict] = []
        self.alarm_seq = 0
        self.kpi_history: list[dict] = []
        self.op_log: list[dict] = []
        self._alarm_fingerprints: set[str] = set()
        self.recompute()

    # ---------------- topology / paths ----------------
    def candidate_paths(self, flow_id: str) -> list:
        return [list(p) for p in self.candidates.get(flow_id, [])]

    # read-only projections live in telemetry.py; these keep the twin API stable
    def observe(self, include_paths: bool = False) -> dict:
        return telemetry.observe(self, include_paths)

    def kpi(self) -> dict:
        return telemetry.kpi(self)

    def config_fingerprint(self) -> str:
        return telemetry.config_fingerprint(self)

    def state_hash(self) -> str:
        return telemetry.state_hash(self)

    def apply(self, op: dict, snapshot: dict | None = None) -> dict:
        """Apply one operation. Returns {'ok': bool, 'detail': str}."""
        kind = op.get("type")
        try:
            if kind == "link_set":
                l = self.links[op["link"]]
                l["up"] = bool(op["up"])
            elif kind == "link_weight":
                self.links[op["link"]]["weight"] = int(op["weight"])
            elif kind == "link_capacity":
                cap = float(op["capacity"])
                if cap < CAPACITY_FLOOR:
                    return {"ok": False,
                            "detail": f"capacity {cap} below physical floor "
                                      f"{CAPACITY_FLOOR} Mbps"}
                self.links[op["link"]]["capacity"] = cap
            elif kind == "qos_set":
                cls = op["cls"]
                if cls not in self.policy:
                    return {"ok": False, "detail": f"unknown class {cls}"}
                if "ceiling" in op:
                    self.policy[cls]["ceiling"] = float(op["ceiling"])
                if "min_share" in op:
                    self.policy[cls]["min_share"] = float(op["min_share"])
            elif kind == "reprovision":
                fid = op["flow"]
                idx = int(op["path_index"])
                paths = self.candidates[fid]
                if not 0 <= idx < len(paths):
                    return {"ok": False, "detail": f"path_index {idx} out of range"}
                self.flows[fid]["path"] = list(paths[idx])
            elif kind == "ack_alarm":
                for a in self.alarms:
                    if a["id"] == op["alarm"]:
                        a["acked"] = True
                        break
                else:
                    return {"ok": False, "detail": "unknown alarm"}
            elif kind == "restore_snapshot":
                if snapshot is None:
                    return {"ok": False, "detail": "no snapshot supplied"}
                self.restore(snapshot)
            else:
                return {"ok": False, "detail": f"unknown op {kind!r}"}
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        self.op_log.append(dict(op, t=self.t))
        return {"ok": True, "detail": "applied"}

    def snapshot(self) -> dict:
        return {"links": copy.deepcopy(self.links), "policy": copy.deepcopy(self.policy),
                "flows": {k: list(v["path"]) for k, v in self.flows.items()}}

    def restore(self, snap: dict) -> None:
        self.links = copy.deepcopy(snap["links"])
        self.policy = copy.deepcopy(snap["policy"])
        for fid, path in snap["flows"].items():
            self.flows[fid]["path"] = list(path)

    # ---------------- simulation ----------------
    def recompute(self) -> None:
        for l in self.links.values():
            l["load"] = 0.0
        eff: dict[str, float] = {}
        for fid, f in self.flows.items():
            ceiling = self.policy[f["cls"]]["ceiling"]
            eff[fid] = min(f["demand"], ceiling)
            for lid in f["path"] or []:
                if not self.links[lid]["up"]:
                    break          # traffic stops at the first failed hop
                self.links[lid]["load"] += eff[fid]
        for l in self.links.values():
            cap = max(l["capacity"], CAPACITY_FLOOR)
            l["util"] = min(l["load"] / cap, UTIL_CEILING)
            over = max(0.0, l["util"] - CONGESTION_UTIL)
            l["loss"] = min(MAX_LOSS, over * LOSS_SLOPE) if l["up"] else 1.0
            l["delay"] = min(l["base_latency"] + 20.0 * over, DELAY_CEILING)
        for fid, f in self.flows.items():
            path = f["path"] or []
            down = any(not self.links[lid]["up"] for lid in path)
            if down or not path:
                f["delivered"], f["delay"], f["loss"] = 0.0, 0.0, 1.0
            else:
                loss = 1.0
                delay = 0.0
                for lid in path:
                    loss *= (1.0 - self.links[lid]["loss"])
                    delay += self.links[lid]["delay"]
                f["loss"] = round(1.0 - loss, 6)
                f["delay"] = delay
                f["delivered"] = eff[fid] * loss
            f["sla_ok"] = bool(
                f["delivered"] >= 0.9 * f["demand"]
                and f["delay"] <= f["sla_delay"]
            )

    def _raise_alarm(self, kind: str, target: str, severity: str, detail: str) -> None:
        fp = f"{kind}|{target}"
        if fp in self._alarm_fingerprints:
            return
        self._alarm_fingerprints.add(fp)
        self.alarm_seq += 1
        self.alarms.append({
            "id": f"A{self.alarm_seq:03d}", "kind": kind, "target": target,
            "severity": severity, "detail": detail, "t": self.t, "acked": False,
        })

    def _generate_alarms(self) -> None:
        for l in self.links.values():
            if not l["up"]:
                self._raise_alarm("LINK_DOWN", l["id"], "critical",
                                  f"link {l['id']} admin/oper down")
            elif l["util"] >= CONGESTION_UTIL:
                self._raise_alarm("LINK_CONGESTION", l["id"], "major",
                                  f"util {l['util']:.2f}")
        for f in self.flows.values():
            if not f["sla_ok"]:
                self._raise_alarm("SLA_VIOLATION", f["id"], "critical",
                                  f"delivered {f['delivered']:.0f}/{f['demand']:.0f} Mbps")
        for cls, pol in self.policy.items():
            demand = sum(f["demand"] for f in self.flows.values() if f["cls"] == cls)
            if demand > 0 and pol["ceiling"] < demand:
                self._raise_alarm("POLICY_THROTTLE", cls, "major",
                                  f"ceiling {pol['ceiling']:.0f} < demand {demand:.0f}")

    def inject(self, fault: dict) -> None:
        """Fault injector entry point (used by scenarios and tests)."""
        kind = fault["type"]
        if kind == "fault_linkset":
            self.links[fault["link"]]["up"] = bool(fault["up"])
        elif kind == "fault_congestion":
            self.links[fault["link"]]["capacity"] *= float(fault["factor"])
        elif kind == "fault_misconfig":
            self.policy[fault["cls"]]["ceiling"] = float(fault["ceiling"])
        elif kind == "fault_alarm_storm":
            r = random.Random(self.seed * 7919 + self.t)
            for _ in range(int(fault["count"])):
                lid = r.choice([l["id"] for l in self.links.values()])
                self.alarm_seq += 1
                self.alarms.append({
                    "id": f"A{self.alarm_seq:03d}", "kind": "STORM_NOISE",
                    "target": lid, "severity": r.choice(["major", "minor", "warning"]),
                    "detail": "storm noise (duplicate suppressed upstream)",
                    "t": self.t, "acked": False})
        else:
            raise ValueError(f"unknown fault {kind!r}")
        self.recompute()

    def refresh(self) -> None:
        """Recompute and (re)generate alarms without advancing the clock."""
        self.recompute()
        self._generate_alarms()
        self.kpi_history.append(self.kpi())

    def diff_ops(self, snap: dict) -> list[dict]:
        """Concrete ops that transform `snap` into the CURRENT configuration.

        Used to keep traces replayable: a rollback is recorded as the effective
        operations, never as an opaque 'restore_snapshot'.
        """
        ops: list[dict] = []
        for lid, l in sorted(self.links.items()):
            o = snap["links"][lid]
            if l["up"] != o["up"]:
                ops.append({"type": "link_set", "link": lid, "up": l["up"]})
            if l["capacity"] != o["capacity"]:
                ops.append({"type": "link_capacity", "link": lid,
                            "capacity": l["capacity"]})
            if l["weight"] != o["weight"]:
                ops.append({"type": "link_weight", "link": lid, "weight": l["weight"]})
        for cls, pol in sorted(self.policy.items()):
            o = snap["policy"][cls]
            if pol != o:
                ops.append({"type": "qos_set", "cls": cls, "ceiling": pol["ceiling"],
                            "min_share": pol["min_share"]})
        for fid, f in sorted(self.flows.items()):
            if list(snap["flows"][fid]) != list(f["path"] or []):
                try:
                    idx = self.candidates[fid].index(list(f["path"]))
                except ValueError:
                    continue
                ops.append({"type": "reprovision", "flow": fid, "path_index": idx})
        return ops

    def tick(self) -> None:
        self.t += 1
        self.recompute()
        self._generate_alarms()
        self.kpi_history.append(self.kpi())

    def clone(self) -> "NetworkEmulator":
        new = NetworkEmulator.__new__(NetworkEmulator)
        new.__dict__ = copy.deepcopy({k: v for k, v in self.__dict__.items()
                                      if k != "rng"})
        return new
