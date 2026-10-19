"""Static topology model: nodes, link/flow inventory, class policy defaults and
the deterministic k-shortest-path generator used to build candidate detours.

Part of the deterministic network digital twin (stdlib only, no external services).

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

import heapq


NODES = ["r0", "r1", "r2", "r3", "r4", "r5"]

# id, node_a, node_b, capacity_mbps, base_latency_ms, routing_weight, production
LINK_SPECS = [
    ("L0", "r0", "r1", 1000.0, 2.0, 1, True),
    ("L1", "r1", "r2", 1000.0, 2.0, 1, True),
    ("L2", "r2", "r3", 800.0, 3.0, 1, True),
    ("L3", "r0", "r4", 400.0, 5.0, 3, True),
    ("L4", "r4", "r5", 600.0, 3.0, 1, True),
    ("L5", "r5", "r3", 400.0, 4.0, 2, True),
    ("L6", "r1", "r5", 300.0, 6.0, 4, False),  # backup
    ("L7", "r2", "r4", 300.0, 5.0, 4, False),  # backup
]

# id, src, dst, class, demand_mbps, sla_delay_ms
FLOW_SPECS = [
    ("F0", "r0", "r3", "gold", 320.0, 60.0),
    ("F1", "r1", "r5", "silver", 220.0, 80.0),
    ("F2", "r0", "r5", "bronze", 160.0, 120.0),
]

CLASSES = ("gold", "silver", "bronze")
DEFAULT_POLICY = {
    "gold": {"ceiling": 1000.0, "min_share": 0.5},
    "silver": {"ceiling": 600.0, "min_share": 0.3},
    "bronze": {"ceiling": 400.0, "min_share": 0.2},
}
TICK_SECONDS = 10
CONGESTION_UTIL = 0.85
CAPACITY_FLOOR = 1.0      # Mbps - no operation may provision a non-physical link
UTIL_CEILING = 5.0        # telemetry is bounded so absurd states cannot leak to CSV
DELAY_CEILING = 60_000.0  # ms
LOSS_SLOPE = 2.0
MAX_LOSS = 0.5




def k_shortest_paths(links: dict, src: str, dst: str, k: int = 4) -> list[list[str]]:
    """Deterministic k shortest simple paths by (total weight, link-id tuple)."""
    adj: dict[str, list[tuple[str, str, int]]] = {n: [] for n in NODES}
    for lid, l in links.items():
        adj[l["a"]].append((l["b"], lid, l["weight"]))
        adj[l["b"]].append((l["a"], lid, l["weight"]))
    for n in adj:
        adj[n].sort(key=lambda x: (x[2], x[1]))
    found: list[tuple[int, tuple[str, ...]]] = []
    heap = [(0, (src,), ())]
    seen = set()
    while heap and len(found) < 200:
        cost, nodes, path = heapq.heappop(heap)
        if nodes[-1] == dst:
            if path not in seen:
                seen.add(path)
                found.append((cost, path))
            continue
        for nxt, lid, w in adj[nodes[-1]]:
            if nxt in nodes:
                continue
            heapq.heappush(heap, (cost + w, nodes + (nxt,), path + (lid,)))
    found.sort(key=lambda x: (x[0], x[1]))
    return [list(p) for _, p in found[:k]]
