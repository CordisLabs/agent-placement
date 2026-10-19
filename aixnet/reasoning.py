"""Diagnosis and fix synthesis: the policy's reasoning, kept separate from the
control loop so that the competence model is auditable in one place.

Evidence classes and their error rates are explicit constants. `rng` is always a
per-(seed, step) stream, so the noise sequence is identical across harness
configurations - only the harness varies.
"""

from __future__ import annotations

from .topology import CONGESTION_UTIL

DEFAULT_CEILING = {"gold": 1000.0, "silver": 600.0, "bronze": 400.0}
TARGET_UTIL = 0.80
INPLACE, DETOUR = "inplace", "detour"

# p(second-best hypothesis wins) per evidence class
P_WRONG = {"alarm_backed": 0.12, "structural": 0.18, "inferred": 0.45, "guess": 0.35}
# p(one root cause of a compound fault is silently dropped)
P_DROP = {"alarm_backed": 0.15, "structural": 0.25, "inferred": 0.45, "guess": 0.35}


def load_of(link: dict) -> float:
    return float(link.get("load_mbps", link["util"] * link["capacity"]))


def required_capacity(obs: dict, link_id: str) -> float:
    """Capacity a link needs so every flow routed over it sits below TARGET_UTIL."""
    total = sum(f["demand_mbps"] for f in obs["flows"]
                if link_id in (f.get("path") or []))
    return round(total / TARGET_UTIL, 1)


def _worst_util(path, flow, links) -> float:
    worst = 0.0
    for pid in path:
        l = links.get(pid)
        if l is None:
            continue
        worst = max(worst, (load_of(l) + flow["demand_mbps"]) / max(l["capacity"], 1e-9))
    return worst


def clean_alt(cands, flow, obs, avoid=None) -> int | None:
    """Index of a candidate path that is down-free AND below target utilisation."""
    links = {l["id"]: l for l in obs["links"]}
    for i, path in enumerate(cands):
        if avoid and avoid in path:
            continue
        if any(pid in links and not links[pid]["up"] for pid in path):
            continue
        if _worst_util(path, flow, links) <= CONGESTION_UTIL:
            return i
    return None


def best_alt(cands, flow, obs, avoid=None) -> int | None:
    """Least-bad candidate path (used when nothing is clean)."""
    links = {l["id"]: l for l in obs["links"]}
    best, best_score = None, None
    for i, path in enumerate(cands):
        if avoid and avoid in path:
            continue
        if any(pid in links and not links[pid]["up"] for pid in path):
            continue
        worst = _worst_util(path, flow, links)
        if worst <= CONGESTION_UTIL:
            return i
        if best_score is None or worst < best_score:
            best, best_score = i, worst
    return best


def diagnose_all(obs, failing, rng) -> list[dict]:
    """Return the root causes the policy believes in, best evidence first."""
    links = {l["id"]: l for l in obs["links"]}
    roots: list[dict] = []
    seen: set = set()

    for f in failing:
        for lid in f["path"]:
            if lid in links and not links[lid]["up"] and lid not in seen:
                seen.add(lid)
                roots.append({"kind": "link_down", "link": lid, "conf": "structural"})
    for a in obs["alarms"]:
        if (a["kind"] == "LINK_DOWN" and a["target"] not in seen
                and links.get(a["target"], {}).get("up") is False):
            seen.add(a["target"])
            roots.append({"kind": "link_down", "link": a["target"],
                          "conf": "alarm_backed"})

    pol = {a["target"] for a in obs["alarms"] if a["kind"] == "POLICY_THROTTLE"}
    for cls in sorted(pol):
        if any(f["cls"] == cls for f in failing):
            roots.append({"kind": "policy", "cls": cls, "conf": "alarm_backed"})

    for f in failing:
        for lid in f["path"]:
            l = links.get(lid)
            if l and l["util"] >= CONGESTION_UTIL and ("cong:" + lid) not in seen:
                seen.add("cong:" + lid)
                roots.append({"kind": "congestion", "link": lid, "conf": "structural"})
    if not roots:
        for f in failing:
            healthy = all(links[lid]["util"] < CONGESTION_UTIL
                          for lid in f["path"] if lid in links)
            if f["delivered_mbps"] < 0.5 * f["demand_mbps"] and healthy:
                roots.append({"kind": "policy", "cls": f["cls"], "conf": "inferred"})
                break
    if not roots:
        return []
    if len(roots) > 1 and rng.random() < P_DROP.get(roots[-1]["conf"], 0.3):
        roots = roots[:-1]                           # compound fault partly missed
    if rng.random() < P_WRONG.get(roots[0]["conf"], 0.3):
        roots[0] = confuse(roots[0], obs, rng)
    return roots


def confuse(root, obs, rng) -> dict:
    wrong = dict(root, wrong=True)
    if root["kind"] == "link_down":
        downs = [l["id"] for l in obs["links"]
                 if not l["up"] and l["id"] != root["link"]]
        if downs:
            wrong["link"] = downs[0]
        else:
            top = max(obs["links"], key=lambda l: (l["util"], l["id"]))
            return {"kind": "congestion", "link": top["id"], "conf": root["conf"],
                    "wrong": True}
    elif root["kind"] == "policy":
        alts = [c for c in DEFAULT_CEILING if c != root["cls"]]
        wrong["cls"] = rng.choice(alts)
    elif root["kind"] == "congestion":
        others = [l["id"] for l in obs["links"]
                  if l["util"] >= CONGESTION_UTIL and l["id"] != root["link"]]
        if others:
            wrong["link"] = others[0]
    return wrong


def variants(root, obs, failing) -> dict:
    """variant name -> op list, or None when it cannot be built without a read."""
    links = {l["id"]: l for l in obs["links"]}
    paths = obs.get("paths") or {}
    kind = root["kind"]

    if kind == "policy":
        cls = root["cls"]
        dem = max([f["demand_mbps"] for f in failing if f["cls"] == cls] or [0.0])
        target = max(DEFAULT_CEILING.get(cls, 400.0), dem)
        cur = ((obs.get("policy") or {}).get(cls) or {}).get("ceiling")
        inplace = ([] if (cur is not None and cur >= dem)
                   else [{"type": "qos_set", "cls": cls, "ceiling": target}])
        out = {INPLACE: inplace, DETOUR: []}
        for f in failing:
            if f["cls"] != cls or not paths.get(f["id"]):
                continue
            idx = best_alt(paths[f["id"]], f, obs)
            if idx is not None:
                out[DETOUR].append({"type": "reprovision", "flow": f["id"],
                                    "path_index": idx})
        return out

    if kind == "congestion":
        l = links.get(root["link"])
        if l is None:
            return {INPLACE: None, DETOUR: None}
        cap = max(l["capacity"] * 2.0, required_capacity(obs, l["id"]))
        out = {INPLACE: [{"type": "link_capacity", "link": l["id"], "capacity": cap}],
               DETOUR: None}
        movers = [f for f in failing if root["link"] in f["path"]]
        if movers and all(paths.get(f["id"]) for f in movers):
            out[DETOUR] = []
            for f in movers:
                idx = best_alt(paths[f["id"]], f, obs, avoid=root["link"])
                if idx is not None:
                    out[DETOUR].append({"type": "reprovision", "flow": f["id"],
                                        "path_index": idx})
        return out

    lid = root["link"]
    targets = [f for f in failing if lid in f["path"]] or failing
    out = {INPLACE: [{"type": "link_set", "link": lid, "up": True}], DETOUR: None}
    if all(paths.get(f["id"]) for f in targets):
        out[DETOUR] = []
        for f in targets:
            idx = best_alt(paths[f["id"]], f, obs, avoid=lid)
            if idx is not None:
                out[DETOUR].append({"type": "reprovision", "flow": f["id"],
                                    "path_index": idx})
    return out


def fix_for(root, obs, failing, prefer, is_ok) -> list[dict] | None:
    """Ops for one root cause; None means "a read is required before deciding"."""
    vs = variants(root, obs, failing)
    order = [prefer] + [k for k in (INPLACE, DETOUR) if k != prefer]
    needs_read = False
    for name in order:
        ops = vs.get(name)
        if ops is None:
            needs_read = True
            continue
        avail = [o for o in ops if is_ok(o)]
        if avail:
            return avail
    return None if needs_read else []


def fix_all(roots, obs, failing, ctx, is_ok, force: str | None = None) -> list[dict]:
    ops: list[dict] = []
    pending_read = False
    for root in roots:
        prefer = force or (INPLACE if ctx.rng.random() < 0.75 else DETOUR)
        sub = fix_for(root, obs, failing, prefer, is_ok)
        if sub is None:
            pending_read = True
            continue
        ops.extend(sub)
    if not ops and pending_read:
        return []
    return ops
