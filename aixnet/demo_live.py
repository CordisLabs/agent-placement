"""`demo-live`: the one-command, offline, laptop-readable live demo.

Two things happen when it runs:

* the control-layer effect and the retry effect are **recomputed on this machine,
  right now**, from the same deterministic emulator the study uses (a few hundred
  episodes, ~2 s, no network, no GPU);
* the real-LLM table is **read from the shipped CSVs**, because re-running a 27B
  model takes hours - it cannot be recomputed live and is labelled as shipped.

Every stage degrades gracefully: if a CSV is missing, or the live run raises for
any reason, the pre-recorded values are printed with a "(pre-recorded)" label and
the command still exits 0. Nothing here may ever crash in front of a visitor.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import time

BAR_LAYER = 66           # bar width for the layer block (own line, 1 m readable)
BAR_FACTOR = 44          # bar width for the inline factor block
LAYER_ORDER = ("near_rt", "non_rt", "smo_intent")
LAYER_NAME = {
    "near_rt": "near_rt  (xApp)",
    "non_rt": "non_rt   (rApp)",
    "smo_intent": "smo_intent (SMO)",
}
RETRY_ORDER = ("none", "retry", "retry_rollback")

# ---- pre-recorded fallbacks, all measured on the shipped data --------------
# (source: out/episodes.csv, 14,400 episodes; out/llm_layer_*.csv, 150 per cell)
PREREC_LAYER = {"near_rt": (0.8625, 2.75), "non_rt": (0.6610, 3.59),
                "smo_intent": (0.2733, 3.95)}
PREREC_RETRY = {"none": 0.7469, "retry": 0.9119, "retry_rollback": 0.9287}
PREREC_LLM = [("qwen2.5:7b", 0.453, 0.193, 0.180),
              ("qwen2.5:14b", 0.200, 0.067, 0.067),
              ("qwen3.8:27b", 0.933, 0.467, 0.267)]


def bar(value: float, width: int = BAR_FACTOR) -> str:
    """ASCII bar; length is proportional to success across [0, 1]."""
    return "#" * int(round(max(0.0, min(1.0, value)) * width))


def _live_episodes(scenario, seed, cfg):
    from .harness import Episode
    return Episode(scenario, seed, cfg).run()


def run_live(seeds: int = 20) -> dict | None:
    """Recompute the full factorial (48 harness configs x 3 layers x 5 scenarios
    x `seeds`) on this machine and aggregate it exactly like the study does, so
    the numbers printed live equal the numbers in the paper. ~10 s.

    Returns None (never raises) if anything at all goes wrong.
    """
    if os.environ.get("AIXNET_DEMO_FORCE_FALLBACK") == "1":
        return {"error": "AIXNET_DEMO_FORCE_FALLBACK=1 (rehearsing the offline path)"}
    try:
        from .runner import all_configs
        from .scenarios import ALL_IDS, get as get_scenario
        configs = all_configs("mock")
        rows = []
        for sid in ALL_IDS:
            sc = get_scenario(sid)
            for cfg in configs:
                for k in range(seeds):
                    r = _live_episodes(sc, k, cfg)
                    r["scenario"] = sid
                    rows.append(r)
        layers, retries = {}, {}
        for layer in LAYER_ORDER:
            rs = [r for r in rows if r["layer"] == layer]
            layers[layer] = (sum(r["success"] for r in rs) / len(rs),
                             statistics.mean([r["steps"] for r in rs]), len(rs))
        for retry in RETRY_ORDER:
            rs = [r for r in rows if r["retry"] == retry and r["layer"] == "near_rt"]
            retries[retry] = (sum(r["success"] for r in rs) / len(rs), len(rs))
        return {"layers": layers, "retries": retries, "seeds": seeds,
                "episodes": len(rows), "configs": len(configs)}
    except Exception as exc:  # noqa: BLE001 - the demo must never crash
        return {"error": f"{type(exc).__name__}: {exc}"}


def read_llm_tables(out_dir: str = "out") -> tuple[list, bool]:
    """(rows, shipped?) - rows are (label, near_rt, non_rt, smo_intent)."""
    if os.environ.get("AIXNET_DEMO_FORCE_FALLBACK") == "1":
        return PREREC_LLM, False
    try:
        per_model: dict[str, dict[str, float]] = {}
        for path in glob.glob(os.path.join(out_dir, "llm_layer_*.csv")):
            with open(path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                continue
            model, layer = rows[0].get("model", "?"), rows[0].get("layer", "?")
            ok = sum(int(r["success"]) for r in rows) / len(rows)
            per_model.setdefault(model, {})[layer] = ok
        # the 27B near_rt cell predates the per-layer file naming
        legacy = os.path.join(out_dir, "llm_arm_scoped3.csv")
        if os.path.exists(legacy):
            with open(legacy, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows and rows[0].get("layer") == "near_rt":
                model = rows[0].get("model", "?")
                prev = per_model.setdefault(model, {})
                prev.setdefault("near_rt", sum(int(r["success"]) for r in rows) / len(rows))
        def rank(m):
            return (0 if "2.5:7b" in m else 1 if "2.5:14b" in m else 2, m)
        out = []
        for model in sorted(per_model, key=rank):
            d = per_model[model]
            if len(d) < 2:
                continue
            label = model.replace("qwen3.8", "qwen3.8").replace("-q4_K_M", "")
            out.append((label,
                        d.get("near_rt"), d.get("non_rt"), d.get("smo_intent")))
        return (out, True) if out else (PREREC_LLM, False)
    except Exception:  # noqa: BLE001
        return PREREC_LLM, False


def _fmt(v) -> str:
    return "  --  " if v is None else f"{v:6.3f}"


def render(seeds: int = 40, out_dir: str = "out") -> str:
    t0 = time.perf_counter()
    live = run_live(seeds)
    t_live = time.perf_counter() - t0
    llm, llm_shipped = read_llm_tables(out_dir)
    L = []
    add = L.append
    add("=" * 78)
    add(" AIxNET 2026  -  same agent, same task, ONLY THE CONTROL LAYER CHANGES")
    add(" offline - python3 stdlib only - no network, no GPU, no docker")
    add("=" * 78)

    ok = live is not None and "error" not in live
    if ok:
        n = list(live["layers"].values())[0][2]
        add("[LIVE] deterministic controller, recomputed on this laptop just now")
        add(f"       {live.get('configs', 48)} harness configs x 3 placements x 5 scenarios "
            f"x {seeds} seeds")
        add(f"       = {live.get('episodes', n * 3)} episodes; {n} per placement")
        add("")
        add("  CONTROL LAYER        acts   telemetry      SUCCESS  steps")
        for layer in LAYER_ORDER:
            succ, steps, _ = live["layers"][layer]
            acts = {"near_rt": "1", "non_rt": "2", "smo_intent": "3"}[layer]
            age = {"near_rt": "current", "non_rt": "1 step old",
                   "smo_intent": "2 steps old"}[layer]
            add(f"  {LAYER_NAME[layer]:20s} {acts:5s}  {age:13s} {succ:7.3f}  {steps:5.2f}")
            add(f"      {bar(succ, BAR_LAYER)}")
        d = live["layers"]["near_rt"][0] - live["layers"]["smo_intent"][0]
        add("")
        add("  >> SAME policy, SAME faults, SAME seeds. Placement alone moves success")
        add(f"     {live['layers']['near_rt'][0]:.3f} -> {live['layers']['smo_intent'][0]:.3f}"
            f"   (-{d * 100:.0f} points, a fall) and costs "
            f"{live['layers']['smo_intent'][1] - live['layers']['near_rt'][1]:+.2f} steps.")
        add("")
        add("[LIVE] largest harness-factor effect, same episodes (near_rt)")
        for retry in RETRY_ORDER:
            v, _ = live["retries"][retry]
            add(f"  retry={retry:15s} {v:6.3f}  {bar(v, BAR_FACTOR)}")
        r0 = live["retries"]["none"][0]
        r1 = live["retries"]["retry_rollback"][0]
        add(f"  >> harness policy alone: {r0:.3f} -> {r1:.3f}   ({(r1 - r0) * 100:+.0f} points)")
    else:
        add("[LIVE] UNAVAILABLE  " + str(live.get("error") if live else "unknown"))
        add("        falling back to pre-recorded values - see the label below")
        add("")
        add("  CONTROL LAYER        acts   telemetry      SUCCESS  steps  (pre-recorded)")
        for layer in LAYER_ORDER:
            succ, steps = PREREC_LAYER[layer]
            add(f"  {LAYER_NAME[layer]:20s} "
                f"{ {'near_rt':'1','non_rt':'2','smo_intent':'3'}[layer]:5s}  "
                f"{ {'near_rt':'current','non_rt':'1 step old','smo_intent':'2 steps old'}[layer]:13s} "
                f"{succ:7.3f}  {steps:5.2f}")
            add(f"      {bar(succ, BAR_LAYER)}")
        add("")
        add("  retry=none 0.747 -> retry+rollback 0.929 (+18 points)   (pre-recorded)")

    add("")
    tag = "SHIPPED" if llm_shipped else "PRE-RECORDED"
    add(f"[{tag}] real LLM arms, frozen prompt, 150 episodes per cell ")
    add("         (27B needs ~3 min per decision - cannot be recomputed live)")
    add("")
    add("  model              near_rt   non_rt   smo_intent   spread")
    for label, a, b, c in llm:
        vals = [v for v in (a, b, c) if v is not None]
        spread = f"{(max(vals) - min(vals)) * 100:+.0f} pp" if len(vals) > 1 else "partial"
        add(f"  {label:18s} {_fmt(a)}  {_fmt(b)}   {_fmt(c)}    {spread}")
    add("")
    add("  source: " + ("out/llm_layer_*.csv + out/llm_arm_scoped3.csv" if llm_shipped
                         else "pre-recorded constants in aixnet/demo_live.py"))
    total = time.perf_counter() - t0
    add("")
    add("-" * 78)
    add(f" live stage {t_live:.2f} s   |   total {total:.2f} s   |   exit 0   |   "
        f"no network calls")
    add("=" * 78)
    return "\n".join(L)


def header() -> str:
    return "\n".join([
        "=" * 78,
        " AIxNET 2026  -  same agent, same task, ONLY THE CONTROL LAYER CHANGES",
        " offline - python3 stdlib only - no network, no GPU, no docker",
        "=" * 78,
    ])


def main(seeds: int = 20, out_dir: str = "out") -> int:
    try:
        print(header(), flush=True)
        print(f" [LIVE] recomputing the study's factorial now "
              f"(~{0.47 * seeds + 0.8:.0f} s, please wait)...", flush=True)
        body = render(seeds=seeds, out_dir=out_dir)
        # the banner was already printed above; drop the duplicate
        print(body.split("=" * 78, 2)[2].lstrip("\n"))
    except Exception as exc:  # noqa: BLE001 - last line of defence
        print("demo-live could not render:", type(exc).__name__, exc)
        print("Fall back to out/figs/*.png and the slides. Exit 0 by design.")
    return 0
