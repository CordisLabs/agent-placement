"""The model-driven arm: same harness factors, but the policy is a real LLM.

Not part of the offline core - it needs a reachable OpenAI-compatible endpoint.
It exists so the harness claim can be tested against a real model when one is
available, and it records the provider's own token accounting alongside the
harness estimate.

The control LAYER is a factor here: the same three arms are re-run at
near_rt / non_rt / smo_intent, which differ only in decision cadence and
telemetry staleness (config.LAYERS), so a real model can be placed at each
layer of the control stack without any per-layer tool or privilege change.

    AIXNET_LLM_BASE_URL=http://127.0.0.1:11434/v1 \
    AIXNET_LLM_MODEL=qwen2.5:7b AIXNET_LLM_TIMEOUT=300 \
    python3 run_demo.py llm-arm --reps 10 --prompt scoped3 --layer near_rt

The output CSV is unique per (endpoint host, model, prompt, layer), and every
row records the provenance (model, endpoint, prompt_style, prompt_sha1, layer)
of the run that produced it. Before a single row is written, the target file is
opened and its recorded provenance is compared with the current run:

  * no target file              -> write it, recording provenance in every row
  * target exists, same run     -> resume: append the missing cells
  * target exists, different run-> raise ProvenanceConflict; the file is NOT
                                   touched and NOT renamed

That last rule exists because a silent clobber already cost this project one
model-axis comparison: `llm-arm` used to name its output `llm_arm_<prompt>.csv`,
so a qwen2.5:14b run overwrote the qwen2.5:7b run. The guard is pinned by
tests/test_llm_artifacts.py (wired into `run_demo.py selfcheck`).

Naming rule for a NEW run:

    <prefix>_layer_<host>_<model>_<prompt>_<layer>.csv

where <prefix> is "llm" (or the controller name), <host> is the sanitised
endpoint netloc including the port (127.0.0.1:11434 -> 127-0-0-1-11434, so an
Ollama server and a vLLM server on one box cannot collide), and <model> is the
sanitised model id (qwen3.8:27b-q4_K_M -> qwen3-8-27b-q4-K-M). The prefix stays
`<prefix>_layer_` (not `llm_arm_`) deliberately: demo_live.py and
out/llm_layer_*.py glob `llm_layer_*.csv`, and those files are outside this
module's write scope. A run that already exists under the older host-less name
is resumed in place rather than duplicated, so a 53-hour replication is never
restarted to satisfy a naming change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from urllib.parse import urlsplit

from .config import LAYERS, HarnessConfig
from .harness import Episode
from .runner import summarise, write_csv
from .scenarios import ALL_IDS, get as get_scenario

# name, harness-config kwargs (layer is added by arm_set), step-budget override
ARM_SPECS = [
    ("L1_reference", {"controller": "llm"}, None),
    ("L2_coarse_planner", {"controller": "llm", "tool_granularity": "coarse",
                           "control": "planner", "guardrails": "lenient",
                           "retry": "none"}, None),
    ("L3_explicit_ack", {"controller": "llm"}, 10),
]

# every run must carry these columns, whatever the harness adds later
REQUIRED_COLS = ("layer", "model", "endpoint", "prompt_style", "seed", "scenario",
                 "success", "steps", "tool_calls", "api_calls",
                 "api_prompt_tokens", "api_completion_tokens", "parse_failures",
                 "wall_s")

# columns that say WHICH run produced a row. Checked against an existing target
# file before writing, and against the freshly written header afterwards.
# `prompt_sha1` is the prompt-template hash (sha1[:12] of the exact system
# prompt; scoped3 -> 30198e736de4). It is deliberately NOT added to
# REQUIRED_COLS: that tuple fixes the on-disk column ORDER, and re-ordering the
# header of an already-published CSV would break readers that split on commas.
PROVENANCE_COLS = ("model", "endpoint", "prompt_style", "prompt_sha1", "layer")

# columns compared for identity of a run; a difference in any of these is a
# hard conflict. `endpoint` is checked separately and only warns, because the
# same server can be addressed as 127.0.0.1 vs localhost.
IDENTITY_COLS = ("model", "prompt_sha1", "layer")

# how many existing rows to sample when reading provenance
_PROV_SAMPLE = 500


class ProvenanceConflict(RuntimeError):
    """An existing CSV was produced by a DIFFERENT run. Refuse to touch it."""


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model or "unknown").strip("-") or "unknown"


def endpoint_host(endpoint: str) -> str:
    """Sanitised netloc of an OpenAI-compatible endpoint, for use in a filename.

    http://127.0.0.1:11434/v1 -> 127-0-0-1-11434 ; empty -> local
    """
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return "local"
    host = urlsplit(endpoint).netloc or endpoint
    return model_slug(host)


def safe_part(part: str, what: str) -> str:
    """Validate a filename component that must NOT be slugged.

    `prompt` and `layer` keep their exact spelling (`near_rt`, not `near-rt`)
    because the shipped files, the legacy resume lookup and the `llm_layer_*`
    globs all depend on it. They come from a fixed vocabulary, so this only
    rejects something a caller typed by hand.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", part or ""):
        raise ValueError(f"unsafe {what} for a filename: {part!r}")
    return part


def csv_path(out_dir: str, model: str, prompt: str, layer: str,
             controller: str = "llm", endpoint: str = "") -> str:
    """Path for a NEW run: unique per (host, model, prompt, layer)."""
    prefix = "llm" if controller == "llm" else controller
    return os.path.join(
        out_dir,
        f"{prefix}_layer_{endpoint_host(endpoint)}_{model_slug(model)}_"
        f"{safe_part(prompt, 'prompt')}_{safe_part(layer, 'layer')}.csv")


def legacy_csv_path(out_dir: str, model: str, prompt: str, layer: str,
                    controller: str = "llm") -> str:
    """The pre-host naming scheme, kept so existing runs resume in place."""
    prefix = "llm" if controller == "llm" else controller
    return os.path.join(
        out_dir,
        f"{prefix}_layer_{model_slug(model)}_"
        f"{safe_part(prompt, 'prompt')}_{safe_part(layer, 'layer')}.csv")


def provenance_of(path: str) -> dict:
    """Provenance recorded in an existing CSV.

    Returns a dict of PROVENANCE_COLS, or {} when the file does not record
    provenance at all (which includes every CSV written before this guard
    existed) -- an unverifiable file is treated as a conflict, never as a
    licence to overwrite. Mixed provenance across rows returns
    {"_inconsistent": ...}: that is the signature of a file that has already
    been clobbered.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        if not set(PROVENANCE_COLS) <= set(cols):
            return {}
        seen = None
        n = 0
        for row in reader:
            got = {k: (row.get(k) or "") for k in PROVENANCE_COLS}
            if seen is None:
                seen = got
            elif got != seen:
                return {"_inconsistent": True, **seen}
            n += 1
            if n >= _PROV_SAMPLE:
                break
    if seen is None:
        return {"_empty": True}
    return seen


def check_compatible(path: str, prov: dict, model: str, prompt: str, layer: str,
                     endpoint: str = "") -> None:
    """Raise ProvenanceConflict unless `path` holds this exact run."""
    def fail(why: str) -> None:
        recorded = prov if prov else "none (no provenance columns: it predates this guard)"
        raise ProvenanceConflict(
            f"refusing to write {path!r}: {why}. This file is an existing "
            f"measurement and will not be overwritten. Either keep it and run "
            f"with a different --out directory, or move the file aside "
            f"yourself. Recorded provenance: {recorded}")

    if not prov:
        fail("it does not record model/prompt provenance")
    if prov.get("_inconsistent"):
        fail("its rows disagree about which run produced them (already clobbered)")
    if prov.get("_empty"):
        return  # header only: nothing to contradict, safe to resume
    for col in IDENTITY_COLS:
        have, want = prov.get(col, ""), {"model": model, "prompt_sha1":
                                         prompt_fingerprint(prompt),
                                         "layer": layer}[col]
        if have != want:
            fail(f"recorded {col}={have!r} but this run is {want!r}")
    if endpoint and prov.get("endpoint") and prov["endpoint"] != endpoint:
        print(f"[warn] {path}: recorded endpoint {prov['endpoint']!r} differs "
              f"from {endpoint!r} (same host); continuing", flush=True)
    want_style = prompt
    if prov.get("prompt_style") and prov["prompt_style"] != want_style:
        print(f"[warn] {path}: recorded prompt_style {prov['prompt_style']!r} "
              f"differs from {want_style!r} but the prompt hash matches",
              flush=True)


def resolve_csv_path(out_dir: str, model: str, prompt: str, layer: str,
                     controller: str = "llm", endpoint: str = "") -> tuple[str, str]:
    """Pick the target CSV and say whether it is new or a resume.

    Returns (path, mode) with mode in {"new", "resume"}. Raises
    ProvenanceConflict rather than ever writing into a different run's file.
    """
    path = csv_path(out_dir, model, prompt, layer, controller, endpoint)
    if os.path.exists(path):
        check_compatible(path, provenance_of(path), model, prompt, layer, endpoint)
        return path, "resume"
    legacy = legacy_csv_path(out_dir, model, prompt, layer, controller)
    if os.path.exists(legacy):
        # A run started before the host was part of the name. Resuming it in
        # place is required: restarting a 53 h replication to satisfy a naming
        # rule would be worse than the naming rule is good.
        check_compatible(legacy, provenance_of(legacy), model, prompt, layer,
                         endpoint)
        print(f"[resume] legacy (host-less) path {legacy}", flush=True)
        return legacy, "resume"
    return path, "new"



def arm_set(layer: str = "near_rt", controller: str = "llm") -> list[tuple]:
    """The three arms of the model-driven study, all placed at one layer.

    controller="mock" gives the deterministic scripted controller on exactly the
    same arms and layers: the matched control for the model arm.
    """
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; expected one of {list(LAYERS)}")
    return [(name, HarnessConfig(layer=layer, controller=controller,
                                 **{k: v for k, v in kw.items() if k != "controller"}),
             budget) for name, kw, budget in ARM_SPECS]


ARMS = arm_set("near_rt")   # default set, kept importable for older callers


def prompt_fingerprint(prompt: str) -> str:
    """Short hash of the exact system prompt text sent to the model.

    This is the `prompt_sha1` column: scoped3 -> 30198e736de4.
    """
    from .llm_client import LLMController
    text = LLMController.PROMPTS.get(prompt, prompt)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _capture(controller) -> list[dict]:
    """Record the model's raw replies for one episode (seed 0 only).

    Wrapping the client here - rather than editing llm_client.py - keeps the
    trace sidecar an observation, not a change to the policy path.
    """
    cl = getattr(controller, "client", None)
    if cl is None or not hasattr(cl, "complete"):
        return []
    log: list[dict] = []
    original = cl.complete

    def complete(messages, temperature: float = 0.0):
        text = original(messages, temperature=temperature)
        log.append({"n": len(log) + 1, "reply": (text or "")[:600]})
        return text

    cl.complete = complete
    return log


def run_arms(reps: int = 10, out_dir: str = "out",
             scenarios: list[str] | None = None, prompt: str = "terse",
             arms=None, layer: str = "near_rt", traces: bool = True,
             controller: str = "llm") -> dict:
    os.environ["AIXNET_LLM_PROMPT"] = prompt
    arms = arms or arm_set(layer, controller=controller)
    if any(cfg.layer != layer for _, cfg, _ in arms):
        raise ValueError(f"arms do not all sit at layer {layer!r}")
    model = (os.environ.get("AIXNET_LLM_MODEL", "") if controller == "llm" else "mock")
    endpoint = (os.environ.get("AIXNET_LLM_BASE_URL", "") if controller == "llm" else "")
    rows: list[dict] = []
    trace_dir = os.path.join(out_dir, "traces_llm", model_slug(model),
                             f"{prompt}_{layer}")
    # Refuses (loudly, without touching anything) if the target file belongs to
    # a different run. See resolve_csv_path / ProvenanceConflict.
    path, mode = resolve_csv_path(out_dir, model, prompt, layer,
                                  controller=controller, endpoint=endpoint)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[run] {mode}: {path}\n"
          f"[run] model={model!r} endpoint={endpoint or '(none)'!r} "
          f"prompt={prompt!r} prompt_sha1={prompt_fingerprint(prompt)} "
          f"layer={layer!r}", flush=True)

    # --- resumability -------------------------------------------------------
    # A full 27B replication costs ~53 h of wall-clock. Losing that to a crash
    # is unacceptable, so we (a) skip cells already present in the CSV, and
    # (b) flush each new cell to disk as it finishes.
    old_rows: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            old_rows = list(csv.DictReader(fh))
        for r in old_rows:
            done.add((r.get("variant"), r.get("scenario"), str(r.get("seed"))))
        print(f"[resume] {len(done)} cells already complete in {path}", flush=True)
        if not set(PROVENANCE_COLS) <= set(old_rows[0] if old_rows else {}):
            print(f"[warn] {path} predates provenance columns; this run will "
                  f"add them to the rows it writes", flush=True)

    guarded = [False]

    def flush() -> None:
        """Write old + new rows to the CSV atomically (tmp then replace)."""
        # Re-check provenance immediately before the first replace. resolve()
        # ran some time ago; if another process created a conflicting file in
        # between, this is the last place we can refuse instead of clobber.
        if not guarded[0] and os.path.exists(path):
            check_compatible(path, provenance_of(path), model, prompt, layer,
                             endpoint)
        cols = ([c for c in REQUIRED_COLS if c in all_rows[0]]
                + [c for c in all_rows[0] if c not in REQUIRED_COLS])
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        os.replace(tmp, path)
        guarded[0] = True

    all_rows = list(old_rows)
    skipped = 0
    for sid in (scenarios or ALL_IDS):
        sc = get_scenario(sid)
        for name, cfg, budget in arms:
            for seed in range(reps):
                if (name, sid, str(seed)) in done:
                    skipped += 1
                    continue
                tpath = (os.path.join(trace_dir, f"{name}-{sid}-s{seed}.jsonl")
                         if traces and seed == 0 else None)
                ep = Episode(sc, seed, cfg, budget=budget, trace_path=tpath)
                replies = _capture(ep.controller) if tpath else []
                t0 = time.perf_counter()
                m = ep.run()
                cl = getattr(ep.controller, "client", None)
                m.update({
                    "variant": name, "prompt_style": prompt,
                    "prompt_sha1": prompt_fingerprint(prompt), "layer": cfg.layer,
                    "wall_s": round(time.perf_counter() - t0, 2),
                    "api_calls": getattr(cl, "calls", 0),
                    "api_prompt_tokens": getattr(cl, "prompt_tokens", 0),
                    "api_completion_tokens": getattr(cl, "completion_tokens", 0),
                    "parse_failures": getattr(ep.controller, "parse_failures", 0),
                    "last_error": (getattr(ep.controller, "last_error", None) or "")[:80],
                    # fall back to the run-level identity so that a controller
                    # with no client (mock) still records provenance that
                    # matches what resolve_csv_path() will check next time
                    "model": getattr(cl, "model", "") or model,
                    "endpoint": getattr(cl, "base_url", "") or endpoint,
                })
                if tpath and replies:
                    with open(tpath + ".replies.jsonl", "w", encoding="utf-8") as fh:
                        for rec in replies:
                            fh.write(json.dumps(rec) + "\n")
                rows.append(m)
                all_rows.append(m)
                done.add((name, sid, str(seed)))
                flush()  # durable after every cell
                print(f"  {name:18s} {sid:20s} seed={seed:2d} succ={m['success']} "
                      f"steps={m['steps']} calls={m['tool_calls']} api={m['api_calls']} "
                      f"ptok={m['api_prompt_tokens']} wall={m['wall_s']}s "
                      f"parse_fail={m['parse_failures']}", flush=True)
    if skipped:
        print(f"[resume] skipped {skipped} already-complete cells", flush=True)
    rows = all_rows
    missing = [c for c in REQUIRED_COLS if rows and c not in rows[0]]
    if missing:
        raise RuntimeError(f"CSV would be missing required columns: {missing}")
    flush()
    # The written header must carry provenance, so that a LATER run can tell
    # what this file is before writing into it.
    if rows:
        with open(path, newline="", encoding="utf-8") as fh:
            header = csv.DictReader(fh).fieldnames or []
        absent = [c for c in PROVENANCE_COLS if c not in header]
        if absent:
            raise RuntimeError(f"{path} would be untraceable: missing "
                               f"provenance columns {absent}")
    return {"episodes": len(rows), "csv": path, "layer": layer, "prompt": prompt,
            "prompt_sha1": prompt_fingerprint(prompt), "trace_dir": trace_dir,
            "model": model, "endpoint": endpoint, "mode": mode,
            "summary": summarise(rows) if rows else []}
