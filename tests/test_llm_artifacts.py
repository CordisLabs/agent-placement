"""Run-artifact traceability tests.

These pin the guard that stops one model's run from silently overwriting
another's CSV. The bug they exist for actually happened: `llm-arm` used to write
`llm_arm_<prompt>.csv`, so a qwen2.5:14b run overwrote the qwen2.5:7b `scoped3`
result and a verified model-axis comparison was lost.

The writer is exercised end to end with controller="mock" (the deterministic
offline policy), so these tests need no endpoint, no key and no network - and
they write only into tempfile.mkdtemp() directories, never into demo/out/,
because everything under demo/out/ is a published measurement.

Wired into `run_demo.py selfcheck` via tests/test_all.py.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _util import check                                          # noqa: E402
from aixnet.llm_arm import (PROVENANCE_COLS, ProvenanceConflict,  # noqa: E402
                            csv_path, endpoint_host, legacy_csv_path,
                            model_slug, prompt_fingerprint, provenance_of,
                            resolve_csv_path, run_arms)

SCEN = ["S1_link_down"]
ENDPOINT = "http://127.0.0.1:11434/v1"


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _digest(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _mock_run(out_dir: str, reps: int = 1) -> dict:
    """One offline episode per arm, through the real writer."""
    return run_arms(reps=reps, out_dir=out_dir, scenarios=SCEN, prompt="scoped3",
                    layer="near_rt", controller="mock", traces=False)


def test_llm_csv_name_carries_host_model_prompt_layer():
    print("\n[test_llm_csv_name_carries_host_model_prompt_layer]")
    base = csv_path("/tmp/x", "qwen2.5:7b", "scoped3", "near_rt",
                    endpoint=ENDPOINT)
    check(os.path.basename(base) ==
          "llm_layer_127-0-0-1-11434_qwen2-5-7b_scoped3_near_rt.csv",
          f"new-run name is host+model+prompt+layer (`{os.path.basename(base)}`)")
    check(":" not in os.path.basename(base) and "/" not in os.path.basename(base),
          "filename is filesystem-safe (no ':' and no '/')")
    check(model_slug("qwen3.8:27b-q4_K_M") == "qwen3-8-27b-q4-K-M",
          "model slug matches the shipped convention "
          "(`qwen3.8:27b-q4_K_M` -> `qwen3-8-27b-q4-K-M`)")
    check(endpoint_host("http://127.0.0.1:11434/v1") == "127-0-0-1-11434",
          "endpoint host slug keeps the port (an Ollama and a vLLM server on "
          "one box cannot collide)")
    other = csv_path("/tmp/x", "qwen2.5:7b", "scoped3", "near_rt",
                     endpoint="http://192.168.1.9:11434/v1")
    check(other != base, "a different endpoint host yields a different path")
    other_layer = csv_path("/tmp/x", "qwen2.5:7b", "scoped3", "non_rt",
                           endpoint=ENDPOINT)
    check(other_layer != base, "a different control layer yields a different path")
    check(legacy_csv_path("/tmp/x", "qwen2.5:7b", "scoped3", "near_rt") != base,
          "the legacy host-less name is a distinct path (kept for resume only)")


def test_prompt_hash_column_is_stable():
    print("\n[test_prompt_hash_column_is_stable]")
    got = prompt_fingerprint("scoped3")
    check(got == "30198e736de4",
          f"prompt_sha1(scoped3) is still the shipped value (`{got}`)")
    check(prompt_fingerprint("scoped") != got,
          "a different prompt version hashes differently")


def test_llm_arm_refuses_to_overwrite_a_different_run():
    print("\n[test_llm_arm_refuses_to_overwrite_a_different_run]")
    with tempfile.TemporaryDirectory() as tmp:
        # the file a mock run will target
        target = csv_path(tmp, "mock", "scoped3", "near_rt", controller="mock",
                          endpoint="")
        cols = list(PROVENANCE_COLS) + ["variant", "scenario", "seed"]
        _write_csv(target, [{"model": "some-other-model", "endpoint": "",
                             "prompt_style": "scoped3", "prompt_sha1": "deadbeef",
                             "layer": "near_rt", "variant": "L1_reference",
                             "scenario": "S1_link_down", "seed": "0"}], cols)
        before = _digest(target)
        raised = None
        try:
            _mock_run(tmp)
        except ProvenanceConflict as exc:
            raised = exc
        check(raised is not None,
              "a conflicting existing CSV makes the writer raise "
              "ProvenanceConflict")
        check(_digest(target) == before,
              "the existing CSV is byte-identical after the refusal (not clobbered)")
        check(raised is not None and "some-other-model" in str(raised),
              "the error names the recorded provenance it refused to overwrite")
        check(not os.path.exists(target + ".tmp"),
              "no temp file is left behind by the refusal")


def test_llm_arm_refuses_a_file_without_provenance():
    print("\n[test_llm_arm_refuses_a_file_without_provenance]")
    with tempfile.TemporaryDirectory() as tmp:
        target = csv_path(tmp, "mock", "scoped3", "near_rt", controller="mock",
                          endpoint="")
        # exactly the shape of the pre-guard shipped files: no provenance cols
        _write_csv(target, [{"variant": "L1_reference", "scenario": "S1_link_down",
                             "seed": "0", "success": "1"}],
                   ["variant", "scenario", "seed", "success"])
        before = _digest(target)
        raised = None
        try:
            _mock_run(tmp)
        except ProvenanceConflict as exc:
            raised = exc
        check(raised is not None,
              "a CSV with no provenance columns is refused, not assumed to be ours")
        check(_digest(target) == before,
              "the unverifiable file is untouched")


def test_llm_arm_resumes_a_matching_run():
    print("\n[test_llm_arm_resumes_a_matching_run]")
    with tempfile.TemporaryDirectory() as tmp:
        first = _mock_run(tmp)
        check(first["mode"] == "new", "the first run reports mode='new'")
        n_first = first["episodes"]
        with open(first["csv"], newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        check(len(rows) == n_first and n_first > 0,
              f"the first run wrote {n_first} rows")
        second = _mock_run(tmp)
        check(second["csv"] == first["csv"],
              "the second identical run reuses the same file")
        check(second["mode"] == "resume",
              "the second identical run reports mode='resume'")
        check(second["episodes"] == n_first,
              f"resuming did not duplicate or drop rows ({second['episodes']})")


def test_llm_arm_rows_record_provenance():
    print("\n[test_llm_arm_rows_record_provenance]")
    with tempfile.TemporaryDirectory() as tmp:
        res = _mock_run(tmp)
        with open(res["csv"], newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols, rows = reader.fieldnames or [], list(reader)
        missing = [c for c in PROVENANCE_COLS if c not in cols]
        check(not missing, f"every row can be traced back to its run (missing: "
                           f"{missing or 'none'})")
        check(all(r["model"] == "mock" for r in rows),
              "the model column records the run's model on every row")
        check(all(r["prompt_style"] == "scoped3" for r in rows),
              "the prompt_style column records the prompt version")
        check(all(r["prompt_sha1"] == prompt_fingerprint("scoped3") for r in rows),
              "the prompt_sha1 column records the prompt template hash")
        check(all(r["layer"] == "near_rt" for r in rows),
              "the layer column records the control-layer placement")
        check(provenance_of(res["csv"]).get("_inconsistent") is None
              and provenance_of(res["csv"]).get("model") == "mock",
              "the written file's provenance re-reads as one consistent run")


def test_llm_arm_resumes_an_existing_hostless_run_in_place():
    print("\n[test_llm_arm_resumes_an_existing_hostless_run_in_place]")
    with tempfile.TemporaryDirectory() as tmp:
        # a run started before the host was part of the name: header only
        legacy = legacy_csv_path(tmp, "mock", "scoped3", "near_rt",
                                 controller="mock")
        _write_csv(legacy, [], list(PROVENANCE_COLS) + ["variant", "scenario",
                                                        "seed"])
        res = _mock_run(tmp)
        check(res["csv"] == legacy,
              "an existing host-less run is resumed in place, not duplicated")
        check(not os.path.exists(csv_path(tmp, "mock", "scoped3", "near_rt",
                                          controller="mock", endpoint="")),
              "no second file is created for the same run")
        check(res["episodes"] > 0, "the resumed legacy file receives the new rows")


def test_llm_arm_two_models_do_not_share_a_file():
    print("\n[test_llm_arm_two_models_do_not_share_a_file]")
    p7 = csv_path("/tmp/x", "qwen2.5:7b", "scoped3", "near_rt", endpoint=ENDPOINT)
    p14 = csv_path("/tmp/x", "qwen2.5:14b", "scoped3", "near_rt", endpoint=ENDPOINT)
    check(p7 != p14,
          "the 7b and 14b runs of one prompt cannot resolve to the same file "
          "(the regression that lost the 7b result)")
    check(os.path.basename(p7) != "llm_arm_scoped3.csv",
          "the pre-guard colliding name is no longer produced")
