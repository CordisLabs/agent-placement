"""Test entry point: `python3 tests/test_all.py`.

Also collected by pytest if it is installed (each test module is importable).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_emulator                                             # noqa: E402
import test_harness                                              # noqa: E402
import test_llm_artifacts                                        # noqa: E402
import test_policy                                               # noqa: E402
from _util import FAILS                                          # noqa: E402

MODULES = (test_emulator, test_harness, test_policy, test_llm_artifacts)


def main() -> int:
    tests = []
    for mod in MODULES:
        tests += [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    print(f"aixnet-2026 selfcheck: {len(tests)} test groups "
          f"across {len(MODULES)} modules")
    for t in tests:
        t()
    print("\n" + "=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} assertion(s)")
        for f in FAILS:
            print("  -", f)
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
