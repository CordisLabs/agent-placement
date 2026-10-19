"""Shared assertion helper: no pytest dependency, plain asserts."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL {msg}")
    else:
        print(f"  ok   {msg}")


def count() -> int:
    return len(FAILS)
