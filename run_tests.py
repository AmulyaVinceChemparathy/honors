#!/usr/bin/env python3
"""
Run all project test scripts only (no training).

Usage (repo root):
  python run_tests.py

Colab: upload/clone the repo, then in a cell:
  !python run_tests.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Order: lightweight notebook checks first, then mp_v3_train tests
_CANDIDATES = ("kaggle_tests.py", "mp_v3_train_tests.py")

def main() -> int:
    code = 0
    for name in _CANDIDATES:
        path = ROOT / name
        if not path.is_file():
            continue
        r = subprocess.run([sys.executable, str(path)], cwd=ROOT)
        if r.returncode != 0:
            code = r.returncode
    if code == 0 and not any((ROOT / n).is_file() for n in _CANDIDATES):
        print("No test scripts found (expected kaggle_tests.py and/or mp_v3_train_tests.py).", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
