#!/usr/bin/env python3
"""Compatibility wrapper for the canonical dataset builder.

The implementation lives in `backend.scripts.builddataset` after the backend
script reorganization. Keep this file so existing docs and local operator
commands can continue to run `python backend/builddataset.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.scripts.builddataset import main


if __name__ == "__main__":
    main()
