#!/usr/bin/env python3
"""Compatibility wrapper for the canonical training script.

The implementation lives in `backend.scripts.train_models` after the backend
script reorganization. Keep this file so existing docs, Heroku/admin tooling,
and local operator commands can continue to run `python backend/train_models.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.scripts.train_models import main


if __name__ == "__main__":
    raise SystemExit(main())
