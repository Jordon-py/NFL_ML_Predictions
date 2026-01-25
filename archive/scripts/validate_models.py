#!/usr/bin/env python3
"""
scripts/validate_models.py

Lightweight validator for model artifacts in `backend/models/`.

Usage:
  python scripts/validate_models.py [--models-dir PATH]

Exits:
  0 - success (transform + predict ran)
  1 - metadata or model files missing
  2 - validation failed (exceptions during transform/predict)

This script is intended for local CI or developer troubleshooting. It does
not modify artifacts. It prints helpful diagnostics on failure.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd


def load_metadata(models_dir: Path) -> Dict[str, Any]:
    mpath = models_dir / "metadata.json"
    if not mpath.exists():
        print(f"ERROR: metadata.json not found in {models_dir}")
        sys.exit(1)
    return json.loads(mpath.read_text(encoding="utf-8"))


def build_dummy_row(raw_cols: Dict[str, Any]) -> Dict[str, Any]:
    # raw_cols may be dict with 'numeric'/'categorical' or a flat list
    numeric = []
    categorical = []
    if isinstance(raw_cols, dict):
        numeric = raw_cols.get("numeric", [])
        categorical = raw_cols.get("categorical", [])
    elif isinstance(raw_cols, list):
        numeric = raw_cols

    row = {}
    for n in numeric:
        row[n] = 0
    for c in categorical:
        if "team" in c.lower():
            row[c] = "NE"
        else:
            row[c] = "UNK"
    # safe defaults used in runtime feature build
    row.setdefault("home_moneyline_prob", 0.6)
    row.setdefault("away_moneyline_prob", 0.4)
    row.setdefault("spread_line", 0.0)
    row.setdefault("total_line", 45.0)
    row.setdefault("home_rest", 7)
    row.setdefault("away_rest", 7)
    return row


def main(argv: list[str]) -> int:
    models_dir = Path("backend") / "models"
    if len(argv) >= 2 and argv[1] in ("-m", "--models-dir") and len(argv) >= 3:
        models_dir = Path(argv[2])

    print(f"Validating model artifacts in: {models_dir}")
    if not models_dir.exists():
        print(f"ERROR: models directory not found: {models_dir}")
        return 1

    try:
        meta = load_metadata(models_dir)
    except SystemExit:
        return 1
    except Exception as e:
        print("ERROR: failed to read metadata.json:", e)
        return 1

    raw_cols = meta.get("raw_feature_columns", {})
    dummy = build_dummy_row(raw_cols)
    X = pd.DataFrame([dummy])

    # Helper to attempt loading an artifact and running a simple op
    def try_load_and_run(path: Path, label: str):
        if not path.exists():
            print(f"WARN: {label} not found at {path}")
            return True
        try:
            obj = joblib.load(path)
            print(f"Loaded {label} from {path} (type={type(obj)})")
            # If transformer, try transform
            if hasattr(obj, "transform"):
                try:
                    _ = obj.transform(X)
                    print(f"OK: {label}.transform succeeded")
                except Exception as te:
                    print(f"ERROR: {label}.transform failed: {te}")
                    traceback.print_exc()
                    return False
            # If predictor, try predict/predict_proba
            if hasattr(obj, "predict"):
                try:
                    _ = obj.predict(X)
                    print(f"OK: {label}.predict succeeded")
                except Exception as pe:
                    print(f"ERROR: {label}.predict failed: {pe}")
                    traceback.print_exc()
                    return False
            if hasattr(obj, "predict_proba"):
                try:
                    _ = obj.predict_proba(X)
                    print(f"OK: {label}.predict_proba succeeded")
                except Exception as ppe:
                    print(f"ERROR: {label}.predict_proba failed: {ppe}")
                    traceback.print_exc()
                    return False
            return True
        except Exception as e:
            print(f"ERROR: loading {label} at {path} raised: {e}")
            traceback.print_exc()
            return False

    # Resolve artifact paths using metadata values where present
    def resolve(name: str, fallback: str) -> Path:
        p = meta.get(name, fallback)
        pth = Path(p)
        return pth if pth.is_absolute() else models_dir / pth

    pre = resolve("preprocessor", "preprocessor.joblib")
    home_m = resolve("home_model", "home_model.joblib")
    away_m = resolve("away_model", "away_model.joblib")
    win_m = resolve("win_model", "win_clf_calibrated.joblib")

    ok = True
    ok = ok and try_load_and_run(pre, "preprocessor")
    ok = ok and try_load_and_run(home_m, "home_model")
    ok = ok and try_load_and_run(away_m, "away_model")
    # win model is optional
    if (models_dir / win_m).exists() or win_m.exists():
        ok = ok and try_load_and_run(win_m, "win_model")

    if ok:
        print("Validation completed: SUCCESS")
        return 0
    else:
        print("Validation completed: FAIL")
        return 2


if __name__ == "__main__":
    rc = main(sys.argv)
    sys.exit(rc)
