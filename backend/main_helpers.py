# ==========================================
# File: backend/main_helpers.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: logging, json, pathlib, dataclasses
# Notes: Shared utilities.
# ==========================================

"""
FILE: backend/main_helpers.py
PURPOSE: Auxiliary functions and classes for main.py (model loading, history, etc).
DATA SHAPES:
  - InferenceBundle: Container for ML artifacts.
KEY FUNCTIONS/CLASSES:
  - load_inference_bundle, load_dataset_df.
  - _append_prediction_history_to_disk (flat history entries).
"""

import logging
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from threading import Lock
from datetime import datetime, timezone
import joblib
import pandas as pd

log = logging.getLogger(__name__)

prediction_history_entries: List[Dict[str, Any]] = []
_prediction_history_lock = Lock()
PREDICTION_HISTORY_MAX = 1000
PREDICTION_HISTORY_PATH = Path("backend/Predictions/prediction_history.json")

@dataclass(frozen=True)
class InferenceBundle:
    meta: Dict[str, Any]
    report: Dict[str, Any]
    preprocessor: Any
    home_model: Any
    away_model: Any
    hist_win_clf: Any

    @property
    def raw_feature_columns(self) -> List[str]:
        cols = self.meta.get("raw_feature_columns", {}) or {}
        if isinstance(cols, dict):
            num = cols.get("numeric", []) or []
            cat = cols.get("categorical", []) or []
            return [*list(num), *list(cat)]
        return self.meta.get("feature_names", []) or []

def load_inference_bundle(models_dir: Path) -> InferenceBundle:
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    report_path = models_dir / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    artifacts = meta.get("artifacts", meta)
    
    def _resolve_path(key: str, default: Optional[str] = None) -> Optional[Path]:
        val = artifacts.get(key) or meta.get(key) or default
        if not val: return None
        p = Path(str(val))
        if p.is_absolute():
            if p.exists():
                return p
            # Fall back to the local models_dir using the basename for portability.
            fallback = models_dir / p.name
            return fallback if fallback.exists() else p
        return models_dir / p

    pre_path = _resolve_path("preprocessor", "preprocessor.joblib")
    home_path = _resolve_path("home_model", "home_model.joblib")
    away_path = _resolve_path("away_model", "away_model.joblib")
    hist_path = _resolve_path("hist_win_model") or _resolve_path("win_model") or _resolve_path("win_clf_calibrated")

    return InferenceBundle(
        meta=meta,
        report=report,
        preprocessor=joblib.load(pre_path),
        home_model=joblib.load(home_path),
        away_model=joblib.load(away_path),
        hist_win_clf=joblib.load(hist_path) if hist_path and hist_path.exists() else None,
    )

def _score_dataset_file(path: Path, expected_features: List[str]) -> Optional[tuple[int, int]]:
    try:
        cols = pd.read_csv(path, nrows=0).columns
    except Exception as e:
        log.warning("Dataset header read failed for %s: %s", path, e)
        return None
    missing = sum(1 for c in expected_features if c not in cols)
    return missing, len(cols)

def _pick_best_dataset(files: List[Path], expected_features: List[str]) -> Optional[tuple[Path, int]]:
    scored: List[tuple[int, int, str, Path]] = []
    for path in files:
        score = _score_dataset_file(path, expected_features)
        if score is None:
            continue
        missing, col_count = score
        scored.append((missing, -col_count, path.name, path))
    if not scored:
        return None
    scored.sort()
    best = scored[0]
    return best[3], best[0]

def load_dataset_df(data_dir: Path, expected_features: Optional[List[str]] = None) -> pd.DataFrame:
    dataset_path = os.getenv("DATASET_PATH")
    if dataset_path:
        p = Path(dataset_path)
        if not p.is_absolute():
            cwd_candidate = p.resolve()
            data_candidate = (data_dir / p).resolve()
            p = cwd_candidate if cwd_candidate.exists() else data_candidate
        else:
            p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"DATASET_PATH not found: {p}")
        if expected_features:
            score = _score_dataset_file(p, expected_features)
            if score is not None:
                missing, _ = score
                if missing:
                    candidates = sorted(data_dir.glob("game_features_*.csv")) if data_dir.is_dir() else []
                    best = _pick_best_dataset(candidates, expected_features) if candidates else None
                    if best and best[1] == 0 and best[0] != p:
                        log.warning(
                            "DATASET_PATH missing %d expected features; using %s",
                            missing,
                            best[0],
                        )
                        return pd.read_csv(best[0])
        return pd.read_csv(p)

    if data_dir.is_file():
        return pd.read_csv(data_dir)

    files = sorted(data_dir.glob("game_features_*.csv"))
    if not files:
        raise FileNotFoundError(f"No game_features_*.csv in {data_dir}")
    if expected_features:
        best = _pick_best_dataset(files, expected_features)
        if best:
            if best[1] > 0:
                log.warning(
                    "No dataset fully matches model features; best missing %d: %s",
                    best[1],
                    best[0],
                )
            return pd.read_csv(best[0])
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one game_features_*.csv in {data_dir}; set DATASET_PATH explicitly."
        )
    return pd.read_csv(files[0])

def _build_game_id_from_request(request_payload: Dict[str, Any]) -> Optional[str]:
    """Best-effort game_id builder to keep history entries consistent."""
    if not isinstance(request_payload, dict):
        return None
    season = request_payload.get("season")
    week = request_payload.get("week")
    home = request_payload.get("home_team") or request_payload.get("home_abbr")
    away = request_payload.get("away_team") or request_payload.get("away_abbr")
    home = str(home).strip().upper() if home is not None else None
    away = str(away).strip().upper() if away is not None else None
    parts = [season, week, home, away]
    normalized = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    return "-".join(normalized) if normalized else None

def _append_prediction_history_to_disk(request_payload: Dict[str, Any], prediction_payload: Dict[str, Any]) -> None:
    """Append a unified, flat prediction entry for frontend history views."""
    global prediction_history_entries
    entry: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(prediction_payload, dict):
        entry.update(prediction_payload)
    if isinstance(request_payload, dict):
        entry.setdefault("season", request_payload.get("season"))
        entry.setdefault("week", request_payload.get("week"))
        home = request_payload.get("home_team") or request_payload.get("home_abbr")
        away = request_payload.get("away_team") or request_payload.get("away_abbr")
        entry.setdefault("home_team", str(home).strip().upper() if home is not None else None)
        entry.setdefault("away_team", str(away).strip().upper() if away is not None else None)
        entry.setdefault("game_id", _build_game_id_from_request(request_payload))
    with _prediction_history_lock:
        prediction_history_entries = [entry] + (prediction_history_entries or [])
        prediction_history_entries = prediction_history_entries[:PREDICTION_HISTORY_MAX]
        try:
            PREDICTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            PREDICTION_HISTORY_PATH.write_text(json.dumps(prediction_history_entries, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"History persist failed: {e}")
