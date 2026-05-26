# ==========================================
# File: backend/main.py
# Role: FastAPI backend for the NFL prediction dashboard.
# Input Data: HTTP requests (JSON payloads), Model artifacts, Dataset CSVs.
# Output Data: JSON responses (Health, Status, Schedule, Predictions, History).
# Dependencies: fastapi, pydantic, pandas, numpy, joblib, nflreadpy, uvicorn
# Notes: Highest-risk edit zone; manages the full runtime lifecycle and API surface.
# ==========================================

# -*- coding: utf-8 -*-
"""
File: backend/main.py

Purpose:
    FastAPI backend for the NFL prediction dashboard.

    Exposes endpoints for:
    - /health            : service + components health
    - /status/overview   : lightweight dashboard summary
    - /schedule/next-week: upcoming week schedule
    - /history           : recent prediction history
    - /predict           : single-game prediction (scores + win probabilities)

Key design points:
    - Loads latest game_features*.csv dataset + trained models at startup.
    - Normalizes team codes (home/away, abbr/name) to uppercase for stable matching.
    - Uses Pydantic models for request/response typing where it matters
      (health + prediction).
    - Keeps in-memory prediction history bounded for the /history endpoint.

Notes:
    - The prediction contract is aligned with the current React client:
        - Request body: { home_team, away_team, season, week }
        - Response: PredictionResponse with home_score, away_score,
          home_win_probability, away_win_probability, point_diff, etc.

Data shapes:
    - Schedule input rows are pandas DataFrames with season/week/team columns,
      optional gameday/gametime columns, and a derived UTC dt column.
    - Schedule responses are List[ScheduleGameResponse] dictionaries consumed
      by frontend/src/api/client.js.

Syntax notes:
    - FastAPI decorators expose route functions directly.
    - Pydantic models below define public JSON contracts.

Important functions (line numbers last refreshed 2026-04-30):
    - _load_schedule_dataframe: around line 2925
    - _select_schedule_slice: around line 3023
    - _schedule_response: around line 3161

Possible bugs:
    - Upstream nflreadpy/network outages can force CSV fallback behavior.
    - A stale SCHEDULE_PATH can hide newer packaged schedules if fallback
      discovery is not allowed to scan sibling schedule CSVs.

Enhancement ideas:
    - Cache normalized schedule frames with a short TTL.
    - Add a season-release job that refreshes packaged schedule CSVs after the
      official NFL schedule release.
"""

import json
import logging
import os
import hashlib
import re
import time
import sys
import threading
import shutil
import subprocess
import csv
import uuid
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Literal, Set
import nflreadpy as nfl
from dotenv import load_dotenv
import numpy as np
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from backend.app.core.settings import get_settings
from backend.prediction_store import (
    append_prediction_record,
    build_prediction_user_context,
    get_prediction_history as load_prediction_history,
    get_prediction_history_count,
    get_prediction_history_summary as load_prediction_history_summary,
)
from backend.score_sync import extract_score_entries_from_dataframe
from backend.schemas import PredictionRequest as StoredPredictionRequest
from backend.services.inference_row import build_model_input_row
from backend.sqlite_store import upsert_game_scores
from backend.utils import functions_for_main as fn_main
from backend.utils.ops_reporting import (
    collect_dataset_versions,
    collect_performance_drift,
    resolve_latest_dataset,
    file_sha256,
    load_latest_dataset_manifest,
)
from backend.utils.cache import LRUCache

_add_kickoff_utc_datetime = fn_main._add_kickoff_utc_datetime
_coerce_season_week = fn_main._coerce_season_week
_normalize_team_columns = fn_main._normalize_team_columns
_prepare_inputs = fn_main._prepare_inputs
_is_pipeline = fn_main._is_pipeline
_align_numeric_df_for_model = fn_main._align_numeric_df_for_model
_normalize_team_code = fn_main._normalize_team_code
_predict_score = fn_main._predict_score
_clamp_score = fn_main._clamp_score
_smooth_win_probability = fn_main._smooth_win_probability
TEAM_ABBR_MAP = getattr(fn_main, "TEAM_ABBR_MAP", {})
_roll_forward_missing_player_stats = getattr(
    fn_main,
    "_roll_forward_missing_player_stats",
    lambda df, row_df, home_team, away_team, season, week: row_df,
)

if hasattr(fn_main, "_get_game_row_with_source"):
    _get_game_row_with_source = fn_main._get_game_row_with_source
else:
    # Backward-compatibility for environments where functions_for_main still exports only `_get_game_row`.
    def _get_game_row_with_source(
        df: pd.DataFrame,
        season: int,
        week: int,
        home_team: str,
        away_team: str,
    ) -> Tuple[pd.DataFrame, Literal["dataset_exact", "dataset_fuzzy"]]:
        row = fn_main._get_game_row(df, season, week, home_team, away_team)
        return row, "dataset_exact"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


# -------------------------------------------------------------------
# Config & Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parent

# Load local backend/.env only outside Heroku dynos.
if not os.getenv("DYNO"):
    load_dotenv(BASE_DIR / ".env")

SETTINGS = get_settings()
logging.getLogger().setLevel(getattr(logging, str(SETTINGS.log_level).upper(), logging.INFO))

DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

try:
    import sklearn
except Exception:  # pragma: no cover - optional runtime guard
    sklearn = None

SKLEARN_RUNTIME_VERSION = getattr(sklearn, "__version__", None)

try:
    import warnings
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except (ImportError, AttributeError):
        InconsistentVersionWarning = None
except (ImportError, AttributeError):  # pragma: no cover - sklearn import already guarded above
    InconsistentVersionWarning = None  # type: ignore[assignment]
JOBS_DIR = DATA_DIR / "jobs"
STAGING_MODELS_DIR = DATA_DIR / "models" / "staging"
CURRENT_MODELS_DIR = DATA_DIR / "models" / "current"
METRICS_HISTORY_PATH = REPORTS_DIR / "drift" / "metrics_history.csv"



# Allow overriding the schedule CSV via env; default to backend/data
schedule_env_path = SETTINGS.resolved_schedule_path
SCHEDULE_PATH = schedule_env_path if schedule_env_path else (DATA_DIR / "Nfl_schedule_2025.csv")

# Required model keys for /predict to be "ready"
REQUIRED_MODELS: Tuple[str, ...] = ("home", "away", "win")
WIN_PROBA_FEATURE = "nn_home_win_proba"
PREDICT_CACHE_TTL_SEC = max(0, int(SETTINGS.predict_cache_ttl_sec))
PREDICT_CACHE_MAX_ITEMS = max(50, int(SETTINGS.predict_cache_max_items))
GAME_SCORE_SYNC_TTL_SEC = 900
# Models directory is resolved at runtime by _find_models_dir().
# Override in production with: MODELS_DIR=/absolute/or/repo-relative/path


# Ensure expected folders exist (safe on repeated calls)
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STAGING_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: MODELS_DIR is resolved via _find_models_dir() (defined below)
PREDICTION_STORAGE = BASE_DIR / "Predictions"
PREDICTION_STORAGE.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------





def _to_pandas_schedule_safe(table: Any) -> pd.DataFrame:
    """Convert schedule table objects (pandas/polars-like) to pandas safely."""
    if table is None:
        return pd.DataFrame()
    if isinstance(table, pd.DataFrame):
        return table

    if hasattr(table, "to_pandas"):
        try:
            return table.to_pandas(use_pyarrow_extension_array=False)
        except TypeError:
            try:
                return table.to_pandas()
            except Exception:
                pass
        except Exception:
            pass

    if hasattr(table, "to_dicts"):
        try:
            return pd.DataFrame(table.to_dicts())
        except Exception:
            pass

    try:
        return pd.DataFrame(table)
    except Exception:
        logging.exception("[Schedule] Failed to coerce schedule table to pandas.")
        return pd.DataFrame()


def _schedule_csv_year(path: Path) -> Optional[int]:
    """Infer a season year from a schedule CSV filename when possible."""
    matches = re.findall(r"(?:19|20|21)\d{2}", path.stem)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _looks_like_schedule_csv(path: Path) -> bool:
    """Return True for CSV files that are likely to contain NFL schedules."""
    name = path.name.lower()
    return path.suffix.lower() == ".csv" and ("schedule" in name or name.startswith("nfl"))


def _schedule_path_sort_key(path: Path, requested_season: Optional[int]) -> Tuple[int, int, float, str]:
    """Sort packaged schedule paths toward the requested or upcoming season."""
    season = _schedule_csv_year(path)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    if requested_season is not None:
        if season == int(requested_season):
            return (0, 0, -mtime, str(path))
        if season is None:
            return (1, 0, -mtime, str(path))
        return (2, abs(season - int(requested_season)), -mtime, str(path))

    current_year = datetime.now(timezone.utc).year
    if season is None:
        return (2, 9999, -mtime, str(path))
    if season >= current_year:
        return (0, season - current_year, -mtime, str(path))
    return (1, current_year - season, -mtime, str(path))


def _find_schedule_paths(requested_season: Optional[int] = None) -> List[Path]:
    """
    Locate schedule CSV files in priority order.

    Priority:
      1. Explicit SCHEDULE_PATH, when present
      2. Packaged backend/data schedule CSVs
      3. Frontend public schedule CSVs for local-dev compatibility

    Multiple files are returned so a stale explicit path cannot hide a newer
    packaged upcoming-season schedule during the offseason.
    """
    candidates: List[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not path.exists() or not path.is_file():
            return
        seen.add(resolved)
        candidates.append(path)

    # 1) explicit path
    add(SCHEDULE_PATH)

    # 2) search backend/data for schedule-like CSVs
    for p in DATA_DIR.glob("*.csv"):
        if _looks_like_schedule_csv(p):
            add(p)

    # 3) local dev fallbacks under frontend/public
    frontend_public = BASE_DIR.parent / "frontend" / "public"
    for p in (frontend_public / "schedules").glob("*.csv"):
        if _looks_like_schedule_csv(p):
            add(p)
    add(frontend_public / "nflSchedule.csv")

    return sorted(candidates, key=lambda p: _schedule_path_sort_key(p, requested_season))


def _find_schedule_path() -> Optional[Path]:
    """Backward-compatible single-path helper for older internal callers."""
    paths = _find_schedule_paths()
    return paths[0] if paths else None


def _models_dir_has_required_artifacts(models_dir: Path) -> bool:
    """Return True if a directory looks like a complete model bundle.

    We treat a bundle as "complete" if it contains home+away regressors and a win classifier,
    either as full pipelines (*_pipe.joblib) or as raw estimators (*_model.joblib / win_clf*.joblib).
    """
    try:
        if not models_dir.exists() or not models_dir.is_dir():
            return False

        has_home = (models_dir / "home_pipe.joblib").exists() or (models_dir / "home_model.joblib").exists()
        has_away = (models_dir / "away_pipe.joblib").exists() or (models_dir / "away_model.joblib").exists()
        has_win = (models_dir / "win_pipe.joblib").exists() or (models_dir / "win_clf_calibrated.joblib").exists()
        return bool(has_home and has_away and has_win)
    except Exception:
        return False


def _models_dir_metadata_tier(models_dir: Path) -> int:
    """Rank model bundle metadata quality without validating runtime compatibility.

    Selection should prefer a strict serving contract even if the current Python
    environment cannot load it. That makes version mismatches visible as startup
    blockers instead of silently falling back to an older metadata-less bundle.
    """
    metadata_path = models_dir / "metadata.json"
    try:
        if not metadata_path.exists():
            return 0
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return 0
        if meta.get("serving_mode") == "pipeline_primary" or meta.get("bundle_contract_version"):
            return 2
        return 1
    except Exception:
        return 0


def _pick_best_models_dir(candidates: List[Path]) -> Optional[Path]:
    strict: List[Path] = []
    legacy_with_metadata: List[Path] = []
    metadata_less: List[Path] = []

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen or not _models_dir_has_required_artifacts(candidate):
            continue
        seen.add(resolved)

        tier = _models_dir_metadata_tier(candidate)
        if tier >= 2:
            strict.append(candidate)
        elif tier == 1:
            legacy_with_metadata.append(candidate)
        else:
            metadata_less.append(candidate)

    for bucket in (strict, legacy_with_metadata, metadata_less):
        if bucket:
            return bucket[0]
    return None


def _find_models_dir() -> Path:
    """Locate the best models directory.

    Priority order:
      1) Env override (recommended for Heroku): MODELS_DIR / MODELS_PATH / MODEL_DIR
      2) The promoted runtime bundle under backend/data/models/current
      3) Strict metadata-backed bundles such as backend/models
      4) Complete legacy bundles under backend/data/models or common repo locations
      5) Fallback: backend/models (even if incomplete, so errors are visible in logs)

    Tip:
      - On Heroku, set MODELS_DIR=backend/models unless a promoted current bundle exists.
      - Run with the repo environment that matches metadata.json's sklearn_version.
    """
    env_path = SETTINGS.resolved_models_dir
    if env_path is None:
        env = (
            os.environ.get("MODELS_DIR")
            or os.environ.get("MODELS_PATH")
            or os.environ.get("MODEL_DIR")
        )
        env_path = Path(env).expanduser() if env else None

    if env_path is not None:
        p = env_path
        if _models_dir_has_required_artifacts(p):
            return p
        if p.exists():
            # Exists but doesn't look complete; still return so logs show what is missing.
            return p

    candidates: List[Path] = []

    # Promoted runtime bundle produced by admin/retrain flows.
    promoted_current = CURRENT_MODELS_DIR
    if _models_dir_has_required_artifacts(promoted_current):
        return promoted_current

    # Repository-local shared bundle used by older deployment packaging.
    packaged_data_models = DATA_DIR / "models"
    candidates.append(packaged_data_models)

    # Common packaged pattern: backend/data/prod-models/models
    direct = BASE_DIR / "data" / "prod-models" / "models"
    candidates.append(direct)

    # Verified repository-local bundle produced by train_models.py.
    local_default = BASE_DIR / "models"
    candidates.append(local_default)

    # Date-stamped training runs: backend/20251215/models (most recent wins)
    for p in sorted(BASE_DIR.glob("20*/models"), key=lambda item: item.stat().st_mtime, reverse=True):
        candidates.append(p)

    # Any nested prod-models/models in the repo
    for p in sorted(BASE_DIR.glob("**/prod-models/models"), key=lambda item: item.stat().st_mtime, reverse=True):
        candidates.append(p)

    best = _pick_best_models_dir(candidates)
    if best is not None:
        return best

    return local_default


# Resolve models directory once at import time so serving code can rely on MODELS_DIR.
MODELS_DIR: Path = _find_models_dir()


def _resolve_team_metadata_path() -> Optional[Path]:
    def _resolve_path(p: Path) -> Optional[Path]:
        if p.exists():
            return p
        if not p.is_absolute():
            for base in (BASE_DIR.parent, BASE_DIR):
                candidate = (base / p).resolve()
                if candidate.exists():
                    return candidate
        return None

    candidates: List[Path] = []
    env = os.environ.get("TEAM_LOGOS_PATH") or os.environ.get("TEAM_LOGO_PATH")
    if env:
        candidates.append(Path(env).expanduser())

    candidates.extend(
        [
            DATA_DIR / "team_logos.csv",
            DATA_DIR / "team_logo.csv",
            DATA_DIR / "team_logo_abbr.json",
            BASE_DIR / "team_logos.csv",
            BASE_DIR / "team_logo.csv",
            BASE_DIR.parent / "team_logos.csv",
            BASE_DIR.parent / "backend" / "team_logos.csv",
            BASE_DIR.parent / "frontend" / "public" / "team_logos.csv",
        ]
    )

    for raw_path in candidates:
        resolved = _resolve_path(raw_path)
        if resolved:
            return resolved
    return None


@lru_cache(maxsize=1)
def _load_team_metadata_map() -> Dict[str, Dict[str, Any]]:
    """Load team branding metadata once and reuse it across schedule + logo endpoints."""
    path = _resolve_team_metadata_path()
    if not path:
        logging.info("[Logos] No team metadata source found; returning empty map.")
        return {}

    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            out: Dict[str, Dict[str, Any]] = {}
            for key, value in payload.items():
                team_key = str(key).strip().upper()
                if not team_key:
                    continue
                meta = value if isinstance(value, dict) else {"logoUrl": value}
                normalized = {
                    "name": str(meta.get("name") or team_key).strip() or team_key,
                    "logoUrl": str(meta.get("logoUrl") or meta.get("logo_url") or "").strip() or None,
                    "primaryColor": str(meta.get("primaryColor") or meta.get("primary_color") or "").strip() or None,
                    "secondaryColor": str(meta.get("secondaryColor") or meta.get("secondary_color") or "").strip() or None,
                    "wordmark": str(meta.get("wordmark") or meta.get("word_mark") or "").strip() or None,
                }
                out[team_key] = normalized
                if TEAM_ABBR_MAP and team_key in TEAM_ABBR_MAP:
                    out.setdefault(TEAM_ABBR_MAP[team_key], dict(normalized))
            if out:
                logging.info("[Logos] Loaded %d team metadata rows from %s", len(out), path)
            return out

        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}

        key_col = next((cols[k] for k in ("abbr", "team", "team_abbr", "team_code") if k in cols), None)
        if not key_col:
            return {}

        name_col = next((cols[k] for k in ("team_name", "name", "team") if k in cols), None)
        logo_col = next(
            (
                cols[k]
                for k in (
                    "team_logo_espn",
                    "team_logo_squared",
                    "team_logo_wikipedia",
                    "team_wordmark",
                    "logo_url",
                    "logo",
                    "url",
                    "image_url",
                    "image",
                )
                if k in cols
            ),
            None,
        )
        primary_col = next((cols[k] for k in ("team_color", "primary_color", "primarycolor") if k in cols), None)
        secondary_col = next((cols[k] for k in ("team_color2", "secondary_color", "secondarycolor") if k in cols), None)
        wordmark_col = next((cols[k] for k in ("team_wordmark", "wordmark") if k in cols), None)

        out: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            team_key = str(row.get(key_col, "")).strip().upper()
            if not team_key:
                continue
            meta = {
                "name": str(row.get(name_col, "")).strip() or team_key,
                "logoUrl": str(row.get(logo_col, "")).strip() if logo_col else "",
                "primaryColor": str(row.get(primary_col, "")).strip() if primary_col else "",
                "secondaryColor": str(row.get(secondary_col, "")).strip() if secondary_col else "",
                "wordmark": str(row.get(wordmark_col, "")).strip() if wordmark_col else "",
            }
            normalized = {
                "name": meta["name"],
                "logoUrl": meta["logoUrl"] or None,
                "primaryColor": meta["primaryColor"] or None,
                "secondaryColor": meta["secondaryColor"] or None,
                "wordmark": meta["wordmark"] or None,
            }
            out[team_key] = normalized
            if TEAM_ABBR_MAP and team_key in TEAM_ABBR_MAP:
                out.setdefault(TEAM_ABBR_MAP[team_key], dict(normalized))

        if out:
            logging.info("[Logos] Loaded %d team metadata rows from %s", len(out), path)
        return out
    except Exception as exc:
        logging.warning("[Logos] Failed reading %s: %s", path, exc)
        return {}


def _load_team_logo_map() -> Dict[str, str]:
    return {
        team_code: str(meta.get("logoUrl") or "").strip()
        for team_code, meta in _load_team_metadata_map().items()
        if str(meta.get("logoUrl") or "").strip()
    }


def _calculate_win_probability(
    win_model: Any,
    full_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    preprocessor: Optional[Any] = None,
    numeric_medians: Optional[pd.Series] = None,
) -> Tuple[float, bool]:
    """Compute home-team win probability with sensible fallbacks.

    Preferred order:
      1) Use a fitted sklearn Pipeline (win_pipe.joblib) on the *raw* DataFrame.
      2) Use a calibrated classifier (win_clf_calibrated.joblib) on *preprocessed* features
         via the standalone preprocessor.joblib, if present.
      3) As a last resort, fall back to `home_moneyline_prob`, else `0.5`.

    Returns:
      (home_win_probability, win_classifier_used)
    """
    def _predict_positive_class_probability(model: Any, features: Any) -> float:
        raw = np.asarray(model.predict_proba(features), dtype=float)
        if raw.ndim != 2 or raw.shape[0] == 0:
            raise ValueError("predict_proba did not return a 2D probability matrix.")
        if raw.shape[1] == 1:
            return float(raw[0][0])

        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            for step in reversed(list(getattr(model, "named_steps", {}).values())):
                classes = getattr(step, "classes_", None)
                if classes is not None:
                    break

        if classes is not None:
            classes_arr = np.asarray(classes)
            positive_matches = np.where(classes_arr == 1)[0]
            positive_idx = int(positive_matches[0]) if len(positive_matches) else int(len(classes_arr) - 1)
        else:
            positive_idx = 1

        return float(raw[0][positive_idx])

    def _align_raw_features(features: pd.DataFrame, model: Any) -> pd.DataFrame:
        aligned = features.copy() if features is not None else pd.DataFrame()
        expected_cols = (
            list(getattr(model, "feature_names_in_", []))
            if hasattr(model, "feature_names_in_")
            else []
        )
        if not expected_cols and preprocessor is not None:
            expected_cols = (
                list(getattr(preprocessor, "feature_names_in_", []))
                if hasattr(preprocessor, "feature_names_in_")
                else []
            )
        if expected_cols:
            aligned = aligned.reindex(columns=expected_cols)
            if numeric_medians is not None and not numeric_medians.empty:
                for col in expected_cols:
                    if col in numeric_medians.index:
                        aligned[col] = pd.to_numeric(
                            aligned[col], errors="coerce"
                        ).fillna(float(numeric_medians[col]))
        return aligned

    def _fallback_probability() -> float:
        try:
            if full_df is not None and not full_df.empty and "home_moneyline_prob" in full_df.columns:
                val = full_df.iloc[0].get("home_moneyline_prob")
                raw = pd.to_numeric(val, errors="coerce") if val is not None else pd.NA
                if pd.notna(raw):
                    return float(np.clip(float(raw), 1e-6, 1 - 1e-6))
        except Exception:
            pass
        return 0.5

    if win_model is not None and hasattr(win_model, "predict_proba"):
        is_pipeline = bool(getattr(win_model, "steps", None) is not None or getattr(win_model, "named_steps", None) is not None)

        # A) Pipeline case: pass raw DataFrame
        if is_pipeline:
            try:
                pipeline_df = _align_raw_features(full_df, win_model)
                win_prob = _predict_positive_class_probability(win_model, pipeline_df)
                return float(np.clip(win_prob, 1e-6, 1 - 1e-6)), True
            except Exception as e:
                logging.warning("[Predict] win_pipe predict_proba failed; falling back to priors: %s", e)

        # B) Classifier-only case: transform then predict_proba
        if (not is_pipeline) and (preprocessor is not None):
            try:
                pre_df = _align_raw_features(full_df, preprocessor)
                X_proc = preprocessor.transform(pre_df)
                win_prob = _predict_positive_class_probability(win_model, X_proc)
                return float(np.clip(win_prob, 1e-6, 1 - 1e-6)), True
            except Exception as e:
                logging.warning("[Predict] win_clf predict_proba failed after preprocessor.transform; falling back: %s", e)

        # C) Last attempt: numeric-only (may work if the model was trained on raw numeric columns)
        if not is_pipeline:
            try:
                if numeric_df is not None and not numeric_df.empty:
                    win_prob = _predict_positive_class_probability(win_model, numeric_df)
                    return float(np.clip(win_prob, 1e-6, 1 - 1e-6)), True
            except Exception as e:
                logging.warning("[Predict] win_clf predict_proba failed on numeric_df; falling back: %s", e)

    return _fallback_probability(), False


def _augment_with_win_probability_feature(
    full_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    win_prob: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Append the raw home-win probability used by stacked score models."""
    prob = float(np.clip(float(win_prob), 1e-6, 1 - 1e-6))
    full_aug = full_df.copy()
    numeric_aug = numeric_df.copy()
    full_aug[WIN_PROBA_FEATURE] = prob
    numeric_aug[WIN_PROBA_FEATURE] = prob
    return full_aug, numeric_aug


def _patch_imputer_compat(obj: Any) -> int:
    """
    Patch sklearn 1.7->1.8 SimpleImputer compatibility (`_fill_dtype`).
    """
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        return 0

    seen: set[int] = set()
    patched = 0

    def walk(node: Any) -> None:
        nonlocal patched
        if node is None:
            return
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)

        if isinstance(node, SimpleImputer):
            if (not hasattr(node, "_fill_dtype")) and hasattr(node, "_fit_dtype"):
                try:
                    node._fill_dtype = node._fit_dtype  # type: ignore[attr-defined]
                    patched += 1
                except Exception:
                    pass

        # Common sklearn container attributes
        for attr in ("steps", "transformers_", "transformer_list", "named_steps"):
            if hasattr(node, attr):
                val = getattr(node, attr)
                if isinstance(val, dict):
                    for item in val.values():
                        walk(item)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, tuple):
                            if len(item) >= 2:
                                walk(item[1])
                        else:
                            walk(item)

        for attr in ("estimator", "base_estimator", "preprocessor"):
            if hasattr(node, attr):
                walk(getattr(node, attr))

    walk(obj)
    return patched


def _bundle_timestamp(meta: Dict[str, Any]) -> Optional[str]:
    for key in ("bundle_timestamp_utc", "timestamp", "training_timestamp_utc"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


def _requires_strict_bundle_contract(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict) or not meta:
        return False
    if meta.get("serving_mode") == "pipeline_primary":
        return True
    if meta.get("bundle_contract_version"):
        return True
    return False


def _validate_bundle_metadata_contract(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict) or not meta:
        raise RuntimeError("Model metadata is missing or invalid.")

    if not _requires_strict_bundle_contract(meta):
        return meta

    required_keys = [
        "serving_mode",
        "feature_manifests",
        "generated_features",
        "dataset_hash",
        "sklearn_version",
    ]
    missing = [key for key in required_keys if not meta.get(key)]
    if not _bundle_timestamp(meta):
        missing.append("bundle_timestamp_utc")
    if missing:
        raise RuntimeError(
            "Model bundle metadata is missing required contract fields: "
            + ", ".join(sorted(set(missing)))
        )

    runtime_version = SKLEARN_RUNTIME_VERSION
    declared_version = str(meta.get("sklearn_version") or "").strip()
    if runtime_version and declared_version and runtime_version != declared_version:
        raise RuntimeError(
            f"Model bundle requires scikit-learn {declared_version}, but runtime has {runtime_version}."
        )

    return meta


def _collect_model_load_warnings(caught: List[warnings.WarningMessage], path: Path) -> List[str]:
    messages: List[str] = []
    for warning_item in caught:
        warning_message = warning_item.message
        text = str(warning_message)
        is_version_warning = bool(
            InconsistentVersionWarning is not None
            and isinstance(warning_message, InconsistentVersionWarning)
        )
        if is_version_warning or "Trying to unpickle estimator" in text:
            messages.append(f"{path.name}: {text}")
    return messages


# -------------------------------------------------------------------
# App State
# -------------------------------------------------------------------


class AppState:
    """
    In-memory state container for:
      - dataset : feature DataFrame used for prediction
      - models  : dict of trained models (home / away / win)
      - history : list of prediction results returned to clients
    """

    def __init__(self) -> None:
        self.started_at: datetime = datetime.now(timezone.utc)
        self.last_prediction_at: Optional[datetime] = None
        self.dataset: Optional[pd.DataFrame] = None
        self.dataset_path: Optional[Path] = None
        self.dataset_hash: Optional[str] = None
        self.dataset_mtime: Optional[float] = None
        self.models: Dict[str, Any] = {}
        self.models_metadata: Dict[str, Any] = {}
        self.dataset_manifest: Dict[str, Any] = {}
        self.dataset_metadata: Dict[str, Any] = {}
        self.feature_manifest: List[str] = []
        # Optional shared preprocessor artifact (may be saved separately from models)
        self.preprocessor: Optional[Any] = None
        self.score_preprocessor: Optional[Any] = None
        self.win_preprocessor: Optional[Any] = None
        self.history: List[Dict[str, Any]] = []
        # Use an in-memory LRU cache with TTL for prediction results
        try:
            self.predict_cache = LRUCache(max_items=PREDICT_CACHE_MAX_ITEMS, ttl=PREDICT_CACHE_TTL_SEC)
        except Exception:
            self.predict_cache = LRUCache(max_items=256, ttl=300)
        self.predict_cache_hits: int = 0
        self.predict_cache_misses: int = 0
        self.retrain_jobs: Dict[str, Dict[str, Any]] = {}
        self.retrain_lock = threading.Lock()
        self.production_blockers: List[str] = []
        self.production_warnings: List[str] = []
        self.model_load_errors: Dict[str, str] = {}
        self.last_game_score_sync_at: Optional[datetime] = None
        self.last_game_score_sync_count: int = 0

        # Cached numeric medians from the loaded dataset (used for stable imputation).
        # Set during _load_dataset().
        self.numeric_medians: Optional[pd.Series] = None
        self.prior_baseline_medians: Optional[pd.Series] = None
        self._recent_history_keys: Set[str] = set()


    # -------------------------
    # Startup Loader
    # -------------------------
    def load(self) -> None:
        """Load dataset + models at startup with defensive logging."""
        self.started_at = datetime.now(timezone.utc)
        self._load_dataset()
        self.sync_game_scores(force=True)
        self._load_models()

    def refresh_dataset_if_changed(self) -> bool:
        """
        Reload dataset when the selected file has changed on disk.
        Returns True when a reload occurred.
        """
        try:
            explicit = SETTINGS.dataset_path
            target_path = resolve_latest_dataset(DATA_DIR, explicit_path=explicit)
        except Exception:
            return False

        if not target_path.exists():
            return False
        target_mtime = float(target_path.stat().st_mtime)

        if self.dataset_path is None:
            self._load_dataset()
            self.sync_game_scores(force=True)
            return True
        if self.dataset_path.resolve() != target_path.resolve():
            self._load_dataset()
            self.sync_game_scores(force=True)
            return True
        if self.dataset_mtime is None or target_mtime > float(self.dataset_mtime):
            self._load_dataset()
            self.sync_game_scores(force=True)
            return True
        return False

    def sync_game_scores(
        self,
        *,
        force: bool = False,
        schedule_df: Optional[pd.DataFrame] = None,
    ) -> int:
        """Sync completed game scores into SQLite and backfill stored predictions."""

        now = datetime.now(timezone.utc)
        if (
            not force
            and schedule_df is None
            and self.last_game_score_sync_at is not None
            and (now - self.last_game_score_sync_at).total_seconds() < GAME_SCORE_SYNC_TTL_SEC
        ):
            return 0

        entries: List[Dict[str, object]] = []
        if self.dataset is not None:
            entries.extend(
                extract_score_entries_from_dataframe(
                    self.dataset,
                    updated_at=now.isoformat(),
                )
            )
        if schedule_df is not None:
            entries.extend(
                extract_score_entries_from_dataframe(
                    schedule_df,
                    updated_at=now.isoformat(),
                )
            )

        if not entries:
            self.last_game_score_sync_at = now
            self.last_game_score_sync_count = 0
            return 0

        try:
            upsert_game_scores(entries)
        except Exception as exc:
            logging.warning("[Scores] Failed to sync completed game scores: %s", exc)
            return 0

        self.last_game_score_sync_at = now
        self.last_game_score_sync_count = len(entries)
        logging.info("[Scores] Synced %d completed game score rows into SQLite.", len(entries))
        return len(entries)

    def _prediction_cache_key(self, *, season: int, week: int, home_team: str, away_team: str) -> str:
        return f"{season}:{week}:{home_team}:{away_team}"

    def _remember_history(self, payload: Dict[str, Any]) -> None:
        """Keep in-memory history bounded and avoid duplicate entries from cache hits."""
        game_id = str(payload.get("game_id") or "")
        generated_at = str(payload.get("generated_at") or "")
        dedupe_key = f"{game_id}:{generated_at}"
        if dedupe_key in self._recent_history_keys:
            return
        self.history.append(payload)
        self._recent_history_keys.add(dedupe_key)
        if len(self.history) > 500:
            dropped = self.history.pop(0)
            dropped_key = f"{dropped.get('game_id', '')}:{dropped.get('generated_at', '')}"
            self._recent_history_keys.discard(dropped_key)

    def valid_team_codes(self) -> Set[str]:
        """Collect a runtime-valid set of team codes for fail-fast input validation."""
        codes: Set[str] = {str(k).upper() for k in TEAM_ABBR_MAP.keys()}
        if self.dataset is not None:
            for col in ("home_team", "away_team", "home_abbr", "away_abbr"):
                if col in self.dataset.columns:
                    codes.update(
                        self.dataset[col].dropna().astype(str).str.upper().unique().tolist()
                    )
        return {code for code in codes if code}

    def _refresh_runtime_readiness(self) -> None:
        blockers: List[str] = []
        warnings_out: List[str] = []

        if self.dataset is None:
            blockers.append("dataset not loaded")

        missing_models = [name for name in REQUIRED_MODELS if name not in self.models]
        if missing_models:
            blockers.append(f"missing models: {', '.join(sorted(missing_models))}")

        if not self.models_metadata:
            warnings_out.append("model metadata unavailable")
        elif not _requires_strict_bundle_contract(self.models_metadata):
            warnings_out.append("legacy model bundle contract")

        for key, message in self.model_load_errors.items():
            formatted = f"{key}: {message}"
            warnings_out.append(formatted)
            if key in REQUIRED_MODELS or key == "metadata":
                blockers.append(formatted)

        for warning_message in self.production_warnings:
            if warning_message not in warnings_out:
                warnings_out.append(warning_message)
        if any("Trying to unpickle estimator" in warning or "version" in warning.lower() for warning in warnings_out):
            warnings_out.append("scikit-learn artifact version mismatch")

        self.production_blockers = sorted(set(blockers))
        self.production_warnings = sorted(set(warnings_out))

    def get_cached_prediction(self, key: str) -> Optional[Dict[str, Any]]:
        if PREDICT_CACHE_TTL_SEC <= 0:
            return None
        val = self.predict_cache.get(key)
        if val is None:
            self.predict_cache_misses += 1
            return None

        self.predict_cache_hits += 1
        # LRUCache stores raw payloads
        if isinstance(val, dict):
            return val.copy()
        return val

    def store_cached_prediction(self, key: str, payload: Dict[str, Any]) -> None:
        if PREDICT_CACHE_TTL_SEC <= 0:
            return
        try:
            self.predict_cache.set(key, payload.copy())
        except Exception:
            # Best-effort: if cache fails, ignore
            pass

    def _load_dataset(self) -> None:
        """Load DATASET_PATH or the most recent game_features*.csv into memory."""
        try:
            explicit = SETTINGS.dataset_path
            self.dataset_manifest = load_latest_dataset_manifest(DATA_DIR)
            self.dataset_metadata = {}
            self.prior_baseline_medians = None
            try:
                path = resolve_latest_dataset(DATA_DIR, explicit_path=explicit)
            except FileNotFoundError:
                if explicit:
                    logging.warning(
                        "[Dataset] DATASET_PATH not found (%s). Falling back to latest game_features*.csv.",
                        explicit,
                    )
                path = resolve_latest_dataset(DATA_DIR, explicit_path=None)

            if not path.exists():
                logging.warning("[Dataset] No dataset found at: %s", path)
                self.dataset = None
                self.dataset_path = None
                self.dataset_hash = None
                self.dataset_mtime = None
                return

            logging.info("[Dataset] Loading dataset from: %s", path)
            df = pd.read_csv(path, low_memory=False)
            resolved_path = path.resolve()

            if self.dataset_manifest:
                manifest_paths: List[Path] = []
                for key in ("clean_dataset_path", "raw_dataset_path"):
                    raw_manifest_path = self.dataset_manifest.get(key)
                    if not raw_manifest_path:
                        continue
                    try:
                        manifest_paths.append(Path(str(raw_manifest_path)).expanduser().resolve())
                    except Exception:
                        continue
                if manifest_paths and resolved_path not in manifest_paths:
                    self.dataset_manifest = {}

            # Normalize key columns for consistent lookups
            df = _coerce_season_week(df)
            df = _normalize_team_columns(df, cols=["home_team", "away_team", "home_abbr", "away_abbr"])

            self.dataset = df
            self.dataset_path = resolved_path
            try:
                self.dataset_hash = file_sha256(resolved_path)
            except Exception:
                self.dataset_hash = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
            self.dataset_mtime = float(resolved_path.stat().st_mtime)

            try:
                self.numeric_medians = df.select_dtypes(include=[np.number]).median(numeric_only=True)
            except Exception:
                self.numeric_medians = None

            metadata_candidates: List[Path] = []
            manifest_metadata_path = self.dataset_manifest.get("metadata_path")
            if manifest_metadata_path:
                metadata_candidates.append(Path(str(manifest_metadata_path)).expanduser())
            metadata_candidates.append(resolved_path.parent / "game_features_metadata.json")
            metadata_candidates.append(DATA_DIR / "datasets" / "game_features_metadata.json")

            for metadata_path in metadata_candidates:
                try:
                    if metadata_path.exists():
                        self.dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        break
                except Exception as exc:
                    logging.warning("[Dataset] Failed to load dataset metadata from %s: %s", metadata_path, exc)

            priors_meta = self.dataset_metadata.get("priors_imputation", {}) if isinstance(self.dataset_metadata, dict) else {}
            baseline_medians = priors_meta.get("baseline_medians", {}) if isinstance(priors_meta, dict) else {}
            if baseline_medians:
                try:
                    prior_series = pd.to_numeric(pd.Series(baseline_medians), errors="coerce").dropna()
                    if not prior_series.empty:
                        self.prior_baseline_medians = prior_series
                        if self.numeric_medians is None or self.numeric_medians.empty:
                            self.numeric_medians = prior_series
                        else:
                            combined = self.numeric_medians.copy()
                            for column, value in prior_series.items():
                                combined.loc[str(column)] = float(value)
                            self.numeric_medians = combined
                except Exception as exc:
                    logging.warning("[Dataset] Failed to apply prior baseline medians: %s", exc)

            logging.info(
                "[Dataset] Loaded %d rows from %s (sha256=%s)",
                len(df),
                resolved_path.name,
                self.dataset_hash,
            )
        except Exception as e:  # pragma: no cover - defensive
            logging.exception("[Dataset] Error while loading dataset: %s", e)
            self.dataset = None
            self.dataset_path = None
            self.dataset_hash = None
            self.dataset_mtime = None
            self.dataset_manifest = {}
            self.dataset_metadata = {}
            self.prior_baseline_medians = None

    def _load_models(self) -> None:
        """Load each required model independently."""
        self.models = {}
        self.models_metadata = {}
        self.preprocessor = None
        self.score_preprocessor = None
        self.win_preprocessor = None
        self.production_blockers = []
        self.production_warnings = []
        self.model_load_errors = {}
        logging.info("[Model] Using models directory: %s", MODELS_DIR)
        # Default model filenames (non-pipeline artifacts)
        model_files: Dict[str, str] = {
            "home": "home_model.joblib",
            "away": "away_model.joblib",
            "win": "win_clf_calibrated.joblib",
        }

        for name, filename in model_files.items():
            # Prefer a full pipeline artifact (e.g. home_pipe.joblib) if present.
            pipe_filename = f"{name}_pipe.joblib" if name != "win" else "win_pipe.joblib"
            pipe_path = MODELS_DIR / pipe_filename
            model_path = MODELS_DIR / filename

            if pipe_path.exists():
                path = pipe_path
                logging.info("[Model] Using pipeline artifact for '%s': %s", name, path)
            elif model_path.exists():
                path = model_path
                logging.info("[Model] Using estimator artifact for '%s': %s", name, path)
            else:
                path = pipe_path
                logging.info("[Model] Estimator missing for '%s'; falling back to pipeline path: %s", name, path)

            if not path.exists():
                self.model_load_errors[name] = f"missing artifact {path.name}"
                logging.warning("[Model] Missing model file for '%s': %s", name, path)
                continue

            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded = joblib.load(path)
                for warning_message in _collect_model_load_warnings(caught, path):
                    if warning_message not in self.production_warnings:
                        self.production_warnings.append(warning_message)
                patched = _patch_imputer_compat(loaded)
                if patched:
                    logging.info("[Model] Applied %d sklearn-imputer compatibility patches for '%s'", patched, name)
                self.models[name] = loaded
                logging.info("[Model] Loaded '%s' successfully from %s", name, path)
                # If this happens to be a sklearn Pipeline, log its steps for visibility
                try:
                    from sklearn.pipeline import Pipeline

                    if isinstance(loaded, Pipeline):
                        steps = list(getattr(loaded, "named_steps", {}).keys())
                        logging.info("[Model] Pipeline '%s' steps: %s", name, steps)
                except Exception:
                    # sklearn may not be available in some analysis contexts; ignore
                    pass
            except Exception as e:  # pragma: no cover - defensive
                self.model_load_errors[name] = str(e).splitlines()[0]
                logging.exception("[Model] Error loading '%s' from %s: %s", name, path, e)

        logging.info("[Model] Loaded model keys: %s", list(self.models.keys()))
        metadata_path = MODELS_DIR / "metadata.json"
        if metadata_path.exists():
            try:
                loaded_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.models_metadata = loaded_meta if isinstance(loaded_meta, dict) else {}
                validated_meta = _validate_bundle_metadata_contract(self.models_metadata)
                self.models_metadata = validated_meta
                if not _requires_strict_bundle_contract(validated_meta):
                    legacy_message = (
                        f"legacy bundle metadata contract: {metadata_path.name} lacks strict serving metadata"
                    )
                    self.production_warnings.append(legacy_message)
                    logging.warning(
                        "[Model] Loaded legacy bundle metadata from %s without strict contract fields. Retrain to enable strict startup validation.",
                        metadata_path,
                    )
            except Exception as e:
                self.model_load_errors["metadata"] = str(e).splitlines()[0]
                self.production_warnings.append(
                    f"metadata validation failed: {self.model_load_errors['metadata']}"
                )
                logging.exception("[Model] Failed to validate metadata.json at %s: %s", metadata_path, e)
        else:
            self.models_metadata = {}
            self.production_warnings.append("metadata.json missing from active models directory")

        def _load_preprocessor_artifact(primary_name: str, fallback_names: Tuple[str, ...]) -> Optional[Any]:
            for filename in (primary_name, *fallback_names):
                prep_path = MODELS_DIR / filename
                if not prep_path.exists():
                    continue
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        loaded = joblib.load(prep_path)
                    for warning_message in _collect_model_load_warnings(caught, prep_path):
                        if warning_message not in self.production_warnings:
                            self.production_warnings.append(warning_message)
                    patched = _patch_imputer_compat(loaded)
                    if patched:
                        logging.info(
                            "[Model] Applied %d sklearn-imputer compatibility patches for %s",
                            patched,
                            filename,
                        )
                    logging.info("[Model] Loaded preprocessor artifact from %s", prep_path)
                    return loaded
                except Exception as e:  # pragma: no cover - defensive
                    logging.exception("[Model] Failed to load preprocessor %s: %s", prep_path, e)
            return None

        self.score_preprocessor = _load_preprocessor_artifact(
            "score_preprocessor.joblib",
            ("preprocessor.joblib",),
        )
        self.win_preprocessor = _load_preprocessor_artifact(
            "win_preprocessor.joblib",
            ("preprocessor.joblib",),
        )
        self.preprocessor = self.score_preprocessor

        if self.score_preprocessor is None and self.win_preprocessor is None:
            logging.info("[Model] No standalone preprocessors found in %s", MODELS_DIR)

        # If we loaded a standalone preprocessor and some models are Pipelines
        # that contain an unfitted preprocessor step (ColumnTransformer), try
        # to patch the pipeline to use the standalone preprocessor. This can
        # resolve situations where the serialized pipeline referenced an
        # unfitted transformer in CI/CD packaging.
        if self.preprocessor is not None:
            try:
                # Import sklearn classes lazily so packaging without sklearn
                # won't fail at module import time.
                from sklearn.pipeline import Pipeline
                from sklearn.compose import ColumnTransformer
            except Exception:
                Pipeline = None  # type: ignore
                ColumnTransformer = None  # type: ignore

            if Pipeline is not None:
                for mname, model in list(self.models.items()):
                    try:
                        if isinstance(model, Pipeline):
                            steps = list(model.steps)
                            replaced = False
                            replacement_preprocessor = (
                                self.win_preprocessor if mname == "win" else self.score_preprocessor
                            ) or self.preprocessor
                            for i, (step_name, step_est) in enumerate(steps):
                                lname = (step_name or "").lower()
                                if lname in ("pre", "prep", "preprocessor", "preprocess"):
                                    # If the pipeline's preprocessor is a ColumnTransformer
                                    # but not fitted (no 'transformers_' attribute), replace
                                    # it with the standalone preprocessor we loaded.
                                    if (
                                        ColumnTransformer is not None
                                        and isinstance(step_est, ColumnTransformer)
                                        and not hasattr(step_est, "transformers_")
                                        and replacement_preprocessor is not None
                                    ):
                                        steps[i] = (step_name, replacement_preprocessor)
                                        replaced = True
                                        logging.info(
                                            "[Model] Replacing unfitted preprocessor step '%s' in pipeline '%s' with standalone preprocessor",
                                            step_name,
                                            mname,
                                        )
                                        break
                            if replaced:
                                try:
                                    from sklearn.pipeline import Pipeline as SKPipeline

                                    new_pipe = SKPipeline(steps)
                                    self.models[mname] = new_pipe
                                    logging.info("[Model] Patched pipeline model '%s' successfully", mname)
                                except Exception as e:
                                    logging.exception("[Model] Failed to rebuild pipeline for %s: %s", mname, e)
                    except Exception:
                        # Be defensive: do not fail model loading due to pipeline patching
                        logging.exception("[Model] Error while attempting to inspect/patch model %s", mname)

        try:
            self.feature_manifest = _feature_manifest()
        except Exception:
            self.feature_manifest = []

        self._refresh_runtime_readiness()


state = AppState()

# -------------------------------------------------------------------
# FastAPI App + Lifespan
# -------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan hook.

    - On startup: load dataset + models into global state.
    - On shutdown: log a simple message (no persistent cleanup necessary).
    """
    logging.info("[App] Starting up; loading dataset and models...")
    state.load()

    # Start a background watcher to reload models when files in MODELS_DIR change.
    def _model_watcher(poll_seconds: int = 30) -> None:
        try:
            last_stamp = None
            while True:
                try:
                    if MODELS_DIR.exists():
                        m = MODELS_DIR.stat().st_mtime
                    else:
                        m = None
                    if last_stamp is None:
                        last_stamp = m
                    elif m is not None and m != last_stamp:
                        logging.info("[ModelWatcher] Detected change in MODELS_DIR; reloading models...")
                        try:
                            state._load_models()
                        except Exception:
                            logging.exception("[ModelWatcher] Failed reloading models after change.")
                        last_stamp = m
                except Exception:
                    logging.exception("[ModelWatcher] Error while polling models dir.")
                try:
                    time.sleep(poll_seconds)
                except Exception:
                    break

        except Exception:
            logging.exception("[ModelWatcher] Stopping due to fatal error.")

    watcher = threading.Thread(target=_model_watcher, daemon=True, name="model-watcher")
    watcher.start()

    yield
    logging.info("[App] Shutdown complete.")


app = FastAPI(lifespan=lifespan)


# CORS configuration
# ------------------
# The browser will send an `Origin` header that looks like:
#   https://new-nfl-predict.vercel.app
#
# On Heroku we control CORS via config vars (recommended):
#   - RESTRICT_CORS     : "true" | "false" (default: true)
#   - ALLOWED_ORIGINS   : comma-separated list of exact origins (scheme + host)
#                         Example:
#                           https://new-nfl-predict.vercel.app,http://localhost:5173
#                         (We also accept bare hostnames and normalize them to https://...)
#   - ALLOW_ORIGIN_REGEX: regex for dynamic preview origins (e.g., Vercel preview URLs)
#                         Example (recommended):
#                           (?i)^https://(?:[a-z0-9-]+\.)+vercel\.app$
#
def _env_flag(name: str, default: str = "true") -> bool:
    """Parse boolean-ish env vars safely."""
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")

ALLOWED_ORIGINS: List[str] = SETTINGS.allowed_origins
ALLOW_ORIGIN_REGEX = SETTINGS.effective_allow_origin_regex

# If RESTRICT_CORS is false, intentionally open CORS for debugging.
if not SETTINGS.restrict_cors:
    ALLOWED_ORIGINS = ["*"]
    ALLOW_ORIGIN_REGEX = None

logging.info("[App] CORS allowed origins: %s", ALLOWED_ORIGINS)
if ALLOW_ORIGIN_REGEX:
    logging.info("[App] CORS allow_origin_regex: %s", ALLOW_ORIGIN_REGEX)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,          # ✅ list (string here breaks CORS matching)
    allow_origin_regex=ALLOW_ORIGIN_REGEX,  # ✅ supports Vercel preview deployments
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, (dict, list)):
        message = "Request failed."
    else:
        message = str(detail) if detail is not None else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": int(exc.status_code),
                "message": message,
            },
            "detail": detail,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "status_code": 422,
                "message": "Request validation failed.",
            },
            "detail": exc.errors(),
            "path": str(request.url.path),
        },
    )

# -------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------


class HealthComponents(BaseModel):
    """Components section for the /health response."""
    dataset: bool
    models: bool
    loaded_models: List[str] = Field(default_factory=list)
    ready_for_production: bool = False
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """
    Canonical health response model.

    Matches README contract while being friendly to the current frontend:
      - status   : "healthy" | "unhealthy"
      - timestamp: auto-filled UTC timestamp
      - mode     : e.g. "production"
      - reason   : optional human-readable message
      - components: dataset/models readiness
    """
    status: Literal["healthy", "unhealthy"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str = "production"
    reason: Optional[str] = None
    production_ready: bool = False
    components: HealthComponents


class PredictRequest(BaseModel):
    """
    Request body for /predict.

    The frontend usually sends team abbreviations (e.g. "BUF", "HOU"), but
    the backend will uppercase whatever is provided and match against the
    normalized dataset.
    """
    home_team: str
    away_team: str
    season: int
    week: int


class PredictionResponse(BaseModel):
    """
    Canonical prediction response for /predict.

    Fields are aligned with the React client (Card/TeamGrid + PredictionContext):
      - home_score, away_score           : predicted scores
      - home_win_probability, away_win_probability : probabilities in [0, 1]
      - point_diff                       : home_score - away_score
      - game_id                          : stable key used across the app
      - generated_at                     : UTC timestamp
      - mode                             : e.g., "production"
      - win_classifier_used              : whether the calibrated classifier was used
    """
    season: int
    week: int
    home_team: str
    away_team: str
    home_name: Optional[str] = None
    away_name: Optional[str] = None
    game_id: str
    home_score: float
    away_score: float
    home_win_probability: float
    away_win_probability: float
    point_diff: float
    predicted_home_points: Optional[float] = None
    predicted_away_points: Optional[float] = None
    predicted_total: Optional[float] = None
    home_win_prob: Optional[float] = None
    prediction_source: str = "pipeline_primary"
    explanation_fields: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime
    mode: str = Field(..., description="Mode of prediction, e.g., 'production'")
    win_classifier_used: bool = Field(
        ..., description="Whether the win probability classifier was used"
    )


class HistoryEntryResponse(PredictionResponse):
    generated_at: Optional[datetime] = None
    mode: Optional[str] = None
    ts: Optional[str] = None
    user_id: Optional[str] = None
    storage_key: Optional[str] = None
    final_home_score: Optional[int] = None
    final_away_score: Optional[int] = None
    game_status: Optional[str] = None
    score_updated_at: Optional[str] = None


class DebugPredictInputResponse(BaseModel):
    selected_row_source: Literal["dataset_exact", "dataset_fuzzy", "synthetic"]
    constructed_row: Dict[str, Any] = Field(default_factory=dict)
    missing_before_impute: List[str] = Field(default_factory=list)
    missing_after_impute: List[str] = Field(default_factory=list)
    missing_prior_count: int = 0
    row_quality_score: float
    row_quality_rules: Dict[str, Any] = Field(default_factory=dict)
    model_feature_manifest: List[str] = Field(default_factory=list)
    expected_raw_columns: List[str] = Field(default_factory=list)
    dataset_hash: Optional[str] = None
    dataset_path: Optional[str] = None


class DatasetPreviewResponse(BaseModel):
    dataset_path: Optional[str] = None
    dataset_hash: Optional[str] = None
    total_rows: int
    filtered_rows: int
    returned_rows: int
    offset: int
    limit: int
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)


class RetrainRequest(BaseModel):
    dataset_path: Optional[str] = None
    splits: int = Field(default=5, ge=2, le=12)
    embargo: int = Field(default=1, ge=0, le=8)
    skip_train: bool = False
    train_extra: List[str] = Field(default_factory=list)


class RetrainResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime


class RetrainJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    gate: Dict[str, Any] = Field(default_factory=dict)


class PromoteResponse(BaseModel):
    job_id: str
    promoted_to: str
    promoted_at: datetime
    status: str


class DatasetStatsResponse(BaseModel):
    rows: int
    path: str
    hash: Optional[str] = None


class HistoryMetricsResponse(BaseModel):
    total_predictions: int
    resolved_games: int = 0
    win_rate: Optional[float] = None
    avg_abs_spread_error: Optional[float] = None
    avg_confidence: Optional[float] = None
    latest_prediction_at: Optional[str] = None
    last_score_sync_at: Optional[str] = None


class HistorySummaryResponse(HistoryMetricsResponse):
    user_id: Optional[str] = None


class HistoryStatsResponse(BaseModel):
    metrics: HistoryMetricsResponse


class StatusOverviewResponse(BaseModel):
    health: HealthResponse
    dataset: DatasetStatsResponse
    history: HistoryStatsResponse


class PredictCacheStatusResponse(BaseModel):
    enabled: bool
    ttl_seconds: int
    max_items: int
    items: int
    hits: int
    misses: int
    hit_rate: Optional[float] = None


class RuntimeStatusResponse(BaseModel):
    generated_at: str
    started_at: str
    uptime_seconds: int
    dataset_path: Optional[str] = None
    dataset_hash: Optional[str] = None
    dataset_modified_at: Optional[str] = None
    dataset_age_seconds: Optional[int] = None
    last_prediction_at: Optional[str] = None
    history_size: int
    production_ready: bool = False
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    predict_cache: PredictCacheStatusResponse


class PerformanceDriftPointResponse(BaseModel):
    run_id: str
    trained_at: str
    brier: Optional[float] = None
    mae: Optional[float] = None
    home_mae: Optional[float] = None
    away_mae: Optional[float] = None
    source_csv: Optional[str] = None


class PerformanceDriftResponse(BaseModel):
    generated_at: str
    count: int
    points: List[PerformanceDriftPointResponse] = Field(default_factory=list)


class OffseasonStatusResponse(BaseModel):
    generated_at: str
    offseason_mode: bool
    current_season: Optional[int] = None
    current_week: Optional[int] = None
    next_known_schedule_date: Optional[str] = None
    days_until_next_game: Optional[int] = None
    data_freshness_seconds: Optional[int] = None
    dataset_hash: Optional[str] = None
    last_trained_at: Optional[str] = None


class StatusResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    environment: str
    version: str
    uptime_seconds: int
    dataset_hash: Optional[str] = None
    dataset_path: Optional[str] = None
    model_keys: List[str] = Field(default_factory=list)
    production_ready: bool = False


class ScheduleGameResponse(BaseModel):
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_name: Optional[str] = None
    away_name: Optional[str] = None
    home_abbr: Optional[str] = None
    away_abbr: Optional[str] = None
    home_logo: Optional[str] = None
    away_logo: Optional[str] = None
    home_color: Optional[str] = None
    away_color: Optional[str] = None
    home_color2: Optional[str] = None
    away_color2: Optional[str] = None
    home_wordmark: Optional[str] = None
    away_wordmark: Optional[str] = None
    kickoff: Optional[str] = None


class TeamLogoMetadataResponse(BaseModel):
    name: str
    logoUrl: Optional[str] = None
    primaryColor: Optional[str] = None
    secondaryColor: Optional[str] = None
    wordmark: Optional[str] = None


class TeamLogosResponse(BaseModel):
    teams: Dict[str, TeamLogoMetadataResponse] = Field(default_factory=dict)


def _json_safe_value(v: Any) -> Any:
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        return None if pd.isna(v) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    try:
        return None if pd.isna(v) else v
    except Exception:
        return v


def _json_safe_row(row_df: pd.DataFrame) -> Dict[str, Any]:
    """Serialize the first row of a DataFrame into JSON-safe primitives."""
    if row_df is None or row_df.empty:
        return {}
    row = row_df.iloc[0].to_dict()
    return {str(k): _json_safe_value(v) for k, v in row.items()}


def _json_safe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    safe_rows: List[Dict[str, Any]] = []
    for row in records:
        safe_row = {str(k): _json_safe_value(v) for k, v in row.items()}
        safe_rows.append(safe_row)
    return safe_rows


def _row_quality_details(
    *,
    selected_row_source: str,
    missing_after_count: int,
    missing_prior_count: int,
    total_cols: int,
) -> Dict[str, Any]:
    """
    Score row quality on a 0-100 scale with explicit penalties.
    """
    score = 100.0
    source_penalty = 0.0
    if selected_row_source == "synthetic":
        source_penalty = 30.0
    elif selected_row_source == "dataset_fuzzy":
        source_penalty = 10.0
    score -= source_penalty

    completeness_penalty = float(min(40.0, max(0, missing_after_count) * 2.0))
    score -= completeness_penalty

    prior_penalty = float(min(40.0, max(0, missing_prior_count) * 5.0))
    score -= prior_penalty

    missing_ratio = (float(missing_after_count) / float(max(1, total_cols))) * 100.0
    ratio_penalty = float(min(20.0, missing_ratio * 0.2))
    score -= ratio_penalty

    score = float(np.clip(score, 0.0, 100.0))
    return {
        "row_quality_score": score,
        "row_quality_rules": {
            "base_score": 100.0,
            "source_penalty": source_penalty,
            "completeness_penalty": completeness_penalty,
            "missing_prior_penalty": prior_penalty,
            "missing_ratio_penalty": ratio_penalty,
        },
    }


def _is_local_request(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    host = host.lower().strip()
    return host in {"127.0.0.1", "::1", "localhost"}


def _require_admin_access(request: Request) -> None:
    if not bool(SETTINGS.enable_admin):
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled.")

    expected_token = (SETTINGS.admin_token or "").strip()
    provided_token = (
        request.headers.get("x-admin-token")
        or request.headers.get("x_admin_token")
        or ""
    ).strip()

    auth_header = (request.headers.get("authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        provided_token = auth_header[7:].strip()

    if expected_token:
        if provided_token != expected_token:
            raise HTTPException(status_code=403, detail="Invalid admin token.")
        return

    if not _is_local_request(request):
        raise HTTPException(
            status_code=403,
            detail="Admin endpoints require localhost unless ADMIN_TOKEN is configured.",
        )


def _safe_mean_col(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _parse_ts_to_iso(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    try:
        dt = pd.to_datetime(str(raw), utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime().isoformat()
    except Exception:
        return None


def _prediction_user_context_from_request(request: Optional[Request]):
    """Resolve the lightweight user identity used for history isolation.

    The app does not use full auth tokens yet. Instead, the frontend sends a
    stable `X-User-Id` header, and both SQLite and JSON history stores key off
    that derived context.
    """
    user_id = None
    if request is not None:
        user_id = request.headers.get("X-User-Id")
    return build_prediction_user_context(user_id)


def _history_total_for_request(request: Optional[Request]) -> int:
    try:
        return int(get_prediction_history_count(_prediction_user_context_from_request(request)))
    except Exception:
        logging.exception("[History] Failed to count persistent history; falling back to memory.")
        return len(state.history)


def _history_summary_for_request(request: Optional[Request]) -> Dict[str, Any]:
    """Return normalized per-user history metrics.

    SQLite is the primary source of truth. If persistent summary lookup fails,
    the route still responds with a safe fallback shape so status endpoints stay
    available during degraded boots.
    """
    context = _prediction_user_context_from_request(request)
    try:
        summary = load_prediction_history_summary(context)
        if isinstance(summary, dict):
            return {
                "total_predictions": int(summary.get("total_predictions") or 0),
                "resolved_games": int(summary.get("resolved_games") or 0),
                "win_rate": summary.get("win_rate"),
                "avg_abs_spread_error": summary.get("avg_abs_spread_error"),
                "avg_confidence": summary.get("avg_confidence"),
                "latest_prediction_at": summary.get("latest_prediction_at"),
                "last_score_sync_at": summary.get("last_score_sync_at"),
            }
    except Exception:
        logging.exception("[History] Failed to summarize persistent history; falling back to memory.")

    return {
        "total_predictions": len(state.history),
        "resolved_games": 0,
        "win_rate": None,
        "avg_abs_spread_error": None,
        "avg_confidence": None,
        "latest_prediction_at": None,
        "last_score_sync_at": None,
    }


def _prediction_readiness_payload() -> Dict[str, Any]:
    state._refresh_runtime_readiness()
    blockers = list(state.production_blockers)
    if state.dataset is None and "dataset not loaded" not in blockers:
        blockers.append("dataset not loaded")
    if not blockers:
        missing = [m for m in REQUIRED_MODELS if m not in state.models]
        if missing:
            blockers.append(f"missing models: {', '.join(sorted(missing))}")

    return {
        "message": "Prediction service unavailable.",
        "blockers": sorted(set(blockers)),
        "loaded_models": sorted(state.models.keys()),
        "warnings": list(state.production_warnings),
    }


def _persist_prediction_for_request(
    request: Optional[Request],
    payload: PredictRequest,
    prediction_payload: Dict[str, Any],
) -> None:
    if request is None:
        return

    try:
        context = _prediction_user_context_from_request(request)
        append_prediction_record(
            context,
            StoredPredictionRequest(
                home_team=str(payload.home_team).strip(),
                away_team=str(payload.away_team).strip(),
                season=int(payload.season),
                week=int(payload.week),
            ),
            prediction_payload,
        )
    except Exception as exc:
        logging.warning("[Predict] Failed to persist prediction history: %s", exc)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_run_metrics(run_models_dir: Path) -> Dict[str, Any]:
    csv_path = run_models_dir / "cv_fold_metrics.csv"
    summary_path = run_models_dir / "training_summary.json"
    metadata_path = run_models_dir / "metadata.json"

    brier: Optional[float] = None
    mae: Optional[float] = None
    home_mae: Optional[float] = None
    away_mae: Optional[float] = None

    if csv_path.exists():
        try:
            fold_df = pd.read_csv(csv_path)
            home_mae = _safe_mean_col(fold_df, "home_mae_val")
            away_mae = _safe_mean_col(fold_df, "away_mae_val")
            brier = _safe_mean_col(fold_df, "win_brier_val")
            vals = [x for x in (home_mae, away_mae) if x is not None]
            if vals:
                mae = float(sum(vals) / len(vals))
        except Exception:
            pass

    summary = _load_json(summary_path) if summary_path.exists() else {}
    if brier is None:
        try:
            brier = float(summary.get("win", {}).get("Brier_mean_val"))
        except Exception:
            brier = None
    if mae is None:
        try:
            h = float(summary.get("home", {}).get("MAE_mean_val"))
            a = float(summary.get("away", {}).get("MAE_mean_val"))
            mae = float((h + a) / 2.0)
        except Exception:
            mae = None

    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    trained_at = (
        _parse_ts_to_iso(metadata.get("timestamp"))
        or _parse_ts_to_iso(metadata.get("training_timestamp_utc"))
        or _parse_ts_to_iso(summary.get("training_timestamp_utc"))
        or datetime.now(timezone.utc).isoformat()
    )
    run_id = run_models_dir.parent.name

    return {
        "run_id": run_id,
        "trained_at": trained_at,
        "brier": brier,
        "mae": mae,
        "home_mae": home_mae,
        "away_mae": away_mae,
        "cv_metrics_csv": str(csv_path) if csv_path.exists() else None,
    }


def _latest_baseline_metrics() -> Dict[str, Any]:
    if METRICS_HISTORY_PATH.exists():
        try:
            hist = pd.read_csv(METRICS_HISTORY_PATH)
            if not hist.empty:
                last = hist.iloc[-1].to_dict()
                return {
                    "source": str(METRICS_HISTORY_PATH),
                    "brier": float(last["brier"]) if pd.notna(last.get("brier")) else None,
                    "mae": float(last["mae"]) if pd.notna(last.get("mae")) else None,
                    "run_id": last.get("run_id"),
                    "trained_at": last.get("trained_at"),
                }
        except Exception:
            pass

    points = collect_performance_drift(BASE_DIR, limit=1)
    if points:
        p = points[-1]
        return {
            "source": "collect_performance_drift",
            "brier": p.get("brier"),
            "mae": p.get("mae"),
            "run_id": p.get("run_id"),
            "trained_at": p.get("trained_at"),
        }
    return {"source": "none", "brier": None, "mae": None, "run_id": None, "trained_at": None}


def _append_metrics_history(row: Dict[str, Any]) -> None:
    METRICS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "job_id",
        "run_id",
        "trained_at",
        "brier",
        "mae",
        "home_mae",
        "away_mae",
        "brier_delta",
        "mae_delta",
        "gate_status",
        "dataset_path",
        "dataset_hash",
        "staging_dir",
    ]
    file_exists = METRICS_HISTORY_PATH.exists()
    with METRICS_HISTORY_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def _scan_latest_run_models_dir(started_at: datetime) -> Optional[Path]:
    candidates = sorted(
        BASE_DIR.glob("20*/models"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    threshold = started_at.timestamp() - 5.0
    for p in candidates:
        if p.stat().st_mtime >= threshold:
            return p
    return candidates[0] if candidates else None


def _update_job(job_id: str, **kwargs: Any) -> None:
    with state.retrain_lock:
        job = state.retrain_jobs.get(job_id)
        if not job:
            return
        job.update(kwargs)
        job["updated_at"] = datetime.now(timezone.utc).isoformat()


def _append_job_log(job_id: str, message: str) -> None:
    with state.retrain_lock:
        job = state.retrain_jobs.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        logs.append(f"{datetime.now(timezone.utc).isoformat()} {message}")
        job["updated_at"] = datetime.now(timezone.utc).isoformat()


def _run_retrain_job(job_id: str, payload: RetrainRequest) -> None:
    started_at = datetime.now(timezone.utc)
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _update_job(job_id, status="RUNNING")
    _append_job_log(job_id, "Started retrain job.")

    weekly_script = BASE_DIR / "scripts" / "weekly_retrain.py"
    if not weekly_script.exists():
        _update_job(job_id, status="FAILED")
        _append_job_log(job_id, f"Missing script: {weekly_script}")
        return

    cmd = [
        sys.executable,
        str(weekly_script),
        "--data-dir",
        str(DATA_DIR),
        "--reports-dir",
        str(REPORTS_DIR),
        "--splits",
        str(payload.splits),
        "--embargo",
        str(payload.embargo),
    ]
    if payload.dataset_path:
        cmd.extend(["--dataset-path", payload.dataset_path])
    if payload.skip_train:
        cmd.append("--skip-train")
    for item in payload.train_extra:
        cmd.extend(["--train-extra", str(item)])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as e:
        _update_job(job_id, status="FAILED")
        _append_job_log(job_id, f"Failed to execute retrain command: {e}")
        return

    (job_dir / "train.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (job_dir / "train.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    (job_dir / "train.command.txt").write_text(" ".join(cmd), encoding="utf-8")

    _append_job_log(job_id, f"Command finished with return code {proc.returncode}.")
    if int(proc.returncode) != 0:
        _update_job(
            job_id,
            status="FAILED",
            artifacts={
                "job_dir": str(job_dir),
                "stdout_log": str(job_dir / "train.stdout.log"),
                "stderr_log": str(job_dir / "train.stderr.log"),
            },
        )
        return

    if payload.skip_train:
        _update_job(
            job_id,
            status="COMPLETED_NO_TRAIN",
            artifacts={
                "job_dir": str(job_dir),
                "automation_summary": str(REPORTS_DIR / "automation" / "weekly_retrain_latest.json"),
            },
        )
        return

    run_models_dir = _scan_latest_run_models_dir(started_at)
    if run_models_dir is None or not run_models_dir.exists():
        _update_job(job_id, status="FAILED")
        _append_job_log(job_id, "Could not locate model run directory after retrain.")
        return

    staging_dir = STAGING_MODELS_DIR / job_id
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    shutil.copytree(run_models_dir, staging_dir)
    _append_job_log(job_id, f"Copied run artifacts to staging: {staging_dir}")

    metrics = _extract_run_metrics(staging_dir)
    baseline = _latest_baseline_metrics()

    metric_brier_raw = metrics.get("brier")
    baseline_brier_raw = baseline.get("brier")
    metric_mae_raw = metrics.get("mae")
    baseline_mae_raw = baseline.get("mae")

    numeric_types = (int, float, np.integer, np.floating)
    metric_brier = float(metric_brier_raw) if isinstance(metric_brier_raw, numeric_types) else None
    baseline_brier = float(baseline_brier_raw) if isinstance(baseline_brier_raw, numeric_types) else None
    metric_mae = float(metric_mae_raw) if isinstance(metric_mae_raw, numeric_types) else None
    baseline_mae = float(baseline_mae_raw) if isinstance(baseline_mae_raw, numeric_types) else None

    brier_delta = (
        (metric_brier - baseline_brier)
        if metric_brier is not None and baseline_brier is not None
        else None
    )
    mae_delta = (
        (metric_mae - baseline_mae)
        if metric_mae is not None and baseline_mae is not None
        else None
    )

    gate_pass = True
    if brier_delta is not None and brier_delta > 0.01:
        gate_pass = False
    if mae_delta is not None and mae_delta > 0.5:
        gate_pass = False

    gate_status = "PASSED" if gate_pass else "FAILED_GATE"
    dataset_path = payload.dataset_path or (str(state.dataset_path) if state.dataset_path else None)
    dataset_hash = None
    if dataset_path:
        try:
            dataset_hash = file_sha256(Path(dataset_path))
        except Exception:
            dataset_hash = state.dataset_hash
    if dataset_hash is None:
        dataset_hash = state.dataset_hash

    staging_meta_path = staging_dir / "metadata.json"
    staging_meta = _load_json(staging_meta_path)
    staging_meta.update(
        {
            "bundle_version": staging_meta.get("bundle_version") or f"staging-{job_id}",
            "dataset_hash": dataset_hash,
            "trained_at": metrics.get("trained_at"),
            "training_script": "backend/train_models.py",
            "feature_manifest": state.feature_manifest,
            "staging_job_id": job_id,
            "selection_metrics": {
                "brier": metrics.get("brier"),
                "mae": metrics.get("mae"),
                "home_mae": metrics.get("home_mae"),
                "away_mae": metrics.get("away_mae"),
            },
        }
    )
    staging_meta_path.write_text(json.dumps(staging_meta, indent=2), encoding="utf-8")

    _append_metrics_history(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "run_id": metrics.get("run_id"),
            "trained_at": metrics.get("trained_at"),
            "brier": metrics.get("brier"),
            "mae": metrics.get("mae"),
            "home_mae": metrics.get("home_mae"),
            "away_mae": metrics.get("away_mae"),
            "brier_delta": brier_delta,
            "mae_delta": mae_delta,
            "gate_status": gate_status,
            "dataset_path": dataset_path,
            "dataset_hash": dataset_hash,
            "staging_dir": str(staging_dir),
        }
    )

    final_status = "READY_FOR_PROMOTION" if gate_pass else "FAILED_GATE"
    _update_job(
        job_id,
        status=final_status,
        metrics=metrics,
        gate={
            "status": gate_status,
            "baseline": baseline,
            "brier_delta": brier_delta,
            "mae_delta": mae_delta,
            "thresholds": {"max_brier_regression": 0.01, "max_mae_regression": 0.5},
        },
        artifacts={
            "job_dir": str(job_dir),
            "run_models_dir": str(run_models_dir),
            "staging_dir": str(staging_dir),
            "stdout_log": str(job_dir / "train.stdout.log"),
            "stderr_log": str(job_dir / "train.stderr.log"),
        },
    )
    _append_job_log(job_id, f"Job completed with gate status: {gate_status}")


def _promote_staged_bundle(job_id: str) -> Path:
    staging_dir = STAGING_MODELS_DIR / job_id
    if not staging_dir.exists():
        raise HTTPException(status_code=404, detail=f"Staging bundle not found for job_id={job_id}")

    CURRENT_MODELS_DIR.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = CURRENT_MODELS_DIR.parent / f"current_tmp_{job_id}"
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    shutil.copytree(staging_dir, tmp_target)

    backup_dir: Optional[Path] = None
    if CURRENT_MODELS_DIR.exists():
        backup_dir = CURRENT_MODELS_DIR.parent / (
            f"current_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        shutil.move(str(CURRENT_MODELS_DIR), str(backup_dir))

    shutil.move(str(tmp_target), str(CURRENT_MODELS_DIR))

    global MODELS_DIR
    MODELS_DIR = CURRENT_MODELS_DIR
    state._load_models()
    _append_job_log(job_id, f"Promoted staged bundle to {CURRENT_MODELS_DIR}.")
    if backup_dir:
        _append_job_log(job_id, f"Previous bundle backup: {backup_dir}")
    return CURRENT_MODELS_DIR


def _feature_manifest(model_key: str = "scores") -> List[str]:
    """Derive expected raw feature columns from metadata/preprocessor."""
    normalized_key = "score" if model_key == "scores" else model_key
    if normalized_key == "score" and getattr(state, "feature_manifest", None):
        return [str(x) for x in state.feature_manifest]

    meta = state.models_metadata if isinstance(state.models_metadata, dict) else {}
    manifest_keys = [normalized_key]
    if normalized_key == "score":
        manifest_keys.append("scores")

    manifests = meta.get("feature_manifests")
    if isinstance(manifests, dict):
        for key in manifest_keys:
            selected = manifests.get(key)
            if isinstance(selected, dict):
                num = selected.get("numeric") or []
                cat = selected.get("categorical") or []
                manifest = [str(x) for x in (list(num) + list(cat))]
                if normalized_key == "score":
                    state.feature_manifest = manifest
                return manifest

    # Preferred metadata shape in existing bundles.
    if normalized_key == "score" and isinstance(meta.get("feature_names"), list):
        manifest = [str(x) for x in meta["feature_names"]]
        if normalized_key == "score":
            state.feature_manifest = manifest
        return manifest
    if normalized_key == "win" and isinstance(meta.get("feature_names_win"), list):
        return [str(x) for x in meta["feature_names_win"]]
    raw_cols = meta.get("raw_feature_columns")
    if isinstance(raw_cols, dict):
        for key in manifest_keys:
            selected = raw_cols.get(key)
            if isinstance(selected, dict):
                num = selected.get("numeric") or []
                cat = selected.get("categorical") or []
                manifest = [str(x) for x in (list(num) + list(cat))]
                if normalized_key == "score":
                    state.feature_manifest = manifest
                return manifest

        num = raw_cols.get("numeric") or []
        cat = raw_cols.get("categorical") or []
        manifest = [str(x) for x in (list(num) + list(cat))]
        if normalized_key == "score":
            state.feature_manifest = manifest
        return manifest

    # Fallback to fitted preprocessor if present.
    preprocessor = state.score_preprocessor if normalized_key == "score" else state.win_preprocessor
    if preprocessor is None:
        preprocessor = state.preprocessor
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        manifest = [str(x) for x in list(getattr(preprocessor, "feature_names_in_", []))]
        if normalized_key == "score":
            state.feature_manifest = manifest
        return manifest

    if normalized_key == "score":
        state.feature_manifest = []
    return []


def _build_synthetic_prediction_row(
    df: pd.DataFrame,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """Build a schema-complete future-game row for strict pipeline bundles."""
    raw_feature_columns = _feature_manifest("win") or _feature_manifest("scores")
    try:
        built = build_model_input_row(
            dataset_df=df,
            preprocessor=state.win_preprocessor or state.preprocessor,
            season=season,
            week=week,
            home_team=home_team,
            away_team=away_team,
            raw_feature_columns=raw_feature_columns,
            impute_medians=getattr(state, "numeric_medians", None),
        )
        row = built[0]
        if isinstance(row, pd.DataFrame) and not row.empty:
            return row
    except Exception as exc:
        logging.warning("[Predict] Rich synthetic row builder failed; using schema fallback: %s", exc)

    columns = list(df.columns) if df is not None and not df.empty else list(raw_feature_columns)
    synthetic = {col: np.nan for col in columns}
    synthetic.update(
        {
            "season": int(season),
            "week": int(week),
            "home_team": home_team,
            "away_team": away_team,
            "time_key": (int(season) * 100) + int(week),
        }
    )
    return pd.DataFrame([synthetic], columns=columns)


# -------------------------------------------------------------------
# Health + Status
# -------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Lightweight health check used by the frontend and deployment checks.

    Returns component readiness + a timestamp and human-friendly reason.
    """
    has_dataset = state.dataset is not None
    models_ok = all(m in state.models for m in REQUIRED_MODELS)
    production_ready = has_dataset and models_ok and not state.production_blockers

    status: Literal["healthy", "unhealthy"]
    status = "healthy" if production_ready else "unhealthy"

    reasons: List[str] = []
    if not has_dataset:
        reasons.append("dataset not loaded")
    if not models_ok:
        missing = [m for m in REQUIRED_MODELS if m not in state.models]
        reasons.append(f"missing models: {', '.join(missing)}")
    reasons.extend(state.production_blockers)
    warning_sample = list(state.production_warnings[:5])
    if len(state.production_warnings) > 5:
        warning_sample.append(f"... {len(state.production_warnings) - 5} additional startup warnings")

    reason_str = ", ".join(dict.fromkeys(reasons)) if reasons else None

    return HealthResponse(
        status=status,
        reason=reason_str,
        production_ready=production_ready,
        components=HealthComponents(
            dataset=has_dataset,
            models=models_ok,
            loaded_models=list(state.models.keys()),
            ready_for_production=production_ready,
            blockers=list(state.production_blockers),
            warnings=warning_sample,
        ),
    )


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    now = datetime.now(timezone.utc)
    health_payload = health()
    uptime_seconds = max(0, int((now - state.started_at).total_seconds()))
    return StatusResponse(
        status=health_payload.status,
        environment=str(SETTINGS.app_env),
        version=os.getenv("APP_VERSION", "dev"),
        uptime_seconds=uptime_seconds,
        dataset_hash=state.dataset_hash,
        dataset_path=str(state.dataset_path) if state.dataset_path else None,
        model_keys=sorted(state.models.keys()),
        production_ready=health_payload.production_ready,
    )


@app.get("/debug")
def debug() -> Dict[str, Any]:
    """Quick diagnostics used by tests and deployment debugging."""
    return {
        "dataset_loaded": state.dataset is not None,
        "dataset_path": str(state.dataset_path) if state.dataset_path else None,
        "dataset_hash": state.dataset_hash,
        "dataset_rows": int(len(state.dataset)) if state.dataset is not None else 0,
        "models_dir": str(MODELS_DIR),
        "loaded_models": sorted(state.models.keys()),
        "production_ready": not state.production_blockers,
        "startup_blockers": list(state.production_blockers),
        "startup_warnings": list(state.production_warnings),
        "cors_origins": ALLOWED_ORIGINS,
        "allow_origin_regex": ALLOW_ORIGIN_REGEX,
        "restrict_cors": SETTINGS.restrict_cors,
        "environment": SETTINGS.app_env,
    }


@app.post("/api/debug/predict-input", response_model=DebugPredictInputResponse)
@app.post("/debug/predict-input", response_model=DebugPredictInputResponse)
def debug_predict_input(request: PredictRequest) -> DebugPredictInputResponse:
    """
    Show the constructed inference row and feature completeness diagnostics.
    """
    state.refresh_dataset_if_changed()
    if state.dataset is None:
        raise HTTPException(status_code=503, detail="Dataset is not loaded.")

    season = int(request.season)
    week = int(request.week)
    home_team = _normalize_team_code(request.home_team)
    away_team = _normalize_team_code(request.away_team)

    selected_row_source: Literal["dataset_exact", "dataset_fuzzy", "synthetic"] = "dataset_exact"
    try:
        row_df, selected_row_source = _get_game_row_with_source(
            state.dataset, season, week, home_team, away_team
        )
    except HTTPException:
        selected_row_source = "synthetic"
        row_df = _build_synthetic_prediction_row(
            state.dataset, season, week, home_team, away_team
        )

    row_df = _roll_forward_missing_player_stats(
        df=state.dataset,
        row_df=row_df,
        home_team=home_team,
        away_team=away_team,
        season=season,
        week=week,
    )

    full_df, numeric_df = _prepare_inputs(row_df)
    try:
        debug_win_prob, _ = _calculate_win_probability(
            state.models.get("win"),
            full_df,
            numeric_df,
            preprocessor=state.win_preprocessor or state.preprocessor,
            numeric_medians=getattr(state, "numeric_medians", None),
        )
    except Exception:
        debug_win_prob = 0.5
    score_full_df, _ = _augment_with_win_probability_feature(full_df, numeric_df, debug_win_prob)
    feature_cols = _feature_manifest("scores")

    if feature_cols:
        view = score_full_df.reindex(columns=feature_cols)
    else:
        view = score_full_df.copy()

    missing_before = [
        c for c in view.columns
        if c not in score_full_df.columns or pd.isna(view.iloc[0][c])
    ]

    # Median-impute numeric columns for diagnostics only.
    imputed = view.copy()
    medians = getattr(state, "numeric_medians", None)
    if medians is not None and not imputed.empty:
        row_label = imputed.index[0]
        for col in imputed.columns:
            if col in medians.index and pd.isna(imputed.iloc[0][col]):
                imputed.at[row_label, col] = medians[col]

    missing_after = [c for c in imputed.columns if pd.isna(imputed.iloc[0][c])]
    prior_cols = [c for c in imputed.columns if ("prior_" in c or "rolling_" in c)]
    missing_prior_count = sum(1 for c in prior_cols if c in missing_after)

    total_cols = max(1, len(imputed.columns))
    quality = _row_quality_details(
        selected_row_source=selected_row_source,
        missing_after_count=len(missing_after),
        missing_prior_count=int(missing_prior_count),
        total_cols=total_cols,
    )

    return DebugPredictInputResponse(
        selected_row_source=selected_row_source,
        constructed_row=_json_safe_row(imputed),
        missing_before_impute=missing_before,
        missing_after_impute=missing_after,
        missing_prior_count=int(missing_prior_count),
        row_quality_score=quality["row_quality_score"],
        row_quality_rules=quality["row_quality_rules"],
        model_feature_manifest=feature_cols,
        expected_raw_columns=feature_cols,
        dataset_hash=state.dataset_hash,
        dataset_path=str(state.dataset_path) if state.dataset_path else None,
    )


@app.get("/api/debug/dataset", response_model=DatasetPreviewResponse)
@app.get("/debug/dataset", response_model=DatasetPreviewResponse)
def debug_dataset_view(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    team: Optional[str] = Query(None),
    columns: Optional[str] = Query(None),
) -> DatasetPreviewResponse:
    """
    Preview rows from the active dataset with lightweight filtering.
    """
    state.refresh_dataset_if_changed()
    if state.dataset is None:
        raise HTTPException(status_code=503, detail="Dataset is not loaded.")

    base_df = state.dataset
    filtered = base_df

    if season is not None:
        for col in ("season", "season_num"):
            if col in filtered.columns:
                col_vals = pd.to_numeric(filtered[col], errors="coerce").astype("Int64")
                filtered = filtered[col_vals == int(season)]
                break

    if week is not None:
        for col in ("week", "week_num"):
            if col in filtered.columns:
                col_vals = pd.to_numeric(filtered[col], errors="coerce").astype("Int64")
                filtered = filtered[col_vals == int(week)]
                break

    if team:
        team_code = _normalize_team_code(team)
        team_cols = [c for c in ("home_team", "away_team", "home_abbr", "away_abbr") if c in filtered.columns]
        if team_cols:
            mask = pd.Series(False, index=filtered.index)
            for col in team_cols:
                mask = mask | (filtered[col].astype(str).str.upper() == team_code)
            filtered = filtered[mask]

    available_columns = list(filtered.columns)
    if columns:
        requested = [c.strip() for c in columns.split(",") if c.strip()]
        selected_columns = [c for c in requested if c in filtered.columns]
    else:
        selected_columns = []

    if not selected_columns:
        preferred = [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "home_abbr",
            "away_abbr",
            "time_key",
            "home_points",
            "away_points",
            "home_win",
        ]
        selected_columns = [c for c in preferred if c in filtered.columns]

    if not selected_columns:
        selected_columns = available_columns[: min(30, len(available_columns))]

    paged = filtered.iloc[offset: offset + limit].reindex(columns=selected_columns)
    rows = _json_safe_records(paged)

    return DatasetPreviewResponse(
        dataset_path=str(state.dataset_path) if state.dataset_path else None,
        dataset_hash=state.dataset_hash,
        total_rows=int(len(base_df)),
        filtered_rows=int(len(filtered)),
        returned_rows=int(len(rows)),
        offset=int(offset),
        limit=int(limit),
        columns=selected_columns,
        rows=rows,
    )


@app.get("/status/overview", response_model=StatusOverviewResponse)
def status_overview(request: Request) -> StatusOverviewResponse:
    """
    Summary endpoint used by StatsPage.jsx.

    Returns:
      - current health (via /health)
      - dataset info (row count)
      - basic history metrics (prediction count placeholder)
    """
    state.refresh_dataset_if_changed()
    state.sync_game_scores()
    if state.dataset is not None:
        dataset_stats = {
            "rows": len(state.dataset),
            "path": str(state.dataset_path) if state.dataset_path else "unknown",
            "hash": state.dataset_hash,
        }
    else:
        dataset_stats = {"rows": 0, "path": "none", "hash": None}

    history_summary = _history_summary_for_request(request)

    return StatusOverviewResponse(
        health=health(),  # reuse typed health response
        dataset=dataset_stats,
        history={
            "metrics": history_summary
        },
    )


def _to_number(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return num


def _to_iso(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _pick_latest_iso(current: Optional[str], candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return current
    if not current:
        return candidate
    try:
        return candidate if pd.to_datetime(candidate) > pd.to_datetime(current) else current
    except Exception:
        return current


def _build_history_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = entries if isinstance(entries, list) else []
    resolved_games = 0
    correct_predictions = 0
    spread_error_total = 0.0
    spread_error_count = 0
    confidence_total = 0.0
    confidence_count = 0
    latest_prediction_at = None
    last_score_sync_at = None

    for entry in rows:
        if not isinstance(entry, dict):
            continue
        latest_prediction_at = _pick_latest_iso(
            latest_prediction_at,
            _to_iso(entry.get("ts") or entry.get("timestamp") or entry.get("created_at") or entry.get("predicted_at")),
        )
        home_prob = _to_number(entry.get("home_win_probability"))
        away_prob = _to_number(entry.get("away_win_probability"))
        if home_prob is not None or away_prob is not None:
            confidence_total += max(home_prob or 0.0, away_prob or 0.0)
            confidence_count += 1

        actual_home = _to_number(entry.get("final_home_score", entry.get("actual_home_score")))
        actual_away = _to_number(entry.get("final_away_score", entry.get("actual_away_score")))
        if actual_home is None or actual_away is None:
            continue
        resolved_games += 1
        last_score_sync_at = _pick_latest_iso(
            last_score_sync_at,
            _to_iso(entry.get("score_updated_at") or entry.get("last_score_sync_at") or entry.get("updated_at")),
        )

        predicted_home = _to_number(entry.get("home_score"))
        predicted_away = _to_number(entry.get("away_score"))
        predicted_diff = predicted_home - predicted_away if predicted_home is not None and predicted_away is not None else _to_number(entry.get("point_diff"))
        actual_diff = actual_home - actual_away

        if predicted_diff is not None:
            spread_error_total += abs(predicted_diff - actual_diff)
            spread_error_count += 1

        predicted_home_wins: Optional[bool] = None
        if predicted_diff is not None:
            predicted_home_wins = predicted_diff >= 0
        elif home_prob is not None or away_prob is not None:
            predicted_home_wins = (home_prob or 0.0) >= (away_prob or 0.0)
        if predicted_home_wins is not None and (predicted_home_wins == (actual_diff >= 0)):
            correct_predictions += 1

    return {
        "total_predictions": len(rows),
        "resolved_games": resolved_games,
        "win_rate": (correct_predictions / resolved_games) if resolved_games else None,
        "avg_abs_spread_error": (spread_error_total / spread_error_count) if spread_error_count else None,
        "avg_confidence": (confidence_total / confidence_count) if confidence_count else None,
        "latest_prediction_at": latest_prediction_at,
        "last_score_sync_at": last_score_sync_at,
    }


@app.get("/history/summary/memory", response_model=HistoryMetricsResponse)
def history_summary_memory() -> HistoryMetricsResponse:
    """Aggregated in-memory prediction quality and recency metrics for diagnostics."""
    return _build_history_metrics(state.history)


@app.get("/status/models")
def status_models() -> Dict[str, Any]:
    """
    Return model bundle readiness + provenance details for observability.
    """
    metadata = state.models_metadata if isinstance(state.models_metadata, dict) else {}
    loaded_models = sorted(state.models.keys())
    missing_required = [m for m in REQUIRED_MODELS if m not in state.models]

    artifacts = {
        "home_pipe": (MODELS_DIR / "home_pipe.joblib").exists(),
        "away_pipe": (MODELS_DIR / "away_pipe.joblib").exists(),
        "win_pipe": (MODELS_DIR / "win_pipe.joblib").exists(),
        "home_model": (MODELS_DIR / "home_model.joblib").exists(),
        "away_model": (MODELS_DIR / "away_model.joblib").exists(),
        "win_model": (MODELS_DIR / "win_clf_calibrated.joblib").exists(),
        "preprocessor": (MODELS_DIR / "preprocessor.joblib").exists(),
        "score_preprocessor": (MODELS_DIR / "score_preprocessor.joblib").exists(),
        "win_preprocessor": (MODELS_DIR / "win_preprocessor.joblib").exists(),
        "metadata": (MODELS_DIR / "metadata.json").exists(),
    }

    dataset_hash = metadata.get("dataset_hash") or state.dataset_hash
    if (not dataset_hash) and state.dataset_path and state.dataset_path.exists():
        try:
            dataset_hash = hashlib.sha256(state.dataset_path.read_bytes()).hexdigest()
        except Exception:
            dataset_hash = None

    return {
        "ready": len(missing_required) == 0 and not state.production_blockers,
        "models_dir": str(MODELS_DIR),
        "current_models_dir": str(CURRENT_MODELS_DIR) if CURRENT_MODELS_DIR.exists() else None,
        "loaded_models": loaded_models,
        "missing_required": missing_required,
        "load_errors": dict(state.model_load_errors),
        "blockers": list(state.production_blockers),
        "warnings": list(state.production_warnings),
        "feature_manifest_size": len(state.feature_manifest),
        "artifacts": artifacts,
        "provenance": {
            "trained_at": metadata.get("timestamp") or metadata.get("training_timestamp_utc"),
            "dataset_hash": dataset_hash,
            "bundle_version": metadata.get("bundle_version"),
            "training_script": metadata.get("training_script"),
            "metadata_path": str(MODELS_DIR / "metadata.json"),
        },
        "dataset_path": str(state.dataset_path) if state.dataset_path else None,
    }


@app.get("/status/runtime", response_model=RuntimeStatusResponse)
def status_runtime() -> RuntimeStatusResponse:
    now = datetime.now(timezone.utc)
    uptime_seconds = max(0, int((now - state.started_at).total_seconds()))

    dataset_modified_at: Optional[str] = None
    dataset_age_seconds: Optional[int] = None
    if state.dataset_path and state.dataset_path.exists():
        try:
            mtime = datetime.fromtimestamp(state.dataset_path.stat().st_mtime, tz=timezone.utc)
            dataset_modified_at = mtime.isoformat()
            dataset_age_seconds = max(0, int((now - mtime).total_seconds()))
        except Exception:
            dataset_modified_at = None
            dataset_age_seconds = None

    cache_total = state.predict_cache_hits + state.predict_cache_misses
    cache_hit_rate = (state.predict_cache_hits / cache_total) if cache_total > 0 else None

    return RuntimeStatusResponse(
        generated_at=now.isoformat(),
        started_at=state.started_at.isoformat(),
        uptime_seconds=uptime_seconds,
        dataset_path=str(state.dataset_path) if state.dataset_path else None,
        dataset_hash=state.dataset_hash,
        dataset_modified_at=dataset_modified_at,
        dataset_age_seconds=dataset_age_seconds,
        last_prediction_at=state.last_prediction_at.isoformat() if state.last_prediction_at else None,
        history_size=len(state.history),
        production_ready=len(state.production_blockers) == 0,
        blockers=list(state.production_blockers),
        warnings=list(state.production_warnings),
        predict_cache={
            "enabled": PREDICT_CACHE_TTL_SEC > 0,
            "ttl_seconds": PREDICT_CACHE_TTL_SEC,
            "max_items": PREDICT_CACHE_MAX_ITEMS,
            "items": len(state.predict_cache),
            "hits": state.predict_cache_hits,
            "misses": state.predict_cache_misses,
            "hit_rate": cache_hit_rate,
        },
    )


@app.get("/status/dataset-versioning")
def status_dataset_versioning(
    limit: int = Query(12, ge=1, le=100)
) -> Dict[str, Any]:
    versions = collect_dataset_versions(DATA_DIR, limit=limit)
    latest = versions[-1] if versions else None
    previous = versions[-2] if len(versions) > 1 else None

    row_delta = None
    col_delta = None
    changed = None
    if latest and previous:
        row_delta = int(latest["rows"] - previous["rows"])
        col_delta = int(latest["columns"] - previous["columns"])
        changed = bool(latest["sha256"] != previous["sha256"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest": latest,
        "previous": previous,
        "row_delta_vs_previous": row_delta,
        "column_delta_vs_previous": col_delta,
        "content_changed_vs_previous": changed,
        "versions": versions,
    }


@app.get("/status/performance-drift", response_model=PerformanceDriftResponse)
def status_performance_drift(
    limit: int = Query(52, ge=1, le=520)
) -> PerformanceDriftResponse:
    points = collect_performance_drift(BASE_DIR, limit=limit)
    return PerformanceDriftResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        count=len(points),
        points=[PerformanceDriftPointResponse(**p) for p in points],
    )


@app.get("/api/offseason/status", response_model=OffseasonStatusResponse)
@app.get("/offseason/status", response_model=OffseasonStatusResponse)
def offseason_status() -> OffseasonStatusResponse:
    now = datetime.now(timezone.utc)
    state.refresh_dataset_if_changed()

    next_games = get_schedule()
    next_kickoff: Optional[datetime] = None
    next_season: Optional[int] = None
    next_week: Optional[int] = None
    if next_games:
        first = next_games[0]
        season_val = first.get("season")
        next_season = int(season_val) if season_val is not None else None
        week_val = first.get("week")
        next_week = int(week_val) if week_val is not None else None
        kickoff_raw = first.get("kickoff")
        if kickoff_raw:
            try:
                parsed = pd.to_datetime(kickoff_raw, utc=True, errors="coerce")
                if parsed is not None and not pd.isna(parsed):
                    next_kickoff = parsed.to_pydatetime()
            except Exception:
                next_kickoff = None

    days_until_next = None
    if next_kickoff is not None:
        days_until_next = int((next_kickoff - now).total_seconds() // 86400)

    stale_next_slate = days_until_next is not None and days_until_next < 0
    offseason_mode = bool(
        (not next_games)
        or (next_kickoff is None)
        or stale_next_slate
        or (days_until_next is not None and days_until_next > 45)
    )

    if stale_next_slate:
        next_kickoff = None
        days_until_next = None

    last_trained = (
        state.models_metadata.get("timestamp")
        or state.models_metadata.get("training_timestamp_utc")
    )

    dataset_age_seconds = None
    if state.dataset_path and state.dataset_path.exists():
        try:
            mtime = datetime.fromtimestamp(state.dataset_path.stat().st_mtime, tz=timezone.utc)
            dataset_age_seconds = int((now - mtime).total_seconds())
        except Exception:
            dataset_age_seconds = None

    return OffseasonStatusResponse(
        generated_at=now.isoformat(),
        offseason_mode=offseason_mode,
        current_season=next_season,
        current_week=next_week,
        next_known_schedule_date=next_kickoff.isoformat() if next_kickoff else None,
        days_until_next_game=days_until_next,
        data_freshness_seconds=dataset_age_seconds,
        dataset_hash=state.dataset_hash,
        last_trained_at=last_trained,
    )


# -------------------------------------------------------------------
# Admin Retrain + Promotion
# -------------------------------------------------------------------


@app.post("/admin/retrain", response_model=RetrainResponse)
def admin_retrain(payload: RetrainRequest, request: Request) -> RetrainResponse:
    _require_admin_access(request)

    now = datetime.now(timezone.utc)
    job_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    with state.retrain_lock:
        payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        state.retrain_jobs[job_id] = {
            "job_id": job_id,
            "status": "QUEUED",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "payload": payload_dict,
            "logs": [],
            "metrics": {},
            "artifacts": {},
            "gate": {},
        }

    thread = threading.Thread(
        target=_run_retrain_job,
        args=(job_id, payload),
        daemon=True,
        name=f"retrain-{job_id}",
    )
    thread.start()

    return RetrainResponse(job_id=job_id, status="QUEUED", created_at=now)


@app.get("/admin/retrain/{job_id}", response_model=RetrainJobStatus)
def admin_retrain_status(job_id: str, request: Request) -> RetrainJobStatus:
    _require_admin_access(request)

    job = state.retrain_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return RetrainJobStatus(
        job_id=job_id,
        status=str(job.get("status", "UNKNOWN")),
        created_at=pd.to_datetime(job.get("created_at") or "", utc=True).to_pydatetime(),
        updated_at=pd.to_datetime(job.get("updated_at") or "", utc=True).to_pydatetime(),
        logs=list(job.get("logs", [])),
        metrics=dict(job.get("metrics", {})),
        artifacts=dict(job.get("artifacts", {})),
        gate=dict(job.get("gate", {})),
    )


@app.post("/admin/promote/{job_id}", response_model=PromoteResponse)
def admin_promote(job_id: str, request: Request) -> PromoteResponse:
    _require_admin_access(request)

    job = state.retrain_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if str(job.get("status")) != "READY_FOR_PROMOTION":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not promotable (status={job.get('status')}).",
        )

    promoted_path = _promote_staged_bundle(job_id)
    promoted_at = datetime.now(timezone.utc)
    _update_job(job_id, status="PROMOTED", promoted_at=promoted_at.isoformat(), promoted_to=str(promoted_path))

    return PromoteResponse(
        job_id=job_id,
        promoted_to=str(promoted_path),
        promoted_at=promoted_at,
        status="PROMOTED",
    )


# -------------------------------------------------------------------
# Schedule
# -------------------------------------------------------------------


def _load_schedule_dataframe(requested_season: Optional[int] = None) -> pd.DataFrame:
    """Load schedule rows from live data first, then fall back to packaged CSVs.

    This keeps schedule routes usable in three modes:
    1. normal runtime with `nflreadpy`
    2. local or packaged deployments with bundled CSVs
    3. degraded startup where prediction models may be unavailable but the API
       should still serve schedule and status information
    """
    frames: List[pd.DataFrame] = []
    candidate_seasons: List[int] = []
    current_year = datetime.now(timezone.utc).year

    if requested_season is not None:
        candidate_seasons.append(int(requested_season))
    else:
        candidate_seasons.extend([current_year - 1, current_year, current_year + 1, 2025, 2026])

    seen: set[int] = set()
    for season in candidate_seasons:
        if season in seen:
            continue
        seen.add(season)
        frame = pd.DataFrame()
        for attempt in (
            lambda s=season: nfl.load_schedules(seasons=[s]),
            lambda s=season: nfl.load_schedules(seasons=s),
            lambda s=season: nfl.load_schedules(s),
        ):
            try:
                schedule_table = attempt()
                frame = _to_pandas_schedule_safe(schedule_table)
                if frame is not None and not frame.empty:
                    break
            except Exception:
                continue
        if frame is not None and not frame.empty:
            frames.append(frame)

    fallback_frames: List[pd.DataFrame] = []
    for fallback in _find_schedule_paths(requested_season=requested_season):
        try:
            fallback_df = pd.read_csv(fallback)
            if fallback_df is not None and not fallback_df.empty:
                fallback_frames.append(fallback_df)
                logging.info("[Schedule] Loaded fallback schedule CSV: %s", fallback)
        except Exception as exc:
            logging.warning("[Schedule] Failed fallback schedule CSV load from %s: %s", fallback, exc)

    if frames:
        df = pd.concat(frames, ignore_index=True)

        # If live data only produced stale seasons, append packaged schedules
        # so offseason mode can still show the upcoming season when bundled.
        if fallback_frames:
            live_seasons = set(
                pd.to_numeric(df.get("season", pd.Series(dtype=int)), errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            missing_fallbacks: List[pd.DataFrame] = []
            for fallback_df in fallback_frames:
                fallback_seasons = set(
                    pd.to_numeric(fallback_df.get("season", pd.Series(dtype=int)), errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
                if not fallback_seasons or not fallback_seasons.issubset(live_seasons):
                    missing_fallbacks.append(fallback_df)
                    live_seasons.update(fallback_seasons)
            if missing_fallbacks:
                df = pd.concat([df, *missing_fallbacks], ignore_index=True)
    elif fallback_frames:
        df = pd.concat(fallback_frames, ignore_index=True)
    else:
        df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _coerce_season_week(df)
    df = df.infer_objects()
    df = _normalize_team_columns(
        df, cols=["home_abbr", "away_abbr", "home_team", "away_team"]
    )
    dedupe_cols = [
        col
        for col in ("season", "week", "home_team", "away_team", "home_abbr", "away_abbr")
        if col in df.columns
    ]
    if {"season", "week"}.issubset(dedupe_cols) and len(dedupe_cols) >= 4:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")
    df = _add_kickoff_utc_datetime(df)
    return df


def _select_schedule_slice(
    df: pd.DataFrame,
    season: Optional[int] = None,
    week: Optional[int] = None,
    now_utc: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Optional[int], Optional[int]]:
    """Choose the requested slate or the backend's best "next slate".

    Rules:
    - If `season` and `week` are provided, return that exact slice.
    - If only `season` is provided, return the next upcoming week inside that season.
    - If no future games remain and a current/future season is bundled, return
      that upcoming season's earliest week instead of a stale archived slate.
    - If no current/future season exists, fall back to the latest available week.
    """
    if df is None or df.empty:
        return pd.DataFrame(), season, week

    s_col = "season_num" if "season_num" in df.columns else "season"
    w_col = "week_num" if "week_num" in df.columns else "week"
    working = df
    now_ts = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    if season is not None:
        working = working[pd.to_numeric(working[s_col], errors="coerce") == int(season)]
    if working.empty:
        return pd.DataFrame(), season, week

    if week is not None:
        week_df = working[pd.to_numeric(working[w_col], errors="coerce") == int(week)]
        return week_df, season, week

    if "dt" in working.columns:
        future = working[working["dt"].notna() & (working["dt"] > now_ts)].sort_values(
            by=["dt", s_col, w_col]
        )
    else:
        future = pd.DataFrame()

    if not future.empty:
        next_row = future.iloc[0]
        current_year = int(now_ts.year)
        target_season = int(next_row.get(s_col, next_row.get("season", current_year)))
        target_week = int(next_row.get(w_col, next_row.get("week", 1)))
    else:
        season_values = (
            pd.to_numeric(working[s_col], errors="coerce")
            .dropna()
            .astype(int)
            .sort_values()
        )
        current_or_future_seasons = [
            int(value) for value in season_values.unique().tolist() if int(value) >= int(now_ts.year)
        ]
        if season is None and current_or_future_seasons:
            target_season = min(current_or_future_seasons)
        elif season is not None and int(season) >= int(now_ts.year):
            target_season = int(season)
        else:
            target_season = int(season_values.max())
        season_rows = working[pd.to_numeric(working[s_col], errors="coerce") == target_season]
        week_values = pd.to_numeric(season_rows[w_col], errors="coerce").dropna().astype(int)
        positive_weeks = week_values[week_values > 0]
        if positive_weeks.empty:
            return pd.DataFrame(), target_season, None
        if target_season >= int(now_ts.year):
            target_week = int(positive_weeks.min())
        else:
            target_week = int(positive_weeks.max())

    week_df = working[
        (pd.to_numeric(working[s_col], errors="coerce") == target_season)
        & (pd.to_numeric(working[w_col], errors="coerce") == target_week)
    ]
    return week_df, target_season, target_week


def _serialize_schedule_rows(
    week_df: pd.DataFrame,
    target_season: Optional[int],
    target_week: Optional[int],
) -> List[Dict[str, Any]]:
    if week_df is None or week_df.empty or target_season is None or target_week is None:
        return []

    team_meta_map = _load_team_metadata_map()
    results: List[Dict[str, Any]] = []

    for _, row in week_df.iterrows():
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        home_abbr = row.get("home_abbr", home_team)
        away_abbr = row.get("away_abbr", away_team)
        home_code = str(home_abbr or home_team or "").upper()
        away_code = str(away_abbr or away_team or "").upper()
        if not home_code or not away_code:
            continue

        home_meta = team_meta_map.get(home_code) or team_meta_map.get(str(home_team or "").upper()) or {}
        away_meta = team_meta_map.get(away_code) or team_meta_map.get(str(away_team or "").upper()) or {}

        kickoff_val: Optional[str] = None
        if ("dt" in row) and pd.notna(row["dt"]):
            try:
                kickoff_val = row["dt"].isoformat()
            except Exception:
                kickoff_val = None

        results.append(
            {
                "game_id": f"{int(target_season)}_{int(target_week)}_{home_code}_{away_code}",
                "season": int(target_season),
                "week": int(target_week),
                "home_team": str(home_team or home_code).upper(),
                "away_team": str(away_team or away_code).upper(),
                "home_name": home_meta.get("name"),
                "away_name": away_meta.get("name"),
                "home_abbr": home_code,
                "away_abbr": away_code,
                "home_logo": row.get("home_logo") or row.get("home_logo_url") or home_meta.get("logoUrl"),
                "away_logo": row.get("away_logo") or row.get("away_logo_url") or away_meta.get("logoUrl"),
                "home_color": home_meta.get("primaryColor"),
                "away_color": away_meta.get("primaryColor"),
                "home_color2": home_meta.get("secondaryColor"),
                "away_color2": away_meta.get("secondaryColor"),
                "home_wordmark": home_meta.get("wordmark"),
                "away_wordmark": away_meta.get("wordmark"),
                "kickoff": kickoff_val,
            }
        )

    return results


def _schedule_response(season: Optional[int] = None, week: Optional[int] = None) -> List[Dict[str, Any]]:
    """Serialize one schedule slice into the frontend-facing response shape."""
    df = _load_schedule_dataframe(requested_season=season)
    if df is None or df.empty:
        logging.warning("[Schedule] No schedule data available; returning empty list.")
        return []

    state.sync_game_scores(schedule_df=df)
    week_df, target_season, target_week = _select_schedule_slice(df, season=season, week=week)
    results = _serialize_schedule_rows(week_df, target_season, target_week)
    logging.info(
        "[Schedule] Returning %d games for season=%s week=%s",
        len(results),
        target_season,
        target_week,
    )
    return results


@app.get("/schedule", response_model=List[ScheduleGameResponse])
def get_schedule_by_query(
    season: Optional[int] = Query(None, ge=1990, le=2100),
    week: Optional[int] = Query(None, ge=1, le=30),
) -> List[Dict[str, Any]]:
    if week is not None and season is None:
        raise HTTPException(status_code=400, detail="Provide season when querying a specific week.")
    return _schedule_response(season=season, week=week)


@app.get("/schedule/next-week", response_model=List[ScheduleGameResponse])
def get_schedule() -> List[Dict[str, Any]]:
    return _schedule_response()


@app.get("/api/predict/next-week")
@app.get("/predict/next-week")
def predict_next_week() -> Dict[str, Any]:
    """Backward-compatible wrapper around the next-week schedule route."""
    return {"games": get_schedule()}


@app.get("/api/teams/logos", response_model=TeamLogosResponse)
@app.get("/teams/logos", response_model=TeamLogosResponse)
def get_team_logos() -> TeamLogosResponse:
    """Return cached team branding metadata for frontend enrichment."""
    return TeamLogosResponse(
        teams={
            code: TeamLogoMetadataResponse(**meta)
            for code, meta in _load_team_metadata_map().items()
        }
    )


# -------------------------------------------------------------------
# Prediction History
# -------------------------------------------------------------------


@app.get("/history", response_model=List[HistoryEntryResponse])
def get_prediction_history(
    request: Request,
    limit: int = Query(100, ge=1, le=1000)
) -> List[Dict[str, Any]]:
    """
    Return the last N prediction results for the active user context.

    Used by StatsPage/PredictionContext as a history source.
    """
    if limit <= 0:
        return []
    state.refresh_dataset_if_changed()
    state.sync_game_scores()
    try:
        response = load_prediction_history(
            _prediction_user_context_from_request(request),
            limit=limit,
        )
        entries = response.entries if hasattr(response, "entries") else []
        return [
            entry.model_dump(mode="json") if hasattr(entry, "model_dump") else dict(entry)
            for entry in entries
        ]
    except Exception:
        logging.exception("[History] Persistent history lookup failed; falling back to memory.")
        return state.history[-limit:]


@app.get("/history/summary", response_model=HistorySummaryResponse)
def get_prediction_history_summary(request: Request) -> Dict[str, Any]:
    state.refresh_dataset_if_changed()
    state.sync_game_scores()
    context = _prediction_user_context_from_request(request)
    return {
        **_history_summary_for_request(request),
        "user_id": context.user_id,
    }


# -------------------------------------------------------------------
# /predict (final enhanced)
# -------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictRequest, request: Request) -> Dict[str, Any]:
    """
    Predict home/away score and win probability for a single game.

    Enhancements:
      - Cleaner structure (helpers instead of nested functions)
      - Robust row matching (team code OR abbr columns)
      - Stable imputation using dataset medians (computed once)
      - Feature alignment for non-pipeline estimators (feature_names_in_)
      - Output smoothing for probability + sanity clamping for scores
    """
    # ----- Readiness -----
    state.refresh_dataset_if_changed()
    readiness = _prediction_readiness_payload()
    if readiness["blockers"]:
        raise HTTPException(status_code=503, detail=readiness)

    # ----- Validate request -----
    season = int(payload.season)
    week = int(payload.week)

    if week < 1 or week > 30:
        raise HTTPException(status_code=400, detail="Invalid week. Expected 1..30.")
    if season < 1990 or season > 2100:
        raise HTTPException(status_code=400, detail="Invalid season. Expected a realistic year.")
    if not payload.home_team or not payload.away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team are required.")

    home_team = _normalize_team_code(payload.home_team)
    away_team = _normalize_team_code(payload.away_team)
    if home_team == away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")
    valid_codes = state.valid_team_codes()
    invalid_codes = sorted([code for code in (home_team, away_team) if code not in valid_codes])
    if invalid_codes:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown team code(s).",
                "invalid_team_codes": invalid_codes,
                "hint": "Use canonical NFL abbreviations (e.g., KC, BUF, DAL).",
            },
        )

    cache_key = state._prediction_cache_key(
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,
    )
    cached = state.get_cached_prediction(cache_key)
    if cached is not None:
        state.last_prediction_at = datetime.now(timezone.utc)
        _persist_prediction_for_request(request, payload, cached)
        state._remember_history(cached)
        logging.info(
            "[Predict] Cache hit for %s vs %s (season=%s week=%s)",
            home_team,
            away_team,
            season,
            week,
        )
        return cached

    df = state.dataset
    assert df is not None  # guarded above

    # ----- Find feature row -----
    selected_row_source: Literal["dataset_exact", "dataset_fuzzy", "synthetic"] = "dataset_exact"
    try:
        row, selected_row_source = _get_game_row_with_source(
            df, season, week, home_team, away_team
        )
    except HTTPException as e:
        if e.status_code != 404:
            raise
        logging.info(
            "[Predict] No exact dataset row for %s vs %s (season=%s week=%s); using synthetic fallback.",
            home_team,
            away_team,
            season,
            week,
        )
        row = _build_synthetic_prediction_row(
            df, season, week, home_team, away_team
        )
        selected_row_source = "synthetic"

    # ----- Fill missing player stats (safe) -----
    row = _roll_forward_missing_player_stats(
        df=df,
        row_df=row,
        home_team=home_team,
        away_team=away_team,
        season=season,
        week=week,
    )

    # ----- Prepare model inputs -----
    full_df, numeric_df = _prepare_inputs(row)
    home_model = state.models["home"]
    away_model = state.models["away"]
    win_model = state.models.get("win")

    # ----- Predict win probability first -----
    try:
        win_prob_raw, clf_used = _calculate_win_probability(
            win_model,
            full_df,
            numeric_df,
            preprocessor=state.win_preprocessor or state.preprocessor,
            numeric_medians=getattr(state, "numeric_medians", None),
        )
    except Exception as win_err:
        logging.warning("[Predict] Win probability calc failed; defaulting to 0.5: %s", str(win_err).splitlines()[0])
        win_prob_raw, clf_used = 0.5, False

    if _requires_strict_bundle_contract(state.models_metadata) and not clf_used:
        detail = _prediction_readiness_payload()
        blockers = list(detail.get("blockers") or [])
        blocker = "win classifier unavailable for strict model bundle"
        if blocker not in blockers:
            blockers.append(blocker)
        detail.update(
            {
                "message": "Prediction service unavailable.",
                "blockers": blockers,
                "models_dir": str(MODELS_DIR),
            }
        )
        logging.error(
            "[Predict] Strict model bundle at %s could not produce a classifier-backed win probability for %s vs %s.",
            MODELS_DIR,
            home_team,
            away_team,
        )
        raise HTTPException(status_code=503, detail=detail)

    score_full_df, score_numeric_df = _augment_with_win_probability_feature(
        full_df,
        numeric_df,
        win_prob_raw,
    )

    feature_cols = _feature_manifest("scores")
    view_for_quality = score_full_df.reindex(columns=feature_cols) if feature_cols else score_full_df.copy()
    quality_imputed = view_for_quality.copy()
    medians = getattr(state, "numeric_medians", None)
    if medians is not None and not quality_imputed.empty:
        row_label = quality_imputed.index[0]
        for col in quality_imputed.columns:
            if col in medians.index and pd.isna(quality_imputed.iloc[0][col]):
                quality_imputed.at[row_label, col] = medians[col]
    missing_after_quality = [
        c for c in quality_imputed.columns if pd.isna(quality_imputed.iloc[0][c])
    ]
    prior_cols = [c for c in quality_imputed.columns if ("prior_" in c or "rolling_" in c)]
    missing_prior_count = sum(1 for c in prior_cols if c in missing_after_quality)
    quality = _row_quality_details(
        selected_row_source=selected_row_source,
        missing_after_count=len(missing_after_quality),
        missing_prior_count=int(missing_prior_count),
        total_cols=max(1, len(quality_imputed.columns)),
    )

    # ----- Predict scores using the raw win probability as an input feature -----
    try:
        h_score = _predict_score(
            home_model,
            score_full_df,
            score_numeric_df,
            preprocessor=state.score_preprocessor or state.preprocessor,
            numeric_medians=getattr(state, "numeric_medians", None),
            model_name="home_model",
        )
        a_score = _predict_score(
            away_model,
            score_full_df,
            score_numeric_df,
            preprocessor=state.score_preprocessor or state.preprocessor,
            numeric_medians=getattr(state, "numeric_medians", None),
            model_name="away_model",
        )
    except Exception as model_err:
        msg = str(model_err) or "unknown error"
        # Keep messages concise for clients, verbose details stay in logs.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {msg.splitlines()[0]}") from model_err

    # Clamp to sane ranges
    h_score = _clamp_score(h_score)
    a_score = _clamp_score(a_score)

    point_diff = float(h_score - a_score)

    if clf_used:
        win_prob = float(np.clip(win_prob_raw, 1e-6, 1 - 1e-6))
    else:
        win_prob = float(np.clip(1.0 / (1.0 + np.exp(-0.28 * point_diff)), 0.02, 0.98))

    # ----- Build response -----
    game_id = f"{season}_{week}_{home_team}_{away_team}"
    predicted_total = float(h_score + a_score)
    generated_at = datetime.now(timezone.utc)
    team_meta_map = _load_team_metadata_map()
    home_meta = team_meta_map.get(home_team) or {}
    away_meta = team_meta_map.get(away_team) or {}
    prediction_source = str(
        state.models_metadata.get("serving_mode")
        or state.models_metadata.get("bundle_version")
        or "pipeline_primary"
    )

    result: Dict[str, Any] = {
        "season": season,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "home_name": home_meta.get("name"),
        "away_name": away_meta.get("name"),
        "game_id": game_id,
        "home_score": float(h_score),
        "away_score": float(a_score),
        "home_win_probability": float(win_prob),
        "away_win_probability": float(1.0 - win_prob),
        "point_diff": float(point_diff),
        "predicted_home_points": float(h_score),
        "predicted_away_points": float(a_score),
        "predicted_total": predicted_total,
        "home_win_prob": float(win_prob),
        "prediction_source": prediction_source,
        "explanation_fields": {
            "selected_row_source": selected_row_source,
            "row_quality_score": quality["row_quality_score"],
            "row_quality_rules": quality["row_quality_rules"],
            "dataset_hash": state.dataset_hash,
            "dataset_path": str(state.dataset_path) if state.dataset_path else None,
            "missing_prior_count": int(missing_prior_count),
            "missing_after_impute_count": len(missing_after_quality),
        },
        "generated_at": generated_at,
        "mode": "production",
        "win_classifier_used": bool(clf_used),
    }

    state.last_prediction_at = generated_at
    state.store_cached_prediction(cache_key, result)
    _persist_prediction_for_request(request, payload, result)

    # ----- History (bounded) -----
    state._remember_history(result)

    logging.info(
        "[Predict] %s vs %s (season=%s week=%s source=%s quality=%.1f hash=%s) -> home=%.1f away=%.1f total=%.1f win_p=%.3f (clf_used=%s)",
        home_team,
        away_team,
        season,
        week,
        selected_row_source,
        quality["row_quality_score"],
        state.dataset_hash,
        h_score,
        a_score,
        predicted_total,
        win_prob,
        clf_used,
    )

    return result


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
