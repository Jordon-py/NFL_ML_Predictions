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
"""

import json
import logging
import os
import hashlib
import time
import sys
import threading
import shutil
import subprocess
import csv
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple, Literal
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
from backend.utils import functions_for_main as fn_main
from backend.utils.ops_reporting import (
    collect_dataset_versions,
    collect_performance_drift,
    resolve_latest_dataset,
    file_sha256,
)

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


def _find_schedule_path() -> Optional[Path]:
    """
    Locate a schedule CSV file.

    Priority:
      1. Explicit SCHEDULE_PATH (env override or default backend/data/Nfl_schedule_2025.csv)
      2. Any CSV in backend/data/ that looks like a schedule
      3. Frontend public copy at ../frontend/public/nflSchedule.csv (local dev)
    """

    # 1) explicit path
    if SCHEDULE_PATH.exists():
        return SCHEDULE_PATH

    # 2) search backend/data for schedule-like CSVs
    candidates: List[Path] = []
    for p in DATA_DIR.glob("*.csv"):
        name = p.name.lower()
        if "schedule" in name or name.startswith("nfl"):
            candidates.append(p)
    if candidates:
        # Prefer most recently modified
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    # 3) local dev fallback: frontend/public
    frontend_sched = BASE_DIR.parent / "frontend" / "public" / "nflSchedule.csv"
    if frontend_sched.exists():
        return frontend_sched

    return None


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


def _find_models_dir() -> Path:
    """Locate the best models directory.

    Priority order:
      1) Env override (recommended for Heroku): MODELS_DIR / MODELS_PATH / MODEL_DIR
      2) A complete bundle in common repo locations (prod-models/models, dated runs, etc.)
      3) Fallback: backend/models (even if incomplete, so errors are visible in logs)

    Tip:
      - On Heroku, always set MODELS_DIR to a path that exists *in the slug*.
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

    # Common packaged pattern: backend/data/prod-models/models
    direct = BASE_DIR / "data" / "prod-models" / "models"
    if _models_dir_has_required_artifacts(direct):
        candidates.append(direct)

    # Common local pattern: backend/models
    local_default = BASE_DIR / "models"
    if _models_dir_has_required_artifacts(local_default):
        candidates.append(local_default)

    # Date-stamped training runs: backend/20251215/models (most recent wins)
    for p in BASE_DIR.glob("20*/models"):
        if _models_dir_has_required_artifacts(p):
            candidates.append(p)

    # Any nested prod-models/models in the repo
    for p in BASE_DIR.glob("**/prod-models/models"):
        if _models_dir_has_required_artifacts(p):
            candidates.append(p)

    if candidates:
        # Prefer the most recently modified bundle
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return local_default


# Resolve models directory once at import time so serving code can rely on MODELS_DIR.
MODELS_DIR: Path = _find_models_dir()


def _load_team_logo_map() -> Dict[str, str]:
    """Load a {TEAM_ABBR: logo_url} mapping if available.

    Why:
      - Your schedule CSV typically contains team abbreviations (PHI, DAL, ...),
        but does NOT include logo URLs.
      - The frontend can only render logos if we attach a URL per game.

    Expected shapes:
      - CSV (recommended): `team_logos.csv` with columns `team_abbr` + `team_logo_espn`
      - JSON: {"KC": "https://...", ...}
    """
    def _resolve_path(p: Path) -> Optional[Path]:
        if p.exists():
            return p
        if not p.is_absolute():
            # Allow relative paths from either repo root or backend/ dir
            for base in (BASE_DIR.parent, BASE_DIR):
                candidate = (base / p).resolve()
                if candidate.exists():
                    return candidate
        return None

    candidates: List[Path] = []
    env = os.environ.get("TEAM_LOGOS_PATH") or os.environ.get("TEAM_LOGO_PATH")
    if env:
        candidates.append(Path(env).expanduser())

    # Prefer explicit data dir, then local backend/frontend fallbacks used in this repo.
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
        p = _resolve_path(raw_path)
        if not p:
            continue

        try:
            # JSON map support: {"KC": "https://..."}
            if p.suffix.lower() == ".json":
                with open(p, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if not isinstance(obj, dict):
                    continue
                out: Dict[str, str] = {}
                for k, v in obj.items():
                    key = str(k).strip().upper()
                    val = str(v).strip()
                    if not key or not val or val.lower() == "nan":
                        continue
                    out[key] = val
                    # Also store canonical alias (e.g., LA->LAR) for lookups.
                    if TEAM_ABBR_MAP and key in TEAM_ABBR_MAP:
                        out.setdefault(TEAM_ABBR_MAP[key], val)
                if out:
                    logging.info("[Logos] Loaded %d team logos from %s", len(out), p)
                    return out
                continue

            # CSV support (flexible schema)
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}

            key_col = None
            for k in ("abbr", "team", "team_abbr", "team_code"):
                if k in cols:
                    key_col = cols[k]
                    break

            # Prefer ESPN logos, per request.
            val_col = None
            for v in (
                "team_logo_espn",
                # fallbacks
                "team_logo_squared",
                "team_logo_wikipedia",
                "team_wordmark",
                "logo_url",
                "logo",
                "url",
                "image_url",
                "image",
            ):
                if v in cols:
                    val_col = cols[v]
                    break

            if not key_col or not val_col:
                continue

            out: Dict[str, str] = {}
            for _, r in df.iterrows():
                key = str(r.get(key_col, "")).strip().upper()
                val = str(r.get(val_col, "")).strip()
                if not key or not val or val.lower() == "nan":
                    continue
                out[key] = val
                if TEAM_ABBR_MAP and key in TEAM_ABBR_MAP:
                    out.setdefault(TEAM_ABBR_MAP[key], val)

            if out:
                logging.info("[Logos] Loaded %d team logos from %s (col=%s)", len(out), p, val_col)
                return out
        except Exception as e:
            logging.warning("[Logos] Failed reading %s: %s", raw_path, e)

    logging.info("[Logos] No team logo map found; schedule will return null logos.")
    return {}


def _calculate_win_probability(
    win_model: Any,
    full_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    preprocessor: Optional[Any] = None,
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
    def _fallback_probability() -> float:
        try:
            if full_df is not None and not full_df.empty and "home_moneyline_prob" in full_df.columns:
                raw = pd.to_numeric(full_df.iloc[0].get("home_moneyline_prob"), errors="coerce")
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
                win_prob = float(win_model.predict_proba(full_df)[0][1])
                return float(np.clip(win_prob, 1e-6, 1 - 1e-6)), True
            except Exception as e:
                logging.warning("[Predict] win_pipe predict_proba failed; falling back to priors: %s", e)

        # B) Classifier-only case: transform then predict_proba
        if (not is_pipeline) and (preprocessor is not None):
            try:
                X_proc = preprocessor.transform(full_df)
                win_prob = float(win_model.predict_proba(X_proc)[0][1])
                return float(np.clip(win_prob, 1e-6, 1 - 1e-6)), True
            except Exception as e:
                logging.warning("[Predict] win_clf predict_proba failed after preprocessor.transform; falling back: %s", e)

        # C) Last attempt: numeric-only (may work if the model was trained on raw numeric columns)
        if not is_pipeline:
            try:
                if numeric_df is not None and not numeric_df.empty:
                    win_prob = float(win_model.predict_proba(numeric_df)[0][1])
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
        self.feature_manifest: List[str] = []
        # Optional shared preprocessor artifact (may be saved separately from models)
        self.preprocessor: Optional[Any] = None
        self.score_preprocessor: Optional[Any] = None
        self.win_preprocessor: Optional[Any] = None
        self.history: List[Dict[str, Any]] = []
        self.predict_cache: Dict[str, Dict[str, Any]] = {}
        self.predict_cache_hits: int = 0
        self.predict_cache_misses: int = 0
        self.retrain_jobs: Dict[str, Dict[str, Any]] = {}
        self.retrain_lock = threading.Lock()

        # Cached numeric medians from the loaded dataset (used for stable imputation).
        # Set during _load_dataset().
        self.numeric_medians: Optional[pd.Series] = None


    # -------------------------
    # Startup Loader
    # -------------------------
    def load(self) -> None:
        """Load dataset + models at startup with defensive logging."""
        self.started_at = datetime.now(timezone.utc)
        self._load_dataset()
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
            return True
        if self.dataset_path.resolve() != target_path.resolve():
            self._load_dataset()
            return True
        if self.dataset_mtime is None or target_mtime > float(self.dataset_mtime):
            self._load_dataset()
            return True
        return False

    def _prediction_cache_key(self, *, season: int, week: int, home_team: str, away_team: str) -> str:
        return f"{season}:{week}:{home_team}:{away_team}"

    def get_cached_prediction(self, key: str) -> Optional[Dict[str, Any]]:
        if PREDICT_CACHE_TTL_SEC <= 0:
            return None

        entry = self.predict_cache.get(key)
        if not entry:
            self.predict_cache_misses += 1
            return None

        ts = float(entry.get("stored_ts", 0.0))
        age = time.time() - ts
        if age > PREDICT_CACHE_TTL_SEC:
            self.predict_cache.pop(key, None)
            self.predict_cache_misses += 1
            return None

        self.predict_cache_hits += 1
        payload = entry.get("payload")
        if isinstance(payload, dict):
            return payload.copy()
        return None

    def store_cached_prediction(self, key: str, payload: Dict[str, Any]) -> None:
        if PREDICT_CACHE_TTL_SEC <= 0:
            return

        if len(self.predict_cache) >= PREDICT_CACHE_MAX_ITEMS:
            # Remove oldest entry first.
            oldest_key = min(
                self.predict_cache.keys(),
                key=lambda k: float(self.predict_cache[k].get("stored_ts", 0.0)),
            )
            self.predict_cache.pop(oldest_key, None)

        self.predict_cache[key] = {
            "stored_ts": time.time(),
            "payload": payload.copy(),
        }

    def _load_dataset(self) -> None:
        """Load DATASET_PATH or the most recent game_features*.csv into memory."""
        try:
            explicit = SETTINGS.dataset_path
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
            df = pd.read_csv(path)

            # Normalize key columns for consistent lookups
            df = _coerce_season_week(df)
            df = _normalize_team_columns(df, cols=["home_team", "away_team", "home_abbr", "away_abbr"])

            self.dataset = df
            self.dataset_path = path
            try:
                self.dataset_hash = file_sha256(path)
            except Exception:
                self.dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.dataset_mtime = float(path.stat().st_mtime)


            # Cache numeric medians once so prediction-time imputations are stable


            # even when the matched row contains many missing values (future games).


            try:


                self.numeric_medians = df.select_dtypes(include=[np.number]).median(numeric_only=True)


            except Exception:


                self.numeric_medians = None

            logging.info(
                "[Dataset] Loaded %d rows from %s (sha256=%s)",
                len(df),
                path.name,
                self.dataset_hash,
            )
        except Exception as e:  # pragma: no cover - defensive
            logging.exception("[Dataset] Error while loading dataset: %s", e)
            self.dataset = None
            self.dataset_path = None
            self.dataset_hash = None
            self.dataset_mtime = None

    def _load_models(self) -> None:
        """Load each required model independently."""
        self.models = {}
        self.models_metadata = {}
        self.preprocessor = None
        self.score_preprocessor = None
        self.win_preprocessor = None
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
                logging.warning("[Model] Missing model file for '%s': %s", name, path)
                continue

            try:
                loaded = joblib.load(path)
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
                logging.exception("[Model] Error loading '%s' from %s: %s", name, path, e)

        logging.info("[Model] Loaded model keys: %s", list(self.models.keys()))
        metadata_path = MODELS_DIR / "metadata.json"
        if metadata_path.exists():
            try:
                self.models_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as e:
                logging.warning("[Model] Failed to parse metadata.json at %s: %s", metadata_path, e)
                self.models_metadata = {}
        else:
            self.models_metadata = {}

        def _load_preprocessor_artifact(primary_name: str, fallback_names: Tuple[str, ...]) -> Optional[Any]:
            for filename in (primary_name, *fallback_names):
                prep_path = MODELS_DIR / filename
                if not prep_path.exists():
                    continue
                try:
                    loaded = joblib.load(prep_path)
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
    yield
    logging.info("[App] Shutdown complete.")


app = FastAPI(lifespan=lifespan)


# CORS configuration
# ------------------
# The browser will send an `Origin` header that looks like:
#   https://nfl-ml-predictions.vercel.app
#
# On Heroku we control CORS via config vars (recommended):
#   - RESTRICT_CORS     : "true" | "false" (default: true)
#   - ALLOWED_ORIGINS   : comma-separated list of exact origins (scheme + host)
#                         Example:
#                           https://nfl-ml-predictions.vercel.app,http://localhost:5173
#                         (We also accept bare hostnames and normalize them to https://...)
#   - ALLOW_ORIGIN_REGEX: regex for dynamic preview origins (e.g., Vercel preview URLs)
#                         Example (recommended):
#                           ^https://.*\.vercel\.app$
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
    explanation_fields: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime
    mode: str = Field(..., description="Mode of prediction, e.g., 'production'")
    win_classifier_used: bool = Field(
        ..., description="Whether the win probability classifier was used"
    )


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
    win_rate: Optional[float] = None
    resolved_games: int = 0
    avg_abs_spread_error: Optional[float] = None
    avg_confidence: Optional[float] = None
    latest_prediction_at: Optional[str] = None
    last_score_sync_at: Optional[str] = None


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


class ScheduleGameResponse(BaseModel):
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_abbr: Optional[str] = None
    away_abbr: Optional[str] = None
    home_logo: Optional[str] = None
    away_logo: Optional[str] = None
    kickoff: Optional[str] = None


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
    return {k: _json_safe_value(v) for k, v in row.items()}


def _json_safe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    safe_rows: List[Dict[str, Any]] = []
    for row in records:
        safe_rows.append({k: _json_safe_value(v) for k, v in row.items()})
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

    status: Literal["healthy", "unhealthy"]
    status = "healthy" if has_dataset and models_ok else "unhealthy"

    reasons: List[str] = []
    if not has_dataset:
        reasons.append("dataset not loaded")
    if not models_ok:
        missing = [m for m in REQUIRED_MODELS if m not in state.models]
        reasons.append(f"missing models: {', '.join(missing)}")

    reason_str = ", ".join(reasons) if reasons else None

    return HealthResponse(
        status=status,
        reason=reason_str,
        components=HealthComponents(
            dataset=has_dataset,
            models=models_ok,
            loaded_models=list(state.models.keys()),
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
        row_df = pd.DataFrame(
            [
                {
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                    "time_key": (season * 100) + week,
                }
            ]
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
def status_overview() -> StatusOverviewResponse:
    """
    Summary endpoint used by StatsPage.jsx.

    Returns:
      - current health (via /health)
      - dataset info (row count)
      - basic history metrics (prediction count placeholder)
    """
    if state.dataset is not None:
        dataset_stats = {
            "rows": len(state.dataset),
            "path": str(state.dataset_path) if state.dataset_path else "unknown",
            "hash": state.dataset_hash,
        }
    else:
        dataset_stats = {"rows": 0, "path": "none", "hash": None}

    history_metrics = _build_history_metrics(state.history)

    return {
        "health": health(),  # reuse typed health response
        "dataset": dataset_stats,
        "history": {
            "metrics": history_metrics,
        },
    }


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


@app.get("/history/summary", response_model=HistoryMetricsResponse)
def history_summary() -> HistoryMetricsResponse:
    """Aggregated prediction quality and recency metrics for premium dashboard UX."""
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
        "metadata": (MODELS_DIR / "metadata.json").exists(),
    }

    dataset_hash = metadata.get("dataset_hash") or state.dataset_hash
    if (not dataset_hash) and state.dataset_path and state.dataset_path.exists():
        try:
            dataset_hash = hashlib.sha256(state.dataset_path.read_bytes()).hexdigest()
        except Exception:
            dataset_hash = None

    return {
        "ready": len(missing_required) == 0,
        "models_dir": str(MODELS_DIR),
        "current_models_dir": str(CURRENT_MODELS_DIR) if CURRENT_MODELS_DIR.exists() else None,
        "loaded_models": loaded_models,
        "missing_required": missing_required,
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

    return {
        "generated_at": now.isoformat(),
        "started_at": state.started_at.isoformat(),
        "uptime_seconds": uptime_seconds,
        "dataset_path": str(state.dataset_path) if state.dataset_path else None,
        "dataset_hash": state.dataset_hash,
        "dataset_modified_at": dataset_modified_at,
        "dataset_age_seconds": dataset_age_seconds,
        "last_prediction_at": state.last_prediction_at.isoformat() if state.last_prediction_at else None,
        "history_size": len(state.history),
        "predict_cache": {
            "enabled": PREDICT_CACHE_TTL_SEC > 0,
            "ttl_seconds": PREDICT_CACHE_TTL_SEC,
            "max_items": PREDICT_CACHE_MAX_ITEMS,
            "items": len(state.predict_cache),
            "hits": state.predict_cache_hits,
            "misses": state.predict_cache_misses,
            "hit_rate": cache_hit_rate,
        },
    }


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
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(points),
        "points": points,
    }


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
        next_season = int(first.get("season")) if first.get("season") is not None else None
        next_week = int(first.get("week")) if first.get("week") is not None else None
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

    offseason_mode = bool(
        (not next_games)
        or (next_kickoff is None)
        or (days_until_next is not None and days_until_next > 45)
    )

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

    return {
        "generated_at": now.isoformat(),
        "offseason_mode": offseason_mode,
        "current_season": next_season,
        "current_week": next_week,
        "next_known_schedule_date": next_kickoff.isoformat() if next_kickoff else None,
        "days_until_next_game": days_until_next,
        "data_freshness_seconds": dataset_age_seconds,
        "dataset_hash": state.dataset_hash,
        "last_trained_at": last_trained,
    }


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
        created_at=pd.to_datetime(job.get("created_at"), utc=True).to_pydatetime(),
        updated_at=pd.to_datetime(job.get("updated_at"), utc=True).to_pydatetime(),
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
# Schedule: Next Week
# -------------------------------------------------------------------


@app.get("/schedule/next-week", response_model=List[ScheduleGameResponse])
def get_schedule() -> List[ScheduleGameResponse]:
    """
    Return the schedule for the "next" NFL week based on the schedule CSV.

    Logic:
      - Resolve schedule path via _find_schedule_path().
      - Normalize season/week + team abbreviations.
      - If 'gameday' is present, interpret as kickoff datetime (UTC-aware).
      - Determine the next slate using the earliest future game; fall back
        to the latest season/week in the file if all games are in the past.
    """
    df = pd.DataFrame()
    try:
        schedule_table = nfl.load_schedules(seasons=2025)
        df = _to_pandas_schedule_safe(schedule_table)
    except Exception as e:
        logging.warning("[Schedule] nfl.load_schedules failed: %s", e)

    if df is None or df.empty:
        fallback = _find_schedule_path()
        if fallback and fallback.exists():
            try:
                df = pd.read_csv(fallback)
                logging.info("[Schedule] Loaded fallback schedule CSV: %s", fallback)
            except Exception as e:
                logging.warning("[Schedule] Failed fallback schedule CSV load: %s", e)

    if df is None or df.empty:
        logging.warning("[Schedule] No schedule data available; returning empty list.")
        return []

    df = _coerce_season_week(df)
    df = df.infer_objects()
    df = _normalize_team_columns(
        df, cols=["home_abbr", "away_abbr", "home_team", "away_team"]
    )
    # Attach logos if we have a mapping file. This is optional; missing logo
    logo_map = _load_team_logo_map()
    df = _add_kickoff_utc_datetime(df)  # uses 'gameday' column if present

    # Decide which (season, week) is "next"
    now_utc = pd.Timestamp.now(tz="UTC")
    future = df[df["dt"].notna() & (df["dt"] > now_utc)].sort_values(by=["dt", "season", "week"])

    if not future.empty:
        next_row = future.iloc[0]
        target_s = int(next_row.get("season_num", next_row.get("season", 2024)))
        target_w = int(next_row.get("week_num", next_row.get("week", 1)))
    else:
        # Fallback: last season/week in file
        season_series = df.get("season", df.get("season_num"))
        week_series = df.get("week", df.get("week_num"))
        target_s = int(season_series.max()) if season_series is not None else 2025
        target_w = int(week_series.max()) if week_series is not None else 1

    s_col = "season_num" if "season_num" in df.columns else "season"
    w_col = "week_num" if "week_num" in df.columns else "week"

    week_df = df[(df[s_col] == target_s) & (df[w_col] == target_w)]

    results: List[Dict[str, Any]] = []
    for _, row in week_df.iterrows():
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        home_abbr = row.get("home_abbr", home_team)
        away_abbr = row.get("away_abbr", away_team)

        # Logos: prefer explicit schedule columns, else use logo_map keyed by team abbr
        home_logo = (
            row.get("home_logo")
            or row.get("home_logo_url")
            or logo_map.get(str(home_abbr).upper())
            or logo_map.get(str(home_team).upper())
        )
        away_logo = (
            row.get("away_logo")
            or row.get("away_logo_url")
            or logo_map.get(str(away_abbr).upper())
            or logo_map.get(str(away_team).upper())
        )

        kickoff_val: Optional[str] = None
        if ("dt" in row) and pd.notna(row["dt"]):
            try:
                kickoff_val = row["dt"].isoformat()
            except Exception:
                kickoff_val = None

        results.append(
            {
                "game_id": f"{target_s}_{target_w}_{home_team}_{away_team}",
                "season": int(target_s),
                "week": int(target_w),
                "home_team": home_team,
                "away_team": away_team,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_logo": home_logo,
                "away_logo": away_logo,
                "kickoff": kickoff_val,
            }
        )

    logging.info(
        "[Schedule] Returning %d games for season=%s week=%s",
        len(results),
        target_s,
        target_w,
    )
    return results


@app.get("/api/predict/next-week")
@app.get("/predict/next-week")
def predict_next_week() -> Dict[str, Any]:
    """Backward-compatible wrapper around the next-week schedule route."""
    return {"games": get_schedule()}


# -------------------------------------------------------------------
# Prediction History
# -------------------------------------------------------------------


@app.get("/history", response_model=List[PredictionResponse])
def get_prediction_history(
    limit: int = Query(100, ge=1, le=1000)
) -> List[Dict[str, Any]]:
    """
    Return the last N prediction results recorded in memory.

    Used by StatsPage/PredictionContext as a history source.
    """
    if limit <= 0:
        return []
    return state.history[-limit:]


# -------------------------------------------------------------------
# /predict (final enhanced)
# -------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest) -> Dict[str, Any]:
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
    models_ok = all(m in state.models for m in REQUIRED_MODELS)
    if state.dataset is None or not models_ok:
        missing = [m for m in REQUIRED_MODELS if m not in state.models]
        raise HTTPException(status_code=503, detail=f"Not ready: missing models {missing}")

    # ----- Validate request -----
    season = int(request.season)
    week = int(request.week)

    if week < 1 or week > 30:
        raise HTTPException(status_code=400, detail="Invalid week. Expected 1..30.")
    if season < 1990 or season > 2100:
        raise HTTPException(status_code=400, detail="Invalid season. Expected a realistic year.")
    if not request.home_team or not request.away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team are required.")

    home_team = _normalize_team_code(request.home_team)
    away_team = _normalize_team_code(request.away_team)
    if home_team == away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")

    cache_key = state._prediction_cache_key(
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,
    )
    cached = state.get_cached_prediction(cache_key)
    if cached is not None:
        state.last_prediction_at = datetime.now(timezone.utc)
        state.history.append(cached)
        state.history = state.history[-500:]
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
        row = pd.DataFrame(
            [
                {
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                    "time_key": (season * 100) + week,
                }
            ]
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
        )
    except Exception as win_err:
        logging.warning("[Predict] Win probability calc failed; defaulting to 0.5: %s", str(win_err).splitlines()[0])
        win_prob_raw, clf_used = 0.5, False

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

    # Smooth probability (no retraining)
    win_prob = _smooth_win_probability(win_prob_raw, point_diff, clf_used=clf_used)

    # ----- Build response -----
    game_id = f"{season}_{week}_{home_team}_{away_team}"
    predicted_total = float(h_score + a_score)
    generated_at = datetime.now(timezone.utc)

    result: Dict[str, Any] = {
        "season": season,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
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

    # ----- History (bounded) -----
    state.history.append(result)
    state.history = state.history[-500:]

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
