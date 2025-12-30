"""
NFL ML Predictions API — Backend Server
=====================================

FastAPI backend serving ML predictions for NFL game outcomes.

Endpoints implemented in THIS file:
  GET  /health
  GET  /debug
  GET  /schedule/next-week
  POST /predict
  POST /predict/explain
  POST /llm/chat
  GET  /history
  GET  /status/overview

Key environment variables:
  MODELS_DIR                 Path to model artifacts directory (contains metadata.json)
  DATA_DIR                   Path to directory containing game_features_*.csv
  DATASET                    Optional: direct path to a specific engineered features CSV
  ALLOWED_ORIGINS            Optional: comma-separated allowed origins for CORS
  ALLOW_ORIGIN_REGEX         Optional: regex for dynamic CORS (default: vercel.app)
  PREDICTION_HISTORY_MAX     Max number of history entries to keep (default 1000)
  ALLOW_FALLBACK_PREDICTIONS If false, missing game rows will raise instead of roll-forward (default true)
  SCHEDULE_SEASON            Season year used by nflreadpy schedule fetch (default current year)
  MC_SIMS                    Monte Carlo simulation count (default 2000)

Design principles:
- Prefer clarity over cleverness.
- One “readiness” gate for all endpoints.
- Never crash because of missing optional components (Ollama is best-effort).

Run locally:

uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000  

Test endpoints:
curl http://127.0.0.1:8000/teams/$body = @{ home_team="KC"; away_team="BUF"; season=2025; week=15 } | ConvertTo-Json -Depth 10; Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body

curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/debug
curl http://127.0.0.1:8000/schedule/next-week
curl http://127.0.0.1:8000/history?limit=5
curl http://127.0.0.1:8000/status/overview

$body = @{ home_team="KC"; away_team="BUF"; season=2025; week=15 } | ConvertTo-Json -Depth 10; Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body


curl -X POST http://127.0.0.1:8000/predict/explain \\
  -H "Content-Type: application/json" \\
  -d "{\\"home_team\\":\\"KC\\",\\"away_team\\}":{\\"BUF\\",\\"season\\":2025,\\"week\\":1}"


curl -X POST http://127.0.0.1:8000/llm/chat \\
  -H "Content-Type: application/json" \\
  -d "{\\"messages\\": [{\\"role\\": \\"user\\", \\"content\\": \\"What is the best team in the NFL?\\"}]}

"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Tuple
import dotenv
import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from backend.team_assets import load_team_assets_map, TeamAsset, normalize_abbr
from backend.config import BACKEND_DIR, _load_env

# Load environment variables early
_load_env(dotenv_path=BACKEND_DIR / ".env")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------
# Global state (simple + explicit)
# ---------------------------

model_objects: Optional["InferenceBundle"] = None
dataset_df: Optional[pd.DataFrame] = None

prediction_history_entries: List[Dict[str, Any]] = []
_prediction_history_lock = Lock()

PREDICTION_HISTORY_MAX = int(os.getenv("PREDICTION_HISTORY_MAX", "1000"))
PREDICTION_HISTORY_PATH = (BACKEND_DIR / "Predictions" / "prediction_history.json").resolve()

# Path configuration
DEFAULT_DATA_DIR = (BACKEND_DIR / "data" / "heroku-models").resolve()
DEFAULT_MODELS_DIR = (DEFAULT_DATA_DIR / "models").resolve()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_env_path(raw: str, base_dir: Path) -> Path:
    """Resolve relative paths against a base dir, keep absolute paths as-is."""
    p = Path(str(raw)).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw and raw.strip():
        return _resolve_env_path(raw.strip(), base_dir=BACKEND_DIR)
    return default


DATA_DIR = _path_from_env("DATA_DIR", DEFAULT_DATA_DIR)
MODELS_DIR = _path_from_env("MODELS_DIR", DEFAULT_MODELS_DIR)

ALLOW_FALLBACK_PREDICTIONS = _bool_env("ALLOW_FALLBACK_PREDICTIONS", default=False)

# ---------------------------
# Core data structures
# ---------------------------

@dataclass(frozen=True)
class InferenceBundle:
    """Loaded model artifacts + metadata used for inference."""
    meta: Dict[str, Any]
    report: Dict[str, Any]
    preprocessor: Any
    home_model: Any
    away_model: Any
    hist_win_clf: Any

    @property
    def raw_feature_columns(self) -> List[str]:
        cols = self.meta.get("raw_feature_columns", {}) or {}
        num = cols.get("numeric", []) or []
        cat = cols.get("categorical", []) or []
        return [*list(num), *list(cat)]

    @property
    def home_rmse(self) -> float:
        # Defaults match your original intent
        return float(self.report.get("home_model_metrics", {}).get("rmse", 5.5))

    @property
    def away_rmse(self) -> float:
        return float(self.report.get("away_model_metrics", {}).get("rmse", 5.2))


class MonteCarloSimulator:
    """
    Simple Monte Carlo layer to add realism.
    Keeps output deterministic per matchup (stable seed) so repeated calls don't “wiggle.”
    """
    def __init__(self, bundle: InferenceBundle):
        self.n_sims = int(os.getenv("MC_SIMS", "2000"))
        self.home_sd = float(os.getenv("MC_HOME_SD", str(bundle.home_rmse)))
        self.away_sd = float(os.getenv("MC_AWAY_SD", str(bundle.away_rmse)))

    @staticmethod
    def _stable_seed(key: str) -> int:
        # Stable 32-bit seed derived from a matchup key
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def simulate(self, home_mu: float, away_mu: float, key: str) -> Dict[str, Any]:
        rng = np.random.default_rng(self._stable_seed(key))

        home = rng.normal(loc=float(home_mu), scale=self.home_sd, size=self.n_sims)
        away = rng.normal(loc=float(away_mu), scale=self.away_sd, size=self.n_sims)

        # NFL scores are non-negative; we can clamp to reduce nonsense tails
        home = np.clip(home, 0, 80)
        away = np.clip(away, 0, 80)

        win_prob = float(np.mean(home > away))
        return {
            "sim_home_score": float(np.mean(home)),
            "sim_away_score": float(np.mean(away)),
            "sim_win_prob": float(np.clip(win_prob, 0.0, 1.0)),
            "sim_n": int(self.n_sims),
            "sim_home_sd": float(self.home_sd),
            "sim_away_sd": float(self.away_sd),
            # A couple of helpful percentiles for UI/UX (harmless additive fields)
            "sim_home_p10": float(np.quantile(home, 0.10)),
            "sim_home_p90": float(np.quantile(home, 0.90)),
            "sim_away_p10": float(np.quantile(away, 0.10)),
            "sim_away_p90": float(np.quantile(away, 0.90)),
        }


def _feature_helpers():
    """Lazy import to avoid circular imports (kept from your original)."""
    import backend.utils.feature_helpers as fh
    return fh
# Try to import live predictor service
try:
    from backend.services.live_predictor import build_live_row, infer_from_row
    USE_LIVE_PREDICTOR = True
except ImportError as e:
    logging.warning(f"Live predictor unavailable ({e}); falling back to static features.")
    USE_LIVE_PREDICTOR = False

# ---------------------------
# Loading
# ---------------------------

def _latest_game_features_csv(data_dir: Path) -> Path:
    files = sorted(data_dir.glob("game_features_*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No game_features_*.csv found in: {data_dir}")
    return files[0]


def _load_metadata(models_dir: Path) -> Dict[str, Any]:
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_training_report(models_dir: Path) -> Dict[str, Any]:
    report_path = models_dir / "training_report.json"
    if not report_path.exists():
        log.warning("training_report.json not found; using default RMSE")
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_inference_bundle(models_dir: Path = MODELS_DIR) -> InferenceBundle:
    meta = _load_metadata(models_dir)
    report = _load_training_report(models_dir)

    # Direct paths from metadata.json (same behavior as original)
    pre_path = models_dir / meta["preprocessor"]
    home_path = models_dir / meta["home_model"]
    away_path = models_dir / meta["away_model"]
    hist_path = models_dir / meta["hist_win_model"]

    return InferenceBundle(
        meta=meta,
        report=report,
        preprocessor=joblib.load(pre_path),
        home_model=joblib.load(home_path),
        away_model=joblib.load(away_path),
        hist_win_clf=joblib.load(hist_path),
    )


def load_dataset_df(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    dataset_override = os.getenv("DATASET", "").strip()
    if dataset_override:
        candidate = _resolve_env_path(dataset_override, base_dir=BACKEND_DIR)
        csv_path = candidate if candidate.exists() else _latest_game_features_csv(data_dir)
        if not candidate.exists():
            log.warning("DATASET override not found: %s; falling back to latest CSV.", candidate)
    else:
        csv_path = _latest_game_features_csv(data_dir)

    df = pd.read_csv(csv_path)

    # Best-effort season/week coercion (same as original intent)
    fh = _feature_helpers()
    if fh is not None and hasattr(fh, "coerce_season_week"):
        df = fh.coerce_season_week(df)
    else:
        df = df.copy()
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        if "week" in df.columns:
            df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # Normalize key identifiers
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].astype(str).str.upper().str.strip()
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].astype(str).str.upper().str.strip()

    return df

# ---------------------------
# Prediction helpers
# ---------------------------

def _require_ready() -> Tuple[InferenceBundle, pd.DataFrame]:
    """
    Single source of truth for readiness.
    Use 503 for “service unavailable / not initialized”.
    """
    if model_objects is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized. Check backend startup logs.")
    if dataset_df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded. Check backend startup logs.")
    return model_objects, dataset_df


def _compute_game_id(season: int, week: int, home_team: str, away_team: str) -> str:
    return f"{int(season)}_{int(week)}_{str(home_team).strip().upper()}_{str(away_team).strip().upper()}"


def _as_1row_df(row: pd.Series) -> pd.DataFrame:
    return row.to_frame().T


def _safe_fill(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def _predict_regressor(model: Any, pre: Any, X_raw: pd.DataFrame) -> float:
    # Try raw first (works if model is a Pipeline that includes preprocessing)
    try:
        pred = model.predict(X_raw)
        return float(np.ravel(pred)[0])
    except Exception:
        X_tx = pre.transform(_safe_fill(X_raw))
        pred = model.predict(X_tx)
        return float(np.ravel(pred)[0])


def _pick_positive_class_index(clf: Any) -> int:
    classes = getattr(clf, "classes_", None)
    if classes is None:
        return 1
    cls = list(classes)
    for label in (1, True, "HOME", "home", "home_win", "1", "True"):
        if label in cls:
            return cls.index(label)
    return 1 if len(cls) > 1 else 0


def _predict_home_win_prob(bundle: InferenceBundle, X_raw: pd.DataFrame, point_diff: float) -> Tuple[float, bool]:
    """
    Returns (probability, used_fallback).
    Fallback is a logistic curve on point_diff.
    """
    clf = bundle.hist_win_clf

    if hasattr(clf, "predict_proba"):
        # raw
        try:
            proba = clf.predict_proba(X_raw)
            idx = _pick_positive_class_index(clf)
            return float(np.clip(float(proba[0][idx]), 0.0, 1.0)), False
        except Exception as e:
            log.warning("[Predict] hist_win_clf predict_proba(raw) failed: %s", e)

        # transformed
        try:
            X_tx = bundle.preprocessor.transform(_safe_fill(X_raw))
            proba = clf.predict_proba(X_tx)
            idx = _pick_positive_class_index(clf)
            return float(np.clip(float(proba[0][idx]), 0.0, 1.0)), False
        except Exception as e:
            log.warning("[Predict] hist_win_clf predict_proba(preprocessed) failed: %s", e)

    # logistic fallback
    p = 1.0 / (1.0 + math.exp(-0.25 * float(point_diff)))
    return float(np.clip(p, 0.0, 1.0)), True


def _get_feature_columns(bundle: InferenceBundle) -> Tuple[List[str], List[str], List[str]]:
    raw = bundle.meta.get("raw_feature_columns", {}) if bundle.meta else {}
    numeric = list(raw.get("numeric", []) or [])
    categorical = list(raw.get("categorical", []) or [])

    # Defensive fallback (same as your original)
    if not numeric and not categorical:
        all_cols = bundle.raw_feature_columns
        return all_cols, [], all_cols

    return numeric, categorical, numeric + categorical


def _dataset_means(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    means: Dict[str, float] = {}
    for col in numeric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            m = series.mean()
            if not pd.isna(m):
                means[col] = float(m)
    return means


def _roll_forward_team_features(
    df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    target_side: str,
    numeric_cols: List[str],
) -> Dict[str, float]:
    """
    Roll forward numeric features from the most recent completed game for a team.
    """
    if df is None or df.empty:
        return {}
    if "season" not in df.columns or "week" not in df.columns:
        return {}

    season_num = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    week_num = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    time_key = season_num * 100 + week_num
    cutoff = int(season) * 100 + int(week)

    team_mask = ((df.get("home_team") == team) | (df.get("away_team") == team)) & (time_key < cutoff)

    # Keep “completed games only” heuristic if points exist
    if "home_points_for" in df.columns and "away_points_for" in df.columns:
        team_mask &= df["home_points_for"].notna() & df["away_points_for"].notna()

    if not bool(team_mask.any()):
        return {}

    last_idx = time_key[team_mask].idxmax()
    last_game = df.loc[last_idx]
    last_side = "home" if str(last_game.get("home_team")) == team else "away"

    out: Dict[str, float] = {}
    target_prefix = f"{target_side}_"
    source_prefix = f"{last_side}_"

    for col in numeric_cols:
        if not col.startswith(target_prefix):
            continue
        source_col = source_prefix + col[len(target_prefix):]
        if source_col in last_game and pd.notna(last_game[source_col]):
            try:
                out[col] = float(last_game[source_col])
            except Exception:
                continue

    return out


def _build_future_row(
    df: pd.DataFrame,
    bundle: InferenceBundle,
    home: str,
    away: str,
    season: int,
    week: int,
) -> pd.Series:
    """
    Build a row for future games:
      - roll-forward last known team features
      - set identifiers (home/away/season/week)
      - fill remaining numeric with dataset means
    """
    numeric_cols, categorical_cols, _ = _get_feature_columns(bundle)
    means = _dataset_means(df, numeric_cols)

    features: Dict[str, Any] = {}
    features.update(_roll_forward_team_features(df, home, season, week, "home", numeric_cols))
    features.update(_roll_forward_team_features(df, away, season, week, "away", numeric_cols))

    # Explicitly set season/week if present (fixes silent mean-fill bug)
    if "season" in numeric_cols:
        features["season"] = int(season)
    if "week" in numeric_cols:
        features["week"] = int(week)

    # Team identifiers (categorical)
    for col in categorical_cols:
        if col == "home_team":
            features[col] = home
        elif col == "away_team":
            features[col] = away
        elif col == "has_home_team":
            features[col] = True
        elif col.startswith("home_team_"):
            features[col] = (col == f"home_team_{home}")
        elif col.startswith("away_team_"):
            features[col] = (col == f"away_team_{away}")

    # Neutral defaults for common market/rest features if missing
    if "home_moneyline_prob" in numeric_cols and pd.isna(features.get("home_moneyline_prob")):
        features["home_moneyline_prob"] = means.get("home_moneyline_prob", 0.5)
    if "away_moneyline_prob" in numeric_cols and pd.isna(features.get("away_moneyline_prob")):
        features["away_moneyline_prob"] = means.get("away_moneyline_prob", 0.5)
    if "home_rest" in numeric_cols and pd.isna(features.get("home_rest")):
        features["home_rest"] = means.get("home_rest", 7.0)
    if "away_rest" in numeric_cols and pd.isna(features.get("away_rest")):
        features["away_rest"] = means.get("away_rest", 7.0)

    # Simple derived diffs
    if "moneyline_prob_diff" in numeric_cols:
        h, a = features.get("home_moneyline_prob"), features.get("away_moneyline_prob")
        if pd.notna(h) and pd.notna(a):
            features["moneyline_prob_diff"] = float(h) - float(a)

    if "rest_diff" in numeric_cols:
        h, a = features.get("home_rest"), features.get("away_rest")
        if pd.notna(h) and pd.notna(a):
            features["rest_diff"] = float(h) - float(a)

    if "elo_diff_pre" in numeric_cols:
        h, a = features.get("home_elo_pre"), features.get("away_elo_pre")
        if pd.notna(h) and pd.notna(a):
            features["elo_diff_pre"] = float(h) - float(a)

    # General “home_minus_away_*” features
    for col in numeric_cols:
        if not col.startswith("home_minus_away_"):
            continue
        suffix = col[len("home_minus_away_"):]
        h_col = f"home_{suffix}"
        a_col = f"away_{suffix}"
        h, a = features.get(h_col), features.get(a_col)
        if pd.notna(h) and pd.notna(a):
            features[col] = float(h) - float(a)

    # Fill remaining numeric gaps
    for col in numeric_cols:
        if col not in features or pd.isna(features.get(col)):
            features[col] = means.get(col, 0.0)

    return pd.Series(features)


# ---------------------------
# History persistence
# ---------------------------

def _load_prediction_history_from_disk() -> None:
    global prediction_history_entries

    if not PREDICTION_HISTORY_PATH.exists():
        prediction_history_entries = []
        return

    try:
        obj = json.loads(PREDICTION_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            entries = [e for e in obj if isinstance(e, dict)]
            prediction_history_entries = entries[:PREDICTION_HISTORY_MAX]
        else:
            prediction_history_entries = []
    except Exception as e:
        log.warning("Failed to load prediction history from %s: %s", PREDICTION_HISTORY_PATH, e)
        prediction_history_entries = []


def _append_prediction_history_to_disk(request_payload: Dict[str, Any], prediction_payload: Dict[str, Any]) -> None:
    global prediction_history_entries

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request": request_payload,
        "prediction": prediction_payload,
    }

    with _prediction_history_lock:
        prediction_history_entries = [entry] + (prediction_history_entries or [])
        prediction_history_entries = prediction_history_entries[:PREDICTION_HISTORY_MAX]

        try:
            PREDICTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            PREDICTION_HISTORY_PATH.write_text(
                json.dumps(prediction_history_entries, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning("Failed to persist prediction history to %s: %s", PREDICTION_HISTORY_PATH, e)

# ---------------------------
# API models
# ---------------------------

class TeamCard(BaseModel):
    key: str
    value: Any
   

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    season: int
    week: int
    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
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
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str
    prediction_source: str
    win_classifier_used: bool
    simulation_metrics: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(from_attributes=True)


class ExplainRequest(BaseModel):
    home_team: str
    away_team: str
    season: int
    week: int
    prediction: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(from_attributes=True)


class ExplainResponse(BaseModel):
    game_id: str
    used_llm: bool
    llm_model: Optional[str] = None
    explanation: str
    bullets: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    prediction: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    used_llm: bool
    llm_model: Optional[str] = None
    reply: str
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    status: str
    mode: str
    reason: str
    model_config = ConfigDict(from_attributes=True)


class ScheduleGame(BaseModel):
    season: int
    week: int
    home_team: str
    away_team: str
    game_id: Optional[str] = None
    kickoff: Optional[str] = None
    home_score: Optional[float] = None
    away_score: Optional[float] = None


class ScheduleResponse(BaseModel):
    games: List[ScheduleGame]


class HistoryResponse(BaseModel):
    entries: List[Dict[str, Any]]
    total: int


class StatusOverviewResponse(BaseModel):
    health: HealthResponse
    dataset: Dict[str, Any]
    history: Dict[str, Any]

# ---------------------------
# LLM helper (best-effort)
# ---------------------------

def _fallback_explain(pred: Dict[str, Any]) -> Dict[str, Any]:
    home = str(pred.get("home_team", "")).upper()
    away = str(pred.get("away_team", "")).upper()
    hs = pred.get("home_score")
    as_ = pred.get("away_score")
    p_home = pred.get("home_win_probability")
    pdiff = pred.get("point_diff")

    bullets: List[str] = []
    if isinstance(pdiff, (int, float)):
        fav = home if float(pdiff) >= 0 else away
        bullets.append(f"{fav} is favored by about {abs(float(pdiff)):.1f} points (model estimate).")
    if isinstance(p_home, (int, float)):
        bullets.append(f"Home win probability is ~{100.0 * float(p_home):.0f}% (calibrated/ensemble output).")
    if isinstance(hs, (int, float)) and isinstance(as_, (int, float)):
        bullets.append(f"Projected score: {home} {hs}  •  {away} {as_}.")

    caveats = [
        "This is a pre-game estimate. Late injuries, weather, and market moves can shift reality.",
        "If the game row was missing, features may be rolled-forward/mean-filled (see prediction_source).",
    ]

    favored = home if (isinstance(pdiff, (int, float)) and float(pdiff) >= 0) else away
    explanation = f"{home} vs {away}: model leans {favored} based on learned pre-game feature patterns."
    return {"explanation": explanation, "bullets": bullets, "caveats": caveats}


def _build_chat_context(pred: Optional[Dict[str, Any]]) -> Optional[str]:
    if not pred:
        return None

    home = str(pred.get("home_team", "")).upper()
    away = str(pred.get("away_team", "")).upper()

    lines = [
        "You are an NFL predictions assistant.",
        "Use this prediction context when answering if relevant:",
    ]
    if home or away:
        lines.append(f"matchup: {home} vs {away}")
    if pred.get("season") or pred.get("week"):
        lines.append(f"season_week: {pred.get('season')} / {pred.get('week')}")
    if isinstance(pred.get("home_score"), (int, float)) and isinstance(pred.get("away_score"), (int, float)):
        lines.append(f"predicted_score: {home} {pred.get('home_score')} - {away} {pred.get('away_score')}")
    if isinstance(pred.get("home_win_probability"), (int, float)):
        lines.append(f"home_win_probability: {pred.get('home_win_probability')}")
    if pred.get("prediction_source"):
        lines.append(f"prediction_source: {pred.get('prediction_source')}")

    return "\\n".join(lines)


def _normalize_chat_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        content = str(msg.content or "").strip()
        if not content:
            continue
        role = msg.role if msg.role in {"user", "assistant", "system"} else "user"
        normalized.append({"role": role, "content": content})
    return normalized


async def _try_ollama_explain(pred: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from backend.ollama.llm_ollama import explain_prediction
    except Exception as e:
        return {"used_llm": False, "error": f"ollama helper not available: {e}"}
    return await explain_prediction(pred)


async def _try_ollama_chat(messages: List[Dict[str, str]], context_prompt: Optional[str]) -> Dict[str, Any]:
    try:
        from backend.ollama.llm_ollama import chat_messages
    except Exception as e:
        return {"used_llm": False, "error": f"ollama helper not available: {e}"}
    return await chat_messages(messages, system_prompt=context_prompt)



def get_team_asset(team_abbr: str) -> TeamAsset:
    """
    Core lookup logic (kept separate so it’s testable).
    """
    team = normalize_abbr(team_abbr)
    assets = load_team_assets_map()

    asset = assets.get(team)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Team not found: {team}")

    if not asset.preferred_logo:
        # Clear error: client asked for team but we have no usable logo fields
        raise HTTPException(status_code=404, detail=f"No logo available for team: {team}")

    return asset

              
   
        
    
    
    
    
    
# ---------------------------
# Lifespan
# ---------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_objects, dataset_df

    log.info("=" * 60)
    log.info("STARTUP: NFL Prediction API (simplified)")
    log.info("MODELS_DIR=%s", MODELS_DIR)
    log.info("DATA_DIR=%s", DATA_DIR)
    log.info("=" * 60)

    try:
        model_objects = load_inference_bundle()
        log.info("✓ Models loaded successfully")
    except Exception as e:
        model_objects = None
        log.error("✗ Failed to load models: %s", e, exc_info=True)

    try:
        dataset_df = load_dataset_df()
        log.info("✓ Dataset loaded successfully (%d rows)", len(dataset_df))
    except Exception as e:
        dataset_df = pd.DataFrame()
        log.error("✗ Failed to load dataset: %s", e, exc_info=True)

    try:
        _load_prediction_history_from_disk()
        log.info("✓ Prediction history loaded (%d entries)", len(prediction_history_entries))
    except Exception as e:
        log.warning("Prediction history load failed: %s", e)

    log.info("STARTUP COMPLETE")
    yield
    log.info("SHUTDOWN: done")

# ---------------------------
# FastAPI app + CORS
# ---------------------------

app = FastAPI(
    title="NFL ML Predictions API",
    version="2.1.0",
    lifespan=lifespan,
)

DEFAULT_ALLOWED_ORIGINS = [
    "https://nfl-ml-predictions.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

raw_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
allowed_origins = raw_origins or DEFAULT_ALLOWED_ORIGINS  # fix: empty env should not produce [""]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=os.getenv("ALLOW_ORIGIN_REGEX", r"^https://.*\\.vercel\\.app$"),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------------------------
# Routes
# ---------------------------




@app.get("/teams/{team_abbr}", response_model=TeamAsset)
def teams_get(team_abbr: str) -> TeamAsset:
    """
    Get a team’s branding assets (preferred non-square logo included).

    Example:
      GET /teams/LAR
    """
    response_model = get_team_asset(team_abbr) 
    if response_model is None:
        raise HTTPException(status_code=404, detail=f"Team not found: {team_abbr}") 
    return response_model



@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if model_objects is not None:
        return HealthResponse(status="healthy", mode="production", reason="models and dataset loaded")
    reason = []
    if model_objects is None:
        reason.append("models_not_loaded")
    return HealthResponse(status="unhealthy", mode="none", reason=";".join(reason) or "not_ready")


@app.get("/debug")
async def get_debug_info():
    return {
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "MODELS_DIR": str(MODELS_DIR),
            "DATA_DIR": str(DATA_DIR),
            "PREDICTION_HISTORY_PATH": str(PREDICTION_HISTORY_PATH),
            "ALLOW_FALLBACK_PREDICTIONS": str(ALLOW_FALLBACK_PREDICTIONS),
        },
        "metadata": model_objects.meta if model_objects else None,
        "dataset_info": {
            "shape": dataset_df.shape if dataset_df is not None else None,
            "columns": list(dataset_df.columns) if dataset_df is not None else [],
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_game(payload: PredictionRequest) -> PredictionResponse:   
    try:
        if USE_LIVE_PREDICTOR:
            # Live feature generation path 
            # 1. Build the dataframe row dynamically
            row_df = build_live_row(
                home_team=normalize_abbr(payload.home_team),
                away_team=normalize_abbr(payload.away_team),
                season=payload.season,
                week=payload.week
            )
            
            # 2. Get the model bundle
            bundle, _ = _require_ready() # We ignore the static dataset now
            
            # 3. Predict using the row
            # We can use a simplified inference helper since we have the row
            result, feature_fallback = infer_from_row(row_df, bundle)
            
        else:
            # Fallback to invalid static path (shouldn't happen if deps installed)
            raise HTTPException(status_code=503, detail="Live predictor service unavailable.")

        # Ensure JSON-compliant response
        for k, v in result.items():
            if isinstance(v, (np.integer, np.floating)):
                result[k] = float(v) if isinstance(v, np.floating) else int(v)

        # 4. Correct mapping to PredictionResponse
        return PredictionResponse(
            season=payload.season,
            week=payload.week,
            home_team=payload.home_team,
            away_team=payload.away_team,
            game_id=_compute_game_id(payload.season, payload.week, payload.home_team, payload.away_team),
            home_score=result["predicted_home_score"],
            away_score=result["predicted_away_score"],
            home_win_probability=result["win_probability"],
            away_win_probability=result["away_win_probability"],
            point_diff=float(result["predicted_home_score"] - result["predicted_away_score"]),
            mode="production",
            prediction_source="live" if USE_LIVE_PREDICTOR else "model",
            win_classifier_used=not result.get("prob_used_fallback", False),
            simulation_metrics=result.get("details")
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("[Predict] Inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference engine failure: {e}")

    game_id = _compute_game_id(payload.season, payload.week, result["home_team"], result["away_team"])

    source_parts = []
    if feature_fallback:
        source_parts.append("feature_fallback")
    if result.get("prob_used_fallback"):
        source_parts.append("win_fallback")
    prediction_source = "+".join(source_parts) if source_parts else "model"

    response_data = {
        **result,
        "game_id": game_id,
        "point_diff": result["home_score"] - result["away_score"],
        "ts": datetime.now(timezone.utc),
        "mode": "production",
        "prediction_source": prediction_source,
        "win_classifier_used": not bool(result["prob_used_fallback"]),
    }

    _append_prediction_history_to_disk(payload.model_dump(), response_data)
    return PredictionResponse(**response_data)


@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(payload: ExplainRequest) -> ExplainResponse:
    # If caller provides a prediction dict, explain that exact payload.
    if payload.prediction and isinstance(payload.prediction, dict):
        pred = payload.prediction
        game_id = str(pred.get("game_id") or _compute_game_id(payload.season, payload.week, payload.home_team, payload.away_team))
    else:
        bundle, df = _require_ready()
        result, feature_fallback = infer_prediction_from_dataset(
            dataset_df=df,
            bundle=bundle,
            home_team=payload.home_team,
            away_team=payload.away_team,
            season=payload.season,
            week=payload.week,
        )
        game_id = _compute_game_id(payload.season, payload.week, result["home_team"], result["away_team"])
        pred = {
            **result,
            "game_id": game_id,
            "point_diff": result["home_score"] - result["away_score"],
            "prediction_source": "feature_fallback" if feature_fallback else "model",
        }

    # Always have a deterministic fallback explanation
    fb = _fallback_explain(pred)
    used_llm = False
    llm_model = None
    latency_ms = None
    error = None
    explanation = fb["explanation"]
    bullets = fb["bullets"]
    caveats = fb["caveats"]

    if _bool_env("ENABLE_OLLAMA_EXPLAIN", default=False):
        llm = await _try_ollama_explain(pred)
        used_llm = bool(llm.get("used_llm"))
        llm_model = llm.get("model")
        latency_ms = llm.get("latency_ms")
        error = llm.get("error")
        if used_llm and llm.get("explanation"):
            explanation = llm.get("explanation")
            bullets = llm.get("bullets") or bullets
            caveats = llm.get("caveats") or caveats

    return ExplainResponse(
        game_id=game_id,
        used_llm=used_llm,
        llm_model=llm_model,
        explanation=explanation,
        bullets=bullets,
        caveats=caveats,
        latency_ms=latency_ms,
        error=error,
    )


@app.post("/llm/chat", response_model=ChatResponse)
async def llm_chat(payload: ChatRequest) -> ChatResponse:
    messages = _normalize_chat_messages(payload.messages)
    context_prompt = _build_chat_context(payload.prediction)

    # Minimal fallback reply if LLM not enabled
    fallback_reply = "Ask me about a matchup, a spread, or why the model likes a side. If you include a prediction payload, I can explain it."
    if payload.prediction:
        fb = _fallback_explain(payload.prediction)
        reply_parts = [fb["explanation"]]
        if fb["bullets"]:
            reply_parts.append("Key points:\\n- " + "\\n- ".join(fb["bullets"]))
        if fb["caveats"]:
            reply_parts.append("Caveats:\\n- " + "\\n- ".join(fb["caveats"]))
        fallback_reply = "\\n\\n".join(reply_parts)

    used_llm = False
    llm_model = None
    latency_ms = None
    error = None
    reply = fallback_reply

    if _bool_env("ENABLE_OLLAMA_CHAT", default=False):
        llm = await _try_ollama_chat(messages, context_prompt)
        used_llm = bool(llm.get("used_llm"))
        llm_model = llm.get("model")
        latency_ms = llm.get("latency_ms")
        error = llm.get("error")
        if used_llm and llm.get("reply"):
            reply = llm.get("reply")

    return ChatResponse(
        used_llm=used_llm,
        llm_model=llm_model,
        reply=reply,
        latency_ms=latency_ms,
        error=error,
    )

# ---- Schedule helpers + endpoint (kept behavior, reduced fragility)

def _resolve_kickoff(row: pd.Series) -> str:
    k = row.get("kickoff")
    if pd.notna(k) and str(k).strip():
        k_str = str(k)
        if "T" in k_str or "+" in k_str or "Z" in k_str:
            return k_str
        if " " in k_str:
            return k_str.replace(" ", "T") + "-05:00"
        return k_str

    d = row.get("gameday") or row.get("game_date")
    t = row.get("gametime") or row.get("game_time") or row.get("time")
    if pd.notna(d) and pd.notna(t):
        return f"{str(d)}T{str(t)}:00-05:00"
    return str(d) if pd.notna(d) else ""


@app.get("/schedule/next-week", response_model=ScheduleResponse)
async def get_next_week_schedule():
    """
    Fetch next week's schedule using nflreadpy with fallback to CSV/Dataset.
    (Same strategy as your original; season is now configurable.)
    """
    try:
        season = int(os.getenv("SCHEDULE_SEASON", str(datetime.now(timezone.utc).year)))

        # 1) Live schedule
        try:
            sch = nfl.load_schedules(seasons=[season])
            df = sch.to_pandas() if hasattr(sch, "to_pandas") else None
            if isinstance(df, pd.DataFrame) and "home_score" in df.columns and "week" in df.columns:
                future = df[df["home_score"].isna()]
                if not future.empty:
                    min_week = int(future["week"].min())
                    next_week_games = future[future["week"] == min_week]
                    games = [
                        ScheduleGame(
                            season=int(r["season"]),
                            week=int(r["week"]),
                            home_team=str(r["home_team"]),
                            away_team=str(r["away_team"]),
                            game_id=str(r.get("game_id", "")),
                            kickoff=_resolve_kickoff(r),
                        )
                        for _, r in next_week_games.iterrows()
                    ]
                    return ScheduleResponse(games=games)
        except Exception as e:
            log.warning("Live schedule fetch failed: %s. Falling back to CSV/Dataset...", e)

        # 2) Local CSV
        csv_candidates = [
            Path("NFL_Schedule.csv"),
            BACKEND_DIR.parent / "NFL_Schedule.csv",
            BACKEND_DIR / "data" / "NFL_Schedule.csv",
            Path("backend/NFL_Schedule.csv"),
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                if "home_score" in df.columns and "week" in df.columns:
                    future = df[df["home_score"].isna()]
                    if not future.empty:
                        min_week = int(future["week"].min())
                        next_week_games = df[(df["week"] == min_week) & (df["home_score"].isna())]
                        games = [
                            ScheduleGame(
                                season=int(r["season"]),
                                week=int(r["week"]),
                                home_team=str(r["home_team"]),
                                away_team=str(r["away_team"]),
                                game_id=str(r.get("game_id", f"{r['season']}_{r['week']}_{r['home_team']}_{r['away_team']}")),
                                kickoff=_resolve_kickoff(r),
                            )
                            for _, r in next_week_games.iterrows()
                        ]
                        return ScheduleResponse(games=games)
            except Exception as e:
                log.error("Failed to parse local CSV: %s", e)

        # 3) Dataset fallback
        if dataset_df is not None and "home_score" in dataset_df.columns and "week" in dataset_df.columns:
            future = dataset_df[dataset_df["home_score"].isna()]
            if not future.empty:
                min_week = int(future["week"].min())
                next_week_games = future[future["week"] == min_week]
                games = [
                    ScheduleGame(
                        season=int(r["season"]),
                        week=int(r["week"]),
                        home_team=str(r["home_team"]),
                        away_team=str(r["away_team"]),
                        game_id=f"DS_{r['season']}_{r['week']}_{r['home_team']}_{r['away_team']}",
                    )
                    for _, r in next_week_games.iterrows()
                ]
                return ScheduleResponse(games=games)

        return ScheduleResponse(games=[])

    except Exception as e:
        log.error("Critical failure in get_next_week_schedule: %s", e, exc_info=True)
        return ScheduleResponse(games=[])


@app.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = 100):
    with _prediction_history_lock:
        data = prediction_history_entries[:limit]
        return HistoryResponse(entries=data, total=len(prediction_history_entries))


@app.get("/status/overview", response_model=StatusOverviewResponse)
async def get_status_overview():
    h = await health()

    dataset_info = {
        "rows": int(len(dataset_df)) if dataset_df is not None else 0,
        "features": int(len(model_objects.raw_feature_columns)) if model_objects else 0,
    }

    # Keep placeholder but label it clearly
    with _prediction_history_lock:
        history_metrics = {
            "total_predictions": int(len(prediction_history_entries)),
            "win_rate": None,  # unknown (we don't store actual outcomes here)
            "note": "win_rate requires actual game outcomes; currently not tracked in history store",
        }

    return StatusOverviewResponse(health=h, dataset=dataset_info, history=history_metrics)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
