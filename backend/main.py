from __future__ import annotations
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


FILE: /main.py
PURPOSE: FastAPI application for NFL ML Predictions.
DATA SHAPES:
  - PredictionRequest: { home_team: str, away_team: str, season: int, week: int }
  - UnifiedPredictionResponse: { home_score, away_score, point_diff, probabilities, ... }
KEY FUNCTIONS/CLASSES:
  - lifespan: Preloads models and datasets on startup.
  - predict: Unified inference endpoint delegating to PredictionService.
  - legacy routes: APIRouter mounted under /legacy for backward compatibility.
SIDE EFFECTS / I/O: Loads ML artifacts from MODELS_DIR, reads dataset from DATA_DIR, reads team logos CSV.
ERROR HANDLING: 404 for missing games, 503 for uninitialized models.
DEPENDENCIES: FastAPI, Pydantic, PredictionService, InferenceBundle.
"""
# -------------------------------------
# IMPORTS
# -------------------------------------
# /main.py



import logging
import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    UnifiedPredictionResponse,
    HealthResponse,
    StatusOverviewResponse,
    HistoryResponse,
    ScheduleResponse,
    ScheduleEntry,
)
from .services.prediction_service import PredictionService
from .services.inference_row import build_model_input_row
from .config import DATA_DIR as CFG_DATA_DIR, MODELS_DIR as CFG_MODELS_DIR, resolve_cors, TRUTHY
from .main_helpers import (
    load_inference_bundle,
    load_dataset_df,
    _append_prediction_history_to_disk,
    prediction_history_entries,
    _prediction_history_lock,
    get_schedule,
    select_next_week_rows,
    get_team_meta,
    parse_kickoff,
    _pick_col,
    _HOME_COLS,
    _AWAY_COLS,
    _GAME_ID_COLS,
    _STADIUM_COLS,
)
from .ollama.llm_ollama import explain_prediction as llm_explain_prediction, chat_messages as llm_chat_messages
from .routes import (
    TeamLogosResponse,
    router as legacy_router,
)
if __name__ == "__main__" and __package__ is None:
    # Allow running as a script by ensuring repo root is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# -------------------------------------
# GLOBALS
# -------------------------------------
# Load environment variables
load_dotenv(dotenv_path=".env")

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

setup_logging()
log = logging.getLogger(__name__)

# Global state
state: Dict[str, Any] = {
    "bundle": None,
    "dataset": None,
    "service": None,
    "model_metadata": None,
    "model_metadata_path": None,
    "dataset_path": None,
    "team_logos": None,
}

ADMIN_ENABLED = os.getenv("ENABLE_ADMIN", "false").strip().lower() in TRUTHY


def _build_game_id(season: int, week: int, home_team: str, away_team: str) -> str:
    parts = [season, week, home_team, away_team]
    return "-".join(str(p) for p in parts if p is not None and str(p).strip())

def _normalize_team_code(value: str) -> str:
    """Normalize team abbreviation to uppercase."""
    return str(value or "").strip().upper()


def _get_team_meta_map() -> Dict[str, Dict[str, str]]:
    """Load team metadata once and cache it for schedule/prediction enrichment."""
    cached = state.get("team_logos")
    if isinstance(cached, dict):
        return cached

    _dir = Path(__file__).resolve().parent
    repo_root = _dir.parent
    candidates = [
        _dir / "team_logo.csv",
        _dir / "team_logos.csv",
        _dir / "data" / "team_logo.csv",
        _dir / "data" / "team_logos.csv",
        repo_root / "team_logo.csv",
        repo_root / "team_logos.csv",
        repo_root / "data" / "team_logo.csv",
        repo_root / "data" / "team_logos.csv",
    ]
    team_map: Dict[str, Dict[str, str]] = {}
    for csv_path in candidates:
        if csv_path.exists():
            team_map = get_team_meta(csv_path)
            break

    state["team_logos"] = team_map or {}
    return state["team_logos"]


# -------------------------------------
# FUNCTIONS -----
# -------------------------------------
def _build_prediction_payload(req: PredictionRequest, res: PredictionResponse) -> Dict[str, Any]:
    """Flatten model output into the unified API response shape."""
    home_code = str(req.home_team).strip().upper()
    away_code = str(req.away_team).strip().upper()
    home_score = float(res.scores.home_score)
    away_score = float(res.scores.away_score)
    team_meta = _get_team_meta_map()
    home_meta = team_meta.get(home_code, {})
    away_meta = team_meta.get(away_code, {})

    payload = {
        "home_score": home_score,
        "away_score": away_score,
        "point_diff": home_score - away_score,
        "home_win_probability": float(res.winner.proba_home),
        "away_win_probability": float(res.winner.proba_away),
        "prediction_source": res.prediction_source,
        "win_classifier_used": res.win_classifier_used,
        "simulation_metrics": (
            res.simulation_metrics.model_dump() if res.simulation_metrics is not None else None
        ),
        "game_id": _build_game_id(req.season, req.week, home_code, away_code),
        "season": req.season,
        "week": req.week,
        "home_team": home_code,
        "away_team": away_code,
        "home_name": home_meta.get("name") or home_code,
        "away_name": away_meta.get("name") or away_code,
    }
    return payload

def _flatten_raw_feature_columns(raw: Any) -> list[str]:
    """Normalize 'raw_feature_columns' shapes into a flat list of column names.

    Supports:
      - list[str]
      - {"numeric":[...], "categorical":[...]}
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        nums = raw.get("numeric") or []
        cats = raw.get("categorical") or []
        out: list[str] = []
        out.extend([str(c) for c in nums if c is not None])
        out.extend([str(c) for c in cats if c is not None])
        return out
    if isinstance(raw, (list, tuple, set)):
        return [str(c) for c in raw if c is not None]
    return []

def _find_latest_metadata_json(models_dir: Path) -> Path | None:
    """Find the most recently modified metadata.json under a models directory."""
    try:
        root = Path(models_dir)
    except Exception:
        return None
    if root.is_file():
        return root if root.name == "metadata.json" else None

    candidates: list[Path] = []
    direct = root / "metadata.json"
    if direct.exists():
        candidates.append(direct)
    # Common patterns: MODELS_DIR/YYYYMMDD/metadata.json or MODELS_DIR/models/metadata.json
    candidates.extend([p for p in root.glob("**/metadata.json") if p.is_file()])
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
    clf = bundle.win_pipe

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



TEAM_ALIAS = {
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS",
}

def norm_team(team: str) -> str:
    """Normalize team abbreviations."""
    t = str(team).strip().upper()
    return TEAM_ALIAS.get(t, t)

def _find_inference_rows(df: pd.DataFrame, home: str, away: str, season: int, week: int) -> pd.DataFrame:
    """Find specific rows in the dataframe matching the matchup."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Pre-normalization
    h_norm = norm_team(home)
    a_norm = norm_team(away)
    
    # Safe casting
    season_col = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    week_col = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    
    # Masking
    mask = (season_col == int(season)) & (week_col == int(week))
    
    # Match team names (handling potential raw discrepancies if needed, but norm_team usually enough)
    # We apply norm logic to the DF columns on the fly if needed, but for speed assume standard abbrs usually match.
    # If standard retrieval fails, we can try more aggressive matching.
    mask &= (df["home_team"] == h_norm) & (df["away_team"] == a_norm)
    
    return df.loc[mask]

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
      1) Lookup existing row in dataset (preferred).
      2) Roll-forward if missing (fallback).
    """
    # 1. Try finding exact match
    matches = _find_inference_rows(df, home, away, season, week)
    if not matches.empty:
        return matches.iloc[0]

    # 2. Fallback: Roll forward
    numeric_cols, categorical_cols, _ = _get_feature_columns(bundle)
    means = _dataset_means(df, numeric_cols)

    features: Dict[str, Any] = {}
    features.update(_roll_forward_team_features(df, home, season, week, "home", numeric_cols))
    features.update(_roll_forward_team_features(df, away, season, week, "away", numeric_cols))

    # Explicitly set season/week
    if "season" in numeric_cols: features["season"] = int(season)
    if "week" in numeric_cols: features["week"] = int(week)

    # Team identifiers
    features["home_team"] = home
    features["away_team"] = away
    features["has_home_team"] = True
    
    # Dynamic categorical columns (e.g. home_team_ARI)
    for col in categorical_cols:
        if col.startswith("home_team_"):
            features[col] = (col == f"home_team_{home}")
        elif col.startswith("away_team_"):
            features[col] = (col == f"away_team_{away}")

    # Fill gaps with means
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



class ExplainRequest(BaseModel):
    home_team: str
    away_team: str
    season: int
    week: int
    prediction: Optional[Dict[str, Any]] = None


class ExplainResponse(BaseModel):
    game_id: str
    used_llm: bool
    llm_model: Optional[str] = None
    explanation: str
    bullets: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    error: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    prediction: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    used_llm: bool
    llm_model: Optional[str] = None
    reply: str
    latency_ms: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    mode: str
    reason: str


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

    return "\n".join(lines)

def _load_model_metadata(models_dir: Path) -> tuple[Path | None, Dict[str, Any] | None]:
    md_path = _find_latest_metadata_json(models_dir)
    if md_path is None:
        return None, None
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            return md_path, json.load(f)
    except Exception as e:
        log.warning("Could not read metadata.json at %s: %s", md_path, e)
        return md_path, None

def _find_latest_dataset_csv(data_dir: Path=os.getenv("DATA_DIR", Path("./data/dataset"))) -> Path | None:
    """Find the newest game_features_*.csv (or any .csv) under DATA_DIR."""
    try:
        root = Path(data_dir)
    except Exception:
        return None
    if root.is_file():
        return root if root.suffix.lower() == ".csv" else None

    patterns = ("game_features_*.csv", "*.csv")
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend([p for p in root.glob(pat) if p.is_file()])
    # allow nested datasets/YYYYMMDD/game_features_YYYYMMDD.csv layouts
    for pat in patterns:
        candidates.extend([p for p in root.glob(f"**/{pat}") if p.is_file()])

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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

def _resolve_expected_features(bundle: Any, metadata: Dict[str, Any] | None = None) -> list[str]:
    """Resolve the *raw* input feature columns expected by the preprocessor/model."""
    pre = getattr(bundle, "preprocessor", None)
    features_in = getattr(pre, "feature_names_in_", None)
    if features_in is not None:
        expected = [str(x) for x in list(features_in)]
        if expected:
            return expected

    # Prefer explicit lists from training metadata (stable across sklearn versions)
    for cand in (
        (metadata or {}).get("feature_names"),
        getattr(bundle, "feature_names", None),
    ):
        if isinstance(cand, (list, tuple)) and len(cand) > 0:
            return [str(x) for x in cand if x is not None]

    # Fall back to 'raw_feature_columns' (either list or {"numeric","categorical"})
    raw = getattr(bundle, "raw_feature_columns", None)
    if metadata and "raw_feature_columns" in metadata:
        raw = metadata.get("raw_feature_columns")
    return _flatten_raw_feature_columns(raw)

def _validate_feature_schema(bundle: Any, dataset: pd.DataFrame, metadata: Dict[str, Any] | None = None) -> None:
    expected = _resolve_expected_features(bundle, metadata=metadata)
    if not expected:
        raise RuntimeError("Model feature list missing; cannot validate schema.")
    missing = [c for c in expected if c not in dataset.columns]
    if missing:
        sample = ", ".join(missing[:25])
        raise RuntimeError(f"Dataset missing {len(missing)} model features. Sample: {sample}")
def _extract_prediction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Support explain payloads with either {prediction:{...}} or flat fields."""
    pred = payload.get("prediction")
    if isinstance(pred, dict):
        return pred
    if any(k in payload for k in ("home_score", "away_score", "home_win_probability")):
        return payload
    return {}


def _predict_home_win_prob(bundle: InferenceBundle, X_raw: pd.DataFrame, point_diff: float) -> Tuple[float, bool]:
    """
    Returns (probability, used_fallback).
    Since we use full pipelines (preprocessor included), we pass raw data directly.
    """
    clf = bundle.win_pipe or bundle.hist_win_clf  # prefer win_pipe (pipeline) logic
    
    if hasattr(clf, "predict_proba"):
        try:
            # Direct prediction using the pipeline
            proba = clf.predict_proba(X_raw)
            idx = _pick_positive_class_index(clf)
            val = float(proba[0][idx])
            return float(np.clip(val, 0.0, 1.0)), False
        except Exception as e:
            log.warning("[Predict] Pipeline predict_proba failed: %s", e)

    # logistic fallback
    if pd.isna(point_diff):
        return 0.5, True
    p = 1.0 / (1.0 + math.exp(-0.25 * float(point_diff)))
    return float(np.clip(p, 0.0, 1.0)), True

# Startup / Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and dataset
    try:
        log.info(f"Starting up: Loading model bundle from {CFG_MODELS_DIR}...")
        
        # 1. Load Bundle (Models)
        state["bundle"] = load_inference_bundle(CFG_MODELS_DIR)
        
        # 2. Load Metadata (if separate)
        # Assuming metadata.json is always in MODELS_DIR as enforced by config
        meta_path = CFG_MODELS_DIR / "metadata.json"
        if meta_path.exists():
            state["model_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
            state["model_metadata_path"] = str(meta_path)
        
        expected_features = _resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))
        
        # 3. Load Dataset
        # Config enforces DATASET_PATH if specific file is needed.
        log.info(f"Loading dataset from {CFG_DATA_DIR}...")
        state["dataset"] = load_dataset_df(CFG_DATA_DIR, expected_features=expected_features)
        
        # Validate
        _validate_feature_schema(state["bundle"], state["dataset"], metadata=state.get("model_metadata"))
        
        # Init Service
        state["service"] = PredictionService(state["bundle"], state["dataset"])
        _get_team_meta_map()
        
        # Expose state
        app.state.dataset = state["dataset"]
        app.state.models = {"models_dir": str(CFG_MODELS_DIR)}
        app.state.team_logos = state.get("team_logos") or {}
        app.state.started_at = datetime.now(timezone.utc).isoformat()
        
        log.info("Startup complete: Models and dataset ready.")
    except Exception as e:
        log.error(f"Startup failed: {e}", exc_info=True)
        # We raise here because the user wants to fix the traceback, meaning we should fail if invalid.
        raise
    
    yield

app = FastAPI(title="NFL ML Predictions API", lifespan=lifespan)

# CORS Middleware
cors_origins, cors_origin_regex = resolve_cors()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------------------------
# Routes
# ---------------------------




@app.get("/teams/{team_abbr}", response_model=TeamCard)
def teams_get(team_abbr: str) -> TeamCard:
    """
    Get a team’s branding assets (preferred non-square logo included).

    Example:
      GET /teams/LAR
    """
    response_model = get_team_asset(team_abbr)
    if response_model is None:
        raise HTTPException(status_code=404, detail=f"Team not found: {team_abbr}")
    return response_model




# Mount legacy router with a safe prefix to avoid clobbering unified endpoints.
app.include_router(legacy_router, prefix="/legacy")

def _require_ready() -> PredictionService:
    if state["service"] is None:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized.")
    return state["service"]

@app.get("/health", response_model=HealthResponse)
async def health():
    ready = state.get("service") is not None
    status = "healthy" if ready else "unhealthy"
    mode = "ml-inference" if ready else "initializing"
    reason = "models_loaded" if ready else "prediction engine not initialized"
    return HealthResponse(status=status, mode=mode, reason=reason)



@app.get("/schedule/next-week", response_model=ScheduleResponse)
async def get_next_week_schedule(season: int | None = None) -> ScheduleResponse:
    df = get_schedule(season=season)
    
    # Fallback to local JSON if CSV fails/empty (legacy behavior)
    if not isinstance(df, pd.DataFrame) or df.empty:
        fallback_path = Path(__file__).resolve().parent / "post_schedule.json"
        if fallback_path.exists():
            try:
                raw = json.loads(fallback_path.read_text(encoding="utf-8"))
                games_list = raw.get("games")
                if isinstance(games_list, list) and games_list:
                    df = pd.DataFrame(games_list)
            except Exception as exc:
                log.warning("Schedule fallback read failed for %s: %s", fallback_path, exc)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
            
    df_next, use_season, use_week = select_next_week_rows(df)
    team_meta = _get_team_meta_map()

    home_col = _pick_col(df_next, _HOME_COLS)
    away_col = _pick_col(df_next, _AWAY_COLS)
    game_id_col = _pick_col(df_next, _GAME_ID_COLS)
    stadium_col = _pick_col(df_next, _STADIUM_COLS)

    games: list[ScheduleEntry] = []
    for _, row in df_next.iterrows():
        home = str(row.get(home_col, "") if home_col else row.get("home", "")).strip().upper()
        away = str(row.get(away_col, "") if away_col else row.get("away", "")).strip().upper()
        if not home or not away:
            continue

        home_info = team_meta.get(home, {})
        away_info = team_meta.get(away, {})

        # Resolve Game ID
        game_id = ""
        if game_id_col:
            raw_id = row.get(game_id_col)
            if pd.notna(raw_id):
                game_id = str(raw_id).strip()
        if not game_id:
            game_id = _build_game_id(use_season, use_week, home, away)

        stadium = row.get(stadium_col, "") if stadium_col else row.get("stadium", "")

        games.append(
            ScheduleEntry(
                season=int(use_season),
                week=int(use_week),
                kickoff=parse_kickoff(row),
                home_team=home,
                away_team=away,
                game_id=game_id,
                home_abbr=home,
                away_abbr=away,
                home_logo=home_info.get("logoUrl"),
                away_logo=away_info.get("logoUrl"),
                # Prefer name from metadata -> schedule -> abbr
                home_name=home_info.get("name") or str(row.get("home_team_name", "")) or home,
                away_name=away_info.get("name") or str(row.get("away_team_name", "")) or away,
                stadium=stadium,
            )
        )

    return ScheduleResponse(games=games)

@app.get("api/teams/logos", response_model=TeamLogosResponse)
async def get_team_logos() -> TeamLogosResponse:
    return TeamLogosResponse(teams=_get_team_meta_map())

@app.get("api/debug")
async def debug() -> Dict[str, Any]:
    origins, origin_regex = resolve_cors()
    dataset = state["dataset"]
    rows = int(len(dataset)) if dataset is not None else 0
    cols = int(dataset.shape[1]) if dataset is not None else 0
    return {
        "status": "ok" if state["service"] else "initializing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "models_dir": str(CFG_MODELS_DIR),
            "data_dir": str(CFG_DATA_DIR),
            "offline_mode": os.getenv("OFFLINE_MODE", "false"),
        },
        "dataset_info": {
            "rows": rows,
            "cols": cols,
            "shape": [rows, cols],
            "sample_cols": list(dataset.columns[:25]) if dataset is not None else [],
        },
    }


@app.post("api/predict", response_model=PredictionResponse)
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


@app.post("api/predict/explain", response_model=ExplainResponse)
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

cors_origins = os.getenv("CORS_ORIGINS", "").split(",")
cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX", "")
restrict_cors = os.getenv("RESTRICT_CORS", "false").strip().lower() in TRUTHY

@app.post("api/debug/predict-input")
async def debug_predict_input(req: PredictionRequest) -> Dict[str, Any]:
    service = _require_ready()
    bundle = state.get("bundle")
    dataset = state.get("dataset")
    if bundle is None or dataset is None:
        raise HTTPException(status_code=503, detail="Models or dataset not loaded.")

    schedule_df = None
    if hasattr(service, "_get_schedule_df"):
        try:
            schedule_df = service._get_schedule_df(req.season)
        except Exception:
            schedule_df = None

    row_df, source, debug_info = build_model_input_row(
        dataset_df=dataset,
        preprocessor=getattr(bundle, "preprocessor", None),
        season=req.season,
        week=req.week,
        home_team=req.home_team,
        away_team=req.away_team,
        schedule_df=schedule_df,
        raw_feature_columns=getattr(bundle, "raw_feature_columns", None),
        team_history_cache=getattr(service, "_team_history_cache", None),
        debug=True,
    )

    log.info(
        "Debug input %s@%s W%s: source=%s missing_after=%s missing_home_prior=%s missing_away_prior=%s",
        req.away_team,
        req.home_team,
        req.week,
        source,
        debug_info.get("missing_after_impute"),
        debug_info.get("missing_home_prior_count"),
        debug_info.get("missing_away_prior_count"),
    )

    return {
        "models_dir": str(CFG_MODELS_DIR),
        "prediction_source": source,
        "debug": debug_info,
    }

@app.post("api/predict", response_model=UnifiedPredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    service = _require_ready()
    res = service.predict(req)
    payload = _build_prediction_payload(req, res)
    _append_prediction_history_to_disk(req.model_dump(), payload)
    return payload

@app.get("api/predict/next-week")
async def predict_next_week() -> Dict[str, Any]:
    service = _require_ready()
    schedule = await get_next_week_schedule()
    games: list[Dict[str, Any]] = []

    for game in schedule.games:
        req = PredictionRequest(
            home_team=game.home_team,
            away_team=game.away_team,
            season=game.season,
            week=game.week,
        )
        prediction = _build_prediction_payload(req, service.predict(req))
        item = game.model_dump()
        item["prediction"] = prediction
        games.append(item)

    return {"games": games}

@app.post("api/predict/explain")
async def explain(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    pred = _extract_prediction_payload(payload)
    home_team = payload.get("home_team") or pred.get("home_team")
    away_team = payload.get("away_team") or pred.get("away_team")
    season_raw = payload.get("season") or pred.get("season")
    week_raw = payload.get("week") or pred.get("week")

    try:
        season = int(season_raw) if season_raw is not None else None
    except (TypeError, ValueError):
        season = None
    try:
        week = int(week_raw) if week_raw is not None else None
    except (TypeError, ValueError):
        week = None

    needs_prediction = (
        not pred
        or pred.get("home_score") is None
        or pred.get("away_score") is None
        or pred.get("home_win_probability") is None
    )
    if needs_prediction:
        if not (home_team and away_team and season is not None and week is not None):
            raise HTTPException(status_code=400, detail="prediction or full game context required")
        req = PredictionRequest(home_team=home_team, away_team=away_team, season=season, week=week)
        service = _require_ready()
        pred = _build_prediction_payload(req, service.predict(req))

    if home_team:
        pred["home_team"] = home_team
    if away_team:
        pred["away_team"] = away_team
    if season is not None:
        pred["season"] = season
    if week is not None:
        pred["week"] = week

    game_id = pred.get("game_id")
    if not game_id and season is not None and week is not None and home_team and away_team:
        game_id = _build_game_id(season, week, home_team, away_team)

    llm_result = await llm_explain_prediction(pred)
    return {
        "game_id": game_id,
        "used_llm": bool(llm_result.get("used_llm")),
        "llm_model": llm_result.get("model"),
        "explanation": llm_result.get("explanation", ""),
        "bullets": llm_result.get("bullets", []) or [],
        "caveats": llm_result.get("caveats", []) or [],
        "error": llm_result.get("error"),
    }

@app.post("api/llm/chat")
async def llm_chat(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    messages = payload.get("messages")
    prediction = payload.get("prediction")
    system_prompt = None

    if isinstance(prediction, dict) and prediction:
        home = prediction.get("home_team") or prediction.get("home_abbr")
        away = prediction.get("away_team") or prediction.get("away_abbr")
        season = prediction.get("season")
        week = prediction.get("week")
        system_prompt = (
            "You are an NFL prediction assistant. "
            f"Context: {home} vs {away}, season {season}, week {week}. "
            f"Prediction snapshot: {prediction}."
        )

    result = await llm_chat_messages(messages if isinstance(messages, list) else [], system_prompt=system_prompt)
    reply = result.get("reply") or ""
    if not reply and result.get("error"):
        reply = f"Error: {result.get('error')}"
    return {
        "reply": reply,
        "used_llm": bool(result.get("used_llm")),
        "llm_model": result.get("model"),
        "error": result.get("error"),
    }

@app.get("api/history", response_model=HistoryResponse)
async def get_history(limit: int = 100):
    with _prediction_history_lock:
        data = prediction_history_entries[:limit]
        return HistoryResponse(entries=data, total=len(prediction_history_entries))


@app.get("api/status/models")
async def get_status_models() -> Dict[str, Any]:
    """Return model + dataset provenance (from training metadata), plus expected feature schema."""
    bundle = state.get("bundle")
    md = state.get("model_metadata") or {}
    expected = _resolve_expected_features(bundle, metadata=md) if bundle is not None else []
    return {
        "health": "ok" if state.get("service") else "initializing",
        "models_dir": str(CFG_MODELS_DIR),
        "metadata_path": str(state.get("model_metadata_path") or ""),
        "dataset_path": state.get("dataset_path") or "",
        "expected_features_count": len(expected),
        "expected_features_sample": expected[:25],
        "metadata": md,
    }

@app.post("api/admin/reload")
async def admin_reload() -> Dict[str, Any]:
    """Reload model bundle + dataset without restarting the server (local/dev only)."""
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    log.info("Admin reload: reloading models + dataset...")
    state["bundle"] = load_inference_bundle(CFG_MODELS_DIR)
    state["model_metadata_path"], state["model_metadata"] = _load_model_metadata(CFG_MODELS_DIR)
    expected_features = _resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))

    # Dataset reload
    try:
        state["dataset"] = load_dataset_df(CFG_DATA_DIR, expected_features=expected_features)
    except Exception:
        ds_path = _find_latest_dataset_csv(CFG_DATA_DIR)
        if ds_path is None:
            raise
        state["dataset"] = pd.read_csv(ds_path)

    state["dataset_path"] = str(_find_latest_dataset_csv(CFG_DATA_DIR) or "")

    _validate_feature_schema(state["bundle"], state["dataset"], metadata=state.get("model_metadata"))
    state["service"] = PredictionService(state["bundle"], state["dataset"])

    return {
        "reloaded": True,
        "models_dir": str(CFG_MODELS_DIR),
        "metadata_path": str(state.get("model_metadata_path") or ""),
        "dataset_path": state.get("dataset_path") or "",
    }

@app.post("api/admin/retrain")
async def admin_retrain(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Train models on the newest dataset and hot-reload them (local/dev only).

    NOTE: training can take minutes and will block this request while it runs.
    """
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    dataset_path = payload.get("dataset_path") or (state.get("dataset_path") or "")
    if not dataset_path:
        ds = _find_latest_dataset_csv(CFG_DATA_DIR)
        dataset_path = str(ds) if ds else ""

    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=400, detail=f"dataset_path not found: {dataset_path}")

    out_dir = payload.get("out_dir") or str(CFG_MODELS_DIR)
    log.info("Admin retrain: dataset=%s out_dir=%s", dataset_path, out_dir)

    # Import lazily to avoid import cycles during normal startup
    try:
        from .train_models import main as train_main
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not import training script: {e}")

    train_main(data_path=str(dataset_path), out_dir=str(out_dir))

    # Reload freshly trained artifacts
    await admin_reload()

    md = state.get("model_metadata") or {}
    return {
        "trained": True,
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "metadata_path": str(state.get("model_metadata_path") or ""),
        "dataset_hash": md.get("dataset_hash"),
        "training_timestamp_utc": md.get("training_timestamp_utc"),
        "production_ready": md.get("production_ready"),
    }

@app.get("api/status/overview", response_model=StatusOverviewResponse)
async def get_status_overview():
    h = await health()
    dataset_info = {
        "rows": len(state["dataset"]) if state["dataset"] is not None else 0,
        "features": (len(_resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))) if state["bundle"] else 0),
    }
    with _prediction_history_lock:
        history_metrics = {
            "total_predictions": len(prediction_history_entries),
            "win_rate": None,
            "note": "win_rate requires actual outcomes",
        }
    return StatusOverviewResponse(health=h, dataset=dataset_info, history=history_metrics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
