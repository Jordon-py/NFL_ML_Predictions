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
from datetime import datetime, timezone, timedelta, date
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import subprocess
import asyncio
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
    ScoreEntry,
    SeasonContextResponse,
    ExplainPredictionRequest,
    ExplainPredictionResponse,
    ChatRequest,
    ChatResponse,
    AdminRetrainRequest,
)
from .pipeline_models import TrainingExecutionResult
from .services.prediction_service import PredictionService
from .services.inference_row import build_model_input_row
from .config import DATA_DIR as CFG_DATA_DIR, MODELS_DIR as CFG_MODELS_DIR, resolve_cors, TRUTHY
from .utils.team_codes import normalize_team_code
from .main_helpers import (
    InferenceBundle,
    load_inference_bundle,
    load_dataset_df,
    get_schedule,
    select_next_week_rows,
    get_team_meta,
    parse_kickoff,
    _pick_col,
    _HOME_COLS,
    _AWAY_COLS,
    _STADIUM_COLS,
    _SEASON_COLS,
    _WEEK_COLS,
    load_prediction_history,
)
from .prediction_store import (
    append_prediction_record,
    build_prediction_user_context,
    get_prediction_history,
    get_prediction_history_count,
)
from .ollama.llm_ollama import explain_prediction as llm_explain_prediction, chat_messages as llm_chat_messages
from .sqlite_store import get_game_scores, upsert_game_scores
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
    "init_error": None,
    "model_metadata": None,
    "model_metadata_path": None,
    "dataset_path": None,
    "team_logos": None,
}

ADMIN_ENABLED = os.getenv("ENABLE_ADMIN", "false").strip().lower() in TRUTHY
SCORE_SYNC_TZ = os.getenv("SCORE_SYNC_TZ", "UTC")
SCORE_SYNC_DAYS = os.getenv("SCORE_SYNC_DAYS", "sun,mon,thu")
SCORE_SYNC_ENABLED = os.getenv("DISABLE_SCORE_SYNC", "false").strip().lower() not in TRUTHY
SCORE_SYNC_LOOKBACK = max(0, min(int(os.getenv("SCORE_SYNC_LOOKBACK", "2")), 5))
SCOREBOARD_API_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
LOCAL_SCORE_DIR = Path(__file__).resolve().parent / "data" / "scores"


def get_logos(home_team, away_team):
    team_logos = _get_team_meta_map()
    home_logo = team_logos.get(home_team)
    away_logo = team_logos.get(away_team)
    return home_logo, away_logo

def _normalize_team_code(value: str) -> str:
    """Normalize team identifiers to canonical abbreviations."""
    return normalize_team_code(value)


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

def _clean_s(val: Any) -> Optional[str]:
    """Convert nan or empty values to None for Pydantic Optional[str]."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
        return None
    return str(val).strip()


def _derive_season_phase(df_next: pd.DataFrame) -> tuple[str, str]:
    """
    Infer broad NFL season phase from next-slate rows.
    Returns (phase, human_label) where phase is one of:
    - in_season
    - postseason
    - offseason
    """
    if isinstance(df_next, pd.DataFrame) and not df_next.empty:
        game_type_col = _pick_col(df_next, ["game_type", "season_type", "type"])
        if game_type_col:
            game_types = (
                df_next[game_type_col]
                .dropna()
                .astype(str)
                .str.upper()
                .str.strip()
                .unique()
                .tolist()
            )
            has_post = any(gt not in {"REG", "R"} for gt in game_types)
            if has_post:
                return ("postseason", "Postseason")
        return ("in_season", "Regular Season")

    month = datetime.now(timezone.utc).month
    # Typical NFL offseason window: Feb-Jul (inclusive)
    if 2 <= month <= 7:
        return ("offseason", "Offseason")
    # Aug with no schedule is effectively preseason prep for users.
    if month == 8:
        return ("offseason", "Preseason Build-Up")
    return ("offseason", "Offseason")


# -------------------------------------
# FUNCTIONS -----
# -------------------------------------
def _build_game_id(season: int, week: int, home_team: str, away_team: str) -> str:
    """Build a stable game identifier used across schedule, prediction, and history views."""

    return f"{int(season)}-{int(week)}-{str(home_team).strip().upper()}-{str(away_team).strip().upper()}"

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

def _build_next_slate_games(season: int | None = None) -> tuple[list[ScheduleEntry], pd.DataFrame, int, int]:
    """Build the next available slate in the canonical schedule shape used by the UI."""

    df = get_schedule(season=season)
    df_next, use_season, use_week = select_next_week_rows(df)
    team_meta = _get_team_meta_map()

    games = _map_schedule_entries(df_next, use_season, use_week, team_meta)

    return games, df_next, int(use_season), int(use_week)

def _parse_score_value(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _build_scoreboard_entries(events: List[Dict[str, Any]]) -> list[Dict[str, object]]:
    entries: list[Dict[str, object]] = []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        week_data = event.get("week", {})
        week_value = competition.get("week") or week_data.get("number")
        season_value = (
            competition.get("season", {}).get("year")
            or event.get("season", {}).get("year")
            or datetime.now(timezone.utc).year
        )
        competitors = competition.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away or week_value is None:
            continue
        home_team = _normalize_team_code(
            home.get("team", {}).get("abbreviation") or home.get("team", {}).get("shortDisplayName")
        )
        away_team = _normalize_team_code(
            away.get("team", {}).get("abbreviation") or away.get("team", {}).get("shortDisplayName")
        )
        home_score = _parse_score_value(home.get("score"))
        away_score = _parse_score_value(away.get("score"))
        status = event.get("status", {})
        status_type = status.get("type", {})
        state = str(status_type.get("state", "")).lower()
        is_final = (
            state in {"post", "final", "complete"}
            or status_type.get("completed") is True
            or str(status_type.get("name", "")).lower() == "final"
        )
        if not is_final:
            continue

        game_id = _build_game_id(season_value, week_value, home_team, away_team)
        entries.append(
            {
                "game_id": game_id,
                "season": season_value,
                "week": week_value,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "status": status_type.get("description") or status_type.get("name") or "Final",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return entries

def _map_schedule_entries(
    df: pd.DataFrame,
    season_value: int,
    week_value: int,
    team_meta: Dict[str, Dict[str, str]],
) -> list[ScheduleEntry]:
    home_col = _pick_col(df, _HOME_COLS)
    away_col = _pick_col(df, _AWAY_COLS)
    stadium_col = _pick_col(df, _STADIUM_COLS)
    games: list[ScheduleEntry] = []
    for _, row in df.iterrows():
        home = _normalize_team_code(row.get(home_col, "") if home_col else row.get("home", ""))
        away = _normalize_team_code(row.get(away_col, "") if away_col else row.get("away", ""))
        if not home or not away:
            continue

        home_info = team_meta.get(home, {})
        away_info = team_meta.get(away, {})
        stadium = row.get(stadium_col, "") if stadium_col else row.get("stadium", "")

        games.append(
            ScheduleEntry(
                season=int(season_value),
                week=int(week_value),
                kickoff=parse_kickoff(row),
                home_team=home,
                away_team=away,
                game_id=_build_game_id(season_value, week_value, home, away),
                home_abbr=home,
                away_abbr=away,
                home_logo=home_info.get("logoUrl"),
                away_logo=away_info.get("logoUrl"),
                home_name=home_info.get("name") or str(row.get("home_team_name", "")) or home,
                away_name=away_info.get("name") or str(row.get("away_team_name", "")) or away,
                stadium=stadium,
            )
        )
    return games

def _build_schedule_for_week(season: int | None = None, week: int | None = None) -> tuple[list[ScheduleEntry], pd.DataFrame, int, int]:
    """Return a specific week vs season slate (falls back to next-week when not provided)."""

    if week is None:
        return _build_next_slate_games(season=season)

    df = get_schedule(season=season)
    if df is None or df.empty:
        return [], pd.DataFrame(), season or datetime.now(timezone.utc).year, week

    season_col = _pick_col(df, _SEASON_COLS)
    week_col = _pick_col(df, _WEEK_COLS)

    filtered = df.copy()
    use_season = season or datetime.now(timezone.utc).year
    if season_col:
        if season is not None:
            filtered = filtered[pd.to_numeric(filtered[season_col], errors="coerce") == season]
            use_season = season
        else:
            seasons = pd.to_numeric(filtered[season_col], errors="coerce").dropna().astype(int)
            if not seasons.empty:
                use_season = int(seasons.max())
                filtered = filtered[pd.to_numeric(filtered[season_col], errors="coerce") == use_season]

    if week_col:
        filtered = filtered[pd.to_numeric(filtered[week_col], errors="coerce") == week]
    else:
        filtered = filtered.iloc[0:0]

    team_meta = _get_team_meta_map()
    games = _map_schedule_entries(filtered, use_season, week, team_meta)
    return games, filtered, use_season, week

async def _fetch_remote_scores_for_date(date_str: str) -> list[Dict[str, object]]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{SCOREBOARD_API_URL}?dates={date_str}")
        resp.raise_for_status()
        payload = resp.json()
    events = payload.get("events") or []
    return _build_scoreboard_entries(events)

def _load_local_scores_for_date(date_str: str) -> list[Dict[str, object]]:
    path = LOCAL_SCORE_DIR / f"{date_str}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to parse local scoreboard for %s: %s", date_str, exc)
        return []
    return _build_scoreboard_entries(data.get("events") or [])

def _score_sync_dates() -> list[date]:
    today = datetime.now(timezone.utc).date()
    return [today - timedelta(days=i) for i in range(0, SCORE_SYNC_LOOKBACK + 1)]

async def _sync_scores_job() -> None:
    entries: list[Dict[str, object]] = []
    target_dates = _score_sync_dates()
    for target in target_dates:
        date_str = target.strftime("%Y%m%d")
        try:
            remote_entries = await _fetch_remote_scores_for_date(date_str)
            if remote_entries:
                entries.extend(remote_entries)
                continue
        except Exception as exc:
            log.debug("Score sync remote fetch failed for %s: %s", date_str, exc)

        local_entries = _load_local_scores_for_date(date_str)
        if local_entries:
            entries.extend(local_entries)
    if entries:
        upsert_game_scores(entries)
        log.info("Score sync ingested %d entries", len(entries))

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

def _filter_expected_features(features: list[str]) -> list[str]:
    """Drop empty/duplicate names and pandas index placeholders (e.g., 'Unnamed: 0')."""
    if not features:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for name in features:
        s = str(name).strip()
        if not s:
            continue
        if s.lower().startswith("unnamed:"):
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned

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

    latest_manifest = root / "latest_dataset.json"
    if latest_manifest.exists():
        try:
            payload = json.loads(latest_manifest.read_text(encoding="utf-8"))
            clean_dataset_path = payload.get("clean_dataset_path")
            if clean_dataset_path:
                manifest_dataset = Path(clean_dataset_path)
                if manifest_dataset.exists():
                    return manifest_dataset
        except Exception as exc:
            log.warning("Could not read latest_dataset.json from %s: %s", latest_manifest, exc)

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
















# ---------------------------
# Lifespan
# ---------------------------

def _resolve_expected_features(bundle: Any, metadata: Dict[str, Any] | None = None) -> list[str]:
    """Resolve the *raw* input feature columns expected by the preprocessor/model."""
    pre = getattr(bundle, "preprocessor", None)
    features_in = getattr(pre, "feature_names_in_", None)
    if features_in is not None:
        expected = _filter_expected_features([str(x) for x in list(features_in)])
        if expected:
            return expected

    # Prefer explicit lists from training metadata (stable across sklearn versions)
    for cand in (
        (metadata or {}).get("feature_names"),
        getattr(bundle, "feature_names", None),
    ):
        if isinstance(cand, (list, tuple)) and len(cand) > 0:
            expected = _filter_expected_features([str(x) for x in cand if x is not None])
            if expected:
                return expected

    # Fall back to 'raw_feature_columns' (either list or {"numeric","categorical"})
    raw = getattr(bundle, "raw_feature_columns", None)
    if metadata and "raw_feature_columns" in metadata:
        raw = metadata.get("raw_feature_columns")
    return _filter_expected_features(_flatten_raw_feature_columns(raw))

def _materialize_preprocessor_placeholders(bundle: Any, dataset: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic placeholder columns that legacy sklearn artifacts still expect."""

    if dataset is None or dataset.empty:
        return dataset

    pre = getattr(bundle, "preprocessor", None)
    features_in = getattr(pre, "feature_names_in_", None)
    if features_in is None:
        return dataset

    for name in [str(x).strip() for x in list(features_in)]:
        if not name or name in dataset.columns:
            continue
        if name.lower().startswith("unnamed:"):
            dataset[name] = np.arange(len(dataset), dtype=float)
    return dataset

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


def _resolve_user_context(request: Request):
    """Use the signed-in frontend identity when persisting local predictions."""

    return build_prediction_user_context(request.headers.get("X-User-Id"))


def _sync_app_state(app: FastAPI) -> None:
    """Keep `app.state` aligned with the module-level runtime cache."""

    bundle = state.get("bundle")
    app.state.dataset = state.get("dataset")
    app.state.model_metadata = state.get("model_metadata") or {}
    app.state.model_metadata_path = str(state.get("model_metadata_path") or "")
    app.state.dataset_path = state.get("dataset_path") or ""
    app.state.models = {
        "preprocessor": getattr(bundle, "preprocessor", None) if bundle is not None else None,
        "home_model": getattr(bundle, "home_model", None) if bundle is not None else None,
        "away_model": getattr(bundle, "away_model", None) if bundle is not None else None,
        "hist_win_clf": getattr(bundle, "hist_win_clf", None) if bundle is not None else None,
        "models_dir": str(CFG_MODELS_DIR),
    }
    app.state.service = state.get("service")
    app.state.team_logos = state.get("team_logos") or {}
    app.state.started_at = state.get("started_at") or datetime.now(timezone.utc).isoformat()


def _run_preprocessor_smoke_test(bundle: Any, dataset: pd.DataFrame, expected_features: list[str]) -> None:
    """Raise early when sklearn artifacts and the current runtime are incompatible."""

    pre = getattr(bundle, "preprocessor", None)
    if pre is None or not expected_features:
        return
    feature_names_in = getattr(pre, "feature_names_in_", None)
    required_columns = list(feature_names_in) if feature_names_in is not None else expected_features
    sample = dataset.reindex(columns=required_columns).head(1).copy()
    pre.transform(sample)

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler(timezone=SCORE_SYNC_TZ)
    job_added = False
    # Startup: Load models and dataset
    try:
        log.info("Starting up: Loading model bundle and dataset...")
        state["started_at"] = datetime.now(timezone.utc).isoformat()
        state["bundle"] = load_inference_bundle(CFG_MODELS_DIR)
        state["model_metadata_path"], state["model_metadata"] = _load_model_metadata(CFG_MODELS_DIR)
        expected_features = _resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))
        # Dataset: prefer helper loader; fall back to newest CSV under DATA_DIR
        try:
            state["dataset"] = load_dataset_df(CFG_DATA_DIR, expected_features=expected_features)
        except Exception:
            ds_path = _find_latest_dataset_csv(CFG_DATA_DIR)
            if ds_path is None:
                raise
            state["dataset"] = pd.read_csv(ds_path)
        state["dataset"] = _materialize_preprocessor_placeholders(state["bundle"], state["dataset"])
        state["dataset_path"] = str(_find_latest_dataset_csv(CFG_DATA_DIR) or "")
        _validate_feature_schema(state["bundle"], state["dataset"], metadata=state.get("model_metadata"))
        state["service"] = PredictionService(state["bundle"], state["dataset"])
        state["init_error"] = None

        # Smoke-test the fitted preprocessor to fail fast on sklearn version mismatches.
        # Without this, the app can "start" but crash on the first /predict call.
        try:
            _run_preprocessor_smoke_test(state["bundle"], state["dataset"], expected_features)
        except Exception as exc:
            msg = (
                "Model preprocessing smoke test failed (likely scikit-learn version mismatch for joblib artifacts). "
                f"Error: {exc}"
            )
            log.error(msg, exc_info=True)
            state["service"] = None
            state["init_error"] = msg

        _get_team_meta_map()
        _sync_app_state(app)
        log.info("Startup complete: Models and dataset ready.")

        if SCORE_SYNC_ENABLED:
            scheduler.add_job(
                _sync_scores_job,
                trigger="cron",
                id="score-sync",
                day_of_week=SCORE_SYNC_DAYS,
                hour=23,
                minute=45,
                timezone=SCORE_SYNC_TZ,
                replace_existing=True,
                coalesce=True,
                max_instances=1,
            )
            scheduler.start()
            job_added = True
            await _sync_scores_job()
    except Exception as e:
        msg = f"Startup failed: {e}"
        log.error(msg, exc_info=True)
        state["service"] = None
        state["init_error"] = msg
        # Ensure optional app.state bindings exist even when startup fails.
        state["started_at"] = state.get("started_at") or datetime.now(timezone.utc).isoformat()
        _sync_app_state(app)

    try:
        yield
    finally:
        if job_added:
            scheduler.shutdown(wait=False)

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






def _require_ready() -> PredictionService:
    if state["service"] is None:
        raise HTTPException(
            status_code=503,
            detail=state.get("init_error") or "Prediction engine not initialized.",
        )
    return state["service"]

# ---------------------------
# API ROUTES
# ---------------------------

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health():
    """System health check."""
    ready = state.get("service") is not None
    status = "healthy" if ready else "unhealthy"
    mode = "ml-inference" if ready else "initializing"
    reason = "models_loaded" if ready else (state.get("init_error") or "prediction engine not initialized")
    return HealthResponse(status=status, mode=mode, reason=reason)

@app.get("/api/status/overview", response_model=StatusOverviewResponse)
async def get_status_overview(request: Request):
    """High-level system overview."""
    h = await health()
    dataset_info = {
        "rows": len(state["dataset"]) if state["dataset"] is not None else 0,
        "features": (len(_resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))) if state["bundle"] else 0),
    }
    history_metrics = {
        "total_predictions": get_prediction_history_count(_resolve_user_context(request)),
        "win_rate": None,
        "note": "win_rate requires actual outcomes",
    }
    return StatusOverviewResponse(health=h, dataset=dataset_info, history=history_metrics)

@app.get("/api/status/models")
async def get_status_models() -> Dict[str, Any]:
    """Detailed model and dataset metadata."""
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

@app.get("/api/season/context", response_model=SeasonContextResponse)
async def get_season_context(season: int | None = None) -> SeasonContextResponse:
    """
    Return schedule-aware season context so clients can render
    in-season/postseason/offseason UX without guessing.
    """
    games, df_next, use_season, use_week = _build_next_slate_games(season=season)

    phase, label = _derive_season_phase(df_next)
    next_kickoff = None
    kickoff_values = [game.kickoff for game in games if game.kickoff is not None]
    if kickoff_values:
        next_kickoff = min(kickoff_values)

    games_in_next_window = len(games)
    if phase == "in_season" and use_week is not None:
        message = f"{games_in_next_window} game(s) available for Week {int(use_week)}."
    elif phase == "postseason":
        message = f"{games_in_next_window} postseason game(s) available in the next slate."
    else:
        message = "No live weekly slate is available right now."

    return SeasonContextResponse(
        phase=phase,
        label=label,
        message=message,
        current_season=int(use_season),
        display_week=int(use_week) if phase != "offseason" else None,
        games_in_next_window=games_in_next_window,
        next_kickoff=next_kickoff,
        generated_at=datetime.now(timezone.utc),
    )

@app.get("/api/schedule/next-week", response_model=ScheduleResponse)
@app.get("/schedule/next-week", response_model=ScheduleResponse)
async def get_next_week_schedule(season: int | None = None) -> ScheduleResponse:
    games, _, _, _ = _build_next_slate_games(season=season)
    return ScheduleResponse(games=games)


@app.get("/api/schedule/week", response_model=ScheduleResponse)
@app.get("/schedule/week", response_model=ScheduleResponse)
async def get_schedule_for_week(season: int | None = None, week: int | None = None) -> ScheduleResponse:
    games, _, _, _ = _build_schedule_for_week(season=season, week=week)
    return ScheduleResponse(games=games)


@app.get("/api/scores", response_model=List[ScoreEntry])
@app.get("/scores", response_model=List[ScoreEntry])
async def get_scores(season: int | None = None, week: int | None = None):
    return get_game_scores(season=season, week=week)

@app.get("/api/teams/logos", response_model=TeamLogosResponse)
@app.get("/teams/logos", response_model=TeamLogosResponse)
async def get_team_logos() -> TeamLogosResponse:
    return TeamLogosResponse(teams=_get_team_meta_map())

@app.get("/debug")
@app.get("/api/debug")
async def debug() -> Dict[str, Any]:
    """In-depth debugging information."""
    dataset = state["dataset"]
    rows = int(len(dataset)) if dataset is not None else 0
    cols = int(dataset.shape[1]) if dataset is not None else 0
    cors_restrict = os.getenv("RESTRICT_CORS", "false").strip().lower() in TRUTHY
    return {
        "status": "ok" if state["service"] else ("error" if state.get("init_error") else "initializing"),
        "init_error": state.get("init_error"),
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
        "cors_origins": cors_origins,
        "cors_origin_regex": cors_origin_regex,
        "restrict_cors": cors_restrict,
    }

@app.post("/debug/predict-input")
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
        exact_match_index=getattr(service, "_exact_match_index", None),
        impute_medians=getattr(service, "_impute_medians", None),
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

@app.post("/predict", response_model=UnifiedPredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    service = _require_ready()
    res = service.predict(req)
    payload = _build_prediction_payload(req, res)
    append_prediction_record(_resolve_user_context(request), req, payload)
    return payload

@app.get("/predict/next-week")
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

@app.post("/predict/explain", response_model=ExplainPredictionResponse)
async def explain(payload: ExplainPredictionRequest = Body(...)) -> ExplainPredictionResponse:
    payload_data = payload.model_dump(exclude_none=True)
    pred = _extract_prediction_payload(payload_data)
    home_team = payload.home_team or pred.get("home_team")
    away_team = payload.away_team or pred.get("away_team")
    season_raw = payload.season or pred.get("season")
    week_raw = payload.week or pred.get("week")

    try:
        season = int(season_raw) if season_raw is not None else None
        week = int(week_raw) if week_raw is not None else None
    except (TypeError, ValueError):
        season = None
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

    for k, v in [("home_team", home_team), ("away_team", away_team), ("season", season), ("week", week)]:
        if v is not None:
             pred[k] = v

    game_id = pred.get("game_id")
    if not game_id and season is not None and week is not None and home_team and away_team:
        game_id = _build_game_id(season, week, home_team, away_team)

    llm_result = await llm_explain_prediction(pred)
    return ExplainPredictionResponse(
        game_id=game_id,
        used_llm=bool(llm_result.get("used_llm")),
        llm_model=llm_result.get("model"),
        explanation=llm_result.get("explanation", ""),
        bullets=llm_result.get("bullets", []) or [],
        caveats=llm_result.get("caveats", []) or [],
        error=llm_result.get("error"),
    )

@app.post("/llm/chat", response_model=ChatResponse)
async def llm_chat(payload: ChatRequest = Body(...)) -> ChatResponse:
    messages = [message.model_dump() for message in payload.messages]
    prediction = payload.prediction
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
    return ChatResponse(
        reply=reply,
        used_llm=bool(result.get("used_llm")),
        llm_model=result.get("model"),
        error=result.get("error"),
    )

@app.get("/api/history", response_model=HistoryResponse)
@app.get("/history", response_model=HistoryResponse)
async def get_history(request: Request, limit: int = 100):
    return get_prediction_history(_resolve_user_context(request), limit=limit)


@app.get("/status/models")
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

@app.post("/admin/reload")
async def admin_reload(request: Request) -> Dict[str, Any]:
    """Reload model bundle + dataset without restarting the server (local/dev only)."""
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=403, detail="Admin disabled")

    log.info("Admin reload requested.")
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
    state["dataset"] = _materialize_preprocessor_placeholders(state["bundle"], state["dataset"])

    state["dataset_path"] = str(_find_latest_dataset_csv(CFG_DATA_DIR) or "")

    _validate_feature_schema(state["bundle"], state["dataset"], metadata=state.get("model_metadata"))
    _run_preprocessor_smoke_test(state["bundle"], state["dataset"], expected_features)
    state["service"] = PredictionService(state["bundle"], state["dataset"])
    state["init_error"] = None
    _sync_app_state(request.app)

    return {"reloaded": True, "models_dir": str(CFG_MODELS_DIR)}

@app.post("/admin/retrain", response_model=TrainingExecutionResult)
async def admin_retrain(
    request: Request,
    payload: AdminRetrainRequest | None = Body(default=None),
) -> TrainingExecutionResult:
    """Train models on the newest dataset and hot-reload them (local/dev only).

    NOTE: training can take minutes and will block this request while it runs.
    """
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=403, detail="Admin disabled")

    payload = payload or AdminRetrainRequest()

    dataset_path = payload.dataset_path or (state.get("dataset_path") or "")
    if not dataset_path:
         ds = _find_latest_dataset_csv(CFG_DATA_DIR)
         dataset_path = str(ds) if ds else ""

    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=400, detail="dataset path not found")

    out_dir = payload.out_dir or str(CFG_MODELS_DIR)
    log.info("Admin retrain: dataset=%s out_dir=%s", dataset_path, out_dir)

    # Import lazily to avoid import cycles during normal startup
    try:
        from .train_models import main as train_main
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not import training script: {e}")

    training_result = train_main(
        data_path=str(dataset_path),
        out_dir=str(out_dir),
        force_retrain=payload.force,
    )

    if training_result.trained:
        # Reload freshly trained artifacts only when a new run was actually produced.
        await admin_reload(request)

    return training_result

@app.get("/status/overview", response_model=StatusOverviewResponse)
async def get_status_overview(request: Request):
    h = await health()
    dataset_info = {
        "rows": len(state["dataset"]) if state["dataset"] is not None else 0,
        "features": (len(_resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))) if state["bundle"] else 0),
    }
    history_metrics = {
        "total_predictions": get_prediction_history_count(_resolve_user_context(request)),
        "win_rate": None,
        "note": "win_rate requires actual outcomes",
    }
    return StatusOverviewResponse(health=h, dataset=dataset_info, history=history_metrics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
