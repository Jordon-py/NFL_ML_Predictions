# ==========================================
# File: backend/routes.py
# Role: HTTP route handlers for backend endpoints.
# Input Data: HTTP requests.
# Output Data: JSON responses.
# Dependencies: __future__, os, datetime, pathlib
# Notes: Thin controller layer with schedule CSV header normalization.
# ==========================================

from __future__ import annotations

"""
File: backend/routes.py

Routes (APIRouter) for NFL ML Predictions
========================================

File Metrics:
- Purpose: Define the HTTP contract (endpoints) while staying thin.
- Design: Read state from request.app.state (models/dataset caches).
- Stability: Endpoints are “frontend-friendly” and return predictable shapes.

Key Concepts:
- Context endpoints:
  - GET /schedule/next-week  -> games list
  - GET /teams/logos         -> team logos map for UI
- Cognitive endpoints:
  - POST /predict            -> single matchup prediction
  - GET /predict/next-week   -> batch prediction for next week schedule (uses /schedule/next-week)

Learning Checkpoints:
- You should be able to point to where schedule is loaded.
- You should be able to point to where models are accessed.
- You should be able to explain pipeline vs non-pipeline predict flow.

Tips & Next Steps:
- To force offline schedule, set OFFLINE_MODE=true and SCHEDULE_CSV_PATH=.../Nfl_schedule_2025.csv
- If prediction fails, hit /debug for quick diagnosis.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.config import TRUTHY, load_schedule_data_safe

router = APIRouter()
log = logging.getLogger(__name__)


# -------------------------
# Schemas
# -------------------------
class HealthResponse(BaseModel):
    status: str
    reason: str
    started_at: Optional[str] = None

class DebugResponse(BaseModel):
    started_at: Optional[str] = None
    models_loaded: bool
    models_dir: Optional[str] = None
    dataset_rows: int
    dataset_cols: int
    dataset_sample_cols: List[str]

class PredictionRequest(BaseModel):
    home_team: str = Field(..., examples=["KC"])
    away_team: str = Field(..., examples=["BUF"])
    season: int = Field(..., examples=[2025])
    week: int = Field(..., examples=[15])

class PredictionResponse(BaseModel):
    home_score: float
    away_score: float
    point_diff: float
    home_win_probability: float
    away_win_probability: float
    win_classifier_used: bool
    prediction_source: str  # "dataset_row" or "empty_row_fallback"

class ScheduleGame(BaseModel):
    season: int
    week: int
    kickoff: Optional[datetime] = None
    home_team: str
    away_team: str
    game_id: Optional[str] = None
    stadium: Optional[str] = None

class NextWeekGamesResponse(BaseModel):
    games: List[ScheduleGame]

class TeamMeta(BaseModel):
    logoUrl: str
    name: Optional[str] = None

class TeamLogosResponse(BaseModel):
    teams: Dict[str, TeamMeta]

class PredictedGame(BaseModel):
    season: int
    week: int
    kickoff: Optional[datetime] = None
    home_team: str
    away_team: str
    game_id: Optional[str] = None
    prediction: PredictionResponse

class NextWeekPredictionsResponse(BaseModel):
    games: List[PredictedGame]


# -------------------------
# Helpers
# -------------------------
def _is_pipeline(obj: Any) -> bool:
    # Detect direct sklearn `Pipeline` instances, and common wrappers that
    # contain pipelines as their internal estimator (e.g.,
    # `CalibratedClassifierCV(estimator=Pipeline(...))`). If a model is a
    # pipeline or wraps a pipeline, it expects the raw feature DataFrame and
    # will perform its own preprocessing.
    if hasattr(obj, "named_steps"):
        return True
    # wrappers: check common attributes that hold the base estimator
    if hasattr(obj, "estimator") and hasattr(getattr(obj, "estimator"), "named_steps"):
        return True
    if hasattr(obj, "base_estimator") and hasattr(getattr(obj, "base_estimator"), "named_steps"):
        return True
    return False

def _safe_logistic_from_diff(diff: float) -> float:
    return float(1.0 / (1.0 + np.exp(-0.3 * diff)))

_SEASON_COLS = ["season", "season_year", "year"]
_WEEK_COLS = ["week", "week_num", "week_number"]
_HOME_COLS = ["home_team", "home", "home_abbr", "home_code"]
_AWAY_COLS = ["away_team", "away", "away_abbr", "away_code"]
_GAME_ID_COLS = ["game_id", "gameid", "game_key", "gameId"]
_STADIUM_COLS = ["stadium", "venue", "stadium_name"]

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(c.lower())
        if hit:
            return hit
    return None

def _current_season_week() -> Tuple[int, int]:
    try:
        import nflreadpy as nfl  # optional dependency
        return int(nfl.get_current_season()), int(nfl.get_current_week())
    except Exception:
        now = datetime.now(timezone.utc)
        return int(now.year), 1

def _resolve_schedule_candidates(path_str: str, backend_dir: Path, repo_root: Path) -> List[Path]:
    p = Path(path_str)
    if p.is_absolute():
        return [p]
    return [backend_dir / p, repo_root / p]

def _load_schedule_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as exc:
        log.warning("Failed to read schedule CSV %s: %s", csv_path, exc)
        return None

def get_schedule(season: Optional[int] = None) -> pd.DataFrame:
    """
    Load an NFL schedule DataFrame with reasonable fallbacks.

    Order of precedence:
      1) nflreadpy (if available and OFFLINE_MODE=false)
      2) SCHEDULE_PATH environment override
      3) common repo locations
    """
    use_season = season or _current_season_week()[0]
    offline = os.getenv("OFFLINE_MODE", "false").strip().lower() in TRUTHY

    if not offline:
        df = load_schedule_data_safe(use_season)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.columns = [c.strip() for c in df.columns]
            return df

    backend_dir = Path(__file__).resolve().parent
    repo_root = backend_dir.parent
    candidates: List[Path] = []

    csv_path = os.getenv("SCHEDULE_PATH")
    if csv_path:
        candidates.extend(_resolve_schedule_candidates(csv_path, backend_dir, repo_root))

    candidates.extend(
        [
            backend_dir / "data" / f"Nfl_schedule_{use_season}.csv",
            backend_dir / f"Nfl_schedule_{use_season}.csv",
            repo_root / "data" / f"Nfl_schedule_{use_season}.csv",
            repo_root / f"Nfl_schedule_{use_season}.csv",
        ]
    )

    for c in candidates:
        if c.exists():
            df = _load_schedule_csv(c)
            if isinstance(df, pd.DataFrame):
                return df

    return pd.DataFrame()

def _infer_next_week(schedule_df: pd.DataFrame) -> Tuple[int, int]:
    season_default, week_default = _current_season_week()
    if schedule_df is None or schedule_df.empty:
        return (season_default, week_default)

    season_col = _pick_col(schedule_df, _SEASON_COLS)
    week_col = _pick_col(schedule_df, _WEEK_COLS)

    season = season_default
    df_use = schedule_df
    if season_col:
        seasons = pd.to_numeric(schedule_df[season_col], errors="coerce").dropna().astype(int)
        if not seasons.empty:
            season = season_default if season_default in seasons.values else int(seasons.max())
        df_use = schedule_df[pd.to_numeric(schedule_df[season_col], errors="coerce") == season]

    if week_col:
        weeks = pd.to_numeric(df_use[week_col], errors="coerce").dropna().astype(int)
        if weeks.empty:
            return (season, week_default)

        now = datetime.utcnow()
        kickoff_series = df_use.apply(_parse_kickoff, axis=1)
        if kickoff_series.notna().any():
            future_mask = kickoff_series.notna() & (kickoff_series >= now)
            if future_mask.any():
                future_weeks = pd.to_numeric(df_use.loc[future_mask, week_col], errors="coerce").dropna().astype(int)
                if not future_weeks.empty:
                    return (season, int(future_weeks.min()))

        candidate_weeks = weeks[weeks >= week_default]
        if not candidate_weeks.empty:
            return (season, int(candidate_weeks.min()))

        return (season, int(weeks.max()))

    return (season, week_default)

def _select_next_week_rows(schedule_df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    season, week = _infer_next_week(schedule_df)
    df_use = schedule_df.copy() if isinstance(schedule_df, pd.DataFrame) else pd.DataFrame()

    season_col = _pick_col(df_use, _SEASON_COLS)
    week_col = _pick_col(df_use, _WEEK_COLS)

    if season_col:
        df_use = df_use[pd.to_numeric(df_use[season_col], errors="coerce") == season]
    if week_col:
        df_use = df_use[pd.to_numeric(df_use[week_col], errors="coerce") == week]

    return df_use, season, week

def _find_game_row(dataset: pd.DataFrame, home: str, away: str, season: int, week: int) -> Optional[pd.Series]:
    if dataset is None or dataset.empty:
        return None

    needed = {"home_team", "away_team", "season", "week"}
    if not needed.issubset(set(dataset.columns)):
        return None

    m = (
        (dataset["home_team"].astype(str) == str(home)) &
        (dataset["away_team"].astype(str) == str(away)) &
        (pd.to_numeric(dataset["season"], errors="coerce") == season) &
        (pd.to_numeric(dataset["week"], errors="coerce") == week)
    )
    hit = dataset.loc[m]
    if hit.empty:
        return None
    return hit.iloc[0]

def _build_raw_row(pre: Any, dataset_row: Optional[pd.Series], req: PredictionRequest) -> pd.DataFrame:
    raw_cols = list(getattr(pre, "feature_names_in_", []))
    if not raw_cols:
        raise RuntimeError("Preprocessor has no feature_names_in_ (cannot build raw row)")

    base = dataset_row.to_dict() if dataset_row is not None else {}
    base["home_team"] = req.home_team
    base["away_team"] = req.away_team
    base["season"] = req.season
    base["week"] = req.week

    return pd.DataFrame([{c: base.get(c, np.nan) for c in raw_cols}], columns=raw_cols)

def _predict(models: Dict[str, Any], dataset: pd.DataFrame, req: PredictionRequest) -> PredictionResponse:
    pre = models.get("preprocessor")
    home_m = models.get("home_model")
    away_m = models.get("away_model")
    win_m = models.get("win_clf")

    if pre is None or home_m is None or away_m is None:
        raise RuntimeError("Models not fully loaded (need preprocessor/home/away)")

    row = _find_game_row(dataset, req.home_team, req.away_team, req.season, req.week)
    source = "dataset_row" if row is not None else "empty_row_fallback"

    X_raw = _build_raw_row(pre, row, req)

    X_tx = None
    if (not _is_pipeline(home_m)) or (not _is_pipeline(away_m)) or (win_m is not None and not _is_pipeline(win_m)):
        X_safe = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_tx = pre.transform(X_safe)

    def score(m: Any) -> float:
        Xin = X_raw if _is_pipeline(m) else X_tx
        if Xin is None:
            raise RuntimeError("Missing transformed features for non-pipeline model")
        return float(m.predict(Xin)[0])

    home_score = score(home_m)
    away_score = score(away_m)
    diff = home_score - away_score

    win_used = False
    if win_m is not None:
        try:
            Xin = X_raw if _is_pipeline(win_m) else X_tx
            if Xin is None:
                raise RuntimeError("Missing transformed features for win model")
            if hasattr(win_m, "predict_proba"):
                p_home = float(win_m.predict_proba(Xin)[0, 1])
            else:
                p_home = float(win_m.predict(Xin)[0])
            win_used = True
        except Exception:
            p_home = _safe_logistic_from_diff(diff)
    else:
        p_home = _safe_logistic_from_diff(diff)

    p_home = float(np.clip(p_home, 0.0, 1.0))
    return PredictionResponse(
        home_score=home_score,
        away_score=away_score,
        point_diff=diff,
        home_win_probability=p_home,
        away_win_probability=1.0 - p_home,
        win_classifier_used=win_used,
        prediction_source=source,
    )

def _parse_kickoff(row: pd.Series) -> Optional[datetime]:
    # Common schedule columns: gameday + gametime, or a single datetime field
    try:
        gameday = row.get("gameday")
        gametime = row.get("gametime")
        if pd.notna(gameday) and pd.notna(gametime):
            dt = pd.to_datetime(f"{gameday} {gametime}", errors="coerce")
            return None if pd.isna(dt) else dt.to_pydatetime()
        if pd.notna(gameday):
            dt = pd.to_datetime(gameday, errors="coerce")
            return None if pd.isna(dt) else dt.to_pydatetime()
        for col in ("kickoff", "game_datetime", "game_date", "date"):
            if col in row and pd.notna(row.get(col)):
                dt = pd.to_datetime(row.get(col), errors="coerce")
                return None if pd.isna(dt) else dt.to_pydatetime()
    except Exception:
        return None
    return None

def _load_team_logos_map(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Flexible CSV parser: tries to find abbr + logo url columns."""
    df = pd.read_csv(csv_path)
    if df is None or df.empty:
        return {}

    df.columns = [c.strip() for c in df.columns]

    # best-effort column detection
    abbr_candidates = ["team_abbr", "abbr", "team", "team_code", "team_id"]
    logo_candidates = ["logoUrl", "logo_url", "logo", "team_logo", "url"]
    name_candidates = ["name", "team_name", "full_name", "team_full_name"]

    def pick(colnames: List[str]) -> Optional[str]:
        for c in colnames:
            if c in df.columns:
                return c
        # case-insensitive fallback
        lower_map = {c.lower(): c for c in df.columns}
        for c in colnames:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    abbr_col = pick(abbr_candidates)
    logo_col = pick(logo_candidates)
    name_col = pick(name_candidates)

    if not abbr_col or not logo_col:
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        abbr = str(r.get(abbr_col, "")).strip().upper()
        logo = str(r.get(logo_col, "")).strip()
        if not abbr or not logo:
            continue
        item = {"logoUrl": logo}
        if name_col:
            nm = str(r.get(name_col, "")).strip()
            if nm:
                item["name"] = nm
        out[abbr] = item
    return out


def _append_prediction_history(request: Request, req_body: Dict[str, Any], prediction: PredictionResponse) -> None:
    """Append a lightweight prediction entry to in-memory history and persist to disk.

    This is intentionally simple (append-only JSON) so the frontend can read
    recent predictions via `/history`. For production, replace with a proper
    persistent store or an append-only log with rotation.
    """
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "request": req_body,
        "prediction": prediction.model_dump() if hasattr(prediction, "model_dump") else prediction.__dict__,
    }

    # in-memory
    hist = getattr(request.app.state, "prediction_history", []) or []
    hist = [entry] + hist
    # keep only last 1000 entries in memory
    hist = hist[:1000]
    request.app.state.prediction_history = hist

    # persist to disk (best-effort)
    try:
        backend_dir = Path(__file__).resolve().parent
        out_dir = backend_dir / "Predictions"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "prediction_history.json"
        # write the entire history (simple and robust)
        import json

        out_file.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    except Exception:
        # persistence is best-effort; do not fail prediction on disk errors
        pass


# -------------------------
# Routes
# -------------------------
@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    models = getattr(request.app.state, "models", None)
    started_at = getattr(request.app.state, "started_at", None)
    if not models:
        return HealthResponse(status="unhealthy", reason="models_not_loaded", started_at=started_at)
    return HealthResponse(status="healthy", reason="ok", started_at=started_at)

@router.get("/debug", response_model=DebugResponse)
def debug(request: Request) -> DebugResponse:
    models = getattr(request.app.state, "models", None)
    ds: pd.DataFrame = getattr(request.app.state, "dataset", pd.DataFrame())
    started_at = getattr(request.app.state, "started_at", None)
    return DebugResponse(
        started_at=started_at,
        models_loaded=bool(models),
        models_dir=(models.get("models_dir") if models else None),
        dataset_rows=int(0 if ds is None else len(ds)),
        dataset_cols=int(0 if ds is None else ds.shape[1]),
        dataset_sample_cols=list(ds.columns[:30]) if ds is not None and not ds.empty else [],
    )

@router.get("/schedule/next-week", response_model=NextWeekGamesResponse)
def schedule_next_week(request: Request, season: int = 2025) -> NextWeekGamesResponse:
    df = get_schedule(season=season)
    df_next, use_season, use_week = _select_next_week_rows(df)

    home_col = _pick_col(df_next, _HOME_COLS)
    away_col = _pick_col(df_next, _AWAY_COLS)
    game_id_col = _pick_col(df_next, _GAME_ID_COLS)
    stadium_col = _pick_col(df_next, _STADIUM_COLS)

    games: List[ScheduleGame] = []
    for _, r in df_next.iterrows():
        home = str(r.get(home_col, "") if home_col else r.get("home", "")).strip().upper()
        away = str(r.get(away_col, "") if away_col else r.get("away", "")).strip().upper()
        if not home or not away:
            continue
        stadium = str(r.get(stadium_col, "") if stadium_col else r.get("stadium", "")).strip()
        game_id = None
        if game_id_col:
            raw_id = r.get(game_id_col)
            if pd.notna(raw_id):
                game_id = str(raw_id).strip()
        if not game_id:
            game_id = f"{use_season}-{use_week}-{home}-{away}"

        games.append(
            ScheduleGame(
                season=int(use_season),
                week=int(use_week),
                kickoff=_parse_kickoff(r),
                home_team=home,
                away_team=away,
                game_id=game_id,
                stadium=stadium,
            )
        )

    return NextWeekGamesResponse(games=games)

@router.get("/teams/logos", response_model=TeamLogosResponse)
def team_logos(request: Request) -> TeamLogosResponse:
    # cache on app.state so we only parse CSV once per server boot
    cached = getattr(request.app.state, "team_logos", None)
    if isinstance(cached, dict) and cached:
        return TeamLogosResponse(teams=cached)

    backend_dir = Path(__file__).resolve().parent
    # common locations in your repo
    candidates = [
        backend_dir / "team_logo.csv",
        backend_dir / "data" / "team_logo.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break

    if csv_path is None:
        return TeamLogosResponse(teams={})

    teams = _load_team_logos_map(csv_path)
    request.app.state.team_logos = teams
    return TeamLogosResponse(teams=teams)

@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, request: Request) -> PredictionResponse:
    models = getattr(request.app.state, "models", None)
    dataset: pd.DataFrame = getattr(request.app.state, "dataset", pd.DataFrame())

    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded. Check /debug.")

    try:
        pred = _predict(models, dataset, req)

        # record in-memory and persist recent predictions (best-effort)
        try:
            _append_prediction_history(request, req.model_dump() if hasattr(req, "model_dump") else req.__dict__, pred)
        except Exception:
            pass

        return pred
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {type(e).__name__}: {e}")

@router.get("/predict/next-week", response_model=NextWeekPredictionsResponse)
def predict_next_week(request: Request, season: int = 2025) -> NextWeekPredictionsResponse:
    """Batch predict every game returned by /schedule/next-week."""
    models = getattr(request.app.state, "models", None)
    dataset: pd.DataFrame = getattr(request.app.state, "dataset", pd.DataFrame())

    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded. Check /debug.")

    sched = schedule_next_week(request=request, season=season)  # reuse logic
    out: List[PredictedGame] = []

    for g in sched.games:
        req = PredictionRequest(
            home_team=g.home_team,
            away_team=g.away_team,
            season=g.season,
            week=g.week,
        )
        pred = _predict(models, dataset, req)

        # record each prediction in history (best-effort)
        try:
            _append_prediction_history(request, req.model_dump() if hasattr(req, "model_dump") else req.__dict__, pred)
        except Exception:
            pass

        out.append(
            PredictedGame(
                season=g.season,
                week=g.week,
                kickoff=g.kickoff,
                home_team=g.home_team,
                away_team=g.away_team,
                game_id=g.game_id,
                prediction=pred,
            )
        )

    return NextWeekPredictionsResponse(games=out)


@router.get("/status/overview")
def status_overview(request: Request):
    """Lightweight system overview for dashboards.

    Returns health + dataset shape + a small history summary (if available).
    """
    models = getattr(request.app.state, "models", None)
    ds: pd.DataFrame = getattr(request.app.state, "dataset", pd.DataFrame())
    history = getattr(request.app.state, "prediction_history", [])

    return {
        "health": {"status": "healthy" if models else "unhealthy"},
        "dataset": {"rows": 0 if ds is None else len(ds), "cols": 0 if ds is None else ds.shape[1]},
        "history": {"metrics": {"total_predictions": len(history) if isinstance(history, list) else 0}},
    }


@router.get("/history")
def history(request: Request, limit: int = 100):
    """Return recent prediction history entries.

    Currently stored in-memory at `app.state.prediction_history` if available.
    This is a lightweight stub to keep the frontend happy; it can be replaced
    with a persistent store later.
    """
    hist = getattr(request.app.state, "prediction_history", []) or []
    try:
        limit = int(limit)
    except Exception:
        limit = 100
    if isinstance(hist, list):
        return hist[:limit]
    return []


@router.post("/train")
@router.post("/retrain")
def train_stub() -> Dict[str, str]:
    """Training endpoints are intentionally lightweight stubs.

    For reproducible training, run `backend/train_models.py` locally. These
    endpoints return informational responses so UI calls don't 404.
    """
    return {"status": "not_implemented", "message": "Run backend/train_models.py locally to retrain models"}
