#!/usr/bin/env python
"""
NFL Game Prediction API (FastAPI)
Run: uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import logging.config
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
BASE_DIR = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
FRONTEND_BUILD = BASE_DIR / "frontend" / "build"
LOG_DIR = BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            },
            "simple": {"format": "%(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(LOG_DIR / "nfl_prediction.log"),
                "mode": "a",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
        "loggers": {"nfl_prediction": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": False}},
    }
)
logger = logging.getLogger("nfl_prediction")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_DATASET = DATA_DIR / "Nfl_data_sorted.csv"
DEFAULT_SCHEDULE = DATA_DIR / "Nfl_schedule_2025_2026.csv"

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    season: int
    week: int

class PredictionResponse(BaseModel):
    home_score: float
    away_score: float
    home_win_probability: float
    away_win_probability: float
    point_diff: float
    mode: str

class HealthResponse(BaseModel):
    status: str
    mode: Optional[str] = None
    reason: Optional[str] = None

class ScheduleGame(BaseModel):
    season: int
    week: int
    home_team: str
    home_abbr: str
    away_team: str
    away_abbr: str
    kickoff_iso: str
    game_id: str

# -----------------------------------------------------------------------------
# Team canon
# -----------------------------------------------------------------------------
TEAM_ABBREVIATIONS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",  # normalized
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
VALID_ABBRS = set(TEAM_ABBREVIATIONS.values())

def get_team_abbreviation(team_name: str) -> str:
    if team_name in VALID_ABBRS:
        return team_name
    if team_name in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[team_name]
    logger.error("Unknown team: %s", team_name)
    raise ValueError(f"Unknown team name: {team_name}")

# -----------------------------------------------------------------------------
# Feature-name normalization
# -----------------------------------------------------------------------------
def _normalize_feature_cols(raw_cols: dict) -> List[str]:
    numeric = raw_cols.get("numeric", []) or []
    categorical = raw_cols.get("categorical", []) or []
    cols = list(numeric) + list(categorical)
    prefixed = [c for c in cols if c.startswith(("num__", "cat__"))]
    if not prefixed:
        return cols
    def strip_prefix(c: str) -> str:
        return c[5:] if c.startswith(("num__", "cat__")) else c
    return [strip_prefix(c) for c in cols]

# -----------------------------------------------------------------------------
# Model loading (joblib only; NN removed)
# -----------------------------------------------------------------------------
def load_objects() -> Dict[str, Any]:
    import joblib
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text())

    preprocessor = joblib.load(MODELS_DIR / meta["preprocessor"])
    models_meta = meta.get("models", {})
    # Always classic models for now
    home_model_path = MODELS_DIR / models_meta.get("home_model", "home_model.joblib")
    away_model_path = MODELS_DIR / models_meta.get("away_model", "away_model.joblib")
    if not home_model_path.exists():
        raise FileNotFoundError(f"Missing {home_model_path}")
    if not away_model_path.exists():
        raise FileNotFoundError(f"Missing {away_model_path}")
    home_model = joblib.load(home_model_path)
    away_model = joblib.load(away_model_path)

    return {
        "mode": "models",
        "preprocessor": preprocessor,
        "home_model": home_model,
        "away_model": away_model,
        "raw_feature_columns": meta.get("raw_feature_columns", {}),
    }

# -----------------------------------------------------------------------------
# App + lifespan
# -----------------------------------------------------------------------------
model_objects: Optional[Dict[str, Any]] = None
dataset_df: Optional[pd.DataFrame] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_objects, dataset_df
    logger.info("Startup: loading models and dataset")
    model_objects = load_objects()

    ds_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    if not ds_path.exists():
        raise RuntimeError(f"Dataset not found: {ds_path}")
    df = pd.read_csv(ds_path)
    if df.empty:
        raise RuntimeError("Dataset CSV is empty")
    df.columns = [c.strip() for c in df.columns]
    dataset_df = df
    logger.info("Loaded dataset rows=%d cols=%d", len(df), len(df.columns))
    yield
    logger.info("Shutdown complete")

app = FastAPI(
    title="NFL Game Prediction API",
    description="Predict home/away scores and win odds.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # narrow in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_current_nfl_context() -> Dict[str, Any]:
    now = datetime.now()
    current_season = now.year if now.month >= 8 else now.year - 1
    logger.debug("Current season determined as %d", current_season)
    
    global dataset_df
    if (
        dataset_df is not None
        and not dataset_df.empty
        and {"season", "week", "home_points_for", "away_points_for"}.issubset(dataset_df.columns)
    ):
        completed = dataset_df[dataset_df["home_points_for"].notna() & dataset_df["away_points_for"].notna()]
        if not completed.empty:
            ordered = completed.sort_values(["season", "week"], kind="mergesort")
            last = ordered.iloc[-1]
            last_completed_season = int(last["season"])
            last_completed_week = int(last["week"])
            next_season = last_completed_season
            next_week = last_completed_week + 1
            if next_week > 22:
                next_week = 1
                next_season += 1
            return {
                "current_season": current_season,
                "last_completed_season": last_completed_season,
                "last_completed_week": last_completed_week,
                "next_prediction_season": next_season,
                "next_prediction_week": next_week,
                "status": "nfl_season_active" if next_season == current_season else "offseason",
            }
    return {
        "current_season": current_season,
        "last_completed_season": current_season,
        "last_completed_week": 0,
        "next_prediction_season": current_season,
        "next_prediction_week": 1,
        "status": "preseason_or_early",
    }

def build_future_game_features(
    df: pd.DataFrame, home_team: str, away_team: str, season: int, week: int
) -> pd.Series:
    time_key = df["season"].astype(int) * 100 + df["week"].astype(int)
    df = df.assign(time_key=time_key)

    def latest_for(team: str) -> Dict[str, Any]:
        team_mask = (df["home_team"] == team) | (df["away_team"] == team)
        before = df[team_mask & (df["time_key"] < season * 100 + week)]
        if before.empty:
            raise ValueError(f"No prior data available for {team} before {season}-W{week}")
        row = before.loc[before["time_key"].idxmax()]
        # row is a Series for a single game; compare directly without .any()
        if str(row["home_team"]) == team:
            return {
                "prior_pa_avg_3": row.get("home_prior_pa_avg_3"),
                "prior_pa_avg_5": row.get("home_prior_pa_avg_5"),
                "prior_pf_avg_3": row.get("home_prior_pf_avg_3"),
                "prior_pf_avg_5": row.get("home_prior_pf_avg_5"),
                "prior_win_pct_3": row.get("home_prior_win_pct_3"),
                "prior_win_pct_5": row.get("home_prior_win_pct_5"),
            }
        else:
            return {
                "prior_pa_avg_3": row.get("away_prior_pa_avg_3"),
                "prior_pa_avg_5": row.get("away_prior_pa_avg_5"),
                "prior_pf_avg_3": row.get("away_prior_pf_avg_3"),
                "prior_pf_avg_5": row.get("away_prior_pf_avg_5"),
                "prior_win_pct_3": row.get("away_prior_win_pct_3"),
                "prior_win_pct_5": row.get("away_prior_win_pct_5"),
            }

    hf = latest_for(home_team)
    af = latest_for(away_team)

    feature_row: Dict[str, Any] = {}
    for stat in (
        "prior_pa_avg_3",
        "prior_pa_avg_5",
        "prior_pf_avg_3",
        "prior_pf_avg_5",
        "prior_win_pct_3",
        "prior_win_pct_5",
    ):
        feature_row[f"home_{stat}"] = hf[stat]
        feature_row[f"away_{stat}"] = af[stat]

    for base in ("prior_pf_avg", "prior_pa_avg", "prior_win_pct"):
        for wnd in ("3", "5"):
            h = feature_row.get(f"home_{base}_{wnd}")
            a = feature_row.get(f"away_{base}_{wnd}")
            if h is not None and a is not None:
                feature_row[f"home_minus_away_{base}_{wnd}"] = h - a

    return pd.Series(feature_row)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    global model_objects
    if model_objects is None:
        return HealthResponse(status="unhealthy", mode="none", reason="models not loaded")
    return HealthResponse(status="healthy", mode=model_objects.get("mode"), reason="models loaded successfully")


@app.get("/debug")
def debug_info():
    global model_objects, dataset_df
    debug_data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat() + "Z",
        "models_loaded": model_objects is not None,
        "dataset_loaded": dataset_df is not None,
    }
    if model_objects:
        cols = model_objects.get("raw_feature_columns", {})
        debug_data.update(
            {
                "model_mode": model_objects.get("mode"),
                "feature_columns": list(cols.keys()),
                "numeric_features": len(cols.get("numeric", [])),
                "categorical_features": len(cols.get("categorical", [])),
            }
        )
    if dataset_df is not None:
        debug_data.update(
            {
                "dataset_rows": int(len(dataset_df)),
                "dataset_columns": int(len(dataset_df.columns)),
                "season_range": [int(dataset_df["season"].min()), int(dataset_df["season"].max())]
                if "season" in dataset_df else None,
                "week_range": [int(dataset_df["week"].min()), int(dataset_df["week"].max())]
                if "week" in dataset_df else None,
            }
        )
    try:
        mpath = MODELS_DIR / "metadata.json"
        if mpath.exists():
            meta = json.loads(mpath.read_text())
            debug_data["training_metadata"] = {
                "training_timestamp": meta.get("training_timestamp"),
                "dataset_hash": meta.get("dataset_hash"),
                "training_samples": meta.get("training_samples"),
                "model_scores": meta.get("model_scores"),
            }
    except Exception as e:
        debug_data["training_metadata_error"] = str(e)
    return debug_data

@app.get("/schedule/next-week", response_model=List[ScheduleGame])
def get_next_week_schedule():
    context = get_current_nfl_context()
    try:
        schedule_path = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
        if not schedule_path.exists():
            raise HTTPException(status_code=404, detail=f"Schedule not found: {schedule_path}")
        df = pd.read_csv(schedule_path)
        if df.empty:
            raise HTTPException(status_code=404, detail="Schedule is empty")

        # Vectorized kickoff parsing with fallbacks
        gameday = df["gameday"].astype(str).fillna("").str.strip()
        gametime = df["gametime"].astype(str).fillna("").str.strip()
        has_time = df["gameday"].combine_first(df["gametime"]).notna()

        ts_str = np.where(has_time, gameday + " " + gametime, gameday)
        df["kickoff_ts_utc"] = pd.to_datetime(pd.Series(ts_str, index=df.index), errors="coerce", utc=True)

        now = pd.Timestamp.now(tz="UTC")
        future = df[df["kickoff_ts_utc"].notna() & (df["kickoff_ts_utc"] >= now)]
        current_week = int(future["week"].min()) if not future.empty else int(df["week"].max())

        week_games = df[df["week"] == current_week]
        games: List[ScheduleGame] = []
        for _, row in week_games.iterrows():
            ts = row["kickoff_ts_utc"]
            games.append(
                ScheduleGame(
                    season=int(row["season"]),
                    week=int(row["week"]),
                    home_team=str(row["home_team"]),
                    home_abbr=get_team_abbreviation(str(row["home_team"])),
                    away_team=str(row["away_team"]),
                    away_abbr=get_team_abbreviation(str(row["away_team"])),
                    kickoff_iso=(ts.isoformat() if pd.notna(ts) else "TBD"),
                    game_id=str(
                        row.get(
                            "game_id",
                            f"{row['season']}W{row['week']}-{row['away_team']}@{row['home_team']}",
                        )
                    ),
                )
            )
        logger.info("Schedule week %s games=%d", current_week, len(games))
        return games
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Schedule error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load schedule: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict_game(payload: PredictionRequest):
    global model_objects, dataset_df
    if model_objects is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    try:
        home_abbr = get_team_abbreviation(payload.home_team)
        away_abbr = get_team_abbreviation(payload.away_team)
        season = int(payload.season)
        week = int(payload.week)

        mask = (
            (dataset_df["season"] == season)
            & (dataset_df["week"] == week)
            & (dataset_df["home_team"] == home_abbr)
            & (dataset_df["away_team"] == away_abbr)
        )
        rows = dataset_df.loc[mask]
        logger.info("Filtered rows: %s", rows)

        if rows.empty:
            row = build_future_game_features(dataset_df, home_abbr, away_abbr, season, week)
        else:
            row = rows.iloc[0]
            if pd.notna(row.get("home_points_for")) and pd.notna(row.get("away_points_for")):
                raise HTTPException(status_code=400, detail="Game already completed; no prediction produced.")

        raw_cols = model_objects.get("raw_feature_columns", {})
        feature_cols = _normalize_feature_cols(raw_cols)
        if not feature_cols:
            # hard fallback to common set if metadata lacks list
            feature_cols = [
                "home_prior_pf_avg_3","home_prior_pa_avg_3","home_prior_win_pct_3",
                "home_prior_pf_avg_5","home_prior_pa_avg_5","home_prior_win_pct_5",
                "away_prior_pf_avg_3","away_prior_pa_avg_3","away_prior_win_pct_3",
                "away_prior_pf_avg_5","away_prior_pa_avg_5","away_prior_win_pct_5",
            ]
        logger.info("Using feature columns: %s", feature_cols)

        row_cols = set(row.index.tolist())
        missing = [c for c in feature_cols if c not in row_cols]
        if missing:
            raise HTTPException(status_code=500, detail=f"Missing feature columns: {missing}")

        input_df = pd.DataFrame({c: [row[c]] for c in feature_cols})
        X = model_objects["preprocessor"].transform(input_df)

        # Classic models only
        home_score = float(model_objects["home_model"].predict(X)[0])
        away_score = float(model_objects["away_model"].predict(X)[0])

        home_score = round(max(0.0, min(70.0, home_score)), 1)
        away_score = round(max(0.0, min(70.0, away_score)), 1)
        point_diff = round(home_score - away_score, 1)

        k = 0.22
        home_win_prob = 1.0 / (1.0 + np.exp(-k * point_diff))

        return PredictionResponse(
            home_score=home_score,
            away_score=away_score,
            home_win_probability=round(home_win_prob, 3),
            away_win_probability=round(1 - home_win_prob, 3),
            point_diff=point_diff,
            mode="models",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.get("/predict/next-week")
def predict_next_week():
    global model_objects
    if model_objects is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    try:
        context = get_current_nfl_context()
        next_season = context["next_prediction_season"]
        next_week = context["next_prediction_week"]

        schedule_path = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
        if not schedule_path.exists():
            raise HTTPException(status_code=404, detail="Schedule data not found")
        schedule_df = pd.read_csv(schedule_path)
        games_df = schedule_df[(schedule_df["season"] == next_season) & (schedule_df["week"] == next_week)]

        preds: List[Dict[str, Any]] = []
        for _, g in games_df.iterrows():
            try:
                req = PredictionRequest(
                    home_team=str(g["home_team"]),
                    away_team=str(g["away_team"]),
                    season=int(g["season"]),
                    week=int(g["week"]),
                )
                pr = predict_game(req)
                preds.append(
                    {
                        "game_id": str(
                            g.get("game_id", f"{g['season']}W{g['week']}-{g['away_team']}@{g['home_team']}")
                        ),
                        "season": int(g["season"]),
                        "week": int(g["week"]),
                        "home_team": str(g["home_team"]),
                        "away_team": str(g["away_team"]),
                        "kickoff": str(g.get("gameday", "TBD")),
                        "prediction": pr.model_dump_json(),
                    }
                )
            except Exception as e:
                preds.append(
                    {
                        "game_id": str(g.get("game_id", "unknown")),
                        "season": int(g["season"]),
                        "week": int(g["week"]),
                        "home_team": str(g["home_team"]),
                        "away_team": str(g["away_team"]),
                        "error": str(e),
                    }
                )

        return {
            "context": context,
            "games": preds,
            "total_games": len(preds),
            "successful_predictions": len([p for p in preds if "prediction" in p]),
        }
    except Exception as e:
        logger.error("Next-week prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to predict next week: {e}")

@app.post("/retrain")
def retrain():
    global model_objects
    try:
        subprocess.run([sys.executable, str(BACKEND_DIR / "train_models.py")], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e.stderr}")
    model_objects = load_objects()
    return {"detail": "Models retrained successfully."}

@app.post("/update_data")
def update_data():
    try:
        # Build leak-free rolling features using the canonical script
        build = subprocess.run(
            [
                sys.executable,
                str(BACKEND_DIR / "build_csv_datasets.py"),
                "--start","2010","--end","2025","--out-dir",str(DATA_DIR),
            ],
            check=True, capture_output=True, text=True,
        )
        logger.info("build_csvs stdout:\n%s", build.stdout)
        train = subprocess.run(
            [sys.executable, str(BACKEND_DIR / "train_models.py")],
            check=True, capture_output=True, text=True,
        )
        logger.info("train_models stdout:\n%s", train.stdout)
        return {"detail": "Data updated and models retrained."}
    except subprocess.CalledProcessError as e:
        logger.error("Update failed: %s", e.stderr)
        return {"detail": "Update failed", "stderr": e.stderr}

# Serve built frontend (auto-detect dist/build). Mount after routes.
_front = FRONTEND_DIST if FRONTEND_DIST.exists() else (FRONTEND_BUILD if FRONTEND_BUILD.exists() else None)
if _front:
    app.mount("/", StaticFiles(directory=str(_front), html=True), name="app")
    logger.info("Serving frontend from %s", _front)
else:
    logger.warning("No frontend build found; not serving static files")