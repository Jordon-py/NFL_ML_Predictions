"""
NFL Game Prediction API (FastAPI)

Run:
  uvicorn backend.main:app --reload --port 8000

Env:
  DATASET_PATH, SCHEDULE_PATH, CORS_ORIGINS, CORS_ORIGIN_REGEX, SERVE_FRONTEND
"""

from __future__ import annotations

import json
import logging
import logging.config
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env
load_dotenv(Path(__file__).parent / ".env")

# -----------------------
# Paths and constants
# -----------------------
THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
BASE_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
LOG_DIR = BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Use game_features.csv which has the engineered features (prior stats, differentials, betting data)
# merged_game_features.csv only has raw stats and won't work with trained models
DEFAULT_DATASET = DATA_DIR / "game_features.csv"
DEFAULT_SCHEDULE = DATA_DIR / "Nfl_schedule_2025_2026.csv"

FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_BUILD = FRONTEND_DIR / "build"
FRONTEND_DIST = FRONTEND_DIR / "dist"

TRUTHY = {"true", "t", "1", "yes", "y"}
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "false").strip().lower() in TRUTHY

# Logging
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "d": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "d",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "d",
                "filename": str(LOG_DIR / "api.log"),
                "encoding": "utf-8",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)
log = logging.getLogger("api")

# Globals
model_objects: Optional[Dict[str, Any]] = None
dataset_df: Optional[pd.DataFrame] = None

# CORS
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nfl-ml-predictions.vercel.app",
    "https://nfl-predict-frontend.vercel.app",
    "https://www.nfl-predict.com",
    "https://nfl-predict.com",
]
raw_cors = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [
    o.strip() for o in raw_cors.split(",") if o.strip()
] or DEFAULT_CORS_ORIGINS
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app$").strip()

# Teams
TEAM_ABBREVIATIONS = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
TEAM_CODE_FIX = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS"}
VALID_ABBRS = set(TEAM_ABBREVIATIONS.values()) | set(TEAM_CODE_FIX.values())


def get_abbr(name: str) -> str:
    n = str(name).strip()
    if n in VALID_ABBRS:
        return TEAM_CODE_FIX.get(n, n)
    if n in TEAM_CODE_FIX:
        return TEAM_CODE_FIX[n]
    if n in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[n]
    raise ValueError(f"Unknown team: {name}")


# -----------------------
# Feature helpers
# -----------------------
def _normalize_feature_cols(raw_cols: dict) -> List[str]:
    numeric = raw_cols.get("numeric", []) or []
    categorical = raw_cols.get("categorical", []) or []
    cols = list(numeric) + list(categorical)

    def strip_prefix(c: str) -> str:
        return c[5:] if c.startswith(("num__", "cat__")) else c

    if any(c.startswith(("num__", "cat__")) for c in cols):
        cols = [strip_prefix(c) for c in cols]
    return cols


# -----------------------
# Lifespan
# -----------------------
def load_objects() -> Dict[str, Any]:
    """Load model metadata and instantiate reusable predictors for the API."""
    meta_path = MODELS_DIR / "metadata.json"
    log.debug("Loading model metadata from %s", meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def resolve_model_path(meta_key: str, fallback: str) -> Path:
        candidate = Path(meta.get(meta_key, fallback))
        return candidate if candidate.is_absolute() else MODELS_DIR / candidate

    preprocessor = joblib.load(resolve_model_path("preprocessor", "preprocessor.joblib"))
    home_model = joblib.load(resolve_model_path("home_model", "home_model.joblib"))
    away_model = joblib.load(resolve_model_path("away_model", "away_model.joblib"))
    win_model_path = resolve_model_path("win_model", "win_clf_calibrated.joblib")
    win_model = joblib.load(win_model_path) if win_model_path.exists() else None

    return {
        "mode": meta.get("mode", "production"),
        "preprocessor": preprocessor,
        "home_model": home_model,
        "away_model": away_model,
        "win_model": win_model,
        "raw_feature_columns": meta.get("raw_feature_columns", {}),
        "win_threshold_optimal": meta.get("win_threshold_optimal", 0.5),
    }
    # Change log 2025-01-05: Ensured metadata dict access and single-pass model loading for reliable startup.


def _coerce_bool(s: pd.Series) -> pd.Series:
    truthy = {"true", "t", "1", "yes", "y"}
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    return s.astype(str).str.strip().str.lower().isin(truthy)


def _ensure_home_away(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    if {"home_team", "away_team"}.issubset(cols):
        return df
    if {"team", "opponent_team", "is_home"}.issubset(cols):
        is_home = _coerce_bool(df["is_home"])
        return df.assign(
            is_home=is_home,
            home_team=np.where(is_home, df["team"], df["opponent_team"]),
            away_team=np.where(is_home, df["opponent_team"], df["team"]),
        )
    log.warning(
        "Dataset missing home/away columns and team/opponent fallback; synthetic features only."
    )
    return df


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model_objects, dataset_df
    log.info("Startup: loading models and dataset")
    model_objects = load_objects()

    ds_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    if not ds_path.exists():
        raise RuntimeError(f"Dataset not found: {ds_path}")

    df = pd.read_csv(ds_path)
    if df.empty:
        raise RuntimeError("Dataset CSV is empty")

    df.columns = [c.strip() for c in df.columns]
    df = _ensure_home_away(df)
    dataset_df = df

    log.info("Loaded dataset rows=%d cols=%d", len(df), df.shape[1])
    try:
        yield
    finally:
        log.info("Shutdown complete")


# -----------------------
# FastAPI app + CORS + static
# -----------------------
app = FastAPI(title="NFL Game Prediction API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)

if SERVE_FRONTEND:
    for candidate in (FRONTEND_BUILD, FRONTEND_DIST):
        if candidate.exists():
            app.mount(
                "/", StaticFiles(directory=str(candidate), html=True), name="frontend"
            )
            log.info("Serving frontend from %s", candidate)
            break
    else:
        log.warning("SERVE_FRONTEND=true but no frontend build found.")


# -----------------------
# Schemas
# -----------------------
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


# -----------------------
# Helpers
# -----------------------
def get_current_nfl_context() -> Dict[str, Any]:
    now = datetime.now()
    cur_season = now.year if now.month >= 8 else now.year - 1
    if dataset_df is not None and {
        "season",
        "week",
        "home_points_for",
        "away_points_for",
    }.issubset(dataset_df.columns):
        done = dataset_df[
            dataset_df["home_points_for"].notna()
            & dataset_df["away_points_for"].notna()
        ]
        if not done.empty:
            last = done.sort_values(by=["season", "week"]).iloc[-1]
            last_s, last_w = int(last["season"]), int(last["week"])
            nxt_s, nxt_w = last_s, last_w + 1
            if nxt_w > 22:
                nxt_s, nxt_w = last_s + 1, 1
            return {
                "current_season": cur_season,
                "last_completed_season": last_s,
                "last_completed_week": last_w,
                "next_prediction_season": nxt_s,
                "next_prediction_week": nxt_w,
                "status": "nfl_season_active" if nxt_s == cur_season else "offseason",
            }
    return {
        "current_season": cur_season,
        "last_completed_season": cur_season,
        "last_completed_week": 0,
        "next_prediction_season": cur_season,
        "next_prediction_week": 1,
        "status": "preseason_or_early",
    }


def _build_future_row(
    df: pd.DataFrame, home: str, away: str, season: int, week: int
) -> pd.Series:
    """
    Build engineered features for a future game using historical data.
    Computes rolling averages (3-game and 5-game windows) for both teams.
    """
    local = df.copy()
    local["time_key"] = local["season"].astype(int) * 100 + local["week"].astype(int)
    cutoff = season * 100 + week

    def compute_team_features(team: str, prefix: str) -> Dict[str, Any]:
        """Compute prior features for a team using their last N completed games."""
        # Find all games where this team played
        team_mask = (local["home_team"] == team) | (local["away_team"] == team)
        # Only use completed games before the target game
        completed_mask = (
            local["home_points_for"].notna() & 
            local["away_points_for"].notna() & 
            (local["time_key"] < cutoff)
        )
        history = local[team_mask & completed_mask].sort_values("time_key")
        
        if history.empty:
            raise ValueError(f"No prior data for {team} before {season} Week {week}")
        
        features = {}
        
        # Get last 5 games for 5-game averages, last 3 for 3-game averages
        last_5 = history.tail(5)
        last_3 = history.tail(3)
        
        # Helper to extract team's stats from a game row
        def get_team_stats(row, team_abbr):
            is_home = row["home_team"] == team_abbr
            if is_home:
                return {
                    "pf": row.get("home_points_for", np.nan),
                    "pa": row.get("away_points_for", np.nan),
                    "win": 1 if row.get("winner") == team_abbr else 0,
                }
            else:
                return {
                    "pf": row.get("away_points_for", np.nan),
                    "pa": row.get("home_points_for", np.nan),
                    "win": 1 if row.get("winner") == team_abbr else 0,
                }
        
        # Compute 3-game averages
        if len(last_3) >= 1:
            stats_3 = [get_team_stats(row, team) for _, row in last_3.iterrows()]
            features[f"{prefix}prior_pf_avg_3"] = np.mean([s["pf"] for s in stats_3 if not pd.isna(s["pf"])])
            features[f"{prefix}prior_pa_avg_3"] = np.mean([s["pa"] for s in stats_3 if not pd.isna(s["pa"])])
            features[f"{prefix}prior_win_pct_3"] = np.mean([s["win"] for s in stats_3])
        
        # Compute 5-game averages
        if len(last_5) >= 1:
            stats_5 = [get_team_stats(row, team) for _, row in last_5.iterrows()]
            features[f"{prefix}prior_pf_avg_5"] = np.mean([s["pf"] for s in stats_5 if not pd.isna(s["pf"])])
            features[f"{prefix}prior_pa_avg_5"] = np.mean([s["pa"] for s in stats_5 if not pd.isna(s["pa"])])
            features[f"{prefix}prior_win_pct_5"] = np.mean([s["win"] for s in stats_5])
        
        # For advanced stats, try to use the most recent values from the dataset
        # (these are pre-computed in game_features.csv)
        last_game = history.iloc[-1]
        was_home_last = last_game["home_team"] == team
        source_prefix = "home_" if was_home_last else "away_"
        
        # Copy advanced prior stats from last game
        for stat_name in [
            "off_epa_per_play", "off_success_rate", "off_explosive_rate",
            "off_third_down_pct", "off_pass_over_expected",
            "def_success_rate_allowed", "def_explosive_rate_allowed",
            "def_epa_per_play", "def_takeaway_rate", "off_turnover_rate"
        ]:
            for window in ["3", "5"]:
                col_name = f"{source_prefix}prior_{stat_name}_{window}"
                if col_name in last_game.index and pd.notna(last_game[col_name]):
                    features[f"{prefix}prior_{stat_name}_{window}"] = last_game[col_name]
        
        return features
    
    # Get features for both teams
    home_features = compute_team_features(home, "home_")
    away_features = compute_team_features(away, "away_")
    
    # Merge all features
    feature_row = {**home_features, **away_features}
    
    # Compute differential features (home - away)
    for stat_suffix in [
        "pf_avg_3", "pa_avg_3", "win_pct_3",
        "off_epa_per_play_3", "off_success_rate_3", "off_explosive_rate_3",
        "off_third_down_pct_3", "off_pass_over_expected_3",
        "def_success_rate_allowed_3", "def_explosive_rate_allowed_3",
        "def_epa_per_play_3", "def_takeaway_rate_3", "off_turnover_rate_3",
        "pf_avg_5", "pa_avg_5", "win_pct_5",
        "off_epa_per_play_5", "off_success_rate_5", "off_explosive_rate_5",
        "off_third_down_pct_5", "off_pass_over_expected_5",
        "def_success_rate_allowed_5", "def_explosive_rate_allowed_5",
        "def_epa_per_play_5", "def_takeaway_rate_5", "off_turnover_rate_5",
    ]:
        home_key = f"home_prior_{stat_suffix}"
        away_key = f"away_prior_{stat_suffix}"
        if home_key in feature_row and away_key in feature_row:
            h_val = feature_row[home_key]
            a_val = feature_row[away_key]
            if not pd.isna(h_val) and not pd.isna(a_val):
                feature_row[f"home_minus_away_{stat_suffix}"] = h_val - a_val
    
    # Add betting/rest features with neutral defaults
    feature_row["home_moneyline_prob"] = 0.5  # Neutral betting line
    feature_row["away_moneyline_prob"] = 0.5
    feature_row["moneyline_prob_diff"] = 0.0
    feature_row["spread_line"] = 0.0  # Pick'em
    feature_row["total_line"] = 45.0  # Average NFL total
    feature_row["home_rest"] = 7  # Standard week rest
    feature_row["away_rest"] = 7
    feature_row["rest_diff"] = 0
    feature_row["home_game_date"] = f"{season}-W{week:02d}"  # Categorical feature
    
    log.debug("Built future row for %s vs %s: %d features", home, away, len(feature_row))
    return pd.Series(feature_row)


# -----------------------
# Routes
# -----------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if model_objects is None:
        return HealthResponse(
            status="unhealthy", mode="none", reason="models not loaded"
        )
    return HealthResponse(
        status="healthy", mode=model_objects.get("mode"), reason="models loaded"
    )


@app.get("/debug")
def debug_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "active",
        "cors_origins": CORS_ORIGINS,
        "cors_origin_regex": CORS_ORIGIN_REGEX,
    }
    try:
        mpath = MODELS_DIR / "metadata.json"
        if mpath.is_file():
            out["metadata"] = json.loads(mpath.read_text(encoding="utf-8"))
        tr = MODELS_DIR / "training_report.json"
        out["training_report_present"] = tr.is_file()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@app.get("/report/training")
def report_training() -> Dict[str, Any]:
    tr = MODELS_DIR / "training_report.json"
    if not tr.exists():
        raise HTTPException(404, "training_report.json not found")
    return json.loads(tr.read_text(encoding="utf-8"))


@app.get("/report/calibration")
def report_calibration() -> Dict[str, Any]:
    tr = MODELS_DIR / "training_report.json"
    if not tr.exists():
        raise HTTPException(404, "training_report.json not found")
    j = json.loads(tr.read_text(encoding="utf-8"))
    win = j.get("models", {}).get("win_clf", {})
    return {
        "reliability_bins": win.get("reliability_bins", []),
        "auc_val": win.get("auc_val"),
        "brier_val": win.get("brier_val"),
        "logloss_val": win.get("logloss_val"),
        "optimal_threshold": win.get("optimal_threshold"),
        "optimal_threshold_f1": win.get("optimal_threshold_f1"),
        "optimal_threshold_acc": win.get("optimal_threshold_acc"),
    }


@app.get("/schedule/next-week", response_model=List[ScheduleGame])
def get_next_week_schedule() -> List[ScheduleGame]:
    spath = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
    if not spath.exists():
        raise HTTPException(status_code=404, detail=f"Schedule not found: {spath}")
    df = pd.read_csv(spath)

    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace(TEAM_CODE_FIX)

    kickoff = pd.to_datetime(
        (
            df["gameday"].astype(str).str.strip()
            + " "
            + df["gametime"].astype(str).str.strip()
        ),
        errors="coerce",
        utc=True,
    )
    date_only = pd.to_datetime(df["gameday"], errors="coerce", utc=True)
    df["kickoff_ts_utc"] = kickoff.where(kickoff.notna(), date_only)

    now = pd.Timestamp.now(tz="UTC")
    future = df[df["kickoff_ts_utc"].notna() & (df["kickoff_ts_utc"] >= now)]
    current_week = (
        int(future["week"].min()) if not future.empty else int(df["week"].max())
    )
    week_games = df[df["week"] == current_week].copy()

    games: List[ScheduleGame] = []
    for _, r in week_games.iterrows():
        h, a = get_abbr(r["home_team"]), get_abbr(r["away_team"])
        games.append(
            ScheduleGame(
                season=int(r["season"]),
                week=int(r["week"]),
                home_team=str(r["home_team"]),
                home_abbr=h,
                away_team=str(r["away_team"]),
                away_abbr=a,
                kickoff_iso=(
                    r["kickoff_ts_utc"].isoformat()
                    if pd.notna(r["kickoff_ts_utc"])
                    else "TBD"
                ),
                game_id=str(r.get("game_id", f"{r['season']}W{r['week']}-{a}@{h}")),
            )
        )
    log.info("Schedule week %s games=%d", current_week, len(games))
    return games


@app.post("/predict", response_model=PredictionResponse)
def predict_game(payload: PredictionRequest) -> PredictionResponse:
    if model_objects is None or dataset_df is None:
        raise HTTPException(500, "Models or dataset not loaded.")

    try:
        h = get_abbr(payload.home_team)
        a = get_abbr(payload.away_team)
        season, week = int(payload.season), int(payload.week)

        mask = (
            (dataset_df["season"] == season)
            & (dataset_df["week"] == week)
            & (dataset_df["home_team"] == h)
            & (dataset_df["away_team"] == a)
        )
        if "is_home" in dataset_df.columns:
            mask &= dataset_df["is_home"].astype(bool)

        rows = dataset_df.loc[mask]
        
        # Try to get existing row from dataset
        if not rows.empty:
            row = rows.iloc[0]
            # Check if game is already completed
            if pd.notna(row.get("home_points_for")) and pd.notna(row.get("away_points_for")):
                raise HTTPException(400, "Game completed; no prediction needed.")
        else:
            # Game not in dataset - build features dynamically for future game
            log.info("Building features for future game: %s vs %s (%d Week %d)", h, a, season, week)
            try:
                row = _build_future_row(dataset_df, h, a, season, week)
            except ValueError as e:
                raise HTTPException(
                    400,
                    f"Cannot predict {h} vs {a} ({season} Week {week}): {e}"
                )
        
        # Now extract features for model input
        # Note: game_features.csv columns are the actual feature names we need
        all_dataset_cols = set(dataset_df.columns) | set(row.index)
        
        # Build feature vector using all available prior/differential columns
        feature_cols = [
            col for col in all_dataset_cols 
            if any(col.startswith(prefix) for prefix in [
                "home_prior_", "away_prior_", "home_minus_away_",
                "home_moneyline", "away_moneyline", "moneyline_prob",
                "spread_", "total_", "rest_", "home_game_date"
            ])
        ]
        
        # Create feature DataFrame
        data = {}
        for c in feature_cols:
            if c in row.index:
                val = row[c]
                data[c] = [val if not pd.isna(val) else np.nan]
            else:
                data[c] = [np.nan]
        
        if not data:
            raise HTTPException(
                400,
                f"No valid features could be extracted for {h} vs {a}. "
                f"Historical data may be insufficient."
            )

        # Models are Pipelines that include preprocessing, so pass raw DataFrame directly
        X = pd.DataFrame(data)

        # Score regressors
        # Models are small ensembles saved as dicts: {"hgbr", "ridge", "weight"} from training
        def _reg_predict(bundle: Any) -> np.ndarray:
            log.debug("Model bundle type: %s, hasattr predict: %s", type(bundle), hasattr(bundle, "predict"))
            if isinstance(bundle, dict):
                log.debug("Model bundle keys: %s", list(bundle.keys()) if hasattr(bundle, 'keys') else 'no keys method')
                if {"hgbr", "ridge", "weight"}.issubset(bundle):
                    weight = float(bundle["weight"])
                    preds_hgbr = bundle["hgbr"].predict(X)
                    preds_ridge = bundle["ridge"].predict(X)
                    return weight * preds_hgbr + (1.0 - weight) * preds_ridge
                delegate = bundle.get("model") or bundle.get("estimator")
                if delegate is not None and hasattr(delegate, "predict"):
                    return delegate.predict(X)
                # If dict but no expected structure, try to find any predictor
                for key, value in bundle.items():
                    if hasattr(value, "predict"):
                        log.debug("Using predictor from dict key: %s", key)
                        return value.predict(X)
            if not isinstance(bundle, dict) and hasattr(bundle, "predict"):
                return bundle.predict(X)
            raise AttributeError(f"Score model lacks predict method. Type: {type(bundle)}")

        home_score = float(
            np.clip(
                _reg_predict(model_objects["home_model"])[0],
                0.0,
                70.0,
            )
        )
        away_score = float(
            np.clip(
                _reg_predict(model_objects["away_model"])[0],
                0.0,
                70.0,
            )
        )
        point_diff = round(home_score - away_score, 1)

        # Win probability from calibrated classifier if present, else sigmoid on margin
        if model_objects["win_model"] is not None:
            home_prob = float(model_objects["win_model"].predict_proba(X)[0, 1])
        else:
            home_prob = 1.0 / (1.0 + math.exp(-0.25 * point_diff))

        return PredictionResponse(
            home_score=round(home_score, 1),
            away_score=round(away_score, 1),
            home_win_probability=round(home_prob, 3),
            away_win_probability=round(1.0 - home_prob, 3),
            point_diff=point_diff,
            mode="models",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(400, f"Prediction failed: {e}")


@app.get("/predict/next-week")
def predict_next_week() -> Dict[str, Any]:
    if model_objects is None:
        raise HTTPException(500, "Models not loaded.")
    try:
        ctx = get_current_nfl_context()
        spath = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
        if not spath.exists():
            raise HTTPException(404, "Schedule data not found")
        s = pd.read_csv(spath)
        games = s[
            (s["season"] == ctx["next_prediction_season"])
            & (s["week"] == ctx["next_prediction_week"])
        ]

        out: List[Dict[str, Any]] = []
        for _, g in games.iterrows():
            try:
                pr = predict_game(
                    PredictionRequest(
                        home_team=str(g["home_team"]),
                        away_team=str(g["away_team"]),
                        season=int(g["season"]),
                        week=int(g["week"]),
                    )
                )
                out.append(
                    {
                        "game_id": str(
                            g.get(
                                "game_id",
                                f"{g['season']}W{g['week']}-{g['away_team']}@{g['home_team']}",
                            )
                        ),
                        "season": int(g["season"]),
                        "week": int(g["week"]),
                        "home_team": str(g["home_team"]),
                        "away_team": str(g["away_team"]),
                        "kickoff": str(g.get("gameday", "TBD")),
                        "prediction": pr.model_dump(),
                    }
                )
            except Exception as e:
                out.append(
                    {"game_id": str(g.get("game_id", "unknown")), "error": str(e)}
                )

        return {
            "context": ctx,
            "games": out,
            "total_games": len(out),
            "successful_predictions": sum(1 for p in out if "prediction" in p),
        }
    except Exception as e:
        log.error("Next-week prediction error: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to predict next week: {e}")
