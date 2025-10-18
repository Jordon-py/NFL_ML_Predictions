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

DEFAULT_DATASET = DATA_DIR / "merged_game_features.csv"
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
    local = df.copy()
    local["time_key"] = local["season"].astype(int) * 100 + local["week"].astype(int)
    cutoff = season * 100 + week

    def latest_team_features(team: str, out_prefix: str) -> Dict[str, Any]:
        mask = (local["home_team"] == team) | (local["away_team"] == team)
        history = local[mask & (local["time_key"] < cutoff)]
        if history.empty:
            raise ValueError(f"No prior data for {team} before {season}-W{week}")
        last = history.sort_values("time_key").iloc[-1]
        was_home = str(last["home_team"]) == team
        source_prefix = "home_" if was_home else "away_"
        out: Dict[str, Any] = {}
        for col in local.columns:
            if not col.startswith(("home_prior_", "away_prior_")):
                continue
            if col.startswith(source_prefix):
                suffix = col[len(source_prefix) :]
                out[f"{out_prefix}{suffix}"] = last.get(col)
        return out

    home_features = latest_team_features(home, "home_")
    away_features = latest_team_features(away, "away_")
    feature_row: Dict[str, Any] = {**home_features, **away_features}

    prior_suffixes = {
        k[len("home_prior_") :] for k in feature_row if k.startswith("home_prior_")
    }
    for suf in prior_suffixes:
        hk = f"home_prior_{suf}"
        ak = f"away_prior_{suf}"
        if hk in feature_row and ak in feature_row:
            h_val, a_val = feature_row.get(hk), feature_row.get(ak)
            if h_val is not None and a_val is not None:
                feature_row[f"home_minus_away_{suf}"] = h_val - a_val

    for static_col in (
        "home_moneyline_prob",
        "away_moneyline_prob",
        "moneyline_prob_diff",
        "spread_line",
        "total_line",
        "home_rest",
        "away_rest",
        "rest_diff",
    ):
        feature_row.setdefault(static_col, np.nan)

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
        if rows.empty:
            row = _build_future_row(dataset_df, h, a, season, week)
        else:
            row = rows.iloc[0]
            if pd.notna(row.get("home_points_for")) and pd.notna(
                row.get("away_points_for")
            ):
                raise HTTPException(400, "Game completed; no prediction.")
        

        feature_names = _normalize_feature_cols(model_objects["raw_feature_columns"])
        data = {c: [row[c]] if c in row.index else [np.nan] for c in feature_names}
        if missing := [c for c in feature_names if c not in row.index]:
            log.warning("Missing features filled with NaN: %s", missing)

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
            if hasattr(bundle, "predict"):
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
