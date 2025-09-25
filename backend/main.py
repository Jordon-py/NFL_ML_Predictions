#!/usr/bin/env python
"""
NFL Game Prediction API (FastAPI)
Run: uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json, logging, logging.config, os, subprocess, sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------- Paths ----------
THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
BASE_DIR = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
FRONTEND_BUILD = BASE_DIR / "frontend" / "build"
LOG_DIR = BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Logging ----------
logging.config.dictConfig({
    "version": 1, "disable_existing_loggers": False,
    "formatters": {"d":{"format": "%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d - %(message)s"}},
    "handlers": {
        "console": {"class": "logging.StreamHandler","level": "INFO","formatter":"d","stream":"ext://sys.stdout"},
        "file": {"class":"logging.FileHandler","level":"DEBUG","formatter":"d","filename": str(LOG_DIR/"api.log"),"mode":"a"},
    },
    "root": {"level":"DEBUG","handlers":["console","file"]}
})
log = logging.getLogger("api")

# ---------- Constants ----------
DEFAULT_DATASET = DATA_DIR / "Nfl_data_sorted.csv"
DEFAULT_SCHEDULE = DATA_DIR / "Nfl_schedule_2025_2026.csv"

# ---------- Schemas ----------
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

# ---------- Teams ----------
TEAM_ABBREVIATIONS = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"
}
VALID_ABBRS = set(TEAM_ABBREVIATIONS.values())
def get_abbr(name: str) -> str:
    if name in VALID_ABBRS: return name
    if name in TEAM_ABBREVIATIONS: return TEAM_ABBREVIATIONS[name]
    raise ValueError(f"Unknown team: {name}")

# ---------- Feature normalization ----------
def _normalize_feature_cols(raw_cols: dict) -> List[str]:
    numeric = raw_cols.get("numeric", []) or []
    categorical = raw_cols.get("categorical", []) or []
    cols = list(numeric) + list(categorical)
    prefixed = [c for c in cols if c.startswith(("num__","cat__"))]
    if not prefixed: return cols
    def strip(c: str) -> str: return c[5:] if c.startswith(("num__","cat__")) else c
    return [strip(c) for c in cols]

# ---------- Model loading ----------
model_objects: Optional[Dict[str, Any]] = None
dataset_df: Optional[pd.DataFrame] = None

def load_objects() -> Dict[str, Any]:
    import joblib
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists(): raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text())

    pre = joblib.load(MODELS_DIR / meta["preprocessor"])
    mdl = meta.get("models", {"training_report":"training_report.json"})
    home_p_path = MODELS_DIR / "home_model.joblib"
    away_p_path = MODELS_DIR / "away_model.joblib"
    win_p_path  = MODELS_DIR / "win_clf_calibrated.joblib"
    for p in [home_p_path, away_p_path, win_p_path]:
        if not p.exists(): raise FileNotFoundError(f"Missing model file: {p}")

    return {
        "mode": "production",
        "preprocessor": pre,
        "home_model": joblib.load(home_p_path),
        "away_model": joblib.load(away_p_path),
        "win_model": joblib.load(win_p_path),
        "raw_feature_columns": meta.get("raw_feature_columns", {}),
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_objects, dataset_df
    log.info("Startup: loading models and dataset")
    model_objects = load_objects()
    ds_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    if not ds_path.exists(): raise RuntimeError(f"Dataset not found: {ds_path}")
    df = pd.read_csv(ds_path)
    if df.empty: raise RuntimeError("Dataset CSV is empty")
    df.columns = [c.strip() for c in df.columns]
    dataset_df = df
    log.info("Loaded dataset rows=%d cols=%d", len(df), len(df.columns))
    yield
    log.info("Shutdown complete")

# Get CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://localhost:3000").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

log.info(f"CORS Origins configured: {CORS_ORIGINS}")

app = FastAPI(title="NFL Game Prediction API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=CORS_ORIGINS,
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# ---------- Helpers ----------
def get_current_nfl_context() -> Dict[str, Any]:
    now = datetime.now()
    cur_season = now.year if now.month >= 8 else now.year - 1
    if dataset_df is not None and {"season","week","home_points_for","away_points_for"}.issubset(dataset_df.columns):
        done = dataset_df[dataset_df["home_points_for"].notna() & dataset_df["away_points_for"].notna()]
        if not done.empty:
            last = done.sort_values(["season","week"]).iloc[-1]
            last_s, last_w = int(last["season"]), int(last["week"])
            nxt_s, nxt_w = last_s, last_w + 1
            if nxt_w > 22: nxt_s, nxt_w = last_s + 1, 1
            return {"current_season": cur_season,"last_completed_season": last_s,"last_completed_week": last_w,
                    "next_prediction_season": nxt_s,"next_prediction_week": nxt_w,
                    "status": "nfl_season_active" if nxt_s == cur_season else "offseason"}
    return {"current_season": cur_season,"last_completed_season": cur_season,"last_completed_week": 0,
            "next_prediction_season": cur_season,"next_prediction_week": 1,"status": "preseason_or_early"}

def _build_future_row(df: pd.DataFrame, home: str, away: str, season: int, week: int) -> pd.Series:
    time_key = df["season"].astype(int)*100 + df["week"].astype(int)
    df = df.assign(time_key=time_key)
    def latest(team: str) -> Dict[str, Any]:
        mask = (df["home_team"]==team) | (df["away_team"]==team)
        before = df[mask & (df["time_key"] < season*100 + week)]
        if before.empty: raise ValueError(f"No prior data for {team} before {season}-W{week}")
        row = before.loc[before["time_key"].idxmax()]
        if str(row["home_team"]) == team:
            return {k: row.get(k) for k in [
                "home_prior_pa_avg_3","home_prior_pa_avg_5","home_prior_pf_avg_3","home_prior_pf_avg_5",
                "home_prior_win_pct_3","home_prior_win_pct_5"
            ]}
        else:
            return { 
                "home_prior_pa_avg_3": row.get("away_prior_pa_avg_3"),
                "home_prior_pa_avg_5": row.get("away_prior_pa_avg_5"),
                "home_prior_pf_avg_3": row.get("away_prior_pf_avg_3"),
                "home_prior_pf_avg_5": row.get("away_prior_pf_avg_5"),
                "home_prior_win_pct_3": row.get("away_prior_win_pct_3"),
                "home_prior_win_pct_5": row.get("away_prior_win_pct_5")
            }
    
    h = latest(home); a_full = latest(away)
    # rename away side
    a = {
        "away_prior_pa_avg_3": a_full["home_prior_pa_avg_3"],
        "away_prior_pa_avg_5": a_full["home_prior_pa_avg_5"],
        "away_prior_pf_avg_3": a_full["home_prior_pf_avg_3"],
        "away_prior_pf_avg_5": a_full["home_prior_pf_avg_5"],
        "away_prior_win_pct_3": a_full["home_prior_win_pct_3"],
        "away_prior_win_pct_5": a_full["home_prior_win_pct_5"],
    }
    feature_row = {}
    feature_row.update(h); feature_row.update(a)
    for base in ("pf_avg","pa_avg","win_pct"):
        for wnd in ("3","5"):
            H = feature_row.get(f"home_prior_{base}_{wnd}")
            A = feature_row.get(f"away_prior_{base}_{wnd}")
            if H is not None and A is not None:
                feature_row[f"home_minus_away_{base}_{wnd}"] = H - A
    return pd.Series(feature_row)

# ---------- Routes ----------
@app.get("/health", response_model=HealthResponse)
def health():
    if model_objects is None:
        return HealthResponse(status="unhealthy", mode="none", reason="models not loaded")
    return HealthResponse(status="healthy", mode=model_objects.get("mode"), reason="models loaded")

@app.get("/cors-debug")
def cors_debug():
    """Debug endpoint to check CORS configuration"""
    return {
        "cors_origins": CORS_ORIGINS,
        "env_cors_origins": os.getenv("CORS_ORIGINS", "not set"),
        "status": "active"
    }

@app.get("/debug")
def debug_info():
    debug = {"timestamp": datetime.now().isoformat()+"Z"}
    try:
        mpath = MODELS_DIR / "metadata.json"
        if mpath.exists(): debug["metadata"] = json.loads(mpath.read_text())
        tr = MODELS_DIR / "training_report.json"
        if tr.exists(): debug["training_report_present"] = "true"
    except Exception as e:
        debug["error"] = str(e)
    return debug

@app.get("/schedule/next-week", response_model=List[ScheduleGame])
def get_next_week_schedule():
    ctx = get_current_nfl_context()
    try:
        path = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
        if not path.exists(): raise HTTPException(404, f"Schedule not found: {path}")
        df = pd.read_csv(path)
        df["kickoff_ts_utc"] = pd.to_datetime(
            (df["gameday"].astype(str).str.strip()+" "+df["gametime"].astype(str).str.strip()).str.strip(),
            errors="coerce", utc=True
        )
        now = pd.Timestamp.now(tz="UTC")
        future = df[df["kickoff_ts_utc"].notna() & (df["kickoff_ts_utc"] >= now)]
        cur_wk = int(future["week"].min()) if not future.empty else int(df["week"].max())
        week_df = df[df["week"] == cur_wk]
        out: List[ScheduleGame] = []
        for _, r in week_df.iterrows():
            out.append(ScheduleGame(
                season=int(r["season"]), week=int(r["week"]),
                home_team=str(r["home_team"]), home_abbr=get_abbr(str(r["home_team"])),
                away_team=str(r["away_team"]), away_abbr=get_abbr(str(r["away_team"])),
                kickoff_iso=(r["kickoff_ts_utc"].isoformat() if pd.notna(r["kickoff_ts_utc"]) else "TBD"),
                game_id=str(r.get("game_id", f"{r['season']}W{r['week']}-{r['away_team']}@{r['home_team']}")),
            ))
        log.info("Schedule week %s games=%d", cur_wk, len(out))
        return out
    except HTTPException:
        raise
    except Exception as e:
        log.error("Schedule error: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to load schedule: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict_game(payload: PredictionRequest):
    if model_objects is None or dataset_df is None:
        raise HTTPException(500, "Models or dataset not loaded.")
    try:
        h = get_abbr(payload.home_team); a = get_abbr(payload.away_team)
        season, week = int(payload.season), int(payload.week)
        mask = (dataset_df["season"]==season) & (dataset_df["week"]==week) & \
               (dataset_df["home_team"]==h) & (dataset_df["away_team"]==a)
        rows = dataset_df.loc[mask]
        if rows.empty:
            row = _build_future_row(dataset_df, h, a, season, week)
        else:
            row = rows.iloc[0]
            if pd.notna(row.get("home_points_for")) and pd.notna(row.get("away_points_for")):
                raise HTTPException(400, "Game completed; no prediction.")
        raw_cols = model_objects.get("raw_feature_columns", {})
        features = _normalize_feature_cols(raw_cols)
        missing = [c for c in features if c not in row.index]
        if missing:
            raise HTTPException(500, f"Missing feature columns: {missing}")
        X = model_objects["preprocessor"].transform(pd.DataFrame({c:[row[c]] for c in features}))
        # Score regressors
        home_score = float(np.clip(model_objects["home_model"].predict(X)[0], 0.0, 70.0))
        away_score = float(np.clip(model_objects["away_model"].predict(X)[0], 0.0, 70.0))
        point_diff = round(home_score - away_score, 1)
        # Win probability from calibrated classifier
        home_prob = float(model_objects["win_model"].predict_proba(X)[0,1])
        return PredictionResponse(
            home_score=round(home_score,1),
            away_score=round(away_score,1),
            home_win_probability=round(home_prob,3),
            away_win_probability=round(1.0-home_prob,3),
            point_diff=point_diff,
            mode="models",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(400, f"Prediction failed: {e}")

@app.get("/predict/next-week")
def predict_next_week():
    if model_objects is None: raise HTTPException(500, "Models not loaded.")
    try:
        ctx = get_current_nfl_context()
        spath = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
        if not spath.exists(): raise HTTPException(404, "Schedule data not found")
        s = pd.read_csv(spath)
        games = s[(s["season"]==ctx["next_prediction_season"]) & (s["week"]==ctx["next_prediction_week"])]
        out = []
        for _, g in games.iterrows():
            try:
                pr = predict_game(PredictionRequest(
                    home_team=str(g["home_team"]), away_team=str(g["away_team"]),
                    season=int(g["season"]), week=int(g["week"])
                ))
                out.append({"game_id": str(g.get("game_id", f"{g['season']}W{g['week']}-{g['away_team']}@{g['home_team']}")),
                            "season": int(g["season"]), "week": int(g["week"]),
                            "home_team": str(g["home_team"]), "away_team": str(g["away_team"]),
                            "kickoff": str(g.get("gameday","TBD")), "prediction": pr.model_dump()})
            except Exception as e:
                out.append({"game_id": str(g.get("game_id","unknown")), "error": str(e)})
        return {"context": ctx, "games": out, "total_games": len(out),
                "successful_predictions": sum(1 for p in out if "prediction" in p)}
    except Exception as e:
        log.error("Next-week prediction error: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to predict next week: {e}")

@app.post("/retrain")
def retrain():
    global model_objects
    try:
        subprocess.run([sys.executable, str(BACKEND_DIR / "train_models.py")], check=True, capture_output=True, text=True)
        model_objects = load_objects()
        return {"detail": "Models retrained successfully."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Retraining failed: {e.stderr}")

# Frontend compatibility alias
@app.post("/train")
def train_alias():
    return retrain()

# Reports
@app.get("/report/training")
def report_training():
    tr = MODELS_DIR / "training_report.json"
    if not tr.exists(): raise HTTPException(404, "training_report.json not found")
    return json.loads(tr.read_text())

@app.get("/report/errors")
def report_errors(limit: int = 50):
    err = MODELS_DIR / "validation_errors.csv"
    if not err.exists(): raise HTTPException(404, "validation_errors.csv not found")
    df = pd.read_csv(err)
    df = df.sort_values("abs_error", ascending=False).head(int(limit))
    return {"rows": df.to_dict(orient="records"), "limit": int(limit)}

# Serve built frontend
_front = FRONTEND_DIST if FRONTEND_DIST.exists() else (FRONTEND_BUILD if FRONTEND_BUILD.exists() else "frontend/build")
    # comment: if (FRONTEND_BUILD if FRONTEND_BUILD.exists() else "frontend/build")
if _front:
    app.mount("/", StaticFiles(directory=str(_front), html=True), name="nfl-predict")
    log.info("Serving frontend from %s", _front)
else:
    log.warning("No frontend build found; not serving static files")
