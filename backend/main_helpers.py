# ==========================================
# File: backend/main_helpers.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: logging, json, pathlib, dataclasses
# Notes: Shared utilities.
# ==========================================

"""
FILE: backend/main_helpers.py
PURPOSE: Auxiliary functions and classes for main.py (model loading, history, etc).
DATA SHAPES:
  - InferenceBundle: Container for ML artifacts.
KEY FUNCTIONS/CLASSES:
  - load_inference_bundle, load_dataset_df.
  - _append_prediction_history_to_disk (flat history entries).
"""

import logging
import json
import os
from pathlib import Path, PureWindowsPath
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from threading import Lock
from datetime import datetime, timezone
import joblib
import pandas as pd
import numpy as np
from backend.config import TRUTHY, load_schedule_data_safe
from backend.utils.team_codes import TEAM_ABBR_ALIASES, normalize_team_code

log = logging.getLogger(__name__)

prediction_history_entries: List[Dict[str, Any]] = []
_prediction_history_lock = Lock()
PREDICTION_HISTORY_MAX = 1000
PREDICTION_HISTORY_PATH = Path("backend/Predictions/prediction_history.json")

@dataclass(frozen=True)
class InferenceBundle:
    meta: Dict[str, Any]
    report: Dict[str, Any]
    preprocessor: Any
    home_model: Any
    away_model: Any
    hist_win_clf: Any

    @property
    def raw_feature_columns(self) -> List[str]:
        cols = self.meta.get("raw_feature_columns", {}) or {}
        if isinstance(cols, dict):
            num = cols.get("numeric", []) or []
            cat = cols.get("categorical", []) or []
            return [*list(num), *list(cat)]
        return self.meta.get("feature_names", []) or []

def load_inference_bundle(models_dir: Path) -> InferenceBundle:
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    report_path = models_dir / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    artifacts = meta.get("artifacts", meta)

    def _normalize_path_value(value: Any) -> str:
        val_str = str(value).strip()
        if len(val_str) >= 2 and val_str[0] == val_str[-1] and val_str[0] in ("'", '"'):
            val_str = val_str[1:-1].strip()
        return val_str

    def _looks_like_windows_abs(value: str) -> bool:
        try:
            win_path = PureWindowsPath(value)
        except Exception:
            return False
        return bool(win_path.drive) and win_path.is_absolute()

    def _resolve_path(key: str, default: Optional[str] = None) -> Optional[Path]:
        val = artifacts.get(key) or meta.get(key) or default
        if not val:
            return None
        val_str = _normalize_path_value(val)
        if not val_str:
            return None
        p = Path(val_str)
        if p.is_absolute() and p.exists():
            return p

        candidate = models_dir / p
        if candidate.exists():
            return candidate

        if _looks_like_windows_abs(val_str):
            win_path = PureWindowsPath(val_str)
            return models_dir / win_path.name

        return models_dir / p.name

    pre_path = _resolve_path("preprocessor", "preprocessor.joblib")
    home_path = _resolve_path("home_model", "home_model.joblib")
    away_path = _resolve_path("away_model", "away_model.joblib")
    hist_path = _resolve_path("hist_win_model") or _resolve_path("win_model") or _resolve_path("win_clf_calibrated")

    return InferenceBundle(
        meta=meta,
        report=report,
        preprocessor=joblib.load(pre_path),
        home_model=joblib.load(home_path),
        away_model=joblib.load(away_path),
        hist_win_clf=joblib.load(hist_path) if hist_path and hist_path.exists() else None,
    )

def _score_dataset_file(path: Path, expected_features: List[str]) -> Optional[tuple[int, int]]:
    try:
        cols = pd.read_csv(path, nrows=0).columns
    except Exception as e:
        log.warning("Dataset header read failed for %s: %s", path, e)
        return None
    missing = sum(1 for c in expected_features if c not in cols)
    return missing, len(cols)

def _pick_best_dataset(files: List[Path], expected_features: List[str]) -> Optional[tuple[Path, int]]:
    scored: List[tuple[int, int, str, Path]] = []
    for path in files:
        score = _score_dataset_file(path, expected_features)
        if score is None:
            continue
        missing, col_count = score
        scored.append((missing, -col_count, path.name, path))
    if not scored:
        return None
    scored.sort()
    best = scored[0]
    return best[3], best[0]

def load_dataset_df(data_dir: Path, expected_features: Optional[List[str]] = None) -> pd.DataFrame:
    latest_manifest = data_dir / "latest_dataset.json"
    if latest_manifest.exists():
        try:
            payload = json.loads(latest_manifest.read_text(encoding="utf-8"))
            clean_dataset_path = payload.get("clean_dataset_path")
            if clean_dataset_path:
                manifest_path = Path(clean_dataset_path)
                if manifest_path.exists():
                    return pd.read_csv(manifest_path)
        except Exception as exc:
            log.warning("Could not load latest_dataset.json from %s: %s", latest_manifest, exc)

    dataset_path = os.getenv("DATASET_PATH")
    if dataset_path:
        p = Path(dataset_path)
        if not p.is_absolute():
            cwd_candidate = p.resolve()
            data_candidate = (data_dir / p).resolve()
            p = cwd_candidate if cwd_candidate.exists() else data_candidate
        else:
            p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"DATASET_PATH not found: {p}")
        if expected_features:
            score = _score_dataset_file(p, expected_features)
            if score is not None:
                missing, _ = score
                if missing:
                    candidates = sorted(data_dir.glob("game_features_*.csv")) if data_dir.is_dir() else []
                    best = _pick_best_dataset(candidates, expected_features) if candidates else None
                    if best and best[1] == 0 and best[0] != p:
                        log.warning(
                            "DATASET_PATH missing %d expected features; using %s",
                            missing,
                            best[0],
                        )
                        return pd.read_csv(best[0])
        return pd.read_csv(p)

    if data_dir.is_file():
        return pd.read_csv(data_dir)

    files = sorted(data_dir.glob("game_features_*.csv"))
    if not files:
        files = sorted(data_dir.glob("**/game_features_*.csv"))
    if not files:
        raise FileNotFoundError(f"No game_features_*.csv in {data_dir}")
    if expected_features:
        best = _pick_best_dataset(files, expected_features)
        if best:
            if best[1] > 0:
                log.warning(
                    "No dataset fully matches model features; best missing %d: %s",
                    best[1],
                    best[0],
                )
            return pd.read_csv(best[0])
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one game_features_*.csv in {data_dir}; set DATASET_PATH explicitly."
        )
    return pd.read_csv(files[0])

def _build_game_id_from_request(request_payload: Dict[str, Any]) -> Optional[str]:
    """Best-effort game_id builder to keep history entries consistent."""
    if not isinstance(request_payload, dict):
        return None
    season = request_payload.get("season")
    week = request_payload.get("week")
    home = request_payload.get("home_team") or request_payload.get("home_abbr")
    away = request_payload.get("away_team") or request_payload.get("away_abbr")
    home = str(home).strip().upper() if home is not None else None
    away = str(away).strip().upper() if away is not None else None
    parts = [season, week, home, away]
    normalized = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    return "-".join(normalized) if normalized else None

def _append_prediction_history_to_disk(request_payload: Dict[str, Any], prediction_payload: Dict[str, Any]) -> None:
    """Append a unified, flat prediction entry for frontend history views."""
    global prediction_history_entries
    entry: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(prediction_payload, dict):
        entry.update(prediction_payload)
    if isinstance(request_payload, dict):
        entry.setdefault("season", request_payload.get("season"))
        entry.setdefault("week", request_payload.get("week"))
        home = request_payload.get("home_team") or request_payload.get("home_abbr")
        away = request_payload.get("away_team") or request_payload.get("away_abbr")
        entry.setdefault("home_team", str(home).strip().upper() if home is not None else None)
        entry.setdefault("away_team", str(away).strip().upper() if away is not None else None)
        entry.setdefault("game_id", _build_game_id_from_request(request_payload))
    with _prediction_history_lock:
        prediction_history_entries = [entry] + (prediction_history_entries or [])
        prediction_history_entries = prediction_history_entries[:PREDICTION_HISTORY_MAX]
        try:
            PREDICTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            PREDICTION_HISTORY_PATH.write_text(json.dumps(prediction_history_entries, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"History persist failed: {e}")

# ----------------------------------------------------
# Shared Schedule & Team Helpers (Referenced by main.py and routes.py)
# ----------------------------------------------------

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

def _current_season_week() -> tuple[int, int]:
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
            backend_dir / "data" / "NFL_Schedule.csv",
            backend_dir / "NFL_Schedule.csv",
            repo_root / "data" / "NFL_Schedule.csv",
            repo_root / "NFL_Schedule.csv",
        ]
    )

    for c in candidates:
        if c.exists():
            df = _load_schedule_csv(c)
            if isinstance(df, pd.DataFrame):
                return df

    return pd.DataFrame()

def parse_kickoff(row: pd.Series) -> Optional[datetime]:
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

def _infer_next_week(schedule_df: pd.DataFrame) -> tuple[int, int]:
    """Determine the next week of games from a schedule DataFrame."""
    season_default, week_default = _current_season_week()
    if schedule_df is None or schedule_df.empty:
        return (season_default, week_default)

    season_col = _pick_col(schedule_df, _SEASON_COLS)
    week_col = _pick_col(schedule_df, _WEEK_COLS)

    # Determine season
    season = season_default
    df_use = schedule_df
    if season_col:
        seasons = pd.to_numeric(schedule_df[season_col], errors="coerce").dropna().astype(int)
        if not seasons.empty:
            season = season_default if season_default in seasons.values else int(seasons.max())
        df_use = schedule_df[pd.to_numeric(schedule_df[season_col], errors="coerce") == season]

    if not week_col:
        return (season, week_default)

    weeks = pd.to_numeric(df_use[week_col], errors="coerce").dropna().astype(int)
    if weeks.empty:
        return (season, week_default)

    # Try to find the first week with future games
    kickoff_series = df_use.apply(parse_kickoff, axis=1)
    if kickoff_series.notna().any():
        try:
            ts_series = pd.to_datetime(kickoff_series, utc=True)
            ts_now = pd.Timestamp.now(timezone.utc)
            future_mask = ts_series >= ts_now
            if future_mask.any():
                future_weeks = pd.to_numeric(df_use.loc[future_mask, week_col], errors="coerce").dropna().astype(int)
                if not future_weeks.empty:
                    return (season, int(future_weeks.min()))
        except Exception:
            pass

    # Fallback: first week >= current week, or max week
    candidate_weeks = weeks[weeks >= week_default]
    if not candidate_weeks.empty:
        return (season, int(candidate_weeks.min()))

    return (season, int(weeks.max()))


def _dedupe_schedule_rows(schedule_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame() if schedule_df is None else schedule_df

    home_col = _pick_col(schedule_df, _HOME_COLS)
    away_col = _pick_col(schedule_df, _AWAY_COLS)
    game_id_col = _pick_col(schedule_df, _GAME_ID_COLS)
    keep_index: List[Any] = []
    seen_keys: set[str] = set()
    duplicates_removed = 0

    for index, row in schedule_df.iterrows():
        home = normalize_team_code(row.get(home_col, "") if home_col else row.get("home", ""))
        away = normalize_team_code(row.get(away_col, "") if away_col else row.get("away", ""))
        key = f"{season}-{week}-{home}-{away}" if home and away else ""

        if not key and game_id_col:
            raw_id = row.get(game_id_col)
            if pd.notna(raw_id):
                key = str(raw_id).strip().replace("_", "-").upper()

        if key and key in seen_keys:
            duplicates_removed += 1
            continue
        if key:
            seen_keys.add(key)
        keep_index.append(index)

    if duplicates_removed:
        log.warning(
            "Removed %d duplicate schedule rows for season=%s week=%s.",
            duplicates_removed,
            season,
            week,
        )

    return schedule_df.loc[keep_index].reset_index(drop=True)


def select_next_week_rows(schedule_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    season, week = _infer_next_week(schedule_df)
    df_use = schedule_df.copy() if isinstance(schedule_df, pd.DataFrame) else pd.DataFrame()

    season_col = _pick_col(df_use, _SEASON_COLS)
    week_col = _pick_col(df_use, _WEEK_COLS)

    if season_col:
        df_use = df_use[pd.to_numeric(df_use[season_col], errors="coerce") == season]
    if week_col:
        df_use = df_use[pd.to_numeric(df_use[week_col], errors="coerce") == week]

    return _dedupe_schedule_rows(df_use, season, week), season, week

def get_team_meta(csv_path: Path) -> Dict[str, Dict[str, str]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        log.warning("Failed to read team logos CSV %s: %s", csv_path, exc)
        return {}
    if df is None or df.empty:
        return {}

    df.columns = [c.strip() for c in df.columns]

    abbr_candidates = ["team_abbr", "abbr", "team", "team_code", "team_id"]
    logo_candidates = ["team_logo_squared", "team_logo_espn", "logoUrl", "logo_url", "logo", "team_logo", "url"]
    name_candidates = ["team_name", "name", "team", "home_team"]
    primary_color_candidates = ["team_color", "primary_color", "color", "color1"]
    secondary_color_candidates = ["team_color2", "secondary_color", "color2"]
    wordmark_candidates = ["team_wordmark", "wordmark"]

    def pick(colnames: List[str]) -> Optional[str]:
        for c in colnames:
            if c in df.columns:
                return c
        lower_map = {c.lower(): c for c in df.columns}
        for c in colnames:
            hit = lower_map.get(c.lower())
            if hit:
                return hit
        return None

    abbr_col = pick(abbr_candidates)
    logo_col = pick(logo_candidates)
    name_col = pick(name_candidates)
    primary_col = pick(primary_color_candidates)
    secondary_col = pick(secondary_color_candidates)
    wordmark_col = pick(wordmark_candidates)

    if not abbr_col or not logo_col:
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        abbr = normalize_team_code(r.get(abbr_col, ""))
        logo = str(r.get(logo_col, "")).strip()
        if not abbr or not logo:
            continue
        item: Dict[str, str] = {"logoUrl": logo}
        if name_col:
            nm = str(r.get(name_col, "")).strip()
            if nm: item["name"] = nm
        if primary_col:
            primary = str(r.get(primary_col, "")).strip()
            if primary: item["primaryColor"] = primary
        if secondary_col:
            secondary = str(r.get(secondary_col, "")).strip()
            if secondary: item["secondaryColor"] = secondary
        if wordmark_col:
            wordmark = str(r.get(wordmark_col, "")).strip()
            if wordmark: item["wordmark"] = wordmark
        out[abbr] = item

    # Backward-compatible aliases (e.g., WSH -> WAS) so lookups succeed even if callers
    # have not normalized codes on their side.
    for alias, canonical in TEAM_ABBR_ALIASES.items():
        if canonical in out and alias not in out:
            out[alias] = out[canonical]
    return out
