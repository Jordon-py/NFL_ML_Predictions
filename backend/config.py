# ==========================================
# File: backend/config.py
# Role: Backend configuration and env resolution.
# Input Data: Environment variables.
# Output Data: Resolved paths and flags.
# Dependencies: __future__, os, pathlib, pandas
# Notes: Used during startup.
# ==========================================

# backend/config.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

TRUTHY = {"1", "true", "yes", "y", "on"}

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://nfl-ml-predictions.vercel.app",
    "https://nfl-predict.vercel.app",
    "https://new-nfl-predict.vercel.app",
]
DEFAULT_ORIGIN_REGEX = r"https://.*\.vercel\.app$"

def _resolve_data_dir() -> Path:
    env_data_dir = os.getenv("DATA_DIR")
    if env_data_dir:
        p = Path(env_data_dir)
        return (BASE_DIR / p).resolve() if not p.is_absolute() else p.resolve()
    dataset_dir = BASE_DIR / "data" / "dataset"
    if dataset_dir.exists():
        return dataset_dir.resolve()
    default_dir = BASE_DIR / "data" / "datasets"
    if default_dir.exists():
        return default_dir.resolve()
    return (BASE_DIR / "data").resolve()

# Keep your existing behavior: .env is owned by you; we just read env vars.
DATA_DIR = _resolve_data_dir()

def _resolve_models_dir() -> Path:
    env_models_dir = os.getenv("MODELS_DIR")
    if env_models_dir:
        return Path(env_models_dir).resolve()

    candidates = [
        BASE_DIR / "20260102" / "models",
        BASE_DIR / "data" / "models",
        BASE_DIR / "models",
    ]
    for candidate in candidates:
        if (candidate / "metadata.json").exists():
            return candidate.resolve()

    # Fall back to the original default for a clearer error path.
    return (BASE_DIR / "models").resolve()

MODELS_DIR = _resolve_models_dir()

def _split_origins(raw: str) -> list[str]:
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]

def resolve_cors():
    """
    Resolve allowed origins and regex from env, with safe defaults for dev + previews.
    """
    raw_origins = os.getenv("ALLOWED_ORIGINS") or os.getenv("CORS_ORIGINS", "")
    env_origins = _split_origins(raw_origins)
    restrict = os.getenv("RESTRICT_CORS", "false").strip().lower() in TRUTHY

    if restrict:
        origins = env_origins or DEFAULT_ALLOWED_ORIGINS
    else:
        origins = list(dict.fromkeys([*DEFAULT_ALLOWED_ORIGINS, *env_origins]))

    origin_regex = (
        os.getenv("ALLOW_ORIGIN_REGEX")
        or os.getenv("CORS_ORIGINS_REGEX")
        or DEFAULT_ORIGIN_REGEX
    )
    return origins, origin_regex

def load_schedule_data_safe(season: int):
    """
    Optional dependency. Returns a DataFrame or None.
    Tries common nflreadpy signatures to reduce breakage.
    """
    try:
        import nflreadpy as nfl
    except Exception:
        return None

    # Try a few variants (nflreadpy has varied docs/usage)
    for attempt in (
        lambda: nfl.load_schedules(season),                 # positional
        lambda: nfl.load_schedules(seasons=[season]),       # keyword list
        lambda: nfl.load_schedules(seasons=season),         # keyword int
    ):
        try:
            df = attempt()
            return df if isinstance(df, pd.DataFrame) else None
        except Exception:
            continue

    return None
