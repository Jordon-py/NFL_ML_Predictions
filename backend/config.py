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

def _resolve_data_dir() -> Path:
    env_data_dir = os.getenv("DATA_DIR")
    if env_data_dir:
        p = Path(env_data_dir)
        return (BASE_DIR / p).resolve() if not p.is_absolute() else p.resolve()
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
        BASE_DIR / "models",
    ]
    for candidate in candidates:
        if (candidate / "metadata.json").exists():
            return candidate.resolve()

    # Fall back to the original default for a clearer error path.
    return (BASE_DIR / "models").resolve()

MODELS_DIR = _resolve_models_dir()

def resolve_cors():
    """
    Minimal CORS resolver; keep your existing env-driven CORS behavior if you want.
    """
    origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000") or os.getenv("CORS_ORIGINS", "https://.*\\.vercel\\.app$")
    origins = [o.strip() for o in origins_env.split(",")]
    print(len(origins), origins)
   
    if not origins:
        origins = ["*"]
    origin_regex = os.getenv("CORS_ORIGINS_REGEX", "https://.*\\.vercel\\.app$").strip()
    print(len(origin_regex), origin_regex)
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
