"""
Configuration helpers for the NFL ML Predictions backend.

This module centralizes environment loading, common path constants,
feature toggles, and CORS parsing so they can be reused across the app
and supporting scripts.
"""
# -------------------------------------
# IMPORTS -----
# -------------------------------------
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
# -------------------------------------

# Locations
BACKEND_DIR = Path(__file__).parent
BASE_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
LOG_DIR = BACKEND_DIR / "logs"
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"
FRONTEND_BUILD = FRONTEND_DIST  # Alias for compatibility

# Truthy parsing helper
TRUTHY = {"true", "t", "1", "yes", "y"}


def _load_env() -> None:
    """
    Load .env from backend or repo root.
    """
    dotenv_loaded = load_dotenv(BACKEND_DIR / ".env")
    if not dotenv_loaded:
        load_dotenv(BASE_DIR / ".env")
_load_env()


# Data and schedule defaults (lazily loaded to avoid startup failures)
# Use the latest engineered dataset in the backend root; aligns with production path.
DEFAULT_DATASET = BACKEND_DIR / "game_features_20251208.csv"
DEFAULT_SCHEDULE: Optional[object] = None  # Loaded on demand via load_schedule_data()


def load_schedule_data(year: int = 2025):
    """
    Lazily load NFL schedule data. Called only when needed.
    nflreadpy must be installed in the environment calling this.
    """
    try:
        import nflreadpy as nfl
        return nfl.load_schedules(year).to_pandas()
    except ImportError:
        return None

# Feature toggles
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "false").strip().lower() in TRUTHY
ALLOW_FALLBACK_PREDICTIONS = os.getenv("ALLOW_FALLBACK_PREDICTIONS", "false").strip().lower() in TRUTHY

# CORS -----------------------------------------------------------------
# Default curated origins with scheme to match browser origin headers.
DEFAULT_ALLOWED_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nfl-ml-predictions.vercel.app",
    "https://nfl-ml-predictions-pr5uahmqx-christopher-jordons-projects.vercel.app",
    "https://nfl-predict-6fghcp7sx-christopher-jordons-projects.vercel.app",
    "https://new-nfl-predict.vercel.app",
    "https://nfl-predict.vercel.app",
]

ALLOW_ORIGIN_REGEX = os.getenv("ALLOW_ORIGIN_REGEX","https://nfl-ml-predictions.vercel.app")


def _normalize_origin(origin: str) -> str:
    """
    Ensure origins include scheme and no trailing slash so they match
    browser Origin headers.
    """
    if not origin:
        return ALLOW_ORIGIN_REGEX
    candidate = origin.strip()
    # Remove trailing slash
    candidate = candidate[:-1] if candidate.endswith("/") else candidate
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    # Default to https scheme for bare hosts
    return f"https://{candidate}"


def parse_allowed_origins(raw: str | None = None) -> List[str]:
    """
    Parse ALLOWED_ORIGINS env var or fall back to curated defaults.
    """
    source = raw if raw is not None else os.getenv("ALLOWED_ORIGINS", "")
    entries = [_normalize_origin(o) for o in source.split(",") if o.strip()]
    entries = [e for e in entries if e]
    return entries or DEFAULT_ALLOWED_ORIGINS


def resolve_cors() -> Tuple[List[str], str | None]:
    """
    Build the effective allow_origins list and optional regex for FastAPI.
    """
    restrict = os.getenv("RESTRICT_CORS", "false").strip().lower() in TRUTHY
    if restrict:
        origins = parse_allowed_origins()
    else:
        # Broad default: allow curated list and defer to regex for previews.
        origins = DEFAULT_ALLOWED_ORIGINS
    regex = ALLOW_ORIGIN_REGEX or r"https://.*\.vercel\.app"
    return origins, regex

