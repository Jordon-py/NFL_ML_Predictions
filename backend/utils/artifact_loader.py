"""
Artifact readiness helpers for backend startup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from backend.config import MODELS_DIR, DATA_DIR
except ImportError:
    from config import MODELS_DIR, DATA_DIR

log = logging.getLogger(__name__)


def ensure_artifacts() -> None:
    """
    Best-effort startup checks for required paths.
    Creates local writable folders and logs missing inputs.
    """
    predictions_dir = Path("backend") / "Predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    if not Path(MODELS_DIR).exists():
        missing.append(f"MODELS_DIR={MODELS_DIR}")
    if not Path(DATA_DIR).exists():
        missing.append(f"DATA_DIR={DATA_DIR}")

    if missing:
        strict = os.getenv("STRICT_ARTIFACT_CHECKS", "false").strip().lower() in {"1", "true", "yes", "on"}
        message = f"Artifact path(s) not found: {', '.join(missing)}"
        if strict:
            raise FileNotFoundError(message)
        log.warning(message)
