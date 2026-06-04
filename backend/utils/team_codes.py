"""
backend/utils/team_codes.py
---------------------------
Shared, canonical team-code normalization used across:
  - schedule parsing (/schedule/next-week)
  - inference-row building (roll-forward features)
  - team metadata lookup (/teams/logos)

Why this exists:
  Multiple data sources disagree on abbreviations (e.g. WSH vs WAS, LA vs LAR).
  Centralizing normalization prevents subtle UI mismatches (missing logos/names)
  and model-input mismatches (history lookup failing due to alias codes).
"""

from __future__ import annotations

from typing import Any, Dict


# Legacy / alternate abbreviations -> modern canonical codes.
# Keep this map small and explicit; add entries only when encountered in data.
TEAM_ABBR_ALIASES: Dict[str, str] = {
    # Rams
    "LA": "LAR",
    "STL": "LAR",
    "RAMS": "LAR",
    # Chargers
    "SD": "LAC",
    "CHARGERS": "LAC",
    # Raiders
    "OAK": "LV",
    "RAIDERS": "LV",
    # Commanders
    "WSH": "WAS",
    "COMMANDERS": "WAS",
    "REDSKINS": "WAS",
    # Jaguars
    "JAC": "JAX",
    "JAGUARS": "JAX",
}


def normalize_team_code(value: Any) -> str:
    """Normalize team identifiers to canonical uppercase abbreviations."""
    if value is None:
        return ""
    raw = str(value).strip().upper()
    if not raw:
        return ""
    return TEAM_ABBR_ALIASES.get(raw, raw)

