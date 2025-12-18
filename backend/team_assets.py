# ==========================================
# File: backend/team_assets.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: __future__, functools, pathlib, typing
# Notes: Shared utilities.
# ==========================================

"""
teams_assets.py
===============

A tiny, clear API for team branding assets (logos, wordmarks, colors).

Design goals
------------
- Simple resource path: GET /teams/{team_abbr}
- Fast: load CSV(s) once and serve from memory
- Predictable response schema (not raw CSV rows)
- Prefer non-square assets (SVG/wordmark) by default
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field

# Legacy -> modern abbreviation normalization
ALIASES: Dict[str, str] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "LA": "LAR",
}


def normalize_abbr(abbr: str) -> str:
    """Normalize team abbreviations to modern canonical forms."""
    a = (abbr or "").strip().upper()
    return ALIASES.get(a, a)


class TeamAsset(BaseModel):
    """
    What your frontend actually needs.

    - preferred_logo avoids squared logos by design
    - logo_svg is best for crisp rendering if available
    """
    team_abbr: str = Field(..., examples=["LAR"])
    team_name: Optional[str] = None
    team_color: Optional[str] = None
    team_color2: Optional[str] = None

    preferred_logo: Optional[str] = None
    logo_svg: Optional[str] = None
    wordmark: Optional[str] = None
    logo_espn: Optional[str] = None
    logo_wikipedia: Optional[str] = None


def _clean_str(x) -> Optional[str]:
    """Convert NaN/None -> None, else trimmed string."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


@lru_cache(maxsize=1)
def load_team_assets_map() -> Dict[str, TeamAsset]:
    """
    Load team assets from CSV(s) once and return a fast lookup map.

    Expected files in backend/data:
      - team_logos.csv  (your richer meta CSV)
      - team_logo.csv   (official NFL svg CSV, optional but recommended)
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    meta_path = data_dir / "team_logos.csv"
    svg_path = data_dir / "team_logo.csv"  # optional

    if not meta_path.exists():
        raise RuntimeError(f"Missing required file: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Normalize abbreviations for matching
    if "team_abbr" not in df_meta.columns:
        raise RuntimeError("team_logos.csv must contain 'team_abbr' column")

    df_meta["team_abbr"] = df_meta["team_abbr"].astype(str).map(normalize_abbr)

    # Optional: merge NFL SVGs if present
    df_svg = None
    if svg_path.exists():
        df_svg = pd.read_csv(svg_path)
        if "abbr" in df_svg.columns and "logo_url" in df_svg.columns:
            df_svg["abbr"] = df_svg["abbr"].astype(str).map(normalize_abbr)
            df_meta = df_meta.merge(
                df_svg[["abbr", "logo_url"]],
                left_on="team_abbr",
                right_on="abbr",
                how="left",
            )
        else:
            df_svg = None

    # Build map
    assets: Dict[str, TeamAsset] = {}

    for _, row in df_meta.iterrows():
        abbr = normalize_abbr(_clean_str(row.get("team_abbr")) or "")
        if not abbr:
            continue

        logo_svg = _clean_str(row.get("logo_url"))  # from team_logo.csv merge if present
        wordmark = _clean_str(row.get("team_wordmark"))
        logo_espn = _clean_str(row.get("team_logo_espn"))
        logo_wiki = _clean_str(row.get("team_logo_wikipedia"))

        # Preferred logo order:
        # 1) official SVG (best)
        # 2) wordmark (wide, not square)
        # 3) ESPN
        # 4) Wikipedia
        preferred = logo_svg or wordmark or logo_espn or logo_wiki

        assets[abbr] = TeamAsset(
            team_abbr=abbr,
            team_name=_clean_str(row.get("team_name")),
            team_color=_clean_str(row.get("team_color")),
            team_color2=_clean_str(row.get("team_color2")),
            preferred_logo=preferred,
            logo_svg=logo_svg,
            wordmark=wordmark,
            logo_espn=logo_espn,
            logo_wikipedia=logo_wiki,
        )

    return assets


# ---- FastAPI route functions (drop these into your main.py or a router)
