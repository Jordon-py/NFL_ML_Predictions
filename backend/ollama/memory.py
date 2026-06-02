"""Dataset memory for the NFL Ollama agent.

Data shape:
- Input CSV: one NFL game per row with identity columns such as `season`,
  `week`, `home_team`, and `away_team`, plus engineered model features.
- Memory summary: concise text with row count, seasons, teams, key columns, and
  a bounded column preview for the system prompt.
- Relevant context: filtered pandas row preview rendered as text for a user
  question.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

log = logging.getLogger(__name__)

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "datasets" / "game_features_20260531_clean.csv"


class NFLMemory:
    """Load the NFL feature dataset and build bounded prompt context."""

    def __init__(self, csv_path: Optional[str] = None):
        path = csv_path or os.getenv("NFL_DATASET_PATH", str(DEFAULT_CSV))
        self.csv_path = Path(path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"NFL dataset not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        log.info("Loaded %s NFL games from %s", f"{len(self.df):,}", self.csv_path.name)
        self.data_summary = self.summarize_data()

    def summarize_data(self) -> str:
        """Build a concise schema and key-stat summary for the system prompt."""
        df = self.df
        teams = sorted(df["home_team"].dropna().unique()) if "home_team" in df.columns else []
        seasons = sorted(df["season"].dropna().unique()) if "season" in df.columns else []

        key_cols = [
            "season", "week", "game_id", "home_team", "away_team",
            "home_points_for", "away_points_for", "point_diff", "winner",
            "home_prior_win_pct_3", "home_prior_pf_avg_3", "home_prior_pa_avg_3",
            "away_prior_win_pct_3", "away_prior_pf_avg_3", "away_prior_pa_avg_3",
            "home_prior_off_epa_per_play_3", "away_prior_off_epa_per_play_3",
            "spread_line", "total_line", "game_type",
            "home_win_prob_spread", "away_win_prob_spread",
            "surface", "roof", "temp", "wind",
        ]
        available = [c for c in key_cols if c in df.columns]

        return (
            f"DATASET: NFL game features ({len(df):,} rows, {len(df.columns)} columns)\n"
            f"SEASONS: {seasons[0] if seasons else 'unknown'}-{seasons[-1] if seasons else 'unknown'}\n"
            f"TEAMS ({len(teams)}): {', '.join(teams)}\n"
            f"KEY COLUMNS: {', '.join(available)}\n"
            f"ALL COLUMNS: {', '.join(df.columns[:60])}... ({len(df.columns)} total)\n"
        )

    def build_system_prompt(self) -> str:
        """System prompt that turns the LLM into an NFL data analyst."""
        return (
            "You are an NFL data analyst. You have access to a dataset of NFL games.\n"
            "Answer questions using ONLY the data described below.\n"
            "Be concise, use numbers and stats when possible.\n"
            "If a question can't be answered from this data, say so.\n\n"
            f"{self.data_summary}"
        )

    def relevant_context(self, question: str) -> str:
        """Return a small row preview filtered by mentioned team and season."""
        df = self.df
        q_upper = question.upper()

        if "home_team" not in df.columns:
            return "Dataset does not include a home_team column."

        teams = df["home_team"].dropna().unique()
        mentioned = [team for team in teams if team in q_upper]

        seasons = df["season"].dropna().unique() if "season" in df.columns else []
        mentioned_seasons = [season for season in seasons if str(int(season)) in question]

        filtered = df
        if mentioned:
            away_filter = filtered["away_team"].isin(mentioned) if "away_team" in filtered.columns else False
            filtered = filtered[(filtered["home_team"].isin(mentioned)) | away_filter]
        if mentioned_seasons:
            filtered = filtered[filtered["season"].isin(mentioned_seasons)]

        display_cols = [
            "season", "week", "home_team", "away_team",
            "home_points_for", "away_points_for", "winner",
        ]
        display_cols = [column for column in display_cols if column in filtered.columns]

        preview = filtered[display_cols].head(30)
        summary = f"Showing {len(preview)} of {len(filtered)} matching games:\n"
        return summary + preview.to_string(index=False)
