# ==========================================
# File: backend/services/feature_service.py
# Role: Feature building utilities for inference.
# Input Data: Dataset rows and team identifiers.
# Output Data: Model-ready feature vectors.
# Dependencies: pandas, numpy, typing
# Notes: Aligns columns with training schema.
# ==========================================

"""
FILE: backend/services/feature_service.py
PURPOSE: Single source of truth for building ML feature vectors for all models.
DATA SHAPES:
  - Input: pd.Series or Dict (raw game data).
  - Output: pd.DataFrame (model-ready vector).
KEY FUNCTIONS/CLASSES:
  - FeatureBuilder: Logic to roll-forward team stats and align columns.
SIDE EFFECTS / I/O: Reads from internal DataFrames; zero I/O side effects.
ERROR HANDLING: Returns neutral (mean) features if team history missing.
DEPENDENCIES: pandas, numpy, backend.main_helpers.InferenceBundle
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

class FeatureBuilder:
    def __init__(self, bundle: Any, dataset: pd.DataFrame):
        self.bundle = bundle
        self.dataset = dataset
        self.numeric_cols, self.categorical_cols = self._parse_schema()
        self.means = self._compute_means()

    def _parse_schema(self) -> Tuple[List[str], List[str]]:
        raw = self.bundle.meta.get("raw_feature_columns", {})
        if isinstance(raw, dict):
            return list(raw.get("numeric", [])), list(raw.get("categorical", []))
        return self.bundle.meta.get("feature_names", []), []

    def _compute_means(self) -> Dict[str, float]:
        return {col: self.dataset[col].mean() for col in self.numeric_cols if col in self.dataset.columns}

    def build_live_vector(self, home: str, away: str, season: int, week: int) -> pd.DataFrame:
        # Roll forward logic (simplified for consolidation)
        row = self._create_empty_row(home, away, season, week)
        
        # Merge recent stats for home and away
        h_stats = self._get_last_stats(home, season, week, "home")
        a_stats = self._get_last_stats(away, season, week, "away")
        
        row.update(h_stats)
        row.update(a_stats)
        
        # Fill missing with means
        for col in self.numeric_cols:
            if col not in row or pd.isna(row[col]):
                row[col] = self.means.get(col, 0.0)
                
        # Transform using preprocessor if available
        X = pd.DataFrame([row])
        if hasattr(self.bundle, 'preprocessor'):
            return self.bundle.preprocessor.transform(X)
        return X

    def _create_empty_row(self, home, away, season, week) -> Dict[str, Any]:
        row = {"season": season, "week": week}
        for col in self.categorical_cols:
            if col == "home_team": row[col] = home
            elif col == "away_team": row[col] = away
        return row

    def _get_last_stats(self, team: str, season: int, week: int, side: str) -> Dict[str, float]:
        # Minimalist stats extractor
        # Logic: find latest completed game for team, return stats prefixed with side
        mask = ((self.dataset["home_team"] == team) | (self.dataset["away_team"] == team)) & \
               ((self.dataset["season"] < season) | ((self.dataset["season"] == season) & (self.dataset["week"] < week)))
        
        history = self.dataset[mask]
        if history.empty: return {}
        
        latest = history.iloc[-1]
        l_side = "home" if latest["home_team"] == team else "away"
        
        stats = {}
        prefix = f"{side}_"
        l_prefix = f"{l_side}_"
        
        for col in self.numeric_cols:
            if col.startswith(prefix):
                source_col = l_prefix + col[len(prefix):]
                if source_col in latest:
                    stats[col] = latest[source_col]
        return stats
