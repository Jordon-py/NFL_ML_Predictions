# ==========================================
# File: backend/services/prediction_service.py
# Role: Prediction service orchestrating model inference.
# Input Data: PredictionRequest + model artifacts.
# Output Data: PredictionResponse instances.
# Dependencies: logging, typing, numpy, pandas
# Notes: Delegates to feature builders and models.
# ==========================================

# backend/services/prediction_service.py
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    ScorePrediction,
    WinnerPrediction,
    SimulationMetrics,
)
from backend.services.inference_row import build_model_input_row, build_team_history_cache
from backend.config import load_schedule_data_safe

logger = logging.getLogger(__name__)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _extract_home_proba(win_clf: Any, proba_row: np.ndarray) -> Optional[float]:
    """
    Try hard to map classifier probabilities -> P(home wins).
    Handles common class encodings: ['home','away'], ['HOME','AWAY'], [0,1], [False,True]
    """
    classes = getattr(win_clf, "classes_", None)
    if classes is None:
        return None

    classes_list = list(classes)

    # String labels
    lowered = [str(c).strip().lower() for c in classes_list]
    if "home" in lowered:
        return float(proba_row[lowered.index("home")])
    if "away" in lowered and "home" not in lowered and len(lowered) == 2:
        # If only away/home style and home not found, can't safely infer
        return None
    if "true" in lowered:
        return float(proba_row[lowered.index("true")])

    # Numeric labels: assume 1 == home-win
    for key in (1, True):
        if key in classes_list:
            return float(proba_row[classes_list.index(key)])

    return None

class PredictionService:
    """
    Single responsibility: orchestrate inference for /predict.
    """

    def __init__(self, bundle: Any, dataset: pd.DataFrame):
        self.bundle = bundle
        self.dataset = dataset

        # Expected attributes (match your bundle loader)
        self.home_reg = getattr(bundle, "home_model", None) or getattr(bundle, "home_reg", None)
        self.away_reg = getattr(bundle, "away_model", None) or getattr(bundle, "away_reg", None)
        self.win_clf = getattr(bundle, "hist_win_clf", None) or getattr(bundle, "win_clf", None)

        self.preprocessor = getattr(bundle, "preprocessor", None)
        self.raw_feature_columns = getattr(bundle, "raw_feature_columns", None)

        # Cache schedule per season (lazy-loaded)
        self._schedule_cache: dict[int, pd.DataFrame] = {}
        # Cache team history once to avoid re-scanning the full dataset each call.
        self._team_history_cache = build_team_history_cache(dataset)

    def _get_schedule_df(self, season: int) -> Optional[pd.DataFrame]:
        if season in self._schedule_cache:
            return self._schedule_cache[season]
        df = load_schedule_data_safe(season)
        if isinstance(df, pd.DataFrame):
            self._schedule_cache[season] = df
        return df

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        if self.home_reg is None or self.away_reg is None:
            raise RuntimeError("Models not loaded: home_reg/away_reg missing")

        schedule_df = self._get_schedule_df(req.season)

        row_df, source = build_model_input_row(
            dataset_df=self.dataset,
            preprocessor=self.preprocessor,
            season=req.season,
            week=req.week,
            home_team=req.home_team,
            away_team=req.away_team,
            schedule_df=schedule_df,
            raw_feature_columns=self.raw_feature_columns,
            team_history_cache=self._team_history_cache,
        )

        # Transform
        X = row_df
        if self.preprocessor is not None:
            X = self.preprocessor.transform(row_df)

        # Score regressors
        p_home = float(self.home_reg.predict(X)[0])
        p_away = float(self.away_reg.predict(X)[0])
        point_diff = p_home - p_away

        # Winner probabilities
        win_classifier_used = False
        proba_home = None

        if self.win_clf is not None and hasattr(self.win_clf, "predict_proba"):
            probs = np.asarray(self.win_clf.predict_proba(X)[0], dtype=float)
            mapped = _extract_home_proba(self.win_clf, probs)
            if mapped is not None and np.isfinite(mapped):
                proba_home = float(mapped)
                win_classifier_used = True

        # Fallback probability from point diff (simple + stable)
        if proba_home is None:
            # scale 7 ~= one touchdown; keeps logits reasonable
            proba_home = float(_sigmoid(point_diff / 7.0))

        proba_away = float(1.0 - proba_home)

        # Winner label should be TEAM ABBR for your API payload
        home_code = str(req.home_team).strip().upper()
        away_code = str(req.away_team).strip().upper()
        winner_team = home_code if proba_home >= 0.5 else away_code

        return PredictionResponse(
            scores=ScorePrediction(home_score=p_home, away_score=p_away),
            winner=WinnerPrediction(
                winner=winner_team,
                proba_home=proba_home,
                proba_away=proba_away,
                proba_draw=None,
            ),
            # Optional: keep it None to stay “prediction-only”
            simulation_metrics=None,  # SimulationMetrics(...) if you ever re-add MC
            prediction_source=source,
            win_classifier_used=win_classifier_used,
        )
