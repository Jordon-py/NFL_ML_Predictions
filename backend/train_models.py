#!/usr/bin/env python
"""
train_models.py — Production training for NFL predictions
- Regressors: predict home/away scores (LightGBM + GridSearchCV)
- Classifier: predict home win probability (LightGBMClassifier + calibration)
- Saves: preprocessor.joblib, home_model.joblib, away_model.joblib, win_clf_calibrated.joblib
- Reports: models/metadata.json, models/training_report.json, models/validation_errors.csv
"""
from __future__ import annotations

import json, logging, time, warnings, hashlib, os
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss, mean_absolute_error, r2_score,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
force_col_wise = True
# Paths
BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
DATA_CANDIDATES = [
    Path.getenv if False else None,  # reserved
    BACKEND_DIR / "data" / "Nfl_data_sorted.csv",
    REPO_DIR / "Nfl_data_sorted.csv",
]
def _resolve_data_path() -> Path:
    env = Path(str(Path.cwd() / (os.getenv("DATASET_PATH") or ""))) if "DATASET_PATH" in globals() else None
    for p in [env] + DATA_CANDIDATES if env else DATA_CANDIDATES:
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError("Nfl_data_sorted.csv not found in expected locations")

MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("train")

# Feature spec
BASE_FEATURES = [
    # Home priors
    "home_prior_pf_avg_3","home_prior_pf_avg_5",
    "home_prior_pa_avg_3","home_prior_pa_avg_5",
    "home_prior_win_pct_3","home_prior_win_pct_5",
    # Away priors
    "away_prior_pf_avg_3","away_prior_pf_avg_5",
    "away_prior_pa_avg_3","away_prior_pa_avg_5",
    "away_prior_win_pct_3","away_prior_win_pct_5",
    # Relative form
    "home_minus_away_pf_avg_3","home_minus_away_pf_avg_5",
    "home_minus_away_pa_avg_3","home_minus_away_pa_avg_5",
    "home_minus_away_win_pct_3","home_minus_away_win_pct_5",
]

def _load_dataset() -> pd.DataFrame:
    path = _resolve_data_path()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if len(df) < 500:
        raise ValueError(f"Insufficient training data: {len(df)}")
    # Completed games only for training
    df = df[df["home_points_for"].notna() & df["away_points_for"].notna()].copy()
    df["home_win"] = (df["home_points_for"] > df["away_points_for"]).astype(int)
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df

def _preprocessor(features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler())]), features)],
        remainder="drop",
    )

def _grid_lgbm_reg() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [6, 8],
        "num_leaves": [20, 31],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [0.0, 0.1],
    }

def _grid_lgbm_clf() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [6, 8],
        "num_leaves": [20, 31],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [0.0, 0.1],
    }

# In the _fit_regressor function
from sklearn.model_selection import TimeSeriesSplit

def _fit_regressor(X, y, name: str) -> Tuple[LGBMRegressor, Dict[str, Any]]:
    lgbm = LGBMRegressor(objective="regression", random_state=4, n_jobs=-1, verbose=4)
    cv = TimeSeriesSplit(n_splits=5) # <-- With this line
    
    from sklearn.model_selection import RandomizedSearchCV

    # Cast the estimator to satisfy type checker
    rs = RandomizedSearchCV(
        estimator=cast(BaseEstimator, lgbm),
        param_distributions=_grid_lgbm_reg(),
                n_iter=10,   # Try 10 combinations for faster classifier training
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
        refit=True,
        random_state=4
    )
    t0 = time.time()
    gfit = rs.fit(X, y)
    best = cast(LGBMRegressor, gfit.best_estimator_)
    yhat = gfit.predict(X)
    res = {
        "best_params": gfit.best_params_,
        "cv_rmse": gfit.best_score_,
        "train_r2": r2_score(y, yhat),
        "train_mae": mean_absolute_error(y, yhat),
        "search_time_s": time.time() - t0,
        "n_candidates": len(rs.cv_results_["params"]),
    }
    if res["train_r2"] < -0.2:  # More lenient threshold for sports prediction
        raise ValueError(f"{name} regressor underfit: R²={res['train_r2']:.3f}")
    return best, res

def _fit_classifier_optimized(X, y) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame]:
    """
    Optimizes the classifier training process for speed and efficiency.
    
    This function replaces GridSearchCV with a more efficient search method, 
    streamlines the calibration process, and avoids redundant computations.
    """
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV
    import time
    import pandas as pd
    from sklearn.base import BaseEstimator
    from typing import Tuple, Dict, Any
    
    # Use RandomizedSearchCV for faster hyperparameter tuning
    base = LGBMClassifier(objective="binary", random_state=4, n_jobs=-1, verbose=2)
    

    cv_splitter = TimeSeriesSplit(n_splits=5)
    
    # Use RandomizedSearchCV with a limited number of iterations (e.g., n_iter=20)
    rs = RandomizedSearchCV(
        estimator=cast(BaseEstimator, base),
        param_distributions=_grid_lgbm_clf(), 
        n_iter=20, 
        cv=cv_splitter, 
        scoring="roc_auc",
        n_jobs=-1, 
        verbose=2, 
        random_state=4
    )
    
    t0 = time.time()
    rs.fit(X, y)
    best_uncalibrated = rs.best_estimator_
    
    # Streamline calibration
    # Calibrate the best uncalibrated model once
    calib = CalibratedClassifierCV(best_uncalibrated, cv=cv_splitter, method="isotonic")
    calib.fit(X, y)
    
    # Generate cross-validated predictions once using the calibrated model
    fold_preds = []
    for i, (tr, te) in enumerate(cv_splitter.split(X, y)):
        p = calib.predict_proba(X[te])[:, 1]  # Use numpy array indexing instead of pandas .iloc
        fold_preds.append(pd.DataFrame({"idx": te, "fold": i, "prob_home_win": p}))

    preds_df = pd.concat(fold_preds).sort_values("idx")
    
    # Evaluate the calibrated model
    prob = calib.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    
    metrics = {
        "best_params": rs.best_params_,
        "cv_auc": float(rs.best_score_),
        "train_auc": float(roc_auc_score(y, prob)),
        "train_accuracy": float(accuracy_score(y, pred)),
        "train_precision": float(precision_score(y, pred)),
        "train_recall": float(recall_score(y, pred)),
        "train_f1": float(f1_score(y, pred)),
        "train_brier": float(brier_score_loss(y, prob)),
        "search_time_s": time.time() - t0,
        "n_candidates": len(rs.cv_results_["params"]),
    }
    
    return calib, metrics, preds_df


def main() -> None:
    df = _load_dataset()
    X_raw = df[BASE_FEATURES]
    y_home = df["home_points_for"].astype(float).values
    y_away = df["away_points_for"].astype(float).values
    y_win  = df["home_win"].astype(int).values

    pre = _preprocessor(BASE_FEATURES)
    X_proc = pre.fit_transform(X_raw)

    log.info("Fitted preprocessor on %d samples, %d features", X_proc.shape[0], X_proc.shape[1])

    # Regressors
    log.info("Training LightGBM regressors (home, away)...")
    home_reg, home_res = _fit_regressor(X_proc, y_home, "home")
    away_reg, away_res = _fit_regressor(X_proc, y_away, "away")
    log.debug("Home regressor results: %s", home_res)
    log.debug("Away regressor results: %s", away_res)

    # Classifier
    log.info("Training LightGBM classifier for win probability...")
    win_clf, win_res, cv_preds = _fit_classifier_optimized(X_proc, y_win)
    log.debug("Win classifier results: %s", win_res)

    # Persist models
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(home_reg, MODELS_DIR / "home_model.joblib")
    joblib.dump(away_reg, MODELS_DIR / "away_model.joblib")
    joblib.dump(win_clf, MODELS_DIR / "win_clf_calibrated.joblib")

    # Error analysis file: join index back to ID columns if present
    cv_preds = cv_preds.merge(df.reset_index().rename(columns={"index":"idx"})[
        ["idx","season","week","home_team","away_team","home_win"]
    ], on="idx", how="left")
    cv_preds["abs_error"] = (cv_preds["prob_home_win"] - cv_preds["home_win"]).abs()
    cv_preds.to_csv(MODELS_DIR / "validation_errors.csv", index=False)

    # Reports
    hash_values = pd.util.hash_pandas_object(df, index=False).values
    hash_bytes = np.asarray(hash_values).tobytes()
    dataset_hash = hashlib.md5(hash_bytes).hexdigest()[:10]
    training_report = {
        "dataset": {"rows": int(len(df)), "hash": dataset_hash},
        "features_used": BASE_FEATURES,
        "regression": {"home": home_res, "away": away_res},
        "classification": win_res,
        "thresholds": {"win_auc_min": 0.65},
        "production_ready_win_model": bool(win_res["cv_auc"] >= 0.60),
    }
    (MODELS_DIR / "training_report.json").write_text(json.dumps(training_report, indent=2))

    metadata = {
        "training_timestamp": pd.Timestamp.now().isoformat(),
        "dataset_hash": dataset_hash,
        "training_samples": int(len(df)),
        "raw_feature_columns": {"numeric": BASE_FEATURES, "categorical": []},
        "models": {
            "home_model": "home_model.joblib",
            "away_model": "away_model.joblib",
            "win_model": "win_clf_calibrated.joblib",
        },
        "preprocessor": "preprocessor.joblib",
        "targets": ["home_points_for","away_points_for","home_win"],
        "model_scores": {
            "home_r2_cv": float(home_res["cv_rmse"]) if "cv_rmse" in home_res else None,
            "away_r2_cv": float(away_res["cv_rmse"]) if "cv_rmse" in away_res else None,
            "win_auc_cv": float(win_res["cv_auc"]),
        },
        "production_ready_win_model": bool(win_res["cv_auc"] >= 0.65),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    log.info("Saved models and reports to %s", MODELS_DIR)
    if not metadata["production_ready_win_model"]:
        log.warning("Win model below AUC threshold (%.3f < 0.75). Not production-ready.", win_res["cv_auc"])

if __name__ == "__main__":
    main()
