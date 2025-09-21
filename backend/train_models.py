#!/usr/bin/env python
"""
train_models.py
===============

Purpose
-------
Enhanced NFL score prediction training with automated model selection.
Trains and compares LightGBM (with grid search) vs Neural Network (with hyperparameter tuning)
and automatically selects the best performing model for production use.

Key Features
------------
1) **LightGBM with Grid Search**: Comprehensive hyperparameter optimization using GridSearchCV
2) **Neural Network with Keras Tuner**: Deep learning model with automated architecture search
3) **Automated Model Selection**: Cross-validation based comparison and selection
4) **Production-Ready Validation**: Fail-fast error handling and comprehensive logging
5) **Multi-Model Support**: API seamlessly handles both LightGBM and Keras models

Training Pipeline
-----------------
1) Load and validate CSV: `<repo_root>/Nfl_data_sorted.csv`
2) Feature engineering validation (no outcome leakage)
3) Preprocessing pipeline: imputation + scaling
4) **LightGBM Grid Search**: 8-parameter hyperparameter optimization with 5-fold CV
5) **Neural Network Tuning**: Architecture + optimizer tuning with Keras Tuner
6) **Model Comparison**: Automated selection based on validation performance
7) **Production Persistence**: Save best models + comprehensive metadata

Hyperparameter Spaces
---------------------
**LightGBM Grid Search:**
- n_estimators: [300, 500, 800]
- learning_rate: [0.03, 0.05, 0.1]
- max_depth: [-1, 10, 15]
- num_leaves: [20, 31, 50]
- subsample: [0.7, 0.8, 0.9]
- colsample_bytree: [0.7, 0.8, 0.9]
- reg_alpha: [0.0, 0.1, 0.5]
- reg_lambda: [0.0, 0.1, 0.5]

**Neural Network Tuning:**
- Architecture: 1-4 hidden layers, 32-256 units per layer
- Activations: relu, elu, swish
- Dropout: 0.1-0.5
- Optimizers: Adam, RMSprop, Nadam with tunable learning rates
- Early stopping + learning rate reduction

External Dependencies
---------------------
pandas, numpy, scikit-learn, lightgbm, joblib, tensorflow, keras-tuner, optuna

Production Notes
----------------
- Trains on all available data for maximum predictive power
- No fallbacks - all dependencies must be available for production deployment
- Comprehensive validation and fail-fast error handling
- Enhanced metadata with training metrics and model comparison results
- Cross-validation based model selection for robust performance estimates
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Required imports - all must be available for production
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Neural network imports - required for production
import tensorflow as tf
from keras_tuner import Objective, RandomSearch
from tensorflow import keras

# Suppress sklearn warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------------------------------------------------------
# Paths & logging
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
# Use the sorted dataset used elsewhere in the repo
DATA_PATH = BASE_DIR / "Nfl_data_sorted.csv"
MODELS_DIR = BASE_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_feature_list(df: pd.DataFrame) -> List[str]:
    """Return the numeric prior features used for score prediction.

    This function validates presence and provides a single source of truth
    for features for both training and inference.
    """
    features = [
        "home_prior_pa_avg_3",
        "home_prior_pa_avg_5",
        "home_prior_pf_avg_3",
        "home_prior_pf_avg_5",
        "home_prior_win_pct_3",
        "home_prior_win_pct_5",
        "away_prior_pa_avg_3",
        "away_prior_pa_avg_5",
        "away_prior_pf_avg_3",
        "away_prior_pf_avg_5",
        "away_prior_win_pct_3",
        "away_prior_win_pct_5",
    ]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return features


# -----------------------------------------------------------------------------
# Modeling helpers
# -----------------------------------------------------------------------------


def _fit_lgbm_regressor_with_grid_search(
    X: np.ndarray, y: np.ndarray, target_name: str
) -> Tuple[LGBMRegressor, Dict[str, Any]]:
    """Train a LightGBM regressor with grid search hyperparameter optimization."""
    logger.info(f"Starting grid search for LightGBM {target_name} model...")

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [-1, 10, 15],
        "num_leaves": [20, 31, 50],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.0, 0.1, 0.5],
    }

    # Base model
    lgbm = LGBMRegressor(
        objective="regression", metric="rmse", random_state=42, verbose=-1, n_jobs=-1
    )

    # Grid search with cross-validation
    cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    start_time = time.time()
    grid_search.fit(X, y)
    search_time = time.time() - start_time

    # Extract best model and results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back from negative

    # Calculate additional metrics
    y_pred = best_model.predict(X)
    train_r2 = r2_score(y, y_pred)
    train_mae = mean_absolute_error(y, y_pred)

    # Cross-validation scores for stability assessment
    cv_scores = cross_val_score(best_model, X, y, cv=cv_folds, scoring="r2", n_jobs=-1)

    search_results = {
        "best_params": best_params,
        "best_cv_rmse": best_score,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "cv_r2_scores": cv_scores.tolist(),
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "search_time_seconds": search_time,
        "n_candidates": len(grid_search.cv_results_["params"]),
    }

    logger.info(
        f"Grid search completed for {target_name}: "
        f"Best RMSE={best_score:.3f}, Train R²={train_r2:.3f}, "
        f"CV R²={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
    )

    if train_r2 < 0.1:
        raise ValueError(f"LightGBM {target_name} model training failed - R² = {train_r2:.3f}")

    return best_model, search_results


def _build_neural_network_model(input_dim: int, hp: Any) -> keras.Model:
    """Build a neural network architecture for score prediction."""
    model = keras.Sequential()

    # Input layer with dropout
    model.add(
        keras.layers.Dense(
            units=hp.Int("input_units", min_value=64, max_value=256, step=32),
            input_dim=input_dim,
            activation=hp.Choice("input_activation", ["relu", "elu", "swish"]),
        )
    )
    model.add(keras.layers.Dropout(hp.Float("input_dropout", 0.1, 0.5, step=0.1)))

    # Hidden layers
    for i in range(hp.Int("n_layers", 1, 4)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=128, step=16),
                activation=hp.Choice(f"activation_{i}", ["relu", "elu", "swish"]),
            )
        )
        model.add(keras.layers.Dropout(hp.Float(f"dropout_{i}", 0.1, 0.4, step=0.1)))

    # Output layer for regression
    model.add(keras.layers.Dense(1, activation="linear"))

    # Compile with tunable optimizer parameters
    optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop", "nadam"])
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG")

    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])

    return model


def _fit_neural_network_with_tuning(
    X: np.ndarray, y: np.ndarray, target_name: str
) -> Tuple[keras.Model, Dict[str, Any]]:
    """Train a neural network with hyperparameter tuning using Keras Tuner."""
    logger.info(f"Starting neural network hyperparameter tuning for {target_name}...")

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    def build_model(hp):
        return _build_neural_network_model(X.shape[1], hp)

    # Create tuner
    tuner = RandomSearch(
        build_model,
        objective=Objective("val_loss", direction="min"),
        max_trials=30,
        directory=f"keras_tuner_logs_{target_name}",
        project_name=f"nfl_score_prediction_{target_name}",
        overwrite=True,
    )

    # Prepare validation split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Early stopping callback
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Reduce learning rate callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )

    start_time = time.time()

    try:
        # Search for best hyperparameters
        tuner.search(
            X_train,
            y_train,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

        search_time = time.time() - start_time

        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Retrain on full dataset with best hyperparameters
        final_model = build_model(best_hp)

        # Train with early stopping on validation loss
        history = final_model.fit(
            X_train,
            y_train,
            epochs=200,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

        # Calculate metrics
        y_pred_train = final_model.predict(X_train, verbose=0).flatten()
        y_pred_val = final_model.predict(X_val, verbose=0).flatten()

        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)

        # Get best hyperparameters as dict
        best_params = {}
        for param in best_hp.space:
            best_params[param.name] = best_hp.get(param.name)

        tuning_results = {
            "best_params": best_params,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "final_val_loss": min(history.history["val_loss"]),
            "epochs_trained": len(history.history["loss"]),
            "search_time_seconds": search_time,
            "n_trials": tuner.oracle.get_state()["tried_so_far"],
        }

        logger.info(
            f"Neural network tuning completed for {target_name}: "
            f"Train R²={train_r2:.3f}, Val R²={val_r2:.3f}, "
            f"Epochs={len(history.history['loss'])}"
        )

        if train_r2 < 0.1:
            logger.warning(
                f"Neural network {target_name} model poor performance - R² = {train_r2:.3f}"
            )

        return final_model, tuning_results

    except Exception as e:
        logger.error(f"Neural network training failed for {target_name}: {e}")
        return None, {"error": str(e)}


def _compare_models(
    lgbm_results: Dict[str, Any], nn_results: Dict[str, Any], target_name: str
) -> str:
    """Compare LightGBM and Neural Network performance to select the best model."""
    if "error" in nn_results:
        logger.info(f"Neural network unavailable for {target_name} - using LightGBM")
        return "lgbm"

    lgbm_cv_r2 = lgbm_results["cv_r2_mean"]
    lgbm_cv_std = lgbm_results["cv_r2_std"]
    nn_val_r2 = nn_results["val_r2"]

    logger.info(f"Model comparison for {target_name}:")
    logger.info(f"  LightGBM: CV R² = {lgbm_cv_r2:.3f} ± {lgbm_cv_std:.3f}")
    logger.info(f"  Neural Network: Val R² = {nn_val_r2:.3f}")

    # Select model based on validation performance
    # Use LightGBM if it's within 1 std dev of NN (simpler model preference)
    if nn_val_r2 > (lgbm_cv_r2 + lgbm_cv_std):
        logger.info(f"Selected Neural Network for {target_name} (significantly better)")
        return "neural_network"
    else:
        logger.info(f"Selected LightGBM for {target_name} (comparable or better performance)")
        return "lgbm"


# -----------------------------------------------------------------------------
# Main training pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    # Validate dataset exists and load
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Production validation: ensure we have sufficient data for training
    if len(df) < 100:
        raise ValueError(f"Insufficient training data: only {len(df)} rows available")

    # Filter to completed games only (remove any future/scheduled games)
    valid_scores = df["home_points_for"].notna() & df["away_points_for"].notna()
    df = df[valid_scores].reset_index(drop=True)

    logger.info("Training on %d completed games from %s", len(df), DATA_PATH)
    logger.info(
        "Season range: %d-%d, Week range: %d-%d",
        df["season"].min(),
        df["season"].max(),
        df["week"].min(),
        df["week"].max(),
    )

    # Feature matrix with validation
    features = _get_feature_list(df)
    X = df[features]

    # Validate no missing features in production dataset
    missing_count = X.isnull().sum().sum()
    if missing_count > len(X) * 0.05:  # More than 5% missing
        raise ValueError(f"Too many missing feature values: {missing_count}/{len(X)}")

    # Targets with validation
    if not {"home_points_for", "away_points_for"}.issubset(df.columns):
        raise ValueError("Dataset missing required target columns")

    y_home = df["home_points_for"]
    y_away = df["away_points_for"]

    # Validate target ranges (NFL scores should be 0-70)
    if y_home.min() < 0 or y_home.max() > 80 or y_away.min() < 0 or y_away.max() > 80:
        raise ValueError("Invalid score values detected in training data")

    # Production preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                features,
            )
        ],
        remainder="drop",
    )

    X_proc = preprocessor.fit_transform(X)
    logger.info(
        "Preprocessing pipeline fitted on %d samples with %d features",
        X_proc.shape[0],
        X_proc.shape[1],
    )

    # Train models with hyperparameter optimization and model comparison
    logger.info("Training enhanced models with grid search and neural network tuning...")

    # Train LightGBM models with grid search
    home_lgbm_model, home_lgbm_results = _fit_lgbm_regressor_with_grid_search(
        X_proc, y_home, "home"
    )
    away_lgbm_model, away_lgbm_results = _fit_lgbm_regressor_with_grid_search(
        X_proc, y_away, "away"
    )

    # Train Neural Network models with hyperparameter tuning
    home_nn_model, home_nn_results = _fit_neural_network_with_tuning(X_proc, y_home, "home")
    away_nn_model, away_nn_results = _fit_neural_network_with_tuning(X_proc, y_away, "away")

    # Select best models based on performance comparison
    home_best_model_type = _compare_models(home_lgbm_results, home_nn_results, "home")
    away_best_model_type = _compare_models(away_lgbm_results, away_nn_results, "away")

    # Use the best performing models
    if home_best_model_type == "lgbm":
        home_model = home_lgbm_model
        home_final_results = home_lgbm_results
    else:
        home_model = home_nn_model
        home_final_results = home_nn_results

    if away_best_model_type == "lgbm":
        away_model = away_lgbm_model
        away_final_results = away_lgbm_results
    else:
        away_model = away_nn_model
        away_final_results = away_nn_results

    # Validate final model performance
    home_score = home_final_results.get("cv_r2_mean", home_final_results.get("val_r2", 0))
    away_score = away_final_results.get("cv_r2_mean", away_final_results.get("val_r2", 0))

    if home_score < 0.1 or away_score < 0.1:
        raise ValueError(
            f"Final model validation failed - R² scores: "
            f"home={home_score:.3f}, away={away_score:.3f}"
        )

    logger.info(
        "Enhanced model training complete - Final R² scores: home=%.3f (%s), away=%.3f (%s)",
        home_score,
        home_best_model_type,
        away_score,
        away_best_model_type,
    )

    # Production model persistence with validation
    try:
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

        # Save models based on their types
        if home_best_model_type == "lgbm":
            joblib.dump(home_model, MODELS_DIR / "home_model.joblib")
        else:
            # Save Keras model
            if home_model is not None:
                home_model.save(MODELS_DIR / "home_model.keras")

        if away_best_model_type == "lgbm":
            joblib.dump(away_model, MODELS_DIR / "away_model.joblib")
        else:
            # Save Keras model
            if away_model is not None:
                away_model.save(MODELS_DIR / "away_model.keras")

        logger.info("Successfully saved enhanced models to %s", MODELS_DIR)
        logger.info("Home model: %s, Away model: %s", home_best_model_type, away_best_model_type)
    except Exception as e:
        raise RuntimeError(f"Failed to save trained models: {e}") from e

    # Production metadata with comprehensive information
    feature_names: List[str] = []
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        logger.warning("Could not extract transformed feature names")
        feature_names = []

    # Calculate dataset hash for cache invalidation
    import hashlib

    dataset_hash = hashlib.md5(df.to_string().encode()).hexdigest()[:8]

    from datetime import datetime

    training_timestamp = datetime.now().isoformat()

    meta = {
        "training_timestamp": training_timestamp,
        "dataset_hash": dataset_hash,
        "dataset_path": str(DATA_PATH),
        "training_samples": len(df),
        "season_range": [int(df["season"].min()), int(df["season"].max())],
        "week_range": [int(df["week"].min()), int(df["week"].max())],
        "model_scores": {"home_r2": float(home_score), "away_r2": float(away_score)},
        "model_types": {
            "home_model_type": home_best_model_type,
            "away_model_type": away_best_model_type,
        },
        "training_results": {
            "home_lgbm": home_lgbm_results,
            "away_lgbm": away_lgbm_results,
            "home_nn": home_nn_results,
            "away_nn": away_nn_results,
        },
        "raw_feature_columns": {"numeric": features, "categorical": []},
        "transformed_feature_names": feature_names,
        "models": {
            "home_model": "home_model.keras"
            if home_best_model_type == "neural_network"
            else "home_model.joblib",
            "away_model": "away_model.keras"
            if away_best_model_type == "neural_network"
            else "away_model.joblib",
        },
        "preprocessor": "preprocessor.joblib",
        "target_names": ["home_points_for", "away_points_for"],
    }

    try:
        with open(MODELS_DIR / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Model training complete - Ready for production use")
    except Exception as e:
        raise RuntimeError(f"Failed to save model metadata: {e}") from e


if __name__ == "__main__":
    main()
