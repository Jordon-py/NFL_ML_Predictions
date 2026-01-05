# Prediction System Audit Report

## 1. Overview
- Scope: backend prediction endpoints, inference pipeline, training artifacts, evaluation reports, and datasets in this repo.
- Primary runtime path: FastAPI `/predict` -> `PredictionService` -> model bundle in `backend/20260102/models`, dataset in `backend/data/datasets`.
- Evidence reviewed: `backend/20260102/models/training_summary.json`, `backend/20260102/models/cv_fold_metrics.csv`, `backend/20260102/models/training_report.txt`, `backend/20260102/models/feature_metadata.json`, `backend/reports/nflex_v6_report.md`, `backend/reports/dataset_analysis_insights.md`, `backend/data/legacy_data/prod-models/game_features_20251210_holdout_predictions.csv`.
- Summary: The deployed inference path is consistent and guarded by schema validation, but multiple training and legacy inference paths introduce drift risk. Some training features (notably `time_key`) are not computed for synthetic inference rows and are filled by medians, which weakens temporal context for future games.

## 2. Pipeline Map
1. Startup loads dataset and model bundle:
   - `backend/main.py` uses `backend/main_helpers.py` `load_inference_bundle` and `load_dataset_df`.
   - `MODELS_DIR` resolves to `backend/20260102/models` (`backend/config.py`).
2. Request handling:
   - `POST /predict` in `backend/main.py` receives `PredictionRequest`.
3. Row construction:
   - `backend/services/inference_row.py` `build_model_input_row`:
     - Try exact dataset row by season/week/home/away.
     - If missing, build synthetic row and enrich with schedule data (nflreadpy or CSV).
     - Roll forward priors and rolling stats from team history cache.
     - Align to expected features (preprocessor `feature_names_in_` or metadata list).
     - Impute remaining numeric features with dataset medians.
4. Preprocess:
   - `preprocessor.joblib` transforms raw row into model-ready features.
5. Inference:
   - `home_model` and `away_model` predict scores.
   - `win_clf_calibrated` predicts home win probability; fallback sigmoid from point diff if classifier is absent.
6. Response:
   - `backend/main.py` `_build_prediction_payload` builds `UnifiedPredictionResponse`, adds team metadata.
   - History persisted to `backend/Predictions/prediction_history.json`.
7. Batch:
   - `GET /predict/next-week` loops schedule games and applies the same pipeline.

Legacy path:
- `/legacy/predict` and `/legacy/predict/next-week` in `backend/routes.py` use a different inference path that builds a raw row and fills missing values with zeros, not priors.

## 3. Critical Code Review
- `backend/main.py`: main endpoint surface and response flattening; uses `_validate_feature_schema` to ensure dataset contains expected features. Startup can fail if multiple `game_features_*.csv` exist (see `backend/main_helpers.py` `load_dataset_df`).
- `backend/services/prediction_service.py`: core inference logic; uses win classifier if available and sigmoid fallback otherwise. This keeps availability high but can create calibration drift if the classifier is missing or incompatible.
- `backend/services/inference_row.py`: most influential on predictive behavior. Synthetic rows rely on team history and schedule data; missing features are filled with dataset medians.
  - Issue: `time_key` is in the trained feature list but is not computed for synthetic rows, so it is filled with a median value, reducing temporal context for future games.
  - Issue: if schedule data is unavailable, market and rest features are imputed rather than observed.
- `backend/main_helpers.py`: `load_inference_bundle` uses `metadata.json` and joblib artifacts. `training_report.json` is optional and absent in `backend/20260102/models`; metrics live in `training_summary.json` and `training_report.txt`.
- `backend/routes.py`: legacy endpoints use `_predict` with a simpler row builder and `fillna(0)`, which is not aligned with `inference_row` and can produce different outputs than `/predict`.
- `backend/train_models.py` vs `backend/pipeline_enhanced_v3.py`: two different training paths (HistGradientBoosting vs GradientBoosting plus calibration). The active metadata indicates GradientBoostingRegressor plus CalibratedClassifierCV (`backend/20260102/models/metadata.json`), so it likely came from `pipeline_enhanced_v3.py`. This split increases drift risk.
- `backend/services/feature_service.py` and `backend/utils/functions_for_main.py`: alternate feature builders that are not used by the main pipeline; consider retiring or unifying.
- `frontend/src/api/client.js`: frontend calls `/predict`, `/predict/next-week`, `/predict/explain`, `/history`; response contract matches `backend/schemas.py`.

## 4. Performance Correlation

### 4.1 Current model bundle (backend/20260102/models)
Source: `backend/20260102/models/training_summary.json` and `backend/20260102/models/cv_fold_metrics.csv`.
Dataset in `training_report.txt`: 2732 rows, 160 features.

| Metric | Value |
| --- | --- |
| Home MAE (val mean) | 4.4440 |
| Home RMSE (val mean) | 5.8541 |
| Away MAE (val mean) | 4.4229 |
| Away RMSE (val mean) | 5.6749 |
| Win Brier (val mean) | 0.1276 |
| Win LogLoss (val mean) | 0.3975 |
| Win Accuracy (val mean) | 0.8155 |

### 4.2 Holdout predictions (legacy file)
Source: `backend/data/legacy_data/prod-models/game_features_20251210_holdout_predictions.csv` (506 games).

| Metric | Value |
| --- | --- |
| Home MAE | 4.0570 |
| Away MAE | 3.9725 |
| Win Accuracy | 0.8399 |
| Win Brier | 0.1112 |
| Win LogLoss | 0.3553 |
| ROC AUC | 0.9227 |

Observed bias on this holdout file:
- Actual mean scores: home 24.257, away 21.903.
- Predicted mean scores: home 23.708, away 21.539.
- Mean point diff: actual 2.354 vs predicted 2.169.
This suggests mild underprediction of totals and point diff on that snapshot.

### 4.3 NFLEX v6 report (classification only)
Source: `backend/reports/nflex_v6_report.md`.

Cross-validated (training seasons):
- Brier 0.1725 to 0.1803 depending on model.
- LogLoss 0.5011 to 0.5663.
- ROC AUC 0.8102 to 0.8176.

Holdout season:
- Brier 0.1997 to 0.2253.
- LogLoss 0.5824 to 0.6760.
- ROC AUC 0.6984 to 0.7592.

Note: NFLEX v6 metrics appear to be for an earlier model family and may not match the deployed 2026-01-02 bundle, but they provide a useful baseline.

## 5. Visual Analytics
(Placeholders; run later with matplotlib or Plotly.)

```python
# Line chart: CV fold metrics for scores and win probability
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("backend/20260102/models/cv_fold_metrics.csv")
plt.plot(df["fold"], df["home_mae_val"], label="home_mae_val")
plt.plot(df["fold"], df["away_mae_val"], label="away_mae_val")
plt.plot(df["fold"], df["win_brier_val"], label="win_brier_val")
plt.xlabel("Fold")
plt.ylabel("Metric value")
plt.legend()
plt.title("CV Fold Metrics")
plt.show()
```

```python
# Accuracy progression across folds
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("backend/20260102/models/cv_fold_metrics.csv")
plt.plot(df["fold"], df["win_acc_val"], marker="o")
plt.xlabel("Fold")
plt.ylabel("Win accuracy")
plt.title("Win Accuracy by CV Fold")
plt.show()
```

```python
# Confusion matrix from holdout predictions
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("backend/data/legacy_data/prod-models/game_features_20251210_holdout_predictions.csv")
y_true = df["home_win_actual"].astype(int)
y_pred = (df["home_win_prob_model"] >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Away win", "Home win"]).plot()
plt.title("Holdout Confusion Matrix")
plt.show()
```

```python
# Distribution histogram: training vs inference proxies
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("backend/data/datasets/game_features_20260102.csv")
holdout = pd.read_csv("backend/data/legacy_data/prod-models/game_features_20251210_holdout_predictions.csv")

plt.hist(train["home_points_for"].dropna(), bins=30, alpha=0.6, label="train home_points_for")
plt.hist(holdout["home_score_pred"].dropna(), bins=30, alpha=0.6, label="holdout home_score_pred")
plt.legend()
plt.title("Score Distribution: Train vs Holdout Preds")
plt.show()
```

## 6. Optimization Matrix
| Component | Current Behavior | Identified Issue | Suggested Improvement | Expected Benefit |
| --- | --- | --- | --- | --- |
| Synthetic row time context | `time_key` not computed in synthetic inference rows | `time_key` is filled with a median value, losing temporal context | Compute `time_key` from season and week in `build_model_input_row` | Better temporal generalization for future games |
| Training pipeline clarity | Multiple training scripts exist with different models | Ambiguity on which script produced current bundle | Add metadata field `training_script` and deprecate unused scripts | Traceable provenance and less drift |
| Dataset selection | `load_dataset_df` requires exactly one `game_features_*.csv` | Startup fails if multiple datasets exist | Allow explicit `DATASET_PATH` or select latest by timestamp | More reliable deployments |
| Feature parity | Schedule features optional in inference | Missing schedule data causes median imputation | Cache schedule data on startup and surface missing fields in `/debug` | Better consistency and easier diagnosis |
| Legacy endpoints | `/legacy/predict` uses different row builder and `fillna(0)` | Divergent outputs vs `/predict` | Deprecate or align legacy pipeline with `inference_row` | Consistent results across endpoints |
| Model artifact usage | Pipeline artifacts exist (`home_pipe`, `away_pipe`, `win_pipe`) but not used | Risk of confusion or double preprocessing if swapped | Add explicit model type checks and a single inference path | Safer model upgrades |
| Monitoring | Prediction history is stored but not aggregated | No live drift or calibration monitoring | Add lightweight metrics export (MAE, Brier, volume) | Early detection of degradation |

## 7. Reflexive Summary
The deployed prediction system is reasonably well-structured: startup validates model features against the dataset, inference builds rows with priors and rollups, and the API exposes stable response shapes for the frontend. Training artifacts in `backend/20260102/models` show solid CV performance for both score prediction and win probability. However, there are multiple training and inference code paths in the repo, and some features used in training (notably `time_key`) are not explicitly computed in synthetic inference rows, which can reduce fidelity for future games. The holdout predictions file suggests decent accuracy and calibration, with mild underprediction of totals.

Next steps:
1) Align synthetic row construction with training features (compute `time_key` and any other derived fields).
2) Document and standardize the training pipeline that produces the deployed bundle.
3) Add light monitoring for prediction quality and drift using the existing history and holdout utilities.
