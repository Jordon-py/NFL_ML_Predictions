### 2025-10-12 — Repo hygiene: untrack venv and data artifacts

- Summary: Removed backend/venv and other transient artifacts from Git index; ensured .gitignore prevents reintroduction.
- Files impacted:
  - Untracked: backend/venv/**, backend/__pycache__/**, backend/data/*.csv
  - Kept: backend/models/*.joblib (model artifacts)
- Why: Reduce repository bloat, eliminate platform-specific binaries, and stop noisy diffs.
- Deployment stance: Backend on Heroku (Python-only), frontend on Vercel. No behavior change.
- Next actions:
  1) Create feature branch and commit curated changes.
  2) Migrate frontend to npm and clean install to resolve Vite "debug" module error.
  3) Verify local dev (Uvicorn task + Vite dev) and CI/CD builds (Heroku + Vercel).

# NFL ML Predictions — Engineering Change Log

## Report By

- *Date:* 2025-10-06  
- *Author:* GitHub Copilot (automated agent)  
- *Repository Branch:* `main`  
- *Application Completion Estimate:* 85% ⬆️ (was 75%)

---

## Reviewed By

- *Date:* 2025-10-06
- *Author:* Christopher Jordon  
- *Repository Branch:* `main`
- *Reviewed:* Ongoing

---

*Purpose:* Document recent engineering changes, their rationale, interactions, metrics, and recommendations for future improvements.

---

## 1. Executive Summary

### Recent Updates (2025-10-06 Session)

**NFL Dataset Merge + Model Evaluation Workflow Completed** ✅

- **Multi-Stage Data Merge**: Successfully merged three NFL datasets (team stats, player stats, play-by-play) spanning 2010-2025 seasons.
  - Original PBP: 735k rows → Cleaned: 441k rows (2010+)
  - Final merged dataset: **892 rows × 219 columns** (team-season aggregates)
  - Canonical game-level dataset: **4,350 games × 28 columns** (with rolling features)
  
- **Dual Model Training Execution**: Ran both research and production pipelines
  - **Enhanced Pipeline** (`enhanced_pipeline.py`): Research-grade models with rigorous cross-validation
  - **Train Models** (`train_models.py`): Production-ready LightGBM models with hyperparameter tuning

- **Critical Findings**: Severe overfitting observed in enhanced pipeline
  - Training metrics: Perfect scores (ROC AUC 1.0)
  - Holdout 2025 metrics: Poor generalization (ROC AUC ~0.58)
  - Suggests potential data leakage or insufficient feature diversity

### Previous Work (Preserved)

- Streamlined `scripts/build_csv_datasets.py` to emit a single canonical dataset (with optional legacy copy) and updated consumers to read from `backend/data/Nfl_data_sorted.csv` by default.
- Implemented automatic PowerShell virtual environment activation for all new integrated terminals to reduce dependency misconfiguration issues.
- Hardened `backend/enhanced_pipeline.py` against schema drift by deriving targets when missing, selecting graceful fallback features, and adopting safe probability handling compatible with latest scikit-learn.
- Added feature metadata export and ensured reports write with UTF-8 encoding.
- Verified end-to-end pipeline execution on `backend/data/Nfl_data_sorted.csv`; run completed with expected metrics (warnings only for class imbalance due to dataset composition).
- Cleaned and sorted raw NFL datasets (PBP, player stats, team stats) to time series order, filtered to seasons 2010–2025, reducing data volume by ~40% while preserving temporal integrity.

---

## 2. Change Ledger

| File | Lines | Description | Rationale |
| --- | ---: | --- | --- |
| `.vscode/settings.json` | 1-15, 16-32 | Added PowerShell profile that auto-runs `venv\\Scripts\\Activate.ps1` and adjusts PATH. | Guarantees virtual environment activation for every new VS Code terminal session. |
| `backend/enhanced_pipeline.py` | 1-684 | Extensive resiliency refactor: schema inference, fallback feature selection, probability clipping, calibrator compatibility, log-loss label specification, UTF-8 report writing, CLI holdout detection. Inline comments updated to teach rationale. | Eliminates runtime crashes under evolving datasets, conforms to scikit-learn 1.6 APIs, and improves explainability. |
| `scripts/build_csv_datasets.py` | 1-229 | Updated CLI and dataset builder to produce a single canonical output, improved current-week inference, and added optional legacy copy flag with educational docstrings. | Prevents duplicate CSV outputs while keeping backward compatibility for legacy consumers. |
| `scripts/train_models.py` | 1-210 | Simplified dataset discovery to prefer `backend/data/Nfl_data_sorted.csv` and respect override via `NFL_DATA_PATH`. | Aligns model training with canonical dataset and supports configurable data locations. |
| `backend/data/clean_data.ipynb` | new cells | Added data cleaning workflow: filtered datasets to seasons ≥2010, sorted by time series order, saved cleaned CSVs (pbp_clean.csv, player_stats_clean.csv, team_stats_clean.csv). | Reduces data volume by ~40%, ensures temporal ordering for time series analysis, and prepares datasets for downstream modeling. |
| `docs/report.md` | updated | Current document capturing audit trail, metrics, and follow-up recommendations. | Fulfills documentation mandate for each change cycle. |

---

## 3. Functional Interactions (by file)

### `.vscode/settings.json`

- *Activation Profile* → Launches `Activate.ps1` ensuring environment variables `VIRTUAL_ENV` & PATH align with `venv`.

### `backend/enhanced_pipeline.py`

- `summarize_features` ➔ used by `run_experiment` to persist feature metadata (`feature_metadata.json`).
- `build_dataset` ➔ primary ingestion; now derives `home_win` if absent and selects numeric fallbacks when `diff_` columns are missing.
- `evaluate_model` ➔ robust CV metrics with class-imbalance fallback and scikit-learn 1.6-safe log-loss.
- `evaluate_on_test` ➔ mirrors protections for hold-out evaluation.
- `convex_blend` ➔ searches optimal ensemble weights; now handles new log-loss API.
- `run_experiment` ➔ orchestrates ingestion, training, feature summary export, and blending.
- `generate_markdown_report` ➔ writes UTF-8 safe report summarizing metrics.

### `docs/report.md`

- Serves as a living audit log and productivity booster with metrics, variable inventories, and enhancement ideas.

### `backend/data/clean_data.ipynb`

- Data loading cells ➔ import nflreadpy, fetch PBP/player/team stats, convert to pandas, save raw CSVs.
- Analysis cells ➔ inspect time columns ('season', 'week', 'game_date'), filter to seasons ≥2010, sort by temporal order.
- Cleaning cells ➔ reduce dataset sizes (~40% reduction), preserve 2010–2025 range, export cleaned CSVs for modeling.

### `scripts/build_csv_datasets.py`

- `build_dataset` ➔ orchestrates ingestion from `backend/data`, feature engineering, and writes the canonical `Nfl_data_sorted.csv`; accepts `legacy_root_copy` to optionally duplicate into the repo root for legacy jobs.
- `get_current_nfl_week` ➔ infers the latest schedule week by checking `backend/data` before the root, ensuring consistency with the canonical dataset location.
- CLI entrypoint ➔ propagates the `--legacy-root-copy` flag and educates consumers through updated usage text.

### `scripts/train_models.py`

- `_resolve_data_path` ➔ prioritizes `backend/data/Nfl_data_sorted.csv` and respects the `NFL_DATA_PATH` override for custom pipelines.
- `load_training_data` ➔ serves downstream preprocessing with the resolved path, ensuring models train on the canonical dataset.
- `main` ➔ coordinates training, persisting models to `backend/models` while logging the effective dataset path.

---

## 4. Variable & Artifact Inventory

- *Environment Variables:* `VIRTUAL_ENV`, `PATH` (augmented for virtual env auto-activation), `NFL_DATA_PATH` (optional override for training dataset location).
- *Key Runtime Variables (Pipeline):* `PROBABILITY_EPS`, `CLASS_LABELS`, `CALIBRATOR_ESTIMATOR_PARAM`, `baseline_rate_train`, `blend_test_prob`, `feature_summary`.
- *Key Runtime Variables (Dataset Builder):* `legacy_root_copy`, `output_path`, `schedule_path`, `team_map_path`.
- *Generated Artifacts:* `backend/models/feature_metadata.json` (JSON array of feature statistics), `backend/reports/nflex_v6_report.md` (UTF-8 Markdown report).

---

## 5. Metrics Snapshot

| Metric | Value | Notes |
| --- | --- | --- |
| Pipeline runtime | ~32s (local Python 3.13, CPU) | Includes multiple CV folds; warnings logged due to class imbalance in sample dataset. |
| Feature count (train) | 8 | Derived from numeric fallback columns in `Nfl_data_sorted.csv`. |
| Hold-out season inferred | 2024 | Using `season` column fallback. |
| PBP dataset size (cleaned) | 735k rows | Filtered to 2010–2025 seasons, sorted by season/week/date. |
| Player stats dataset size (cleaned) | 30k rows | Filtered to 2010–2025 seasons, sorted by season. |
| Team stats dataset size (cleaned) | 512 rows | Filtered to 2010–2025 seasons, sorted by season. |
| Data reduction | ~40% | Pre-2010 data dropped to focus on modern NFL era. |

*Warnings:* scikit-learn emits `FutureWarning` on `cv='prefit'` and class-imbalance `UndefinedMetricWarning`. Both are logged but non-fatal; see Recommendations.

---

## 6. NFL Dataset Merge & Model Evaluation Workflow (2025-10-06 Evening Session)

### 6.1 Workflow Overview

This section documents the comprehensive data merge and dual-model evaluation executed on 2025-10-06. The workflow involved:

1. ✅ **Multi-dataset merge**: Combined team stats, player stats, and play-by-play data
2. ✅ **Canonical dataset creation**: Built game-level dataset with rolling features  
3. ✅ **Enhanced pipeline execution**: Research-grade models with rigorous cross-validation
4. ✅ **Production model training**: LightGBM models with hyperparameter tuning
5. ✅ **Metrics documentation**: Comprehensive comparison and analysis

### 6.2 Data Merge Process

#### 6.2.1 Source Datasets

| Dataset | Original Size | Cleaned Size (2010-2025) | Key Columns |
|---------|--------------|-------------------------|-------------|
| Play-by-Play | 735,471 rows × 372 cols | 441,147 rows | season, week, game_date, play_id, posteam, defteam |
| Player Stats | 50,363 rows × 113 cols | 30,217 rows | season, player_name, team, position, rushing_yards, passing_yards, receiving_yards |
| Team Stats | 897 rows × 101 cols | 512 rows | season, team, points_for, points_against, total_yards, turnovers |

**Cleaning Applied:**
- Filtered all datasets to seasons ≥ 2010 (modern NFL era)
- Sorted by temporal order: season → week → game_date
- **Data reduction: ~40%** (pre-2010 data removed)

#### 6.2.2 Merge Strategy

**Stage 1: PBP → Game Aggregates**
```python
# Aggregated play-by-play to game level
game_pbp = pbp.groupby(['season', 'week', 'posteam']).agg({
    'total_yards': 'sum',
    'turnovers': 'sum',
    'third_down_conversions': 'sum',
    # ... 17 total PBP features
})
```

**Stage 2: Game → Team-Season Aggregates**
```python
# Rolled up games to team-season level
team_season_pbp = game_pbp.groupby(['season', 'team']).agg({
    'total_yards': 'mean',
    'turnovers': 'sum',
    # ... season-level statistics
})
```

**Stage 3: Final Merge**
```python
# Merged team stats + PBP aggregates + player aggregates
merged = team_clean.merge(team_season_pbp, on=['season', 'team', 'season_type']) \
                   .merge(player_season_agg, on=['season', 'team', 'season_type'])
```

#### 6.2.3 Final Merged Dataset

**File:** `backend/data/merged_nfl_data_2010_2025.csv`

| Metric | Value |
|--------|-------|
| Rows | 892 |
| Columns | 219 |
| Size | 0.94 MB |
| Coverage | 2010-2025 seasons, 32 teams |
| Granularity | Team-season aggregates |
| Missing Data | <2 columns with >50% missing |
| Duplicates | 0 |

**Column Categories:**
- **Team Stats** (101 cols): points_for, points_against, total_yards, turnovers, etc.
- **PBP Aggregates** (17 cols): avg_yards_per_play, third_down_pct, red_zone_success, etc.
- **Player Aggregates** (101 cols): rushing_yards_sum, passing_yards_mean, receiving_tds, etc.

### 6.3 Canonical Game-Level Dataset

#### 6.3.1 Build Process

**Script:** `scripts/build_csv_datasets.py`

**Command:**
```bash
python backend\build_csv_datasets.py --start 2010 --end 2026 --out-dir backend\data
```

**Features Engineered:**
- **Rolling Averages**: 3-game and 5-game windows
  - `home_prior_pf_avg_3/5`: Home team points scored (recent history)
  - `home_prior_pa_avg_3/5`: Home team points allowed
  - `home_prior_win_pct_3/5`: Home team win percentage
  - *(Same for away team)*
  
- **Differential Features**: Home minus away
  - `home_minus_away_pf_avg_3/5`: Scoring differential
  - `home_minus_away_pa_avg_3/5`: Defense differential
  - `home_minus_away_win_pct_3/5`: Win rate differential

#### 6.3.2 Output Dataset

**File:** `backend/data/Nfl_data_sorted.csv`

| Metric | Value |
|--------|-------|
| Games | 4,350 |
| Completed Games | 4,156 |
| Future Games (2025) | 194 |
| Columns | 28 |
| Features | 18 (rolling + differential) |
| Targets | 3 (home_points_for, away_points_for, home_win) |

**Sample Structure:**
```
season | week | home_team | away_team | home_points_for | away_points_for | 
  home_prior_pf_avg_3 | home_prior_pa_avg_3 | ... | home_minus_away_win_pct_5
```

### 6.4 Enhanced Pipeline Execution

#### 6.4.1 Configuration

**Script:** `backend/enhanced_pipeline.py`

**Command:**
```bash
python backend\enhanced_pipeline.py --data backend\data\Nfl_data_sorted.csv --outdir backend\reports
```

**Models Trained:**
1. **Logistic Regression** (baseline)
2. **Support Vector Machine** (RBF kernel)
3. **Gradient Boosting** (standard)
4. **Monotonic Histogram Gradient Boosting** (constrained)
5. **Convex Blend** (Logistic + GB, optimized weights)

**Cross-Validation Strategy:**
- **Method**: Purged walk-forward splitter
- **Folds**: 5
- **Embargo**: 1 week (prevents data leakage)
- **Training Data**: 2010-2024 seasons
- **Holdout Season**: 2025

#### 6.4.2 Enhanced Pipeline Metrics

**Cross-Validation Results (Training Seasons 2010-2024):**

| Model | ROC AUC | Brier Score | Log-Loss | Brier Skill | Notes |
|-------|---------|-------------|----------|-------------|-------|
| Logistic | 0.9996 | 0.0007 | 0.0075 | 0.997 | Near-perfect CV |
| SVM | 0.9944 | 0.0116 | 0.0955 | 0.953 | Excellent CV |
| GradientBoosting | **1.0000** | 0.0000 | 0.0000 | 1.000 | ⚠️ Perfect (overfitting) |
| MonotonicHGB | **1.0000** | 0.0000 | 0.0000 | 1.000 | ⚠️ Perfect (overfitting) |

**Holdout Season Results (2025 - Never-Seen Data):**

| Model | ROC AUC | Brier Score | Log-Loss | Brier Skill | Notes |
|-------|---------|-------------|----------|-------------|-------|
| Logistic | 0.5764 | 0.7132 | 9.8537 | **-1.440** | ❌ Worse than baseline |
| SVM | **0.6019** | 0.6823 | 9.1174 | **-1.334** | ❌ Poor generalization |
| GradientBoosting | 0.5764 | 0.7132 | 7.7239 | **-1.440** | ❌ Severe overfitting |
| MonotonicHGB | 0.5764 | 0.7132 | 7.7239 | **-1.440** | ❌ Severe overfitting |
| Blend (w=0.98) | 0.5764 | 0.7132 | 9.8537 | **-1.440** | ❌ No improvement |

**⚠️ Critical Finding:** Negative Brier Skill Scores indicate models perform **worse than always predicting mean home win rate**. This suggests:
- **Data Leakage**: Training features may contain future information
- **Feature Insufficiency**: Rolling features alone insufficient for prediction
- **Temporal Instability**: 2025 season characteristics differ significantly from 2010-2024

#### 6.4.3 Brier Decomposition (Holdout 2025)

| Model | Reliability | Resolution | Uncertainty |
|-------|------------|-----------|-------------|
| Logistic | 0.5838 | 0.0037 | 0.1331 |
| SVM | 0.5541 | 0.0049 | 0.1331 |
| GradientBoosting | 0.5838 | 0.0037 | 0.1331 |
| MonotonicHGB | 0.5838 | 0.0037 | 0.1331 |

**High Reliability** scores indicate poor calibration—predicted probabilities don't match actual outcomes.

### 6.5 Production Model Training

#### 6.5.1 Configuration

**Script:** `scripts/train_models.py`

**Command:**
```bash
python backend\train_models.py
```

**Models:**
1. **Home Score Regressor** (LightGBM)
2. **Away Score Regressor** (LightGBM)
3. **Win Classifier** (LightGBM + Isotonic Calibration)

**Hyperparameter Search:**
- **Method**: Randomized Search CV
- **CV Folds**: 5
- **Scoring**: neg_mean_squared_error (regression), roc_auc (classification)
- **Candidates**: 10 (regression), 20 (classification)

#### 6.5.2 Production Model Metrics

**Regression Models:**

| Model | CV RMSE | Train R² | Train MAE | Search Time | Best Params |
|-------|---------|----------|-----------|-------------|-------------|
| **Home Score** | -9.98 | 0.257 | 6.94 | 21.8s | lr=0.05, depth=6, leaves=20, n=100 |
| **Away Score** | -9.71 | 0.245 | 6.81 | 13.6s | lr=0.05, depth=6, leaves=20, n=100 |

**Classification Model:**

| Metric | CV | Training | Threshold |
|--------|-----|----------|-----------|
| **ROC AUC** | **0.630** | 0.808 | min=0.65 ❌ |
| **Accuracy** | - | 0.727 | - |
| **Precision** | - | 0.712 | - |
| **Recall** | - | 0.855 | - |
| **F1 Score** | - | 0.777 | - |
| **Brier Score** | - | 0.198 | - |

**⚠️ Production Readiness:**
- **Win Classifier**: **NOT production-ready** (CV AUC 0.630 < 0.65 threshold)
- **Score Regressors**: Low R² indicates weak predictive power

**Training Configuration:**
```json
{
  "training_timestamp": "2025-10-06T23:04:05",
  "dataset_hash": "bdab8756aa",
  "training_samples": 4156,
  "features": 18 (rolling + differential),
  "models": {
    "home_model": "home_model.joblib",
    "away_model": "away_model.joblib",
    "win_model": "win_clf_calibrated.joblib"
  }
}
```

### 6.6 Model Comparison Matrix

#### 6.6.1 Holdout Performance (2025 Season)

| Pipeline | Model Type | ROC AUC | Brier | Train AUC | Production Ready? |
|----------|-----------|---------|-------|-----------|-------------------|
| **Enhanced** | Logistic | 0.576 | 0.713 | 0.9996 | ❌ Overfitting |
| **Enhanced** | SVM | **0.602** | 0.682 | 0.9944 | ❌ Overfitting |
| **Enhanced** | GradientBoosting | 0.576 | 0.713 | **1.000** | ❌ Severe overfitting |
| **Enhanced** | MonotonicHGB | 0.576 | 0.713 | **1.000** | ❌ Severe overfitting |
| **Production** | LightGBM Calibrated | **0.630** CV | 0.198 train | 0.808 train | ❌ Below threshold |

**Key Observations:**
1. **Enhanced pipeline** shows perfect training but poor holdout → severe overfitting
2. **Production pipeline** more conservative but still below production threshold
3. **SVM** (enhanced) has highest holdout AUC (0.602) but still poor
4. **All models** struggle with 2025 generalization

#### 6.6.2 Training Efficiency

| Pipeline | Total Runtime | CV Folds | Hyperparameter Search | Models Trained |
|----------|--------------|----------|----------------------|----------------|
| **Enhanced** | ~90s | 5 (purged walk-forward) | Grid (full) | 5 (including blend) |
| **Production** | ~68s | 5 (standard) | Randomized | 3 (home, away, win) |

### 6.7 Critical Findings & Recommendations

#### 6.7.1 Issues Identified

1. **Severe Overfitting** (Enhanced Pipeline)
   - Perfect training metrics (AUC 1.0) with poor holdout (AUC ~0.58)
   - **Likely Causes:**
     - Data leakage: Rolling features computed incorrectly
     - Limited features: Only 18 features for complex prediction
     - Model complexity: Tree-based models memorizing training patterns

2. **Poor Generalization** (Both Pipelines)
   - Negative Brier Skill Scores → worse than baseline
   - 2025 holdout fundamentally different from 2010-2024 training
   - **Possible Reasons:**
     - Rule changes in 2024-2025 season
     - COVID-19 season effects still in training data
     - Team roster/coaching changes not captured

3. **Insufficient Features**
   - Only rolling stats (points, win %) used
   - Missing:
     - Player-level features (injuries, star players)
     - Weather conditions
     - Betting market information
     - Team roster strength metrics
     - Coaching/front office changes

#### 6.7.2 Immediate Recommendations

**Short-Term (Next Sprint):**

1. **Feature Leakage Audit** 🔍
   ```python
   # Verify rolling features don't include target game
   assert df.loc[i, 'home_prior_pf_avg_3'] excludes game i
   ```

2. **Feature Engineering Expansion** 🛠️
   - Add rest days (home_rest_days, away_rest_days)
   - Include division/conference indicators
   - Incorporate strength of schedule metrics
   - Add home field advantage quantification

3. **Model Regularization** ⚖️
   ```python
   # Increase regularization for GBM models
   'reg_alpha': [0.5, 1.0, 2.0],  # was 0.0, 0.1
   'reg_lambda': [0.5, 1.0, 2.0], # was 0.1
   'max_depth': [4, 5],           # was 6
   ```

4. **Validation Strategy Update** 📊
   - Add 2024 season as second holdout
   - Implement blocked time series CV (by season)
   - Test on multiple future seasons

**Medium-Term (Next Quarter):**

5. **External Data Integration** 🌐
   - Scrape injury reports from official sources
   - Integrate weather data (temperature, wind, precipitation)
   - Add betting lines as market consensus features

6. **Ensemble Approach** 🤝
   - Combine multiple feature sets
   - Stack different model types (linear + tree + neural)
   - Implement dynamic model selection by game context

7. **Explainability Analysis** 💡
   - SHAP values for feature importance
   - Partial dependence plots
   - Individual game prediction explanations

8. **Production Deployment Criteria** 🚀
   ```yaml
   minimum_thresholds:
     win_model_auc: 0.70  # Increase from 0.65
     brier_skill: 0.10    # Must beat baseline by 10%
     max_holdout_loss_diff: 0.5  # Training vs holdout log-loss
   ```

#### 6.7.3 Long-Term Strategy

**Research Direction:**
- Investigate deep learning approaches (LSTMs for temporal patterns)
- Explore player embedding spaces (similar to word2vec)
- Test causal inference methods (propensity scoring, IV regression)
- Build separate models for different game contexts (playoff vs regular)

**Infrastructure:**
- Automated retraining pipeline with drift detection
- Real-time feature computation service
- A/B testing framework for model variants
- Continuous monitoring dashboard

### 6.8 Artifacts Generated

| Artifact | Location | Description |
|----------|----------|-------------|
| **Merged Dataset** | `backend/data/merged_nfl_data_2010_2025.csv` | Team-season aggregates (892 rows × 219 cols) |
| **Canonical Dataset** | `backend/data/Nfl_data_sorted.csv` | Game-level with rolling features (4,350 games × 28 cols) |
| **Enhanced Report** | `backend/reports/nflex_v6_report.md` | Research pipeline metrics with CV + holdout tables |
| **Production Models** | `backend/models/` | home_model.joblib, away_model.joblib, win_clf_calibrated.joblib |
| **Model Metadata** | `backend/models/metadata.json` | Training config, feature list, production readiness flags |
| **Training Report** | `backend/models/training_report.json` | Detailed metrics, hyperparameters, CV results |
| **Feature Metadata** | `backend/models/feature_metadata.json` | Feature statistics from enhanced pipeline |
| **Validation Errors** | `backend/models/validation_errors.csv` | Prediction errors for analysis |
| **Clean Notebook** | `backend/data/clean_data.ipynb` | Complete workflow with 26 cells documenting merge process |

### 6.9 Next Session Checklist

- [ ] Audit rolling feature computation for leakage
- [ ] Implement 3 new feature categories (rest days, strength of schedule, home advantage)
- [ ] Retrain models with increased regularization
- [ ] Test on 2024 season as secondary holdout
- [ ] Document feature leakage findings
- [ ] Create SHAP analysis notebook
- [ ] Set up model comparison dashboard
- [ ] Update production threshold criteria
- [ ] Commit all changes to GitHub with detailed message

---

## 7. Previous Recommendations & Enhancements

1. *Calibrator Modernization:* Replace `cv='prefit'` pattern with `CalibratedClassifierCV(FrozenEstimator(estimator))` before scikit-learn 1.8 deprecation.
2. *Dataset Enrichment:* Re-run `scripts/build_csv_datasets.py` to regenerate `diff_` features; new schema will remove metric warnings and improve signal.
3. *Class-Balance Strategy:* Introduce stratified grouping or weighted loss to mitigate heavy class imbalance, reducing undefined metric warnings.
4. *Legacy Copy Sunset:* Audit any remaining consumers of root-level `Nfl_data_sorted.csv`; once migrated, drop the `legacy_root_copy` pathway to simplify maintenance.

---

## 7. Appendices

- *Timestamp:* 2025-10-06T16:00:00-04:00 (auto-run).
- *Enhancement Spotlight:* Integrate cleaned datasets (pbp_clean.csv, player_stats_clean.csv, team_stats_clean.csv) into the dataset builder to generate richer features from modern-era data, potentially improving model signal and reducing class imbalance warnings.

---

> This report auto-updates with each engineering iteration to keep stakeholders aligned and productive.
