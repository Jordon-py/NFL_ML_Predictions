# Enhanced Pipeline Report

## Overview

The Enhanced Pipeline provides a comprehensive dataset merge and model evaluation workflow for the NFL prediction system. It orchestrates data validation, feature engineering validation, and model evaluation with robust error handling.

## Key Features

### 1. Dataset Validation and Merging

The pipeline validates and merges multiple data sources:

- **Schedule Data**: Historical game scores and future scheduled games
- **Feature Engineering**: Leak-free rolling statistics (3-game and 5-game windows)
- **Team Normalization**: Consistent team abbreviations using `ABBR_FIX` mapping

### 2. Comprehensive Validation Checks

The pipeline performs multiple validation checks:

#### Schema Validation
- Verifies all expected columns are present
- Checks for missing features that could break model predictions

#### Datatype Validation
- Ensures columns have correct datatypes (int, float, string)
- Detects type mismatches that could cause computation errors

#### Team Code Validation
- Validates team abbreviations are properly normalized
- Detects legacy codes (STL, SD, OAK) that should be updated

#### Temporal Order Validation
- Ensures data is chronologically sorted
- Critical for time-series cross-validation

#### Leakage Validation
- Verifies rolling features use shift(1) before rolling()
- Prevents future information from leaking into past predictions

### 3. Model Evaluation Framework

The pipeline evaluates all three production models:

#### Regression Models (Home/Away Score Prediction)
- **Metrics**: R², RMSE, MAE
- **Cross-Validation**: TimeSeriesSplit (respects temporal order)
- **Validation**: Ensures no future information leakage

#### Classification Model (Win Probability)
- **Metrics**: ROC AUC, Accuracy, Precision, Recall, F1, Brier Score
- **Calibration**: Uses CalibratedClassifierCV for probability reliability
- **Cross-Validation**: Stratified TimeSeriesSplit

### 4. Automatic Error Recovery

When errors occur, the pipeline automatically proposes two production-ready recovery solutions:

#### Error Types Handled:

**1. Datatype Mismatch**
   - Solution 1: Explicit type conversion with validation
   - Solution 2: Schema-based validation in data loading pipeline

**2. Join Errors**
   - Solution 1: Fuzzy match with enhanced team normalization
   - Solution 2: Outer join with missing data report

**3. Missing Features**
   - Solution 1: Median imputation with logging
   - Solution 2: Rebuild features from source data

**4. Feature Leakage**
   - Solution 1: Validate shift-before-rolling pattern
   - Solution 2: Time-series split validation

## Usage

### Basic Usage

```bash
# Run full pipeline with default settings
python backend/enhanced_pipeline.py --start 2010 --end 2025 --evaluate-models
```

### Advanced Usage

```bash
# Only create dataset without evaluation
python backend/enhanced_pipeline.py --start 2016 --end 2025 --no-evaluate-models

# Run with custom cross-validation folds
python backend/enhanced_pipeline.py --start 2010 --end 2025 --cv-folds 10
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start` | int | 2010 | Starting season year |
| `--end` | int | 2025 | Ending season year |
| `--evaluate-models` | flag | True | Evaluate models after dataset creation |
| `--no-evaluate-models` | flag | - | Skip model evaluation |
| `--cv-folds` | int | 5 | Number of cross-validation folds |

## Output Files

The pipeline generates several output files in `backend/data/reports/`:

### 1. Pipeline Summary (`pipeline_summary.json`)
```json
{
  "start_time": "2025-01-15T10:30:00",
  "parameters": {
    "start_year": 2010,
    "end_year": 2025,
    "evaluate_models": true,
    "cv_folds": 5
  },
  "merge_report": {
    "output_path": "backend/data/Nfl_data_sorted.csv",
    "rows": 2749,
    "columns": 35,
    "validation": {...}
  },
  "evaluation_results": {
    "home_model": {
      "mean_r2": 0.3245,
      "std_r2": 0.0523
    },
    "away_model": {
      "mean_r2": 0.3189,
      "std_r2": 0.0487
    },
    "win_classifier": {
      "mean_auc": 0.6364,
      "std_auc": 0.0342
    }
  },
  "status": "SUCCESS"
}
```

### 2. Dataset Validation Report (`dataset_validation_report.json`)
Contains detailed validation results for all checks performed.

### 3. Model Evaluation Report (`model_evaluation_report.json`)
Contains cross-validation results and performance summaries.

### 4. Error Reports (`error_report_*.json`)
Generated when errors occur, includes:
- Error type and message
- Context information
- Two production-ready recovery solutions with implementation code

### 5. Pipeline Log (`enhanced_pipeline_*.log`)
Detailed execution log with timestamps and debug information.

## Integration with Existing Workflow

The enhanced pipeline integrates seamlessly with existing components:

### 1. Dataset Building (`build_csv_datasets.py`)
- Uses the same `build_dataset()` function
- Maintains compatibility with existing team normalization
- Preserves leak-free feature engineering patterns

### 2. Model Training (`train_models.py`)
- Evaluates the same models (home, away, win classifier)
- Uses identical feature specifications (`BASE_FEATURES`)
- Maintains TimeSeriesSplit for temporal validation

### 3. API Service (`main.py`)
- Validates the same dataset format expected by the API
- Ensures models can be loaded and used for predictions
- Verifies feature names match API expectations

## Best Practices

### 1. Run Before Retraining
Always run the enhanced pipeline before retraining models to:
- Validate data integrity
- Detect potential issues early
- Establish baseline metrics

### 2. Review Error Reports
When errors occur:
- Check the generated error report JSON
- Review both proposed solutions
- Choose the solution with lower risk level
- Test the solution before deploying

### 3. Monitor Cross-Validation Scores
Track CV scores over time to detect:
- Model degradation
- Dataset drift
- Feature quality issues

Expected baselines:
- Home/Away R² > 0.30 (good performance)
- Win Classifier AUC > 0.60 (acceptable performance)

### 4. Regular Validation
Run the pipeline periodically to:
- Validate data quality remains high
- Ensure no regressions in feature engineering
- Monitor model performance trends

## Troubleshooting

### Issue: "Models not found"
**Solution**: Train models first using `python backend/train_models.py`

### Issue: "Dataset not found"
**Solution**: Build dataset first using `python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data`

### Issue: "Validation failures"
**Check**: Review the error report JSON for specific failures and recovery solutions

### Issue: "Low cross-validation scores"
**Actions**:
1. Check for data quality issues in validation report
2. Verify feature engineering is working correctly
3. Consider retraining models with updated data
4. Review feature importance and selection

## Technical Details

### Architecture

```
Enhanced Pipeline
├── DatasetMerger
│   ├── build_dataset() - from build_csv_datasets.py
│   ├── DatasetValidator
│   │   ├── validate_schema()
│   │   ├── validate_datatypes()
│   │   ├── validate_team_codes()
│   │   ├── validate_temporal_order()
│   │   └── validate_no_leakage()
│   └── generate_report()
│
├── ModelEvaluator
│   ├── load_models()
│   ├── evaluate_regression_model()
│   ├── evaluate_classifier_model()
│   ├── cross_validate_models() - TimeSeriesSplit
│   └── generate_evaluation_report()
│
└── Error Recovery System
    ├── PipelineError
    ├── RecoverySolution
    └── Automatic solution generation
```

### Key Patterns

#### 1. Fail-Fast Validation
```python
# Validation stops on first critical error
# Provides immediate feedback with recovery solutions
if not schema_valid:
    error.log_error_report()
    return False
```

#### 2. TimeSeriesSplit for CV
```python
# Ensures no future information in validation
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Train on past, test on future
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

#### 3. Leak-Free Feature Validation
```python
# Verify first values are NaN after shift(1)
df_sorted = df.sort_values(['season', 'week'])
first_valid = df_sorted[col].first_valid_index()
if first_valid == 0:
    log_potential_leakage(col)
```

## Future Enhancements

Potential improvements for the enhanced pipeline:

1. **Automated Fixing**: Implement automatic application of low-risk recovery solutions
2. **Drift Detection**: Add feature distribution drift detection between training and production data
3. **Performance Tracking**: Store historical CV results for trend analysis
4. **Alert System**: Integrate with monitoring systems for automatic alerts on failures
5. **Parallel Validation**: Run validation checks in parallel for faster execution

## References

- Main dataset builder: `backend/build_csv_datasets.py`
- Model training: `backend/train_models.py`
- API service: `backend/main.py`
- System instructions: `.github/copilot-instructions.md`

## Support

For issues or questions:
1. Check the error report JSON for automatic recovery solutions
2. Review the pipeline log file for detailed execution trace
3. Consult the main README.md for system overview
4. Check existing issues in the GitHub repository

---

**Last Updated**: 2025-01-15  
**Version**: 1.0.0  
**Maintainer**: NFL Prediction System Team
