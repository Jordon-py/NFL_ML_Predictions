# Enhanced Pipeline - Quick Start Guide

## What is the Enhanced Pipeline?

The Enhanced Pipeline is a comprehensive workflow tool that:
- Validates your NFL dataset for integrity and quality issues
- Merges schedule data with engineered features
- Evaluates model performance with time-series cross-validation
- **Automatically provides 2 production-ready recovery solutions when errors occur**

## Quick Start

### 1. Run the Full Pipeline

```bash
cd /path/to/NFL_ML_Predictions
python backend/enhanced_pipeline.py --start 2010 --end 2025 --evaluate-models
```

This will:
1. Build the dataset from seasons 2010-2025
2. Run comprehensive validations
3. Evaluate all three models (home, away, win classifier)
4. Generate detailed reports

### 2. Check the Results

After running, check these files in `backend/data/reports/`:

- `pipeline_summary.json` - Overall results and status
- `dataset_validation_report.json` - Data quality checks
- `model_evaluation_report.json` - Model performance metrics
- `enhanced_pipeline_*.log` - Detailed execution log

### 3. If Errors Occur

The pipeline automatically generates error reports with recovery solutions:

```bash
# Check for error reports
ls backend/data/reports/error_report_*.json
```

Each error report contains:
- Error type and message
- Context information
- **Two production-ready recovery solutions** with implementation code
- Risk level for each solution (LOW/MEDIUM/HIGH)

## Command-Line Options

```bash
python backend/enhanced_pipeline.py [OPTIONS]

Options:
  --start INT              Starting season year (default: 2010)
  --end INT                Ending season year (default: 2025)
  --evaluate-models        Evaluate models after dataset creation (default: True)
  --no-evaluate-models     Skip model evaluation
  --cv-folds INT          Number of cross-validation folds (default: 5)
  --help                   Show help message
```

## Common Use Cases

### Case 1: Full Workflow (Recommended)
```bash
python backend/enhanced_pipeline.py --start 2010 --end 2025 --evaluate-models
```

### Case 2: Dataset Validation Only
```bash
python backend/enhanced_pipeline.py --start 2010 --end 2025 --no-evaluate-models
```

### Case 3: Recent Data with Deep Cross-Validation
```bash
python backend/enhanced_pipeline.py --start 2016 --end 2025 --cv-folds 10
```

## What Gets Validated?

The pipeline checks:

1. **Schema** - All required columns present
2. **Datatypes** - Correct types (int, float, string)
3. **Team Codes** - Legacy codes normalized (STL→LAR, SD→LAC, etc.)
4. **Temporal Order** - Data sorted chronologically
5. **Feature Leakage** - Rolling features use shift(1) before rolling()

## Error Recovery Examples

### Example 1: Datatype Mismatch

If the pipeline detects a datatype mismatch, it provides:

**Solution 1 (LOW risk)**: Explicit type conversion
```python
df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)
```

**Solution 2 (LOW risk)**: Schema validation in data loading
```python
# Add to build_csv_datasets.py
EXPECTED_SCHEMA = {'season': 'int64', 'week': 'int64', ...}
validate_schema(df, EXPECTED_SCHEMA)
```

### Example 2: Join Error

If team codes don't match during joins:

**Solution 1 (LOW risk)**: Enhanced normalization
```python
df['home_team'] = df['home_team'].replace(ABBR_FIX).str.strip().str.upper()
```

**Solution 2 (MEDIUM risk)**: Outer join with debugging
```python
merged = pd.merge(left_df, right_df, on=['team'], how='outer', indicator=True)
# Generate report of unmatched records
```

### Example 3: Missing Features

If required features are missing:

**Solution 1 (LOW risk)**: Median imputation
```python
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())
```

**Solution 2 (MEDIUM risk)**: Rebuild from source
```python
from build_csv_datasets import build_dataset
df = build_dataset(2010, 2025, DATA_DIR, production_mode=True)
```

## Integration with Existing Workflow

The enhanced pipeline works alongside existing tools:

```
Existing Workflow:
1. backend/build_csv_datasets.py  # Build dataset
2. backend/train_models.py        # Train models
3. backend/main.py                # API server

Enhanced Workflow:
1. backend/enhanced_pipeline.py   # Validate + evaluate
2. Review error reports (if any)
3. Apply recovery solutions
4. backend/train_models.py        # Retrain if needed
5. backend/main.py                # Deploy
```

## Model Evaluation Metrics

The pipeline reports these metrics:

**Regression Models (Home/Away Scores)**:
- R² Score (> 0.30 is good)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**Classification Model (Win Probability)**:
- ROC AUC (> 0.60 is acceptable, > 0.65 is good)
- Accuracy, Precision, Recall, F1
- Brier Score (calibration quality)

## Troubleshooting

### "Models not found"
**Solution**: Train models first:
```bash
python backend/train_models.py
```

### "Dataset not found"
**Solution**: Build dataset first:
```bash
python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data
```

### "Validation failures"
**Solution**: Check the error report JSON:
```bash
cat backend/data/reports/error_report_*.json
```
Review the recovery solutions and choose one based on risk level.

### "Low cross-validation scores"
**Actions**:
1. Check data quality in validation report
2. Review feature engineering in build_csv_datasets.py
3. Consider retraining with more data
4. Verify no data leakage

## Interactive Usage (Jupyter)

See `docs/clean_data_example.ipynb` for interactive examples:

```bash
jupyter notebook docs/clean_data_example.ipynb
```

The notebook demonstrates:
- Loading and inspecting data
- Running validation checks
- Applying cleaning operations
- Understanding error recovery

## Documentation

- **Full Guide**: `docs/report.md`
- **Example Notebook**: `docs/clean_data_example.ipynb`
- **Source Code**: `backend/enhanced_pipeline.py`
- **System Architecture**: `.github/copilot-instructions.md`

## Support

For questions or issues:
1. Check error reports for automatic recovery solutions
2. Review the detailed log file
3. Consult `docs/report.md` for comprehensive documentation
4. Check the example notebook for interactive guidance

## Best Practices

1. **Run before retraining** - Validate data before training models
2. **Review error reports** - Always check error reports when failures occur
3. **Choose low-risk solutions first** - Prefer LOW risk solutions over MEDIUM/HIGH
4. **Monitor CV scores** - Track cross-validation scores over time
5. **Regular validation** - Run periodically to catch issues early

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-15  
**Maintainer**: NFL Prediction System Team
