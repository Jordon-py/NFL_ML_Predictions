# Enhanced Pipeline - Full Dataset Merge and Model Evaluation

## Quick Start

Run the complete workflow in one command:

```bash
python backend/enhanced_pipeline.py --full
```

This will:
1. ✅ Validate dataset structure and detect corruption
2. ✅ Automatically fix corrupted headers if needed
3. ✅ Merge datasets (if required)
4. ✅ Train all models with hyperparameter optimization
5. ✅ Evaluate models and check production readiness
6. ✅ Generate comprehensive reports

## Individual Commands

### Validate Dataset
```bash
python backend/enhanced_pipeline.py --validate
```
Checks for:
- File existence and accessibility
- Column structure (expected 28 columns)
- Required columns presence
- Data type consistency
- Header corruption

### Fix Corrupted Headers
```bash
python backend/enhanced_pipeline.py --fix-headers
```
- Automatically detects column count mismatches
- Creates backup before modification
- Applies correct column names
- Validates repair success

### Merge Datasets
```bash
python backend/enhanced_pipeline.py --merge
```
- Checks for multiple dataset files
- Merges if needed (usually not required)
- Validates merge consistency

### Train Models
```bash
python backend/enhanced_pipeline.py --train
```
- Trains 3 models: home_score, away_score, win_probability
- Uses RandomizedSearchCV for hyperparameter tuning
- Generates model artifacts and reports

### Evaluate Models
```bash
python backend/enhanced_pipeline.py --evaluate
```
- Checks all model files exist
- Reads metadata and training reports
- Assesses production readiness
- Generates evaluation report

## Error Handling

The pipeline provides **production-ready recovery solutions** for all error types:

### Example: Missing Dataset
```
ISSUE DETECTED: Dataset not found at backend/data/Nfl_data_sorted.csv

Production-Ready Recovery Solutions:

Solution 1:
Run: python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data

Solution 2:
Ensure nfl-data-py is installed: pip install nfl-data-py
```

### Example: Corrupted Headers
```
ISSUE DETECTED: Dataset has corrupted column headers: ['no need to ask me ']

Production-Ready Recovery Solutions:

Solution 1 - Rebuild dataset:
  python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data

Solution 2 - Fix headers manually:
  1. Backup current file: cp backend/data/Nfl_data_sorted.csv backend/data/Nfl_data_sorted.csv.bak
  2. Open file and replace corrupted header with proper column names
  3. Verify columns match: home_prior_pf_avg_3, home_prior_pf_avg_5, etc.
```

## Output Files

### Training Artifacts
- `backend/models/home_model.joblib` - Home score predictor
- `backend/models/away_model.joblib` - Away score predictor
- `backend/models/win_clf_calibrated.joblib` - Win probability predictor
- `backend/models/preprocessor.joblib` - Feature scaling pipeline

### Reports
- `backend/models/metadata.json` - Model versioning & configuration
- `backend/models/training_report.json` - Performance metrics
- `backend/models/validation_errors.csv` - Per-game prediction errors
- `backend/reports/enhanced_pipeline.log` - Full execution log
- `backend/reports/model_evaluation.json` - Evaluation summary

### Documentation
- `docs/dataset_merge_evaluation_report.md` - Comprehensive analysis

## Production Readiness Criteria

Models are considered production-ready when:
1. ✅ All model files present (4/4)
2. ✅ Win AUC ≥ 0.60 (preferably ≥ 0.65)
3. ✅ Training samples ≥ 500
4. ✅ No critical validation errors

If criteria not met, the pipeline suggests improvements:
- Expand training data range
- Enhanced feature engineering
- Hyperparameter tuning refinement

## Troubleshooting

### Issue: pip install timeout
**Solution:** Install core packages manually
```bash
pip install --user pandas numpy scikit-learn==1.3.2 lightgbm joblib
```

### Issue: Column count mismatch
**Solution:** Run header fix
```bash
python backend/enhanced_pipeline.py --fix-headers
```

### Issue: Missing features during training
**Solution:** Rebuild dataset with all features
```bash
python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data
```

### Issue: Models not production-ready (low AUC)
**Solution:** Expand training data
```bash
python backend/build_csv_datasets.py --start 2005 --end 2025 --out-dir backend/data
python backend/train_models.py
```

## Architecture

```
┌──────────────────────┐
│  enhanced_pipeline   │
│       --full         │
└──────────┬───────────┘
           │
     ┌─────▼──────┐
     │  Validate  │ → Checks dataset integrity
     └─────┬──────┘
           │
      ┌────▼─────┐
      │ Fix (if  │ → Repairs corrupted headers
      │  needed) │
      └────┬─────┘
           │
       ┌───▼──┐
       │ Merge│ → Combines datasets (if needed)
       └───┬──┘
           │
      ┌────▼─────┐
      │  Train   │ → 3 models + hyperparameter search
      └────┬─────┘
           │
     ┌─────▼────────┐
     │   Evaluate   │ → Production readiness check
     └──────────────┘
```

## Key Features

### 1. Fail-Fast Validation
Detects issues early with specific error messages and recovery solutions.

### 2. Automatic Recovery
Attempts to fix common issues (like corrupted headers) automatically.

### 3. Comprehensive Logging
All actions logged to `backend/reports/enhanced_pipeline.log` for debugging.

### 4. Production-Ready Solutions
Every error comes with 2-3 actionable solutions, not just error messages.

### 5. Graceful Degradation
Pipeline continues where possible, reports all issues at end.

## Integration with Existing Workflow

The enhanced pipeline complements existing scripts:

- `build_csv_datasets.py` - Still used for dataset generation
- `train_models.py` - Can be called directly or through pipeline
- `main.py` - FastAPI server uses trained models

**Recommended workflow:**
1. Use `enhanced_pipeline.py --full` for initial setup
2. Use `enhanced_pipeline.py --train --evaluate` for retraining
3. Use `build_csv_datasets.py` directly for custom dataset parameters
4. Use `train_models.py` directly for custom training configurations

## Performance

Typical execution times:
- **Validation:** < 1 second
- **Header Fix:** < 2 seconds
- **Training:** 5-15 minutes (depending on hyperparameter search)
- **Evaluation:** < 5 seconds
- **Full Pipeline:** 5-15 minutes total

## Future Enhancements

Planned features:
- [ ] Parallel hyperparameter search for faster training
- [ ] Automatic feature importance analysis
- [ ] Model versioning and rollback
- [ ] A/B testing framework
- [ ] Real-time monitoring integration
- [ ] Automated dataset refresh scheduling

## Support

For issues or questions:
1. Check logs: `backend/reports/enhanced_pipeline.log`
2. Review error messages and suggested solutions
3. Consult `docs/dataset_merge_evaluation_report.md`
4. Open an issue on GitHub

## Version History

**v1.0 (2025-10-07)**
- Initial release
- Full workflow automation
- Automatic header repair
- Comprehensive error handling
- Production-ready recovery solutions
