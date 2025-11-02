# Engineering Log

## 2025-11-02

### Session Start: `NotFittedError` Investigation

- **Objective**: Diagnose and resolve the `NotFittedError` preventing the backend from starting up.
- **Initial Analysis**: The error originates from the `preprocessor` object, which is loaded from `backend/models/preprocessor.joblib`. Although the file is loaded, the object's internal state is not fitted, causing failures when `transform` is called during prediction.
- **Hypothesis**: The current model artifacts were trained on a dataset with a different feature set than what the current, production-ready code expects. Specifically, the old artifacts expect leakage-prone columns (e.g., `_dom_bin`, `season_home_win_rate`) that are now correctly filtered out by the `train_models.py` script.
- **Next Steps**:
  - Document the chosen fix path in `FIX_NOTES.md`.
  - Execute the model retraining script to generate new, clean artifacts.
  - Validate the fix by restarting the backend server and running health checks.
