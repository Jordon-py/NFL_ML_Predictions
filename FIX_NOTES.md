# Fix Notes

## FIX-001: `NotFittedError` on Startup

- **Date**: 2025-11-02
- **Defect**: `sklearn.exceptions.NotFittedError` from the `preprocessor` during application startup.
- **Root Cause**: The model artifacts in `backend/models/` are stale. They were trained on a dataset containing leakage-prone features that the current codebase correctly identifies and removes. This creates a mismatch between the features the model expects and the features the application provides.

---

### Candidate Fixes

| Option | Description | Pros | Cons | Risk |
|---|---|---|---|---|
| **1. Retrain Models** | Run `backend/train_models.py` to regenerate all model artifacts using the current, correct feature set. | - Correctly aligns models with production code. <br> - Eliminates leakage. <br> - Uses the intended training pipeline. | - May change model performance (for the better). <br> - Takes a few minutes to run. | **Low** |
| **2. Re-engineer Features** | Modify the prediction logic to re-create the old, leaky features that the stale models expect. | - Avoids retraining. <br> - Gets the server running quickly. | - **Propagates a critical design flaw (data leakage).** <br> - Adds technical debt. <br> - Violates the "production-ready" goal. | **High** |
| **3. Disable Preprocessor** | Hack the prediction logic to bypass the preprocessor step entirely. | - None. | - Guarantees prediction failure. <br> - Breaks the entire ML pipeline. | **Critical** |

---

### Chosen Path: Option 1 - Retrain Models

- **Rationale**: This is the only option that addresses the root cause correctly and aligns with the project's goal of a production-ready, reliable system. It eliminates data leakage, ensures consistency between training and inference, and pays down technical debt. The risk is minimal and confined to a predictable change in model performance metrics.
- **Impact**: New `*.joblib` files and a new `metadata.json` will be generated in `backend/models/`. The backend service will become healthy and serve predictions based on a correctly trained, leak-free model.
- **Rollback Plan**: The old model artifacts can be restored from git history if necessary.
