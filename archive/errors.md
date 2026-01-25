# NFL_ML_Predictions — Diagnostic Report (errors.md)

Generated: 2025-11-08

This document records a repository-wide diagnostic pass focusing on correctness, ML-safety (leakage), runtime risks, and maintainability. It is a read-only report: no code changes are made here. Each issue includes file path, line range (approximate), severity, explanation, and an actionable fix suggestion.

## Codebase overview

- Purpose: produce NFL game-level features, train score regressors and a win-probability classifier, expose predictions via a FastAPI backend, and provide a React/Vite frontend.
- Key components:
  - `backend/build_csv_datasets.py` — feature engineering & dataset builder (source of truth for engineered columns).
  - `backend/pipeline_enhanced.py` — enhanced training pipeline, CV, plotting, and artifact management.
  - `backend/train_models.py` — legacy training path / hyperparameter search and metadata writing.
  - `backend/main.py` — FastAPI app: model loading, sanity checks, endpoints (`/predict`, `/schedule/next-week`, `/health`).
  - `backend/transform_dataset.py` — helper converting per-team rows into per-game rows.
  - `frontend/` — React app; interacts with API endpoints (not exhaustively inspected for ML logic here).

For ML prediction quality, the most important files are: `build_csv_datasets.py` (feature correctness & leakage), `train_models.py` and `pipeline_enhanced.py` (CV/training/calibration), and `main.py` (inference and fallback behavior). Problems in any of these can silently corrupt predictions.

---

# Ordered Error List (descending severity)

## ❌ Critical Error #1 — Non-picklable fallback calibration wrapper

**File:** `backend/pipeline_enhanced.py`  
**Line:** ~470–520 (calibration fallback block added)  
**Type:** ML Misconfiguration / Runtime  
**Severity:** Critical

**Description:**
The calibration fallback creates a local class `_UncalibratedWrapper` inside an exception handler and assigns an instance to `win_cal`. Later, `save_artifacts()` calls `_save_joblib(models.win_model, P_WIN)` (joblib.dump). Local (nested) classes are not importable / picklable by joblib/pickle, and attempting to persist `win_cal` will raise a pickling error at runtime.

**Why It Happens:**
Python's pickle mechanism (used by joblib) cannot serialize classes defined inside a function scope because they are not importable by module path. The code defines the wrapper inside the `except` block and then attempts to save the object.

**How to Fix:**

- Move the fallback wrapper to module scope (top-level class) so instances are picklable.
- Prefer to use composition with a small top-level wrapper that delegates to the classifier. Or set `win_cal` to the underlying `LogisticRegression` object (uncalibrated) and record in metadata that calibration failed.

**Example Fix:**

```python
class UncalibratedProbWrapper:
    def __init__(self, clf):
        self.clf_ = clf
    def predict_proba(self, X):
        return self.clf_.predict_proba(X)
    @property
    def base_estimator_(self):
        return self.clf_
    @property
    def method(self):
        return None

# ... inside train_models or pipeline_enhanced exception handler:
win_cal = UncalibratedProbWrapper(win_base)
```

**Logic Behind This Fix:**
Top-level class is importable and picklable, enabling joblib persistence. Alternatively, persist the raw `LogisticRegression` and set metadata flag `calibrated: false`.

**Expected Improvement:**
Prevents hard crashes when saving models; ensures artifact persistence succeeds and downstream loading works.

## ❌ Critical Error #2 — Inconsistent artifact metadata key names (load vs save)

**File:** `backend/train_models.py` (saves metadata) and `backend/main.py` (loads metadata)  
**Line:** train_models metadata write ~line 300; main.py load_objects ~line 220  
**Type:** Runtime / Integration  
**Severity:** Critical

**Description:**
The training writer (`train_models.py`) writes metadata keys like `"win_model": "win_clf_calibrated.joblib"` and `"home_model": "home_model.joblib"`. The API loader in `main.py` calls `resolve_model_path("win_CLF_calibrated", "win_clf_calibrated.joblib")` — note the differing key casing `win_CLF_calibrated` vs `win_model` or `win_clf_calibrated`. This mismatch can cause the API to fail to locate the win-model path even though it exists.

**Why It Happens:**
Inconsistent naming/casing conventions across training and serving code (metadata keys are not standardized). Case-sensitive keys and ad-hoc key names lead to runtime FileNotFound or KeyError when main.py expects a different field.

**How to Fix:**

- Standardize metadata schema across training and serving. Choose canonical keys (e.g., `home_model`, `away_model`, `win_model`, `preprocessor`) and update both training scripts and `main.py` to use them.
- Add a compatibility layer in `load_objects()` that accepts multiple candidate keys but logs a warning when falling back.

**Example Fix (compat wrapper):**

```python
WIN_KEYS = ["win_model", "win_clf_calibrated", "win_CLF_calibrated"]
def _pick_meta_path(meta, keys, fallback):
    for k in keys:
        if k in meta:
            return meta[k]
    return fallback

win_path = _pick_meta_path(meta, WIN_KEYS, 'win_clf_calibrated.joblib')
```

**Expected Improvement:**
Removes brittle integration errors; API will not silently mis-locate artifacts and will produce clearer error messages.

## ❌ Critical Error #3 — Unhandled/incorrect exception branch in predict (variable used but not set)

**File:** `backend/main.py`  
**Line:** predict_game exception handler around win probability fallback (~line 1030)  
**Type:** Bug / Runtime  
**Severity:** Critical

**Description:**
In the exception handling block for computing win probability, there is an apparent stray bare identifier `home_prob` used alone (no assignment). This is likely a typo and causes a NameError or leaves `home_prob` undefined when the code continues. Also, the block toggles `win_fallback_used` and may proceed without setting a numeric `home_prob`.

**Why It Happens:**
Probably a copy/paste error or partial edit left behind during refactor. Code paths can reach here and attempt to return PredictionResponse with undefined variables.

**How to Fix:**

- Replace the stray `home_prob` with an explicit fallback assignment (e.g., `home_prob = 0.5`) or, better, compute a calibrated fallback from regression margin/sigmoid.
- Ensure that when `ALLOW_FALLBACK_PREDICTIONS` is false the function raises an HTTPException and does not proceed.

**Example Fix:**

```python
except Exception:
    log.exception("Unexpected error while computing win probability; using sigmoid fallback")
    if not ALLOW_FALLBACK_PREDICTIONS:
        raise HTTPException(503, detail="Win probability model failed and fallbacks are disabled")
    # deterministic fallback
    home_prob = 0.5
    win_fallback_used = True
    win_classifier_used = False
    win_prob_source = "legacy-sigmoid"
```

**Expected Improvement:**
Avoids runtime NameError and produces predictable fallback probability or rejects request cleanly.

## ❌ Critical Error #4 — Local class and function duplication / non-top-level definitions prevent pickling or reuse

**File:** multiple (notably `pipeline_enhanced.py`)  
**Line:** fallback wrapper class defined inside function (~lines 480–520)  
**Type:** Runtime / Maintainability  
**Severity:** High

**Description:**
Local classes and helper definitions inside functions (or inside exception branches) impair serialization and are harder to unit-test or reuse. We already flagged the Uncalibrated wrapper; similar patterns appear elsewhere (ad-hoc inner helpers). These raise hazards for joblib/pickle and when reloading objects in the API.

**How to Fix:**

- Move helpers that need to be persisted or reused to module-level definitions.

**Expected Improvement:**
Improved serialization compatibility and testability.

## ❌ Critical Error #5 — Duplicate / conflicting function definitions in `build_csv_datasets.py`

**File:** `backend/build_csv_datasets.py`  
**Line:** `make_time_key` defined twice (~lines 70 and 220 in the file excerpt)  
**Type:** Bug / Logic  
**Severity:** High

**Description:**
`make_time_key` appears to be defined more than once (duplicated). Duplicate helper definitions can lead to confusion, unexpected behavior (if two versions diverge), or accidental shadowing. The second variant uses defensive casting and NaN handling while the earlier one assumed integers — this indicates inconsistent behavior.

**Why It Happens:**
Likely due to incremental edits and partial refactors finishing with duplicate copy-pasted functions.

**How to Fix:**

- Consolidate to a single `make_time_key()` implementation (use the defensive version that handles NaNs and casting). Remove duplicates.

**Expected Improvement:**
Predictable time-key computation; fewer surprises when sorting and grouping by time.

## ❌ Critical Error #6 — Potential leakage via overly broad forbidden tokens handling and feature filtering

**File:** `backend/pipeline_enhanced.py` and `backend/build_csv_datasets.py`  
**Line:** leakage filter in pipeline_enhanced.py ~lines 150–190; leak blocklist in train_models.py ~lines 60–90  
**Type:** Data Issue / ML Misconfiguration  
**Severity:** High

**Description:**
There are multiple leak-guard heuristics across scripts that are inconsistent (different blacklists, different tokenization). For example, `pipeline_enhanced.py` chooses to exclude features where token match any of `FORBIDDEN_TOKENS` and `FORBIDDEN_EXACT`, but earlier versions or other scripts had different rules (e.g., `LEAK_BLOCKLIST` in `train_models.py`). Inconsistencies risk either (a) accidentally dropping useful pre-game features (overbroad token), or (b) missing leaky columns (incomplete list).

**Why It Happens:**
Different authors / files implemented their own heuristics. There is no centralized canonical schema for which columns are allowed / blocked.

**How to Fix:**

- Create a single authoritative leak-guard configuration (e.g., `backend/config/leak_guard.json` or a small module) and import it from all training/ingestion scripts.
- Prefer exact-name blacklists over simple substring matches for sensitive tokens; document any substring rules explicitly and test them with unit tests.

**Expected Improvement:**
Reduce chance of leaking post-game info into features, improving model validity.

## ❌ Critical Error #7 — `build_csv_datasets.py` has inconsistent rolling helper usage and a clear bug

**File:** `backend/build_csv_datasets.py`  
**Line:** `_rolling_prior_stats` around the two `if advanced_cols` blocks (~lines 290–330 in excerpt)  
**Type:** Bug / Logic  
**Severity:** High

**Description:**
The function defines `safe_rolling_mean` but later has two `if advanced_cols:` loops. The second loop calls `safe_roll` which is undefined (typo), likely causing NameError at runtime. This will break feature construction when `advanced_cols` is present.

**Why It Happens:**
Copy/paste or refactor error where the helper name changed but not all references updated.

**How to Fix:**

- Replace `safe_roll` with the correct function name (likely `safe_rolling_mean`) and remove the duplicated loop.

**Expected Improvement:**
Fixes runtime crash during dataset build when advanced metrics are present.

## ❌ Critical Error #8 — Hard-coded Windows paths and environment coupling in training scripts

**File:** `backend/train_models.py`  
**Line:** DEFAULT train dataset path near top (~line 40–60) and load_dotenv usage  
**Type:** Maintainability / Portability  
**Severity:** Medium

**Description:**
`TRAIN_DATASET_FILE` is hard-coded to an absolute Windows path pointing to a user directory. This will break on other machines or CI. There are many env-loading patterns across scripts with varying defaults.

**How to Fix:**

- Use relative paths by default (e.g., `backend/data/game_features.csv`) and allow env override. Document recommended venv/working-dir invocation in README.

**Expected Improvement:**
Improved portability and reproducible runs on CI and other developers' machines.

## ❌ Critical Error #9 — Multiple incompatible OneHotEncoder params used across code

**File:** `backend/train_models.py` and `backend/build_csv_datasets.py` / `pipeline_enhanced.py`  
**Line:** `_make_preprocessor` in train_models (~lines 140–170) and other preprocessors  
**Type:** Runtime / Compatibility  
**Severity:** Medium

**Description:**
Different files pass inconsistent parameters to `OneHotEncoder` (`sparse=True`, `sparse_output=True`) depending on scikit-learn versions. This will raise TypeError on older/newer sklearn versions.

**How to Fix:**

- Normalize to a small compatibility helper that creates an OHE with parameters resolved by installed sklearn version. Add a guard or pin the minimum sklearn version in `requirements.txt`.

**Expected Improvement:**
Avoids hard failures when developers use different sklearn releases.

## ❌ Critical Error #10 — `main.py` duplicated helper `_glob_latest` and malformed default regex

**File:** `backend/main.py`  
**Line:** `_glob_latest` duplicated and `ALLOW_ORIGIN_REGEX` default (~lines 70–120)  
**Type:** Maintainability / Runtime  
**Severity:** Medium

**Description:**
`_glob_latest` appears twice (two definitions). Also `ALLOW_ORIGIN_REGEX` default is `r"https://.*//.vercel//.app$"` — this contains extra slashes and is almost certainly not the intended regex. Duplicate function definitions confuse static analysis; malformed regex may allow unexpected CORS origins or mis-block valid ones.

**How to Fix:**

- Remove duplicate `_glob_latest` definitions and keep a single clear implementation.
- Replace `ALLOW_ORIGIN_REGEX` with a correct pattern such as `r"https://.*\\.vercel\\.app$"` and add comments.

**Expected Improvement:**
Cleaner CORS control and fewer maintenance surprises.

## ❌ Critical Error #11 — `predict` alignment logic may mangle expected preprocessing order

**File:** `backend/main.py`  
**Line:** `_get_expected_features` and `_predict_with_fill` logic (~lines 860–940)  
**Type:** ML Misconfiguration / Logic  
**Severity:** Medium

**Description:**
The methods try to extract `feature_names_in_` from the estimator and reindex `Xdf` accordingly. This is brittle across pipeline wrappers (ColumnTransformer, Pipeline, GridSearchCV), and the heuristics try many attribute names. A mismatch between `feature_names_in_` and the preprocessor's expected column order can lead to silent misalignment where columns are filled with NaNs or reordered incorrectly. There is also logic that adds missing columns as NaN which may be acceptable but should be explicit.

**How to Fix:**

- Prefer to persist and use an explicit canonical `feature_list` (the exact ordered list used to fit the preprocessor) in `metadata.json` and rely on that to assemble `X` at prediction time.
- Only fall back to estimator introspection when `feature_list` is absent and log a warning.

**Expected Improvement:**
Reduce silent feature-order/data-mismatch bugs at inference time and increase reproducibility.

## ❌ Critical Error #12 — CLI flags expected by user are missing in `build_csv_datasets.py`

**File:** `backend/build_csv_datasets.py`  
**Line:** `parse_args()` (~near bottom)  
**Type:** UX / Maintainability  
**Severity:** Low

**Description:**
The file's top comments and README mention CLI options like `--save-dominance-matrix`, `--no-calibration-rows`, `--dominance-log`, and `--encode onehot`. However `parse_args()` currently only accepts `--start`, `--end`, `--out-dir`, `--legacy-root-copy`. Passing the additional flags results in argparse `unrecognized arguments` errors (observed by the user).

**How to Fix:**

- Add the missing CLI flags to `parse_args()` and wire them into the `build_dataset()` call. Keep defaults backward-compatible.

**Expected Improvement:**
Improved CLI ergonomics and parity with documentation.

## ❌ Critical Error #13 — Tests not exercising critical failure paths / no integration smoke tests

**File:** repository-wide (tests/)  
**Line:** `backend/tests/*`  
**Type:** Process / Quality  
**Severity:** Medium

**Description:**
Some tests exist but the earlier attempt to run `pytest` reported "no tests ran". The test suite may not cover high-risk areas like serialization of models, prediction fallback behavior, or dataset building with advanced metrics.

**How to Fix:**

- Add targeted unit tests that:
  - Assert that `pipeline_enhanced` can produce `cv_fold_metrics.csv` and `training_summary.json` for a tiny synthetic dataset.
  - Assert that `main.load_objects()` can load models produced by training scripts (roundtrip test using tempdir).
  - Smoke-test the `/predict` endpoint via `TestClient` with both complete historical rows and future-game feature construction.

**Expected Improvement:**
Faster detection of regressions and safer refactors.

---

# Summary and Next Steps

The above diagnostics outline issues that range from serialization/persistence bugs and metadata mismatches (critical) to maintainability and behavioral inconsistencies (medium/low). Fixing the critical items first (1-4) will prevent runtime crashes and ensure that artifacts saved by training code can be loaded by the API. After those, harmonize metadata keys, centralize leak-guarding, fix the dataset builder bugs, and add targeted tests.

Guiding Principles:

- Do not modify code yet. This report is diagnostic-only.  
- Prioritize: serialization and artifact naming -> inference correctness (feature alignment) -> data leakage guards -> CLI parity and tests.  
- Make small, testable commits and add unit tests that cover the exact failure modes described above.

If you want, I can proceed to implement fixes in the recommended order and run the unit/CLI validations. For each change I will: (a) open a narrow PR with one focused change, (b) add a test that demonstrates the fix, (c) run a local smoke run, and (d) update this report with the validation results.

---

End of report.
