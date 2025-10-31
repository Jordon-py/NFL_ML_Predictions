# Maintenance Log

This file tracks errors, optimizations, and suggested improvements for the NFL ML Predictions codebase.

## File: backend/main.py

- Issue: Missing import for `math` module (used in sigmoid calculation for win probability fallback).
- Fix: Add `import math` to the imports section.
- Syntax Example:

  ```python
  import math
  ```

- Issue: Type checker errors for dict access on `model_objects` (line 289, 360, etc.).
- Fix: Add type guards or use defensive access.
- Syntax Example:

  ```python
  mode = model_objects.get("mode", "production") if isinstance(model_objects, dict) else "production"
  ```

- Issue: `reportAttributeAccessIssue` for accessing attributes on potentially None objects.
- Fix: Add null checks before attribute access.
- Syntax Example:

  ```python
  if model_objects and hasattr(model_objects, "mode"):
      mode = model_objects.mode
  ```

- Issue: `reportGeneralTypeIssues` for operations on potentially incompatible types.
- Fix: Add explicit type conversions or checks.
- Syntax Example:

  ```python
  point_diff = float(point_diff)
  ```

## File: backend/train_models.py

- Issue: Incomplete `_fit_classifier` function (missing RandomizedSearchCV setup).
- Fix: Implement full _fit_classifier with RandomizedSearchCV, param_dist, and holdout metrics.
- Syntax Example:

  ```python
  rs = RandomizedSearchCV(
      estimator=Pipeline([('pre', pre), ('clf', LogisticRegression(random_state=random_state, max_iter=1000))]),
      param_distributions=CLF_PARAM_DISTS,
      cv=TimeSeriesSplit(n_splits=N_SPLITS),
      scoring="neg_log_loss",
      n_jobs=-1,
      random_state=random_state,
      verbose=2,
      refit=True,
  )
  rs.fit(X, y)
  best_pipeline = cast(Pipeline, rs.best_estimator_)
  # Compute holdout metrics...
  return best_pipeline, holdout_metrics
  ```

- Issue: `mode` variable not defined in metadata (line 309).
- Fix: Set `production_ready` to `True` or `False` explicitly.
- Syntax Example:

  ```python
  "production_ready": True,
  ```

## File: frontend/src/api/client.js

- Issue: Parameter `base` in `normalizeBase` has implicit any type.
- Fix: Add JSDoc type annotation.
- Syntax Example:

  ```javascript
  /**
   * @param {string} base
   * @returns {string}
   */
  function normalizeBase(base) {
  ```

- Issue: Property `VITE_API_BASE` does not exist on type (import.meta.env).
- Fix: Use optional chaining or default.
- Syntax Example:

  ```javascript
  const ENV_BASE = String(import.meta.env?.VITE_API_BASE ?? "");
  ```

## File: frontend/src/components/TeamGrid.jsx

- Issue: Parameter `row` in `formatKickoffTime` has implicit any type.
- Fix: Add JSDoc or convert to TypeScript.
- Syntax Example:

  ```javascript
  /**
   * @param {Object} row
   * @returns {string}
   */
  const formatKickoffTime = (row) => {
  ```

- Issue: Property `away_score` does not exist on type (in destructuring).
- Fix: Add optional chaining or check existence.
- Syntax Example:

  ```javascript
  const { home_score, away_score } = result || {};
  if (home_score !== undefined && away_score !== undefined) {
  ```

- Issue: Various property access issues on potentially undefined objects.
- Fix: Add null checks.
- Syntax Example:

  ```javascript
  if (game && game.home_abbr) {
  ```

## General Issues

- Issue: Numpy import failure due to corrupted installation.
- Fix: Reinstall numpy cleanly.
- Syntax Example:

  ```bash
  pip uninstall numpy -y
  pip install numpy
  ```

- Issue: Models not fitted, causing dummy predictions.
- Fix: Run training script after fixing _fit_classifier.
- Syntax Example:

  ```bash
  python backend/train_models.py --data backend/data/game_features.csv --out backend/models
  ```

## To-Implement

- Add win probabilities to frontend display (currently only scores shown).
- Implement calibration for win classifier to improve probability estimates.
- Add unit tests for prediction functions.
- Optimize feature building for future games (cache historical stats).

## Unused or Missing Implementations

- Probabilities calculated in backend but not sent to frontend in TeamGrid.jsx.
- Suggested: Modify TeamGrid to display win probabilities beside scores.

## AI Developer Notes

- Dev: Ensure all changes are tested with real data before deployment.
- Dev: Monitor for data leakage in feature engineering.
- Dev: Update this log after each fix with date and time.
- Dev: If issues persist, provide exact error messages for debugging.
- Dev: Prioritize backend fixes before frontend polish.

Last Updated: October 21, 2025
