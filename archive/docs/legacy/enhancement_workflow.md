# NFL Prediction System — Enhancement Workflow (Q4 2025)

_This playbook assumes a focused ~5 hour build session. Timestamps are relative (T+hh:mm) so you can pause/resume while keeping the intended sequence._

| T+ | Milestone | Files/Sections | Why it matters |
| --- | --- | --- | --- |
| 00:00 | Baseline checks & artifact snapshot | `backend/data/enhanced/model_outputs/*.csv` | Know exactly what “better” means before touching production code. |
| 00:20 | Persist champion model artifacts | `backend/data/enhanced/train_new_models.py` (L118-L172) | Create a joblib bundle + metadata for the logistic_blended model. |
| 01:00 | Dual-model loading in FastAPI | `backend/main.py` (L60-L125) | Load both production LightGBM and enhanced logistic stacks with clear modes. |
| 01:45 | Prediction response upgrades | `backend/main.py` (L137-L213) | Surface model version, confidence, and features used to frontend. |
| 02:30 | Frontend trust & UX refresh | `frontend/src/components/*.jsx` | Show version badges, calibration charts, and historical context. |
| 03:45 | Observability & guardrails | `backend/main.py` (L20-L55), `backend/train_models.py` (L200-L260) | Structured logs, drift alerts, and feature validation keep the app reliable. |
| 04:30 | Automation & retraining loop | `scripts/`, GitHub Actions | Weekly retrains with dataset rebuild + notifications to stay ahead of drift. |

---

## Step 1 — T+00:00 to 00:20 · Baseline snapshot

1. Open the latest walk-forward metrics in `backend/data/enhanced/model_outputs/summary_metrics.csv`.
   - **Why**: anchors your improvements with concrete accuracy/ROC/Brier targets.
   - **Check**: logistic_blended should read ~0.63 accuracy / 0.65 ROC AUC.

2. Copy the CSV/JSON pair into an `artifacts/2025-10-03/` folder (git-ignored) for safekeeping.
   - **Command**: `mkdir -p artifacts/2025-10-03 && cp backend/data/enhanced/model_outputs/* artifacts/2025-10-03/`
   - **Reasoning**: keeps a frozen baseline in case regressions slip in later.

3. Log the benchmark in your team notes (Notion/Jira) with a link to `walk_forward_metrics.csv` so everyone knows the baseline.

_Exit criteria_: Everyone agrees on “better than 0.65 ROC, 0.63 accuracy” before edits begin.

---

## Step 2 — T+00:20 to 01:00 · Persist the champion model bundle

1. **Add joblib export** in `backend/data/enhanced/train_new_models.py` right after the logistic block (≈L118-L142).
   - **Insert**: `joblib.dump(logit, OUTPUT_DIR / "logistic_blended.joblib")`
   - **Logic**: the pipeline already includes a `StandardScaler`, so persisting the whole `Pipeline` keeps preprocessing + model coherent.

2. **Write enhanced metadata** near the bottom of the same file (≈L164-L194).
   - Create `metadata = {"timestamp": ..., "features": logit_features, "cv_summary": metrics}`.
   - Dump as `OUTPUT_DIR / "logistic_blended_metadata.json"` using `json.dumps(..., indent=2)`.
   - **Why**: API consumers need feature order + scores to validate compatibility.

3. **Version tag**: generate a hash
   - Use `hashlib.md5(pd.util.hash_pandas_object(df[logit_features], index=False).values).hexdigest()[:10]`.
   - Store it under `"dataset_hash"` so retraining diff is trivial.

4. Re-run `python backend/data/enhanced/train_new_models.py` and confirm the new files appear.
   - **Test**: hash logged in metadata matches actual dataset; `summary_metrics.json` should be unchanged aside from small float jitter.

---

## Step 3 — T+01:00 to 01:45 · Dual-model loading in FastAPI

1. **Extend `load_objects()`** in `backend/main.py` (~L74-L119).
   - Add a new block that checks for `model_outputs/logistic_blended.joblib` and, if found, loads it under `"win_model_enhanced"`.
   - Parse the metadata JSON so you can surface `"mode": "logistic_blended"` when selected.
   - **Syntax tip**: wrap with `try/except FileNotFoundError` to fall back gracefully.

2. **Add a configuration toggle**.
   - At the top (≈L42), add `ACTIVE_WIN_MODEL = os.getenv("WIN_MODEL_MODE", "production")`.
   - Store it inside `model_objects` so routes can check it.
   - **Pitfall**: ensure `ACTIVE_WIN_MODEL` is validated against `{"production", "logistic_blended"}` to avoid typos.

3. **Document** the new behavior in `metadata.json` by appending `"win_model_enhanced": "logistic_blended.joblib"` once the file is deployed (update `backend/train_models.py` to include the key after L237).

4. **Smoke test**: run `uvicorn backend.main:app --reload` and hit `/health`; expect `"mode": "production"` initially.

---

## Step 4 — T+01:45 to 02:30 · Enrich prediction responses

1. In `PredictionResponse` dataclass (`backend/main.py` ≈L96-L112), append fields:
   - `model_version: str`
   - `confidence_interval: tuple[float, float] | None = None`
   - `features_considered: list[str] | None = None`
   - **Logic**: keep defaults optional so older clients keep working.

2. Inside `predict_game` (≈L173-L220):
   - After computing `home_prob`, calculate logit calibration spread using the stored metadata (e.g., ±1 std from `walk_forward_metrics.csv`).
   - Populate the new fields; when `ACTIVE_WIN_MODEL == "logistic_blended"`, use the enhanced pipeline and its feature list.

3. Emit a structured log entry (`log.info`) containing game ID, season/week, model mode, probability, and dataset hash for observability.

4. Update `/predict/next-week` to pass through the same enriched payload so the frontend stays consistent.

5. **Check**: manual request via `httpie` should return the new keys. Example expectation:

   ```json
   {
     "home_win_probability": 0.642,
     "model_version": "logistic_blended@2025-10-03",
     "features_considered": ["home_prior_pf_avg_3", ...]
   }
   ```

---

## Step 5 — T+02:30 to 03:45 · Frontend trust & UX refresh

1. `frontend/src/components/PredictionResult.jsx` (~L16-L48):
   - Add a `modelVersion` badge beneath the title using `<span className="badge">Model: {entry.meta.modelVersion}</span>`.
   - Explain tooltips: use `title` attribute to display ROC/Brier metrics from API response.

2. `frontend/src/components/HistoryChart.jsx` (inspect lines where dataset is built):
   - Overlay a rolling calibration curve by plotting `actual - predicted` residuals.
   - **Syntax reminder**: keep dataset transformation inside `useMemo` to avoid re-renders.

3. `frontend/src/components/TeamGrid.jsx`:
   - Highlight games where confidence interval width > 0.25 with a warning badge.
   - Ensure accessible color contrast (WCAG AA); use CSS variables declared in `TeamGrid.css`.

4. Create a new component `ConfidenceLegend.jsx` under `frontend/src/components/` and mount it inside `DashBoard.jsx` after `PredictionResult` (≈L28).
   - Purpose: explain what the confidence bands mean.

5. Add analytics hook in `PredictionContext.js` to push events (`model_used`, `confidence_span`) to your analytics service (placeholder function now, real implementation later).

6. **Validation**: run `npm run lint` + screenshot updated UI for release notes.

---

## Step 6 — T+03:45 to 04:30 · Observability & guardrails

1. Replace `logging.basicConfig` in `backend/train_models.py` (≈L46) with a `dictConfig` that outputs JSON lines (structured logging).
   - Include keys for `event`, `dataset_hash`, and `model_version`.

2. Add feature drift detection in `train_models.py` after metadata write (≈L246).
   - Use `pandas.DataFrame.corrwith` to compare new vs. previous dataset (load last metadata). If drift > 0.15 on any feature, log a WARNING and write `models/drift_report.json`.

3. In `backend/main.py` startup, validate that the dataset hash recorded in metadata matches the enhanced metadata; if mismatch, raise and stop the app (fail-fast).

4. Wire a Prometheus-compatible `/metrics` endpoint (FastAPI dependency) exporting:
   - Request latency histograms
   - Counter per model mode
   - Gauge for last retrain timestamp

5. **Test**: run `pytest backend/tests/test_health.py` (add if missing) to ensure health endpoint reports the added fields.

---

## Step 7 — T+04:30 to 05:30 · Automation & retraining loop

1. Add a script `scripts/nightly_retrain.ps1` that:
   - Activates `.venv`
   - Runs `python backend/build_csv_datasets.py --start 2010 --end $(Get-Date -Format yyyy) --out-dir backend/data`
   - Runs `python backend/data/enhanced/train_new_models.py`
   - Sends a webhook if metrics dip below thresholds.

2. Create `.github/workflows/retrain.yml`:
   - Windows runner to build dataset and enhanced models weekly (cron `0 12 * * 2`).
   - Upload artifacts (`model_outputs/*.json`, new joblib files) for review.

3. Update `DEPLOYMENT.md` with the new retrain cadence and manual override steps.

4. Configure an on-call Slack alert when the workflow fails (GitHub Actions → Slack app).

5. **Dry run** locally by invoking the PowerShell script; ensure exit codes propagate.

---

### Recap checklist

- [ ] Enhanced logistic model persisted with metadata + hash.
- [ ] FastAPI can swap between LightGBM and logistic pipelines via env flag.
- [ ] Frontend displays model version, confidence cues, and calibration visuals.
- [ ] Observability stack records drift, metrics, and structured logs.
- [ ] Weekly automation keeps models fresh with alerts on degradation.

Complete each step sequentially; if you pause more than a day, re-run Step 1 to refresh baselines before resuming.
# Archived: Enhancement Workflow

This process note has been archived. Current operational steps reside in `docs/RUNBOOK.md` and engineering cadence is reflected in `docs/report.md`.
