# Analysis & Teaching — NFL_ML_Predictions

- Backend: Scheduling logic now prefers upcoming kickoff times, then a calendar-based week, with the dataset tail as a last resort so the frontend shows the real _next_ slate and not archived results.
- Backend: Model loading now tries different candidate file names (e.g., `home_model.joblib`, `home_pipe.joblib`) so startup isn't noisy with `Pipeline not found` messages.

\n### 1) Schedule & Time-based heuristics (Backend)

Task: Add a unit test to `Dashboard` that mocks `predictGame` and asserts `setPrediction` is called with correct payload.

Hints:

- Use `jest.mock('../../api/client')` and spy `predictGame` return value.
- Use `fireEvent.click` on the TeamGrid card and assert that `setLoading` becomes true then false.

\n### 3) CORS & Heroku pitfalls (Backend)

- Concept: Heroku can't guess which origins you want allowed — explicit envs are necessary. `RESTRICT_CORS=true` forces you to list origins using `ALLOWED_ORIGINS` or `CORS_ORIGINS`.
- Code pointers: `backend/main.py` -> CORS section and `/debug` endpoint.
- Practice: In production set `ALLOWED_ORIGINS=https://your-front-end.example.com,http://localhost:3000` and `RESTRICT_CORS=true`. In dev, set `RESTRICT_CORS=false` so localhost is allowed automatically.

Task: After deployment, call `/debug` and verify it returns `cors_origins` and `restrict_cors` that match your env.

Hints:

- If `RESTRICT_CORS=true` and you forget to set `ALLOWED_ORIGINS`, the server will deny all cross-origin requests. This often shows up as CORS error logs on the browser console even though the backend responds.

\n### 4) Prediction payload & contract (Frontend ↔ Backend)

- Concept: Keep the JSON contract stable. `api/client.js` normalizes camelCase -> snake_case for predict. Always validate server response schema client-side.
- Code pointers: `frontend/src/api/client.js` -> `validatePredictionResponse()` and `predictGame()`; `backend/main.py` -> `PredictionRequest` Pydantic model.
- Practice: Add a `prediction_source` and show it in the TeamGrid so users see whether the classifier or fallback was used.

Task: Add a card-level badge for `prediction_source` and style it using CSS variables from `base.css`.

Hints:

- On the backend, the `PredictionResponse` was extended with `prediction_source` and `confidence_score`. Use these fields to inform the UI and to gate tooltips.

\n### 5) Model Loading & Feature Alignment (Backend)

- Concept: Model artifacts may be named differently in training vs deploy; use candidate filename matching and log the chosen file.
- Code pointers: `backend/main.py` -> `ModelManager._load_pipelines()`, `ModelManager._load_metadata()` and `build_feature_frame()`
- Practice: When you update models, incrementally change `MODELS_DIR` or use `--reload` patterns and provide detailed startup logs.

Task: Temporarily rename your joblib in models folder and confirm the loader matches the fallback candidate.

Hints:

- Adding `feature_names_in_` alignment reduces classifier rejections on unseen features; implement imputation and missing columns fallback.

\n### 6) Testing & deploy checks (Ops)

- Concept: Add a debug endpoint (`/debug`) to let CI verify CORS, model presence, and dataset path without exposing secrets.
- Code pointers: `backend/tests/test_api_endpoints.py` shows how to smoke `/health`, `/schedule/next-week`, `/predict`.

Task: Add a CI job to run `pytest backend/tests` and `npm run build` for the front-end.

Hints:

- Use `scripts/verify_api_cors.py` to ensure the server's CORS config matches `ALLOWED_ORIGINS` and works end-to-end with Vercel/Heroku.

---

## Common failure modes & how to fix them

- Backend returns 500 on `/predict`: usually missing required columns in the dataset or a model that didn't load. Check server logs for `Model loading failed` or `Dataset is empty`.
- Dashboard shows old week: server used dataset tail. Confirm `home_game_date` or schedule CSV has upcoming kickoffs; change environment so `SCHEDULE_PATH` points to the updated schedule.
- CORS errors at deploy: Set `ALLOWED_ORIGINS` and `RESTRICT_CORS=true`, then call `/debug` to verify.

---

## Common failure modes & how to fix them

- Backend returns 500 on `/predict`: usually missing required columns or a model didn't load. Inspect server logs for `Model loading failed` or `Dataset is empty`.

- Dashboard shows old week: server likely used dataset tail; confirm `home_game_date` is present or update `SCHEDULE_PATH` to a newer CSV.

- CORS issues after Heroku deploy: ensure `ALLOWED_ORIGINS` includes your frontend domain and `RESTRICT_CORS=true` is set accordingly.

---

## How to run & quick-smoke commands (PowerShell)

Start the backend:

```
cd backend; .\.venv\\Scripts\\Activate.ps1; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend (dev):

```
cd frontend; npm install; npm run dev
```

Smoke endpoints:

```
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri "http://127.0.0.1:8000/schedule/next-week" -Method Get | ConvertTo-Json -Depth 4
$payload = @{ home_team='CLE'; away_team='BAL'; season=2025; week=11 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $payload -ContentType 'application/json' | ConvertTo-Json -Depth 6
```

Run backend tests:

```
cd backend; .\.venv\\Scripts\\Activate.ps1; pytest -q
```

Build frontend:

```
cd frontend; npm run build
```

---

## Where to add E2E tests

Add Playwright or Cypress tests that verify end-to-end: schedule appears, clicking a card triggers `/predict`, the response is shown, and `/history` updates. The test can stub or use a deployed backend.

**Task:** Add Playwright test scaffolding that: visits index, waits for schedule, clicks a matchup, and asserts that `prediction_source` appears on the card.

---

## Top recommended code improvements (low-friction)

- Expose schedule selection metadata (`strategy`, `selection`) in `/schedule/next-week` so the frontend can show provenance for the chosen slate.

- Add a `/status/overview` endpoint (or expand existing) to provide dataset & model metrics to the dashboard.

- Add unit tests for `get_current_nfl_context()` with boundary months (July/August/January) so calendar fallback behaves as intended.

---

## Further reading & references

- FastAPI docs: Pydantic models, lifespans, and middleware
- FastAPI CORSMiddleware documentation
- sklearn: `feature_names_in_` and ColumnTransformer behaviour
- React: Context vs local state, test patterns (react-testing-library)

---

## Summary & next steps

This repository looks stable; the major fixes were schedule & prediction contract, CORS improvements, and model-loading robustness. The next steps are: add E2E tests, add a GitHub Action for CI (pytest + npm build), and small UX polish like schedule provenance on the dashboard.

If you want, I can scaffold the Playwright tests and the CI workflow next.

Developed with insights from Raptor mini (Preview).

Updated: 2025-11-17 (UTC)
