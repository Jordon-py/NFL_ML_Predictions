# Repository Guardian — Copilot instructions (W1)

Big Picture
- FastAPI serves ML predictions; models/datasets live under backend/data/prod-models; Vite/React frontend consumes REST.

Dev Quickstart
- Backend: cd backend; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt; uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
- Frontend: cd frontend; npm install; npm run dev
- Deploy: git push heroku rollback/heroku-endpoint-restore:main

Conventions
- React via hooks/Context only; custom CSS (no frameworks). Logging via warnings/errors; raise HTTPException on API errors. Tests: pytest (backend/tests), vitest (frontend).
- Maintain alfred.log.md tasks; update docs/report.md “Active Enhancements Under Development.”

Snippet (CORS defaults) [backend/main.py](backend/main.py#L75-L111)
```
def _parse_allowed_origins(raw: str) -> List[str]:
    ...
```

Services & Integrations
- Models and metadata in backend/data/prod-models/models; dataset default backend/data/prod-models/game_features_20251210.csv; env overrides MODELS_DIR/DATASET_PATH.
- CORS allow list ALLOWED_ORIGINS + regex r"https://.*\.vercel\.app"; catch-all OPTIONS avoids 400 preflights [backend/main.py](backend/main.py#L563-L571). No DB/cache.
- FastAPI on 8000; Vite dev proxy; Heroku deploy via Procfile.

Cross-Component Communication
- POST /predict {home_team, away_team, season, week} → PredictionResponse (scores, win probabilities, prediction_source, win_classifier_used) [backend/main.py](backend/main.py#L1310-L1525).

Where to Look
- backend/main.py, backend/requirements.txt, frontend/package.json, vite.config.js, Procfile/heroku.yml/vercel.json, docs/report.md, alfred.log.md.

Ambiguities to Confirm
- Which origins beyond localhost/Vercel should be whitelisted?
- Should /history be implemented or stubbed for UI callers?
- When win_model is absent, should sigmoid fallback remain allowed?

Changed since last run
- Fixed constant predictions by sending raw columns into model pipelines [backend/main.py](backend/main.py#L1375-L1495).
- Parsed ALLOWED_ORIGINS properly and added catch-all OPTIONS handler to stop 400 preflights [backend/main.py](backend/main.py#L75-L111, backend/main.py#L563-L571).
- Pending verification: frontend logos render correctly; /history still missing.
