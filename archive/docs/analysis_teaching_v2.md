# Analysis & Teaching — NFL_ML_Predictions (v2)

This file summarizes the key design and teachable issues found in the repo. It complements `docs/report.md` by focusing on how to _reason_ about the code and what to test next.

## Highlights

- Schedule selection problem fixed: use kickoff timestamps first, calendar fallback next.
- Prediction calls were moved from Context to Dashboard for separation of concerns.
- CORS and Procfile fixes for Heroku deployment included — `/debug` helps CI verify allowed origins.
- Model loading logic allowed multiple candidate filenames and logs chosen artifacts.

## Teaching snippets

- Current week heuristics: prefer kickoff timestamps because the dataset may contain many historical rows.
- Use Pydantic to validate and coerce requests on the backend; use `api/client.js` to validate responses in the frontend.
- When training models with sklearn, ensure the preprocessor exposes `feature_names_in_` to help inference with ColumnTransformer.

## Suggested next steps

- Add E2E Playwright tests for the dashboard -> predict -> history flow.
- Publish a GitHub Action that: runs `pytest` on the backend, builds the frontend, then calls `/debug` and asserts `restrict_cors` is true and `ALLOWED_ORIGINS` includes the production URL.

---

Updated: 2025-11-17 (UTC)
