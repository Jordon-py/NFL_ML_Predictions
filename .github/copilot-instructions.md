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

Snippet (CORS defaults) [backend/main.py](../backend/main.py#L75-L111)
```
def _parse_allowed_origins(raw: str) -> List[str]:
    ...
```
<!--
COPILOT INSTRUCTIONS (IMMUTABLE)
Owner: Christopher
Rule: DO NOT MODIFY THIS FILE. DO NOT REWRITE IT. DO NOT “IMPROVE” IT.
If changes are needed, request explicit approval from Christopher FIRST.
-->

# Copilot Collaboration Instructions (Production-First)

## 0) Prime Directive (Non-Negotiable)
Your #1 job is to keep this repository **production-ready** at all times.

**You must:**
- keep the API stable and working
- prevent endpoint drift
- prevent hardcoded-value mismatches
- keep imports, modules, and routing correct
- simplify code only when it improves clarity without reducing quality
- keep top-level docs accurate (**README + MEMORY.md**) after every meaningful change

**You must NOT:**
- change this file
- delete guardrails, validations, or error handling “to make it work”
- introduce breaking changes without documenting them and updating MEMORY.md
- add Redux/Redux Toolkit (explicitly avoided)
- add CSS frameworks (Tailwind/MUI/Bootstrap/etc.) unless Christopher requests it

---

## 1) Working Style (How to Collaborate With Christopher)
Christopher prefers:
- **diff-first** changes (minimal, safe patches)
- **contract-first** development (schemas + examples before implementation)
- **production realism** (correct env vars, clean imports, correct routing, CORS sanity)
- **clarity + maintainability** over cleverness
- **step-by-step** reasoning for non-trivial logic
- **dry-run before destructive cleanup** (branches/deps/files)

### Required Response Format (Every Time)
When proposing work, output in this exact structure:

1) **Intent**: one sentence describing the goal  
2) **Risk Level**: Low / Medium / High  
3) **Patch**: minimal diffs + file list  
4) **API Impact**: endpoints affected + contract changes  
5) **Smoke Tests**: exact commands to verify  
6) **Docs Update**: what to update in README + MEMORY.md  
7) **Follow-ups**: 1–3 improvements (optional)

---

## 2) Production Readiness Checklist (Run Every Time)
Before you consider any change “done,” verify:

### ✅ API Health (FastAPI or Backend)
- All endpoints boot without errors (import paths correct)
- No missing environment variables in production
- CORS configured appropriately (explicit origins in prod)
- Pydantic validation passes (NaN handling, Optional fields when needed)
- Response payloads match documented schemas
- Errors return consistent structured messages (not stack traces)

### ✅ Drift Prevention (Critical)
- If any endpoint accepts JSON payloads → schema must be explicit and stable  
- If any model/features contract exists → schema validation must match raw feature columns  
- If any schedule or dataset pipeline exists → ensure postseason/game types are supported as intended  
- If any “fallback inference” exists → confirm it does not silently regress predictions

### ✅ Imports & File Mapping
- No broken relative imports
- Avoid ambiguous module names (`utils.py` conflicts, missing `__init__.py`)
- Prefer stable package imports (`backend.utils...`) over fragile relative chains
- If moving files: update every import + update docs + update MEMORY.md

### ✅ Hardcoded Value Audit
Hardcoded values must be:
- centralized (constants/config)
- documented (MEMORY.md)
- mapped correctly (env, URL base, file paths, endpoints)

Absolutely flag and fix:
- hardcoded API base URLs
- hardcoded model/data paths
- magic strings for endpoint routing
- hidden defaults that break deployment

---

## 3) API Contract Rules (No Surprises)
If you touch any endpoint:
- define request/response schema in docs
- add at least one example payload
- add a smoke test command (curl/httpie) and include expected output shape
- ensure consistent status codes

### Stability Rules
- never rename fields silently
- never change meaning of a field without versioning/documenting
- prefer additive changes (new optional fields) over breaking changes

---

## 4) Simplification Rules (Quality Without Regression)
You SHOULD simplify when:
- repeated logic can be centralized safely
- complexity is accidental, not essential
- you reduce cognitive load and improve readability

You MUST NOT simplify by:
- removing validation
- removing logging/observability
- removing error handling
- weakening schema guarantees
- merging layers that reduce clarity (e.g., mixing routing + business logic + IO)

---

## 5) Documentation Rules (Always Current)
After meaningful work, you must update:
- **README.md** (how to run, env vars, endpoints, key flows)
- **MEMORY.md** (current repo truth, contracts, decisions)

Docs must include:
- how to run backend + frontend
- base URL + env var rules
- endpoint list + schema examples
- deployment notes (CORS, env vars, build commands)

---

## 6) Security Standards (Strict)
- Never hardcode credentials/API keys
- Validate all user inputs
- Use safe parsing for JSON and file paths
- Do not log secrets
- If an endpoint accepts user-controlled filenames/paths → sanitize

---

## 7) Testing & Smoke Checks (Minimum Standard)
Required minimum checks after backend changes:
- start server locally
- hit `/status` or equivalent health endpoint
- hit at least 1 core endpoint end-to-end

Required minimum checks after frontend changes:
- page loads
- API calls do not fail due to wrong base URL joining
- error states render cleanly

---

## 8) Memory Protocol (MANDATORY)
You must treat `MEMORY.md` as the repo’s living source of truth.

### Every time you make a meaningful change, you MUST:
- update MEMORY.md with:
  - what changed
  - why it changed
  - what contracts/endpoints were affected
  - what was verified (smoke tests)
  - any new risks/tech debt introduced

### Structure Rules
- Do NOT rewrite the MEMORY.md schema
- Append changes under the **Changelog** section
- Keep keys stable and machine-extractable (YAML blocks)

Pointer:
➡ Always update `MEMORY.md` after implementing changes.

---

## 9) Code Review Standards (Baseline)
### Code Quality Essentials
- Functions should be focused and appropriately sized
- Use clear, descriptive naming conventions
- Ensure proper error handling throughout

### Documentation Expectations
- All public functions must include doc comments
- Complex algorithms should have explanatory comments
- README files must be kept up to date

---

## 10) Definition of Done (Strict)
A change is DONE only when:
- API works
- contracts are stable
- smoke tests pass
- docs updated (README + MEMORY.md)
- drift risks are acknowledged or eliminated

Services & Integrations
- Models and metadata in backend/data/prod-models/models; dataset default backend/data/prod-models/game_features_20251210.csv; env overrides MODELS_DIR/DATASET_PATH.
- CORS allow list ALLOWED_ORIGINS + regex r"https://.*\.vercel\.app"; catch-all OPTIONS avoids 400 preflights [backend/main.py](../backend/main.py#L563-L571). No DB/cache.
- FastAPI on 8000; Vite dev proxy; Heroku deploy via Procfile.

Cross-Component Communication
- POST /predict {home_team, away_team, season, week} → PredictionResponse (scores, win probabilities, prediction_source, win_classifier_used) [backend/main.py](../backend/main.py#L1310-L1525).

Where to Look
- backend/main.py, backend/requirements.txt, frontend/package.json, vite.config.js, Procfile/heroku.yml/vercel.json, docs/report.md, alfred.log.md.

Ambiguities to Confirm
- Which origins beyond localhost/Vercel should be whitelisted?
- Should /history be implemented or stubbed for UI callers?
- When win_model is absent, should sigmoid fallback remain allowed?

Changed since last run
- Fixed constant predictions by sending raw columns into model pipelines [backend/main.py](../backend/main.py#L1375-L1495).
- Parsed ALLOWED_ORIGINS properly and added catch-all OPTIONS handler to stop 400 preflights [backend/main.py](../backend/main.py#L75-L111, ../backend/main.py#L563-L571).
- Pending verification: frontend logos render correctly; /history still missing.
