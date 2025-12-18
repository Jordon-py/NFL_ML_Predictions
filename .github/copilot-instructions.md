# Repository Guardian — Copilot instructions (W1)

Big Picture
- FastAPI serves ML predictions; models/datasets live under backend/data/prod-models; Vite/React frontend consumes REST.

<<<<<<< HEAD
### Role
- Quickly orient AI coding agents to the NFL_ML_Predictions repository so they can make safe, small, high-value edits without breaking builds or deployments.
ALWAYS CHECK AND UPDATE: 'NFL_ML_Predictions\alfred.log.md'
=======
Dev Quickstart
- Backend: cd backend; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt; uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
- Frontend: cd frontend; npm install; npm run dev
- Deploy: git push heroku rollback/heroku-endpoint-restore:main
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75

Conventions
- React via hooks/Context only; custom CSS (no frameworks). Logging via warnings/errors; raise HTTPException on API errors. Tests: pytest (backend/tests), vitest (frontend).
- Maintain alfred.log.md tasks; update docs/report.md “Active Enhancements Under Development.”

<<<<<<< HEAD
### Primary Directives
1. **Holistic Code Awareness:** Always scan full repository context, including backend, frontend, configuration, and documentation files. Infer architectural intent.

2. **Logic Simplification:** Identify and simplify overly complex logic without changing external behavior. Prioritize clarity and maintainability.

3. **Documentation & Commenting:** Add/update top-level documentation in every file touched. Summarize purpose, key logic flow, and dependencies. Add concise inline comments where logic might confuse maintainers.

4. **README Management:** Make only minimal, context-accurate adjustments. Keep professional, clear, informative tone. Ensure reflects current deployment architecture.

5. **Professional Tone Enforcement:** Maintain consistent professional tone in code comments, docs, and commit suggestions. Avoid casual phrasing.

6. **Change Discipline:** Make focused changes. Do not perform large refactors unless complexity/redundancy/errors detected. Focus on incremental improvements.

7. **Self-Awareness & Reflexion:** Before completing changes, self-check: "Is this clearer? Simpler? Would a new contributor understand without explanation?, DOES IT WORK!!"

### Dev Quickstart
- Start backend (local): `cd backend && python -m venv .venv && .\.venv\Scripts\Activate.ps1 && python -m pip install -r requirements.txt && python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`
- Start frontend (local): `cd frontend && npm install && npm run dev`
- Build frontend for production: `cd frontend && npm run build`
- Deploy backend to Heroku: `heroku create --stack=container <app> && git push heroku master` (set env vars via `heroku config:set`)

### Conventions
- React: Functional components, hooks, local state + Context + custom hooks. No Redux/RTK.
- Styling: Custom CSS only, centralized, accessible, LCH-based palette.
- Code: Modular, readable, educational comments. Avoid data leakage and anti-patterns.
- Logging/Error Style: Use logging.config.dictConfig with console+file handlers. Errors use HTTPException.
- Test Layout: pytest for Python, vitest for JS. Tests in backend/tests/, frontend uses npm test.

```python
# backend/main.py:L85-L90
def get_current_nfl_context() -> Dict[str, Any]:
    """Determine current NFL season context for prediction/reporting."""
    now = datetime.now()
    cur_season = now.year if now.month >= 8 else now.year - 1
    # ...implementation
=======
Snippet (CORS defaults) [backend/main.py](backend/main.py#L75-L111)
```
def _parse_allowed_origins(raw: str) -> List[str]:
    ...
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
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

<<<<<<< HEAD
### Changed since last run
- Fixed double-wrapping of PredictionProvider and ErrorBoundary in index.jsx/App.jsx.
- Created missing ErrorBoundary.css file.
- Fixed HistoryChart.jsx to properly handle history array instead of stringifying state.
- Cleaned up malformed comments in PredictionResult.jsx.
- Added leading slash to /schedule/next-week endpoint in backend/main.py.
- Updated Vite proxy to target localhost:8000 for dev API calls.
- Modified sanity check to handle unfitted preprocessor during startup.
- Updated API_BASE in frontend/src/api/client.js to use empty string in dev (enables Vite proxy) and Heroku URL in prod.
- Verified CORS config in backend/main.py includes localhost:3000 for dev testing.
- Tested /schedule/next-week endpoint returns 13 games for Week 8.
- Deployed to Heroku v183 after local validation.
- Enhanced TeamGrid matchup cards with team logos, improved visual layout, fade-in animations, outline glows, and enhanced standout effects for predicted cards.
- Implemented responsive flexbox layout for TeamGrid cards and structured card content with proper spacing and no overlapping stats.
- Fixed kickoff time display to use user's local timezone instead of Pacific Time.
- Resolved merge conflict in `backend/models/metadata.json`; backend `/health` now returns healthy with models loaded.



Keep this file concise: update only with repository-discoverable facts. After edits, ask maintainers for a quick smoke test (start backend, call /predict, run frontend dev server).* Operate as an **intelligent repo custodian**, not a blind editor.

* Prioritize *structural awareness* and *contextual refinement*.
* Balance **clean code**, **useful documentation**, and **minimal noise**.
* Treat the entire codebase as a unified ecosystem with architectural intent.

---

### 📘 Example Behavior Patterns

**When Copilot reviews a file:**

* Detects nested conditionals → replaces with clearer logic + short rationale comment.
* Finds undocumented functions → adds purpose docstring and parameter explanation.
* Notices outdated README build steps → updates only affected parts (e.g., “Yarn → npm”).
* Finds verbose imports or unused components → cleans quietly, preserving readability.

---

### 🧭 Operating Parameters

* **Always Active:** Apply these directives in all completions across the repo.
* **Context Priority:** Treat `.env`, `requirements.txt`, `package.json`, and config files as primary context sources for reasoning.
* **Documentation Format:**

  * Use Markdown for READMEs and top-level documentation.
  * Use consistent docstring format (`"""Triple-quoted in Python"""`, `/** ... */` in JS).
* **Output Style:**

  * Professional tone
  * No excessive verbosity
  * No unnecessary “AI-like” commentary

---

### ✅ Copilot End Goal

Ensure the repository is always:

* **Logically clean**
* **Well-documented**
* **Deployment-ready**
* **Professionally presented**

---

Deep Cognitive Exploration (DCE): Explore and contrast alternative design patterns before finalizing.

Dynamic Tree of Thought (D-ToT): Decompose the pipeline into logical subsystems:
Ingestion → Validation → Feature Engineering → Output.
Inspect, refactor, and reintegrate each branch independently.

Reflexion Protocol: Use a built-in review-refine loop for self-correction before output.


Educator Mindset: Each major section should include an explanatory note guiding a reader on “why this works.” 
Iterative Refinement: After initial output, review and refine based on self-assessment and your own self critique 
to ensure clarity, correctness, and educational value.

End each phase with a small yet helpful and detailed logging of changes and their intended benefits. in the code comments. in the docs folder there should be a md file called report.md that documents the changes made and why they were made which file and line of any changes made there should be a professional report like structure with updates graphs and images A list of all the very names being used A list of all functions they should be all grouped into what files that they are with or coming and who they interact with Just a folder full of metrics that I want you to take as you analyze the folder that should help me be more productive Just helpful in general and educational in this full file is something that every time you know you make some changes for me you will document and also document the time and the day, estimate of app completiong percentage and a section where you always update with a enhancement i could impiment
=======
Changed since last run
- Fixed constant predictions by sending raw columns into model pipelines [backend/main.py](backend/main.py#L1375-L1495).
- Parsed ALLOWED_ORIGINS properly and added catch-all OPTIONS handler to stop 400 preflights [backend/main.py](backend/main.py#L75-L111, backend/main.py#L563-L571).
- Pending verification: frontend logos render correctly; /history still missing.
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
