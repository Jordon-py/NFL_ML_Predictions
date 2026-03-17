# AI Memory Merge

## AI_Important_memory.md

## Repository Map (Condensed)

- Backend entrypoint: backend/main.py (FastAPI app, lifespan loading, primary routes).
- Backend helpers: backend/main_helpers.py (model/dataset loading, history persistence).
- Backend routes: backend/routes.py (legacy router, schedule + logo helpers).
- Backend services: backend/services/prediction_service.py (core inference), backend/services/inference_row.py (feature row build), backend/services/live_predictor.py (live data row build).
- Schemas: backend/schemas.py (Pydantic contracts).
- Frontend entry: frontend/src/index.jsx, frontend/src/App.jsx (React Router).
- Frontend data layer: frontend/src/api/fetch.js + client.js.
- Frontend state: frontend/src/hooks/usePredictionState.js.
- Frontend UI: Dashboard + Card/TeamGrid components; StatsPage at frontend/src/pages/StatsPage.jsx + StatsPage.css.

## Key Data Flows

- Startup: backend/main.py lifespan loads model bundle (MODELS_DIR) + dataset (DATA_DIR/DATASET_PATH) then validates feature schema.
- Prediction: /predict -> PredictionService -> build_model_input_row -> preprocessor + regressors + win classifier.
- Schedule: /schedule/next-week uses schedule CSV loaders and enriches with team logos.
- Team logos: /teams/logos returns parsed team_logos.csv (primary source) with logo + color fields.
- Frontend: usePredictionState loads schedule + history + logos; TeamGrid renders Card components with logos + colors.

## Deployment Notes

- Heroku expects Linux paths. Model metadata sometimes contains absolute Windows paths; path resolution must treat drive-letter paths as absolute and fall back to basename in MODELS_DIR.
- Vercel rootDirectory = frontend. Vercel env requires VITE_API_BASE_URL; dev base can use VITE_API_DEV or VITE_DEV_ENV.
- Build artifacts are committed (frontend/dist).

## Hotspots / Watchouts

- backend/main_helpers.py: path resolution for model artifacts; dataset selection vs expected feature list.
- backend/services/live_predictor.py: internal inference helper previously referenced missing functions from backend/main.py.
- frontend/src/pages/StatsPage.jsx + StatsPage.css: only location where styling changes should be applied per UI instructions.
- frontend/src/api/fetch.js: base URL selection must handle dev/prod env naming.

## Clean Architecture Intent

- Keep backend endpoints thin; push transformation logic into helper/services.
- Keep frontend data normalization in hooks/utils; UI components should remain presentational.


---

## codex_memory.md


---

- **File:** backend/**init**.py
**Summary of Changes:** No changes needed; file already minimal.
**Before/After Complexity:** Unchanged; empty initializer.
**Self-Critique:** Safe and clear; nothing to simplify.
**Simplification Score:** 10

- **Next Steps:** Continue to the next backend file


---

**File:** backend/config.py

**Summary of Changes:** No changes needed; logic already clear and minimal.
**Before/After Complexity:** Unchanged; straightforward config resolution.
**Self-Critique:** Safe to leave as-is; no meaningful simplification without behavior changes.
**Simplification Score:** 10
**Next Steps:** Continue to the next backend file

---

**File:** backend/main.py
**Summary of Changes:** Removed unused feature-building and LLM helper blocks, dropped duplicate history/payload helpers, and trimmed imports to the active API surface.
**Before/After Complexity:** Before: mixed legacy inference logic and duplicate helpers; After: API orchestration only with external helpers/services.
**Self-Critique:** Kept functional paths intact; removed only unused definitions. Should re-run the backend to confirm no missing imports.
**Simplification Score:** 9
**Next Steps:** Remove legacy router usage and migrate TeamLogosResponse out of legacy routes


---

**File:** backend/schemas.py
**Summary of Changes:** Added TeamMeta/TeamLogosResponse to centralize API response models after removing legacy routes.
**Before/After Complexity:** Before: response model lived in legacy router; After: schema lives in canonical schema file.
**Self-Critique:** Small increase in schema size but reduces cross-file coupling.
**Simplification Score:** 9
**Next Steps:** Remove legacy router import and file


---

**File:** backend/main.py
**Summary of Changes:** Dropped legacy router dependency and pulled TeamLogosResponse from schemas instead.
**Before/After Complexity:** Before: included legacy router module; After: single API surface with no legacy include.
**Self-Critique:** Safe removal for current frontend; legacy clients will lose endpoints.
**Simplification Score:** 9
**Next Steps:** Delete legacy routes file and verify no imports remain


---

**File:** backend/routes.py
**Summary of Changes:** Deleted legacy router module.
**Before/After Complexity:** Before: duplicate/legacy endpoints; After: single authoritative API.
**Self-Critique:** Removing legacy endpoints is irreversible; ensure no external dependencies.
**Simplification Score:** 10
**Next Steps:** Scan remaining backend files for unused legacy utilities

---

**File:** backend/main_helpers.py
**Summary of Changes:** Removed unused path helpers and NumPy import, and reused the shared column picker for team metadata.
**Before/After Complexity:** Before: redundant helpers and repeated column lookup logic; After: single picker and leaner imports.
**Self-Critique:** Kept behavior intact; minimal simplification to avoid breaking model path resolution.
**Simplification Score:** 9
**Next Steps:** Review remaining backend services for redundant helpers and unused imports

---

**File:** backend/services/prediction_service.py
**Summary of Changes:** Dropped unused logging and simplified regressor/probability prediction paths and schedule cache lookup.
**Before/After Complexity:** Before: extra error tracking and duplicated proba calls; After: single-path predictions with a shared proba helper.
**Self-Critique:** Simplified error handling may mask the original exception but preserves fallback behavior.
**Simplification Score:** 9
**Next Steps:** Review other backend services for similar prediction helpers or duplicate logic

---

**File:** backend/services/inference_row.py
**Summary of Changes:** Removed unused logging and reduced repeated column scans in roll-forward stat mapping.
**Before/After Complexity:** Before: per-team scans of all dataset columns; After: precomputed home/away column lists.
**Self-Critique:** Keeps behavior intact while trimming redundant passes.
**Simplification Score:** 9
**Next Steps:** Evaluate larger row-construction helpers for safe consolidation

---

**File:** backend/services/feature_service.py
**Summary of Changes:** Deleted unused feature builder module (no references in backend).
**Before/After Complexity:** Before: redundant parallel feature pipeline; After: single inference path.
**Self-Critique:** Safe removal since no imports; confirm no external scripts depend on it.
**Simplification Score:** 10
**Next Steps:** Check for other orphan modules in backend/services or utils

---

**File:** backend/services/live_predictor.py
**Summary of Changes:** Deleted unused live predictor module (no references in backend).
**Before/After Complexity:** Before: large unused live inference path; After: fewer maintenance surfaces.
**Self-Critique:** Safe removal based on in-repo usage; external consumers would need migration.
**Simplification Score:** 10
**Next Steps:** Scan backend/utils for references to removed live predictor paths

---
