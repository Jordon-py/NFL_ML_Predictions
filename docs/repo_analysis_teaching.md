# NFL Predictions Repo Analysis

Date: 2026-03-13

Scope:
- Reviewed active code in `backend/`, `frontend/src/`, and the live artifact contract in `backend/models/metadata.json`
- Ignored `archive/`, vendored environments, and `node_modules/`

Verification:
- `frontend/`: `npm run build` passed
- I did not run backend integration tests or model-training jobs

## Major Parts Of The App

1. `backend/main.py`, `backend/config.py`, `backend/schemas.py`
   Backend API, startup lifecycle, CORS, admin endpoints, health/status routes
2. `backend/main_helpers.py`, `backend/services/prediction_service.py`, `backend/services/inference_row.py`
   Model loading, dataset/history access, feature-row assembly, prediction orchestration
3. `backend/build_csv_datasets_v3.py`, `backend/utils/*`
   Dataset construction, feature engineering, export metadata and quality reports
4. `backend/train_models.py`, `backend/models/metadata.json`
   Training flow, artifact export, model metadata contract
5. `frontend/src/App.jsx`, `frontend/src/hooks/usePredictionState.js`, `frontend/src/api/*`
   App shell, routing, shared state, API access
6. `frontend/src/components/*`, `frontend/src/pages/*`, `frontend/src/utils/*`
   Prediction dashboard, cards, history, stats, LLM chat, UI normalization

---

## 1. Backend API And Lifecycle

Purpose:
- This section boots the FastAPI app, loads models/datasets at startup, exposes routes, and owns scheduled rebuild/reload behavior.

Critique:
- `backend/main.py` currently acts as app factory, lifecycle manager, scheduler, route module, admin controller, payload normalizer, and some inference fallback logic all in one file.
- That makes it hard to test route behavior separately from startup behavior, and it raises the risk of state bugs because many concerns mutate the same global `state` dictionary.

### Suggestion 1: Split startup, routers, and shared state into separate modules

- References: `backend/main.py:708`, `backend/main.py:771`, `backend/main.py:825`, `backend/main.py:857`, `backend/main.py:985`, `backend/main.py:1160`
- Problem: one file currently owns scheduler startup, app assembly, route registration, and admin actions.
- Enhancement: move lifecycle code into a `startup.py` or `app_state.py` module, move routes into focused router files like `routers/status.py`, `routers/predict.py`, and keep `main.py` as thin assembly code.
- How to do it:
  1. Create a small state container object that holds `bundle`, `dataset`, `service`, and metadata.
  2. Initialize that container during lifespan startup.
  3. Inject that container into route handlers instead of reading a module-global dict directly.
  4. Keep scheduler startup and shutdown inside the lifespan layer only.
- Syntax explanation:
  - `@app.get("/api/health")` is a decorator. It attaches a Python function to an HTTP route.
  - `@asynccontextmanager` wraps startup code before `yield` and shutdown code after `yield`.
  - When many decorators and route handlers live in one file, the file becomes the app's control center whether you intended that or not.
- Hint tips:
  - If a helper does not need HTTP objects, it should not live in a route file.
  - If a function mutates app state, make that dependency explicit in its parameters.

### Suggestion 2: Replace CLI-only retraining calls with a callable training service

- References: `backend/main.py:1175`, `backend/main.py:1189`, `backend/main.py:1190`, `backend/train_models.py:676`, `backend/train_models.py:695`
- Problem: `admin_retrain()` calls `train_main(data_path=..., out_dir=...)`, but `train_models.main()` is defined as `def main() -> None:` and parses CLI args instead of accepting kwargs.
- Enhancement: expose a library function like `train_from_path(...)` or `train_models.run_training(...)`, and let the CLI wrapper call that function after parsing arguments.
- How to do it:
  1. Keep `argparse` in a thin CLI layer only.
  2. Move the real training body into a callable function with explicit parameters.
  3. Have `/api/admin/retrain` call the callable function or enqueue it as a background job.
  4. Return job status instead of blocking the HTTP request for a long retrain.
- Syntax explanation:
  - `argparse` is meant for command-line entrypoints. `args = parser.parse_args()` reads process arguments, not function arguments.
  - A signature like `def main() -> None:` means the function accepts no named parameters.
  - Calling a CLI-shaped function from app code is brittle because the app and CLI want different interfaces.
- Hint tips:
  - Keep CLI code thin and reusable service code thick.
  - If a function may be called by HTTP, tests, or a script, design the function API first and wrap it with CLI parsing second.

---

## 2. Backend Inference Assembly And Prediction Services

Purpose:
- This section loads artifacts, builds a model input row for a requested matchup, and runs the home/away regressors plus win classifier.

Critique:
- The prediction path is workable, but it mixes row lookup, schedule enrichment, roll-forward logic, one-hot fallback logic, and imputation in the same inference chain.
- It is optimized for getting a prediction out, but not yet structured for maintainability or easy debugging.

### Suggestion 1: Turn row assembly into explicit stages with a named result object

- References: `backend/services/prediction_service.py:119`, `backend/services/prediction_service.py:138`, `backend/services/inference_row.py:377`, `backend/services/inference_row.py:389`, `backend/services/inference_row.py:421`, `backend/services/inference_row.py:457`
- Problem: `build_model_input_row()` returns either a 2-tuple or a 3-tuple depending on `debug`, which makes callers rely on positional unpacking and mode-specific behavior.
- Enhancement: return one named result object every time, for example a dataclass holding `row_df`, `source`, and optional `debug_info`.
- How to do it:
  1. Define a small result type for inference-row assembly.
  2. Break the function into stage helpers: exact-match lookup, schedule enrichment, prior roll-forward, diff recomputation, final alignment.
  3. Collect per-stage metadata in one debug dictionary.
  4. Let the route decide whether to expose the debug part, instead of changing the return shape.
- Syntax explanation:
  - `Tuple[pd.DataFrame, str] | Tuple[pd.DataFrame, str, Dict[str, Any]]` is a union type. It tells readers that the function changes shape depending on runtime mode.
  - A dataclass gives names to fields, so the caller reads `result.source` instead of `payload[1]`.
- Hint tips:
  - When the return shape changes with a flag, that is usually a signal to introduce a named object.
  - Debug data is easier to extend when it is keyed by stage name instead of tuple position.

### Suggestion 2: Pre-index the dataset and separate numeric vs categorical imputation

- References: `backend/services/inference_row.py:100`, `backend/services/inference_row.py:120`, `backend/services/inference_row.py:405`, `backend/services/inference_row.py:440`, `backend/services/inference_row.py:345`, `backend/services/inference_row.py:369`
- Problem: the code repeatedly scans dataset rows and then uses broad `fillna(0)` behavior after `reindex`, which can flatten meaning across numeric and categorical features.
- Enhancement: build fast lookup indexes once and apply dtype-aware fill rules at inference time.
- How to do it:
  1. Build a matchup index keyed by `(season, week, home_team, away_team)` during service initialization.
  2. Build a team-history cache using vectorized groupings rather than row-by-row `iterrows()`.
  3. Keep separate numeric and categorical fill maps from the training metadata.
  4. Only create manual one-hot columns if the artifact contract explicitly says they are needed.
- Syntax explanation:
  - `row_df.reindex(columns=expected_cols)` adds missing columns and orders existing ones to match the expected schema.
  - `fillna(0)` is simple, but on a mixed-schema frame it turns "missing category" and "missing measurement" into the same value.
  - `iterrows()` is easy to read but usually slower and less stable than vectorized grouping or indexed lookups.
- Hint tips:
  - If the same query pattern happens every prediction, precompute it once in `__init__`.
  - Use the training artifact to tell inference how to fill values, instead of inventing inference-time rules ad hoc.

---

## 3. Dataset Builder And Feature Engineering

Purpose:
- This section pulls schedule and stats data, engineers pregame features, exports the modeling CSV, and writes metadata/quality-report side files.

Critique:
- The builder is the heart of the ML system, but it currently carries packaging work, data loading, feature engineering, export, and logging in one very large script.
- It works more like a powerful notebook script than a composable package module.

### Suggestion 1: Remove import-path hacks and standardize package-relative imports

- References: `backend/build_csv_datasets_v3.py:78`, `backend/utils/feature_helpers.py:42`, `backend/main.py:716`, `backend/main.py:728`
- Problem: the app has to inject `PYTHONPATH` when launching the builder because the builder and helper modules use top-level imports like `from utils...` and `from config...`.
- Enhancement: make `backend` a consistent package boundary and use relative imports throughout the backend package.
- How to do it:
  1. Change imports like `from utils.feature_helpers import ...` to relative forms inside the package.
  2. Run the script as a module, for example `python -m backend.build_csv_datasets_v3`.
  3. Remove the `PYTHONPATH` mutation from `build_and_reload_dataset()`.
  4. Add one smoke test that imports the builder from repo root exactly how production will run it.
- Syntax explanation:
  - `from .utils.feature_helpers import ...` means "import from this package hierarchy."
  - `python -m backend.build_csv_datasets_v3` tells Python to resolve imports using package rules instead of treating the file as a loose script.
- Hint tips:
  - If runtime code needs to patch `PYTHONPATH`, the import graph is usually telling you the package boundary is unclear.
  - Pick one execution style for backend scripts and make every script follow it.

### Suggestion 2: Break the builder into named phases and write a structured build manifest

- References: `backend/build_csv_datasets_v3.py:1467`, `backend/build_csv_datasets_v3.py:1877`, `backend/build_csv_datasets_v3.py:2001`, `backend/build_csv_datasets_v3.py:2014`, `backend/build_csv_datasets_v3.py:2030`, `backend/build_csv_datasets_v3.py:2047`
- Problem: `build_dataset()` is responsible for too many steps, and the large logging call near export is trying to dump many unrelated objects at once.
- Enhancement: split the builder into phased functions and write a compact machine-readable manifest per run.
- How to do it:
  1. Separate "load raw sources", "engineer features", "quality checks", and "export artifacts" into distinct functions.
  2. Let each phase return a small summary dictionary.
  3. Write one `build_manifest.json` that includes dataset path, season range, row count, feature count, and quality metrics.
  4. Keep logs human-readable and use JSON files for structured diagnostics.
- Syntax explanation:
  - The `*` in `build_dataset(..., *, encode="onehot", ...)` makes later arguments keyword-only. That is good when a function has many advanced options.
  - `json.dump(...)` is the right tool for structured reports because downstream code can read it without scraping logs.
- Hint tips:
  - Logs should answer "what happened just now?"
  - Manifest files should answer "what was produced and with what settings?"

---

## 4. Training Pipeline And Artifact Contract

Purpose:
- This section trains the regression/classification models, writes model artifacts, and defines the metadata the runtime loader depends on.

Critique:
- The repo currently shows two artifact contracts: the live `backend/models/metadata.json` shape and the shape produced by `backend/train_models.py`.
- That mismatch is the kind of bug that only appears when retraining is attempted, which makes it high-risk for maintenance.

### Suggestion 1: Version the artifact manifest and unify key names across training and loading

- References: `backend/main_helpers.py:68`, `backend/main_helpers.py:83`, `backend/main_helpers.py:106`, `backend/train_models.py:973`, `backend/train_models.py:977`, `backend/train_models.py:983`, `backend/models/metadata.json:4`
- Problem: the live metadata uses top-level keys like `home_model` and `away_model`, while `train_models.py` writes nested `artifacts` keys like `reg_home` and `reg_away`.
- Enhancement: define one manifest schema with a `contract_version` and one canonical set of artifact names.
- How to do it:
  1. Create a manifest model that defines required files and optional files.
  2. Make the trainer write that schema every time.
  3. Make the loader validate that schema before attempting `joblib.load(...)`.
  4. If you must support legacy manifests, add explicit version branches instead of ambiguous fallback guesses.
- Syntax explanation:
  - `meta.get("artifacts", meta)` means "use `meta["artifacts"]` if it exists, otherwise use `meta` itself."
  - That fallback pattern only works safely if both shapes use the same key names.
- Hint tips:
  - Treat model metadata like an API contract, not just a convenience JSON blob.
  - The first field in a long-lived manifest should be a version number.

### Suggestion 2: Add retrain-to-reload compatibility tests

- References: `backend/main_helpers.py:59`, `backend/main_helpers.py:117`, `backend/train_models.py:936`, `backend/train_models.py:1003`, `backend/main.py:1166`, `backend/main.py:1170`
- Problem: there is no obvious automated proof that "train -> write artifacts -> load bundle -> serve prediction" still works as one pipeline.
- Enhancement: add a lightweight compatibility test that trains on a fixture, loads the emitted artifacts, and verifies a prediction request can be assembled.
- How to do it:
  1. Create a tiny fixture dataset with just enough columns to exercise the pipeline.
  2. Run the trainer against that fixture into a temp directory.
  3. Call `load_inference_bundle()` on the produced directory.
  4. Instantiate `PredictionService` and assert one request returns the expected response shape.
- Syntax explanation:
  - The dataclasses in `backend/train_models.py:171-190` are structured records. They are good inputs for report writing and test assertions because `asdict(...)` serializes them predictably.
  - A smoke test is not a full statistical test. It simply proves the interfaces still fit together.
- Hint tips:
  - For ML repos, "artifact compatibility tests" are often more valuable than deep unit tests.
  - Test the contract at the folder level, not just the function level.

---

## 5. Frontend App Shell, Shared State, And API Layer

Purpose:
- This section mounts the React app, defines routes, centralizes prediction state, and talks to the backend.

Critique:
- The frontend shell is functional, but it duplicates responsibility across routing, state, and API fallbacks.
- The biggest issue is drift risk: multiple places try to normalize the same ideas such as schedule shape, history shape, and API base selection.

### Suggestion 1: Replace top-level prop drilling with a provider/store split

- References: `frontend/src/App.jsx:63`, `frontend/src/App.jsx:84`, `frontend/src/App.jsx:93`, `frontend/src/hooks/usePredictionState.js:165`, `frontend/src/hooks/usePredictionState.js:180`, `frontend/src/hooks/usePredictionState.js:305`
- Problem: `App.jsx` destructures a large state object and then redistributes many of those fields and setters back down to route components.
- Enhancement: create a `PredictionStateProvider` and let routes consume only the slices they need via custom hooks.
- How to do it:
  1. Move the hook call into a provider component.
  2. Expose `usePredictionStore()`, `useHealthStatus()`, or similarly narrow hooks.
  3. Keep route components focused on page concerns rather than state plumbing.
  4. Split polling logic from history logic so they can evolve independently.
- Syntax explanation:
  - Object destructuring like `const { schedule, week, predictions, ... } = predictionState` is convenient.
  - Prop drilling happens when those many fields are then passed down again through multiple component layers.
  - A React context provider turns "pass everything through props" into "read what you need where you need it."
- Hint tips:
  - If a route only needs `history`, it should not receive `setPrediction`, `setLoading`, and `seasonContext`.
  - Design hooks around responsibilities, not around one giant state object.

### Suggestion 2: Make the backend the single authority for API base and schedule fallbacks

- References: `frontend/src/api/fetch.js:14`, `frontend/src/api/fetch.js:18`, `frontend/src/api/client.js:136`, `frontend/src/api/client.js:212`, `frontend/src/api/client.js:233`, `README.md:22`, `README.md:52`
- Problem: the frontend ignores the documented dev/base env split and also re-parses a hard-coded `Nfl_schedule_2025.csv` as a fallback schedule source.
- Enhancement: move schedule authority completely to the backend and make API base resolution explicitly honor `VITE_API_DEV` in development.
- How to do it:
  1. Define a clear env precedence order for dev vs production.
  2. Fail fast if the chosen env var is missing.
  3. Remove browser-side CSV schedule parsing and let the backend return offseason/postseason rows.
  4. Keep one contract for `getNextWeekSchedule()` instead of one network path plus one CSV parsing path.
- Syntax explanation:
  - `import.meta.env` is Vite's compile-time environment object.
  - A fallback chain is just ordered selection logic: choose the first valid configuration source and stop there.
  - `Promise.allSettled(...)` is good for partial failure, but it should not hide ownership boundaries.
- Hint tips:
  - Put business rules near the backend data source.
  - Frontend fallbacks should be UI fallbacks, not alternate data pipelines.

---

## 6. Frontend UI Surfaces

Purpose:
- This section renders the dashboard, matchup cards, prediction details, history pages, stats pages, and the LLM chat interface.

Critique:
- The UI is visually rich and it builds successfully, but several components still mix controller work with presentation.
- There are also a few contract issues where component-local state does not reset when the selected game changes.

### Suggestion 1: Keep cards and dashboard views presentational, and remove duplicate DOM ids

- References: `frontend/src/components/DashBoard/Dashboard.jsx:70`, `frontend/src/components/DashBoard/Dashboard.jsx:113`, `frontend/src/components/Card/Card.jsx:262`, `frontend/src/components/Card/Card.jsx:398`, `frontend/src/components/Card/Card.jsx:423`, `frontend/src/components/Card/Card.jsx:447`
- Problem: the dashboard handles async orchestration and offseason generation inline, while `Card.jsx` contains debug logging, localStorage writes, `CustomEvent` dispatching, timers, and repeated static HTML `id` values.
- Enhancement: move action logic into controller hooks and keep cards as pure renderers with callbacks.
- How to do it:
  1. Put prediction-trigger and showcase-trigger logic into dashboard hooks or controller functions.
  2. Make `Card` receive already-prepared props and simple `onPredict` / `onReset` callbacks.
  3. Replace repeated DOM `id` values with `data-*` attributes or no identifier at all.
  4. Keep debug/telemetry behavior behind dev-only hooks instead of embedding it in the visual component.
- Syntax explanation:
  - React `key` is only for React's reconciliation. It does not create a DOM identifier you can safely query later.
  - HTML `id` must be unique in the rendered document.
  - A presentational component should mostly render JSX from props and emit events upward.
- Hint tips:
  - If a card writes to storage or dispatches global events, it is doing controller work.
  - If you need test selectors, prefer `data-testid` or another `data-*` attribute over repeated `id` values.

### Suggestion 2: Reset per-game local state and align history/status DTOs across pages

- References: `frontend/src/components/PredictionResult.jsx:42`, `frontend/src/components/PredictionResult.jsx:66`, `frontend/src/components/LLMChat/LLMChat.jsx:64`, `frontend/src/components/LLMChat/LLMChat.jsx:86`, `frontend/src/pages/StatsPage.jsx:157`, `frontend/src/components/HistoryChart.jsx:92`, `frontend/src/utils/predictionContextUtils.js:12`, `backend/main.py:874`, `backend/schemas.py:83`
- Problem: `PredictionResult` and `LLMChat` keep local state that can outlive the currently selected game, and `StatsPage` reads `safeOverview.history?.metrics` even though the backend exposes flat `history` metrics.
- Enhancement: key or reset local state by `game_id`, and define one shared history/status normalization contract for all pages.
- How to do it:
  1. Watch `entry?.game_id` or `prediction?.game_id` with `useEffect` and clear local explanation/chat state when the game changes.
  2. Normalize history/status payloads once in the API layer instead of per-page.
  3. Make `StatsPage` read the actual backend shape, not a guessed nested shape.
  4. Decide whether `HistoryChart` is a chart or a list and rename or redesign it accordingly.
- Syntax explanation:
  - `useEffect(() => { ... }, [entry?.game_id])` reruns the effect when the selected game changes.
  - Optional chaining like `safeOverview.history?.metrics` avoids crashes, but it can also silently hide a schema mismatch if the path is wrong.
  - DTO means "data transfer object": the agreed shape a layer receives from another layer.
- Hint tips:
  - If a component's state is about one selected record, tie the state lifecycle to that record's identity.
  - Normalize API data once near the API client, not separately in every page.

---

## Highest-Leverage Fix Order

1. Fix the training/retraining contract mismatch between `backend/main.py`, `backend/train_models.py`, and `backend/main_helpers.py`
2. Standardize backend package imports so the scheduler does not need `PYTHONPATH` patching
3. Split `backend/main.py` into startup/state/router modules
4. Remove frontend schedule and API-base duplication
5. Move controller logic out of `Dashboard.jsx` and `Card.jsx`
6. Reset per-game component state in `PredictionResult.jsx` and `LLMChat.jsx`

## Additional Concrete Bugs Worth Fixing Soon

- `backend/main.py:972-980`
  The manual build trigger route checks `ADMIN_ENABLED` but does not block when admin mode is off. The `pass` statement means the background build still runs.
- `frontend/src/api/client.js:136-167`
  The browser fallback expects `frontend/public/Nfl_schedule_2025.csv`, but that file is not present in `frontend/public/`. If the API schedule path fails, this fallback path also fails.
- `frontend/src/pages/StatsPage.jsx:157-169` and `backend/main.py:874-880`
  `StatsPage` reads `safeOverview.history?.metrics`, but the backend returns `history` as a flat object with `total_predictions`, `win_rate`, and `note`.

## Short Teaching Summary

- Backend decorators such as `@app.get(...)` register route functions; they are easier to reason about when the file only contains routing concerns.
- `@asynccontextmanager` is the right syntax for startup/shutdown, but it becomes hard to manage if it also owns scheduler policy and artifact refresh logic.
- `argparse` is for CLI boundaries, not for internal application APIs.
- React custom hooks are best when they each own one concern, not when one hook becomes the app store, polling engine, and persistence layer at the same time.
- Optional chaining and defensive fallbacks prevent crashes, but they should not become substitutes for stable contracts between backend and frontend.
