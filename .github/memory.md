<!--
REPO MEMORY (STRUCTURE LOCKED)
Rule: DO NOT CHANGE THE STRUCTURE OR KEY NAMES.
Additions must be APPENDS or EDITS within existing sections only.
-->

# MEMORY.md — Repository Source of Truth

This file is a **machine-extractable repo memory** used to prevent drift and keep the project production-ready.

**Update rule:** Every meaningful change MUST update this file.

---

## Repo Profile (Stable)

```yaml
repo_profile:
  owner: "Christopher"
  focus: "Production-ready full-stack app with stable API + clean docs"
  priorities:
    - "API always works"
    - "Endpoints stable (no drift)"
    - "No hardcoded mismatches"
    - "Minimal safe diffs"
    - "README always current"
  constraints:
    - "Avoid Redux/RTK"
    - "Avoid CSS frameworks unless explicitly requested"
```

## Changelog

```yaml
changelog:
  - date: "2026-04-09"
    summary: "Removed dead frontend/admin experiment surfaces, added live API+CORS verification script, aligned retrain automation with builddataset.py, and cleaned tracked deployment/runtime artifacts."
    files:
      - "scripts/verify_api_cors.py"
      - "backend/scripts/weekly_retrain.py"
      - "backend/tests/test_weekly_retrain.py"
      - "frontend/package.json"
      - "frontend/package-lock.json"
      - ".github/workflows/ci.yml"
      - ".github/workflows/deploy.yml"
      - "README.md"
      - "docs/ENVIRONMENT.md"
      - "docs/FRONTEND_PREDICTION_FLOW.md"
    api_impact:
      - "Removed stale docs/UI references to /predict/explain, /llm/chat, and /admin/reload from the active app surface"
      - "Added scripts/verify_api_cors.py for current /health, /status/overview, /predict, and production CORS checks"
    verification:
      - "python -m pytest backend/tests -q"
      - "cd frontend && npm test -- --run"
      - "cd frontend && npm run build"
      - "python scripts/verify_api_cors.py --backend-url https://nfl-predict-ecf5a5bd34fe.herokuapp.com --verbose"
    risks:
      - "backend/.env still needs human review and potential secret rotation"
      - "backend/predictions.db remains tracked runtime state and should be handled in a later pass"
  - date: "2026-01-23"
    summary: "Removed legacy /legacy router; centralized team logos schema; history loads from disk; removed VITE_DEV_ENV fallback."
    files:
      - "backend/main.py"
      - "backend/main_helpers.py"
      - "backend/schemas.py"
      - "backend/routes.py"
      - "frontend/src/api/fetch.js"
      - "README.md"
      - "docs/report.md"
      - "alfred.log.md"
      - ".github/memory.md"
    api_impact:
      - "Removed /legacy/* endpoints"
    verification:
      - "Not run (manual)"
    risks:
      - "Any callers using /legacy/* must migrate to root endpoints"
  - date: "2026-01-23"
    summary: "Backend owns postseason schedule; frontend fallback removed; model/dataset defaults consolidated."
    files:
      - "backend/main.py"
      - "backend/main_helpers.py"
      - "backend/config.py"
      - "backend/build_csv_datasets_v3.py"
      - "frontend/src/api/client.js"
      - "frontend/public/post_schedule.json"
      - "README.md"
      - "docs/report.md"
      - "alfred.log.md"
      - ".github/memory.md"
    api_impact:
      - "Schedule responses now rely on backend postseason fallback only"
    verification:
      - "Not run (manual)"
    risks:
      - "If clients depended on frontend postseason JSON, they must use backend schedule"
      - "Non-default model/dataset locations now require explicit MODELS_DIR/DATA_DIR"
  - date: "2026-01-23"
    summary: "Fixed frontend fetch helper by restoring readBody parsing."
    files:
      - "frontend/src/api/fetch.js"
      - "docs/report.md"
      - "alfred.log.md"
      - ".github/memory.md"
    api_impact:
      - "None"
    verification:
      - "Not run (manual)"
    risks:
      - "None"
```
Runtime & Environment (Stable)
runtime:
  backend:
    framework: "FastAPI"
    language: "Python"
    key_files:
      - "backend/main.py"
    env_rules:
      - "No secrets in code"
      - "Prefer env vars for URLs/paths"
  frontend:
    framework: "Vite + React.jsx"
    env_rules:
      - "Use VITE_API_BASE_URL (prod) or Vite proxy (dev)"
  gateway_optional:
    framework: "Node/Express"
    purpose: "LLM/tool-calling gateway (if enabled)"

API Contracts (Stable Keys)
api_contracts:
  required_behaviors:
    - "Endpoints must match documented schemas"
    - "No breaking payload changes without documentation"
    - "Additive changes preferred"
  smoke_endpoints:
    - path: "/status"
      method: "GET"
      purpose: "Health + readiness check"

Config & Hardcoded Audit Map
config_map:
  must_be_env:
    - "API base URLs"
    - "Model directories"
    - "Dataset paths"
    - "3rd party keys"
  must_be_centralized:
    - "Magic strings for endpoint paths"
    - "Team alias/normalization maps"
    - "Schedule/game type filters"

Drift Watchlist (What Commonly Breaks)
drift_watchlist:
  api_drift:
    - "Request/response schema changes"
    - "Renamed fields without versioning"
  data_drift:
    - "Feature schema mismatch (raw vs engineered)"
    - "Missing priors defaulting to zeros"
  infra_drift:
    - "CORS origin mismatch"
    - "Wrong API base URL joining (double slashes / missing scheme)"
  import_drift:
    - "Moved files without updating imports"
    - "Relative import chains breaking in production"

Known Decisions (Lock In)
decisions:
  engineering_style:
    - "Diff-first minimal patches"
    - "Contract-first schemas + examples"
    - "Dry-run before destructive cleanup"
  frontend_state:
    - "No Redux"
    - "Use hooks/context/local state"
  documentation:
    - "README + MEMORY.md always updated"

Verification Log (Last Known Good)
verification_log:
  last_verified_date: "2026-04-09"
  verified_by: "Codex"
  checks_run:
    - "python -m pytest backend/tests -q"
    - "cd frontend && npm test -- --run"
    - "cd frontend && npm run build"
    - "python scripts/verify_api_cors.py --backend-url https://nfl-predict-ecf5a5bd34fe.herokuapp.com --verbose"
  notes: "Live verification passed for production health, status overview, predict contract, and canonical frontend CORS origin."

Open Issues / Tech Debt Queue
tech_debt:
  - id: "TD-001"
    title: "TBD"
    severity: "low|medium|high"
    status: "open"
    notes: ""

Changelog (Append Only)
changelog:
  - date: "YYYY-MM-DD"
    change: "What changed"
    reason: "Why it changed"
    impact:
      endpoints: []
      contracts: []
      files: []
    verification:
      commands: []
      result: "pass|fail|partial"
    risks: []
