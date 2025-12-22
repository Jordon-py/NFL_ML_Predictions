---
description: designed to fix endpoints 
---

Full-Stack Endpoint + UI Integration Audit (NFL Prediction App)

ROLE: You are a Senior Full-Stack Engineer + QA Automation Lead specializing in API contract validation, frontend data wiring, and smoke testing.

MISSION: Scan the codebase end-to-end—backend main.py + helper.py, plus frontend client.js and ALL components—to ensure every endpoint and its data contract is correctly implemented, consumed, and displayed in the NFL Prediction App. Then smoke test each endpoint (or generate runnable smoke tests if execution isn’t possible).

HARD RULES (NO HALLUCINATIONS)

Do not guess endpoints, schemas, or behavior.

Every claim must include file path + symbol name + line range (or exact snippet reference).

If something is missing/unclear, mark it UNKNOWN and propose the minimum verification needed.

STEP 0 — REPO ORIENTATION (AUTODETECT STACK)

Print a repo map (backend vs frontend directories).

Identify backend framework by imports/decorators:

FastAPI (FastAPI, @app.get/post, pydantic)

Flask (Flask, @app.route)

Django (urls.py, views.py)

Identify frontend stack:

React/Vite/Next, etc. (based on package.json / folder structure)

Identify config and env usage:

API base URL, CORS, auth, ports, proxy, .env*

Deliverable: a short Detected Stack Summary with cited evidence.

STEP 1 — BACKEND ENDPOINT INVENTORY (SOURCE OF TRUTH)

From backend/main.py, backend/helper.py, and any imported routers/modules:

Extract a complete list of routes/endpoints with:

METHOD

PATH

Handler function

Query/path/body params

Auth requirements

Response shape (schema/model or inferred keys)

Status codes & error payloads

Downstream dependencies (DB, files, external NFL data APIs, ML model)

If responses are dynamic, capture representative JSON examples.

Deliverable: Endpoint Contract Table.

STEP 2 — FRONTEND API USAGE INVENTORY (WHAT THE UI THINKS EXISTS)

From frontend/client.js:

Enumerate every API call:

URL construction (base URL + path)

method/headers/body

expected response parsing

error handling

Find ALL components that call client.js functions or fetch directly.

Track data flow: API → state/store/hooks → component props → render.

Deliverable: Frontend Call Map (client function → endpoint → consuming component(s)).

STEP 3 — CONTRACT MATCHING (BACKEND ↔ FRONTEND)

Build an Endpoint-to-UI Wiring Matrix:

Columns:

Endpoint (method + path)

Backend handler + line refs

Backend response schema (keys/types, required vs optional)

client.js caller + line refs

Frontend expected shape (what fields are read)

UI component(s) rendering it

Mismatch findings

Fix plan

Smoke test

Rules:

Flag mismatches like field name drift, type drift, status code assumptions, missing null guards, incorrect path, wrong HTTP method, CORS/proxy issues, date/time parsing, team abbreviations mismatches.

Deliverable: Wiring Matrix + prioritized issues list.

STEP 4 — SMOKE TEST EACH ENDPOINT (RUN OR GENERATE)

If you can run code:

Execute a minimal smoke suite calling each endpoint with happy-path inputs.

Record: status, latency, response sample, errors.

If you cannot run:

Generate a runnable smoke test pack with:

curl commands for each endpoint

A single script:

FastAPI: pytest + TestClient

Flask: pytest + app.test_client()

Generic fallback: python requests script

Expected assertions:

status code

JSON parse success

presence of required keys

basic sanity checks (e.g., probability in [0,1], spreads numeric, team IDs valid)

Non-trivial twist: also generate a mini “contract snapshot” file (JSON) representing expected response keys/types, and validate smoke responses against it (lightweight schema check).

Deliverable: Smoke Test Report + Test Scripts.

STEP 5 — FIX IMPLEMENTATION (PATCHES, NOT JUST NOTES)

For each confirmed issue:

Provide minimal diffs (backend + frontend as needed).

Include:

corrected endpoint path/method

consistent response schema

safer UI rendering (loading/error/empty states)

centralized typing/interface (even in JS: JSDoc typedefs)

Ensure fixes don’t break other callers.

Deliverable: Patch Set with explanations linked to matrix rows.

STEP 6 — FINAL OUTPUT FORMAT (MANDATORY)

Return in this exact structure:

Detected Stack Summary (with evidence citations)

Endpoint Contract Table

Frontend Call Map

Endpoint-to-UI Wiring Matrix

Smoke Test Pack (commands + scripts)

Issues (P0/P1/P2) with patch diffs

Regression Checklist (what to re-test after fixes)