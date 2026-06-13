# Review Bucket

Updated: 2026-03-28

This folder contains only high-confidence junk, build output, logs, exported HTML/assets, or obvious scratch files that were moved out of the active tree.

Rule used for this pass:

- Keep the original repo-relative path under `review/` so each item is easy to trace and restore.
- Do not move legacy code, dated model runs, duplicate schedule/logo CSVs, or other uncertain files in this pass.

## Moved Items

| Original path | Review path | Why it was moved |
| --- | --- | --- |
| `backend/uvicorn.out` | `review/backend/uvicorn.out` | Local runtime log. |
| `backend/uvicorn.err` | `review/backend/uvicorn.err` | Local runtime log. |
| `tmp_uvicorn.log` | `review/tmp_uvicorn.log` | Local runtime log. |
| `backend/jup.py` | `review/backend/jup.py` | Scratch utility file with no role in the active backend path. |
| `backend/Untitled-1.ipynb` | `review/backend/Untitled-1.ipynb` | Scratch notebook. |
| `backend/Untitled-1.ps1` | `review/backend/Untitled-1.ps1` | Scratch PowerShell file. |
| `frontend/dist/` | `review/frontend/dist/` | Generated Vite build output. |
| `frontend/public/Script Analysis and Enhancement.html` | `review/frontend/public/Script Analysis and Enhancement.html` | Exported HTML artifact, not app source. |
| `frontend/public/Script Analysis and Enhancement_files/` | `review/frontend/public/Script Analysis and Enhancement_files/` | Exported HTML asset dump, not app source. |
| `frontend/public/index.html` | `review/frontend/public/index.html` | Duplicate HTML shell; the live Vite entrypoint is `frontend/index.html`. |
| `frontend/src/utils/TeamGrid (1).md` | `review/frontend/src/utils/TeamGrid (1).md` | Duplicate scratch note file. |
| `frontend/src/components/data_fetch.log` | `review/frontend/src/components/data_fetch.log` | Local log file, not source code. |

## Not Moved On Purpose

- `backend/2025*/` and `backend/2026*/`: kept in place because model bundle discovery is runtime-sensitive.
- `backend/routes.py`, `backend/services/`, `backend/schemas.py`: likely legacy, but not obvious junk.
- `frontend/src/hooks/`, older UI components, and older API helpers: likely legacy, but not obvious junk.
- Duplicate schedule and logo files: need a deliberate canonicalization pass.
