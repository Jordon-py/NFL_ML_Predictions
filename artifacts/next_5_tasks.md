# Next 5 Tasks

1. [ ] **Audit Backend for Bloat** - Scan `backend/` for unused imports, dead functions, and stale helper files.
2. [ ] **Fix Dataset Generation Script** - Investigate and fix `ValueError: invalid literal for int() with base 10: 'season'` in `build_csv_datasets_v3.py`.
3. [ ] **Run Smoke Tests** - Execute `py smoke_test_endpoints.py --base-url http://127.0.0.1:8000` against a running server.
4. [ ] **Verify LLM Explanations** - Confirm Ollama integration logic in `backend/main.py` works as expected.
5. [ ] **Refactor Backend Tests** - Add robust unit tests for `inference_row.py` and `prediction_service.py`.
