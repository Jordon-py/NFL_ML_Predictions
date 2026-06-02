# Key Architectural Insights

_Last Updated:_ 2026-06-02 05:45:00

## Current Focus
- **Premium AI integration**: `POST /api/premium/explain` and `POST /api/premium/chat` are served by `backend/main.py` and use `NFLAgent` in `backend/ollama/llm_ollama.py`.
- **Prediction-history safety**: Premium endpoints reuse prediction context without writing duplicate prediction-history records.
- **Conversational coach UI**: `Dashboard.jsx` exposes the floating Premium AI Coach panel with deterministic error states and responsive mobile bounds.
- **Card-level analyst reports**: `Card.jsx` exposes collapsible Premium AI breakdowns after a prediction is available.
- **Scheduled retraining**: `.github/workflows/scheduled-retrain.yml` runs the weekly retrain path and uploads artifacts for manual review.
- **Runtime drift guard**: CI verifies that the installed scikit-learn version matches `backend/models/metadata.json`.
- **CORS config**: Backend uses `ALLOWED_ORIGINS` and `ALLOW_ORIGIN_REGEX`; local production-preview smoke tests may need backend calls stubbed unless that local origin is allowed.

## Key Documentation

- [Last 5 Tasks](last_5_tasks.md)
- [Next 5 Tasks](next_5_tasks.md)
- [Dataflow Map](../dataflow.md)
