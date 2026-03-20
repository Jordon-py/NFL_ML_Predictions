# Alfred Log

## Tasks
- [ ] Verify no clients depend on `/legacy/*` endpoints after removal.
- [ ] Validate `/schedule/next-week` serves postseason from backend when regular season ends.
- [ ] Confirm dataset builds land in `backend/data/datasets` and MODELS_DIR aligns with training outputs.
- [ ] Validate frontend API calls after restoring fetch body parsing.
- [ ] Run backend smoke checks (`/health`, `/predict`).
- [ ] Run frontend dev check (`npm run dev`).

## Notes
- 2026-01-23: Created root task log for active work.
- 2026-03-20: Added user-scoped prediction persistence to the active FastAPI app and exposed `/teams/logos` for frontend branding metadata.
- 2026-03-20: Restored dashboard-to-history flow by sending `X-User-Id` on predictions and status/history lookups from the signed-in frontend session.
- 2026-03-20: Shipped two UI polish upgrades on the dashboard/card flow: a slate summary hero and an in-card confidence meter.
