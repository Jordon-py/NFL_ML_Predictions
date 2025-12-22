# Deployment Status — 2025-12-04 18:15 UTC

## Frontend (Vercel)

**Status:** ⚠️ Blocked by Git author permissions

**Issue:**

- Vercel CLI reports: "Git author <codex@example.com> must have access to the team Christopher Jordon's projects"
- Large backup file (721MB `backup-pre-clean-2025-12-02.bundle`) was blocking deployment from repo root

**Resolution Steps Taken:**

1. Updated `.vercelignore` to exclude:
   - `*.bundle`, `*.tar`, `*.gz` (large archives)
   - `backup*/` directories
   - `.git/` folder
   - `venv/`, `.venv/` Python virtualenvs
   - `docs/` (not needed in production)

2. Attempted deployment from `frontend/` subdirectory to bypass root-level large files

**Next Steps:**

- Either:
  - A) Update git author email: `git config user.email "c.jordon@icloud.com"`
  - B) Deploy via Vercel dashboard by connecting the GitHub repo (bypasses CLI auth)
  - C) Add `codex@example.com` to Vercel team collaborators

**Alternative (Manual):**

```powershell
# From frontend directory
npm run build
# Then drag dist/ folder to Vercel dashboard or use GitHub integration
```

---

## Backend (Heroku)

**Status:** 🔴 Failing (Application Error page)

**Last Known Good Commit:** `380a0d8d4` (heroku/master)

**Current Branch:** `rollback/heroku-endpoint-restore` (reset to 380a0d8d4, clean)

**Rollback Prepared:**

- Stashed all WIP changes under:
  - `pre-rollback-wip` (tracked changes)
  - `pre-rollback-untracked` (new files)
- Branch now matches last working Heroku release

**To Complete Rollback:**

```powershell
# Force-push last good commit to Heroku
git push heroku rollback/heroku-endpoint-restore:master --force

# Or use Heroku CLI rollback
heroku releases -a nfl-predict-ecf5a5bd34fe
heroku rollback vXXX -a nfl-predict-ecf5a5bd34fe
```

**Verification After Deploy:**

```powershell
curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health
curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/schedule/next-week
```

---

## npm Vulnerabilities

**Status:** ⚠️ 52 vulnerabilities detected

**Breakdown:**

- 35 critical
- 7 high
- 10 moderate

**Recommendation:**

```powershell
cd frontend
npm audit fix          # Apply safe fixes
npm audit fix --force  # If safe fixes insufficient (may introduce breaking changes)
npm audit              # Review remaining issues
```

**Note:** Many vulnerabilities in dev dependencies (e.g., Vite, esbuild) don't affect production bundle security. Focus on runtime dependencies if manual review needed.

---

## Model Artifacts

**Current Promoted Run:** 2025-12-01 16:33 UTC

- **Location:** `backend/models/` (380a0d8d4 commit)
- **Metrics:** Home MAE 4.45 / RMSE 5.85 • Away MAE 4.36 / RMSE 5.57 • Win Brier 0.123 / LogLoss 0.388 / Acc 0.825
- **Config:** GradientBoostingRegressor (scores) + CalibratedClassifierCV (win prob), 136 features, random_state 4211

**Ledger:** See `docs/training_runs.md` for full history (20251117, 20251123, 20251201)

---

## Repository Health

**Completion Estimate:** 87%

**Pending:**

1. Restore backend uptime (Heroku rollback)
2. Resolve Vercel auth/deploy
3. Address npm vulnerabilities
4. Sync master branch with working state after rollback validation
