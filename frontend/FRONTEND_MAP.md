# Frontend Map

Updated: 2026-03-28

## Status Rubric

- `done`: Active in the current frontend path and supported by current routing/imports.
- `needs work`: Wired or user-visible, but incomplete, weakly covered, duplicated, or carrying obvious cleanup debt.
- `review only`: Not moved in this pass because it may still matter, but it is not part of the live frontend route tree or it needs manual review.
- `moved to review`: High-confidence junk or scratch files that were relocated into `review/`.

## Live Frontend Path

- `frontend/src/index.jsx`: Real React entrypoint and global style loader.
- `frontend/src/App.jsx`: Live route tree.
- `frontend/src/api/client.js`: Live API client used by the current route tree.

Current routes:

- `/` -> `frontend/src/components/DashBoard/Dashboard.jsx`
- `/history` -> `frontend/src/components/HistoryPage.jsx`
- `/stats` -> `frontend/src/pages/StatsPage.jsx`
- `/settings` -> inline placeholder in `frontend/src/App.jsx`

## Feature Map

| Status | Feature | Key files | Notes |
| --- | --- | --- | --- |
| `done` | App bootstrap and route shell | `frontend/src/index.jsx`, `frontend/src/App.jsx`, `frontend/src/components/ErrorBoundary.jsx` | The current app mounts through `index.jsx`, wraps the tree in `ErrorBoundary`, and lazy-loads the main routes from `App.jsx`. |
| `done` | Navigation and health indicator | `frontend/src/components/NavBar/NavBar.jsx`, `frontend/src/components/Hamburger/HamburgerMenu.jsx` | The navbar is active, route-aware, and shows backend health status. |
| `done` | Dashboard schedule and prediction flow | `frontend/src/components/DashBoard/Dashboard.jsx`, `frontend/src/components/Card/TeamGrid.jsx`, `frontend/src/components/Card/Card.jsx`, `frontend/src/utils/gameUtils.js`, `frontend/src/api/client.js` | Supports loading next-week games, predicting a single game, predicting all games, displaying win probabilities/scores, and resetting cards. |
| `done` | History page | `frontend/src/components/HistoryPage.jsx`, `frontend/src/components/HistoryChart.jsx`, `frontend/src/api/client.js` | Loads `/history` and renders the history feed directly at `/history`. |
| `done` | Status page | `frontend/src/pages/StatsPage.jsx`, `frontend/src/components/HistoryChart.jsx`, `frontend/src/api/client.js` | Loads `/schedule/next-week`, `/history`, and `/status/overview` and displays KPI cards plus schedule/history summaries. |
| `done` | Live API wrapper | `frontend/src/api/client.js` | This is the only client imported by the active route tree. |
| `needs work` | Settings route | `frontend/src/App.jsx` | The current `/settings` page is an inline placeholder marked “coming soon.” |
| `needs work` | History chart cleanup | `frontend/src/components/HistoryChart.jsx` | The component currently renders duplicated summary text and duplicated ordered lists, which looks like unfinished cleanup. |
| `needs work` | Frontend tests | `frontend/package.json` | Vitest is configured, but no active frontend test files were found. |

## Review Only

### Older Unwired UI Layer

- `frontend/src/pages/LandingPage.jsx`
- `frontend/src/components/AdminControls.jsx`
- `frontend/src/components/LLMChat/LLMChat.jsx`
- `frontend/src/components/PredictionResult.jsx`
- `frontend/src/components/LoadingState.jsx`
- `frontend/src/components/ErrorDisplay.jsx`
- `frontend/src/components/D_BUTTON.jsx`
- `frontend/src/components/Button/D_Button.jsx`

These files are present but not routed from the live `App.jsx` tree.

### Older State and API Helpers

- `frontend/src/hooks/usePredictionState.js`
- `frontend/src/hooks/predictionSelectors.js`
- `frontend/src/hooks/useAuthSession.js`
- `frontend/src/utils/predictionContextUtils.js`
- `frontend/src/utils/predictionHelpers.js`
- `frontend/src/utils/dataFetcher.js`
- `frontend/src/api/fetch.js`
- `frontend/src/api/debugLog.js`
- `frontend/src/api/nfl.js`

Several of these files depend on `client.js` exports that do not exist in the live app, which strongly suggests they belong to an older, unfinished architecture.

### Likely Dead CSS or Duplicated Styling Paths

- `frontend/src/components/DashBoard/Dashboard.css`
- `frontend/src/components/DashBoard/Dashboard.module.css`
- `frontend/src/components/Card/Card.module.css`
- `frontend/src/components/Hamburger/HamburgerMenu.module.css`
- `frontend/src/pages/StatsPage.css`

These files were not found in the active import path that starts at `frontend/src/index.jsx`.

### Public Assets Left In Place For Manual Review

- `frontend/public/Hamburger.jpg`
- `frontend/public/ham.png`
- `frontend/public/nfl_ham.png`
- `frontend/public/nfl-ham-2.png`
- `frontend/public/myteamdescriptions.csv`

These were not moved in this conservative pass. They may be unused or older variants, but they need a deliberate asset-audit pass rather than an automatic move.

## Moved To Review

The following high-confidence junk or exported artifacts were moved into `review/`:

- `frontend/dist/` -> `review/frontend/dist/`
- `frontend/public/Script Analysis and Enhancement.html` -> `review/frontend/public/Script Analysis and Enhancement.html`
- `frontend/public/Script Analysis and Enhancement_files/` -> `review/frontend/public/Script Analysis and Enhancement_files/`
- `frontend/public/index.html` -> `review/frontend/public/index.html`
- `frontend/src/utils/TeamGrid (1).md` -> `review/frontend/src/utils/TeamGrid (1).md`
- `frontend/src/components/data_fetch.log` -> `review/frontend/src/components/data_fetch.log`

## Gaps Worth Addressing Next

- Replace the inline settings placeholder with a real screen or remove the route until it exists.
- Clean up `HistoryChart.jsx` so it renders one summary block and one list.
- Decide whether to delete or revive the older hook/client/component architecture.
- Add frontend tests around the live route tree and `client.js`.
