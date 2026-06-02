# Next 5 Tasks

1. [ ] **Deployed Browser Smoke** - Re-run dashboard, card breakdown, and Premium AI Coach checks against the deployed Vercel URL after release.
2. [ ] **Heroku Ollama Runtime Config** - Confirm `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, optional `OLLAMA_API_KEY`, and `OLLAMA_TIMEOUT_S` are set for production.
3. [ ] **Premium Response Cache** - Add a backend TTL cache for repeated `premium_explain` matchup requests.
4. [ ] **Pure Prediction Helper** - Refactor `/predict` computation into a shared helper so Premium endpoints can reuse model output without route-level side effects.
5. [ ] **Offline Template Fallback** - Return a concise static model-summary template if all Ollama models are unavailable.
