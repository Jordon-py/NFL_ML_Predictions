# Next 5 Tasks

1. [ ] **Regenerate dataset with EPA metrics** - Run `python backend/build_csv_datasetsv3.py --start 2018 --end 2025 --include-epa` to fix feature mismatch. <!-- id: 10 -->
2. [ ] **Implement missing backend endpoints** - Add `/debug` and `/predict/next-week` to `main.py` for parity with frontend client. <!-- id: 11 -->
3. [ ] **Add robust logging** - Integrate structured logging in `main.py` to capture prediction failures and model performance in real-time. <!-- id: 12 -->
4. [ ] **Create smoke tests** - Develop a comprehensive `smoke_test.py` to validate all API endpoints post-regeneration. <!-- id: 13 -->
5. [ ] **Enhance frontend Dashboard** - Integrate health/metrics polling and prediction queue status in the UI. <!-- id: 14 -->
