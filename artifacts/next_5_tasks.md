# Next 5 Tasks

1. [ ] **Verify roll-forward via debug endpoint** - Call `/debug/predict-input` and confirm missing/filled counts drop for real matchups. <!-- id: 40 -->
2. [ ] **Regenerate dataset with EPA metrics** - Run `python backend/build_csv_datasetsv3.py --start 2018 --end 2025 --include-epa` to fix feature mismatch. <!-- id: 23 -->
3. [ ] **Add robust logging** - Integrate structured logging in `backend/main.py` to capture prediction failures and model performance. <!-- id: 24 -->
4. [ ] **Create smoke tests** - Add a `smoke_test.py` that validates all API endpoints end-to-end. <!-- id: 25 -->
5. [ ] **Validate model loading in prod** - Confirm MODELS_DIR for Heroku is set appropriately for the new 20260102 artifacts. <!-- id: 34 -->
