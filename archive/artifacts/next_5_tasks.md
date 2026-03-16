# 🔜 Next 5 Tasks

1. [ ] **Monitor performance in production** - After restarting backend, observe logs for 5-10 minutes to confirm no PerformanceWarnings appear and predictions complete successfully and accurately. <!-- id: 50 -->
2. [ ] **Verify health polling reduction** - Use browser DevTools Network tab to confirm `/health` requests occur every ~60 seconds instead of every 15 seconds. <!-- id: 51 -->
3. [ ] **Load test predictions** - Generate 10-20 predictions to ensure accuracy is unchanged and response times remain consistent. <!-- id: 52 -->
4. [ ] **Regenerate dataset with EPA metrics** - Run `python backend/build_csv_datasetsv3.py --start 2018 --end 2025 --include-epa` to fix feature mismatch warnings. <!-- id: 23 -->
5. [ ] **Add structured logging** - Integrate structured logging in `backend/main.py` to capture prediction latencies and model performance metrics. <!-- id: 24 -->
