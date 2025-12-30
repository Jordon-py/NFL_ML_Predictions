<<<<<<< HEAD
# Archived: API and CORS Verification Checklist

This document has been archived during documentation consolidation. See `docs/DOCS_CONSOLIDATED.md` for the current, canonical guidance. The original version is available in repository history if needed.
=======
# API and CORS Verification Checklist

## Overview

This checklist helps verify that the NFL ML Predictions backend and frontend are properly configured for API communication and CORS alignment.

**Last Updated:** 2025-10-13  
**Status:** ✅ CORS Configuration Verified and Documented

---

## Pre-Deployment Checklist

### Backend (Heroku)

- [x] **CORS Configuration**
  - File: `.env` (root)
  - Variable: `CORS_ORIGINS`
  - Value: `http://localhost:3000,https://localhost:3000,https://nfl-ml-predictions.vercel.app,https://nfl-predict-frontend.vercel.app`
  - ✅ Includes all frontend origins (localhost and Vercel)

- [x] **Backend Environment File**
  - File: `backend/.env`
  - Created: Yes (for local development)
  - Excluded from git: Yes (via `.gitignore`)
  - Contains CORS_ORIGINS: Yes

- [x] **FastAPI Configuration**
  - File: `backend/main.py`
  - CORS Middleware: Configured (lines 265-278)
  - Reads CORS_ORIGINS: From environment variable
  - Allows credentials: Yes
  - Allows all methods: Yes
  - Allows all headers: Yes

- [x] **Models Present**
  - `backend/models/home_model.joblib`: ✅ Exists
  - `backend/models/away_model.joblib`: ✅ Exists
  - `backend/models/preprocessor.joblib`: ✅ Exists
  - `backend/models/metadata.json`: ✅ Exists

- [ ] **Dataset Present**
  - `backend/data/merged_game_features.csv`: ❌ Missing
  - **Action Required:** Run `python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data`
  - Note: Dataset is excluded from git via `*.csv` in `.gitignore`

- [x] **Schedule Data**
  - `backend/data/Nfl_schedule_2025_2026.csv`: ✅ Exists

### Frontend (Vercel)

- [x] **Production Environment**
  - File: `frontend/.env.production`
  - Variable: `VITE_API_BASE_URL`
  - Value: `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`
  - ✅ Points to Heroku backend (no comma-separated values)

- [x] **Development Environment**
  - File: `frontend/.env`
  - Variable: `VITE_API_BASE_URL`
  - Value: `http://127.0.0.1:8000`
  - ✅ Points to local backend

- [x] **Vite Configuration**
  - File: `frontend/vite.config.js`
  - Proxy configured: Yes (for `/api`, `/schedule`, `/predict`)
  - Target: `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`
  - Change origin: Yes

- [x] **API Client**
  - File: `frontend/src/api/client.js`
  - Uses VITE_API_BASE_URL: Yes
  - Fallback URL: Heroku backend
  - JSON headers: Set by default
  - Error handling: Implemented

- [x] **Vercel Configuration**
  - File: `vercel.json`
  - VITE_API_BASE_URL set: Yes
  - Build command: Configured
  - Output directory: `frontend/build`

---

## Deployment Verification Steps

### Step 1: Deploy Backend to Heroku

```bash
# Ensure CORS_ORIGINS is set on Heroku
heroku config:set CORS_ORIGINS="http://localhost:3000,https://localhost:3000,https://nfl-ml-predictions.vercel.app,https://nfl-predict-frontend.vercel.app" -a nfl-predict

# Push to Heroku
git push heroku main

# Verify deployment
heroku logs --tail -a nfl-predict
```

**Expected in logs:**
```
CORS Origins configured: ['http://localhost:3000', 'https://localhost:3000', ...]
Loaded dataset rows=XXXX cols=XX
```

### Step 2: Test Backend Endpoints

```bash
# Test health endpoint
curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health

# Expected response:
# {"status":"healthy","mode":"production","reason":"models loaded"}

# Test CORS headers
curl -X OPTIONS https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health \
  -H "Origin: https://nfl-ml-predictions.vercel.app" \
  -H "Access-Control-Request-Method: GET" \
  -v

# Expected headers:
# Access-Control-Allow-Origin: https://nfl-ml-predictions.vercel.app
# Access-Control-Allow-Credentials: true
```

### Step 3: Run Verification Script

```bash
# Test production backend
python scripts/verify_api_cors.py

# Test local backend
python scripts/verify_api_cors.py --backend-url http://localhost:8000

# Verbose output
python scripts/verify_api_cors.py --verbose
```

**Expected output:**
```
✓ Health Endpoint: PASSED
✓ CORS Configuration: PASSED
✓ Debug Endpoint: PASSED
✓ Predict Endpoint: PASSED (or warning if dataset missing)
Total: 4/4 tests passed
```

### Step 4: Deploy Frontend to Vercel

```bash
# Ensure VITE_API_BASE_URL is set in Vercel project settings
# Login to Vercel
vercel login

# Deploy
cd frontend
npm run build
vercel --prod
```

### Step 5: Test Frontend-Backend Integration

1. **Open Frontend in Browser**
   - URL: https://nfl-ml-predictions.vercel.app
   - Open browser developer console (F12)

2. **Check API Client Logs** (in console):
   ```
   [API Client] Using BASE_URL: https://nfl-predict-ecf5a5bd34fe.herokuapp.com
   [API Client] Mode: production
   ```

3. **Test Prediction**
   - Select two teams
   - Click "Predict"
   - Check Network tab for:
     - Request to: `https://nfl-predict-ecf5a5bd34fe.herokuapp.com/predict`
     - Response status: 200 OK
     - Response body: Contains `home_score`, `away_score`, `home_win_probability`, `away_win_probability`

4. **Verify No CORS Errors**
   - Console should have NO errors like:
     - "Access to fetch... has been blocked by CORS policy"
     - "No 'Access-Control-Allow-Origin' header"

---

## Troubleshooting

### Issue: CORS Error in Browser

**Error:**
```
Access to fetch at 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com/predict' 
from origin 'https://nfl-ml-predictions.vercel.app' has been blocked by CORS policy
```

**Solution:**

1. Check Heroku CORS_ORIGINS:
   ```bash
   heroku config:get CORS_ORIGINS -a nfl-predict
   ```

2. Update if needed:
   ```bash
   heroku config:set CORS_ORIGINS="http://localhost:3000,https://localhost:3000,https://nfl-ml-predictions.vercel.app,https://nfl-predict-frontend.vercel.app" -a nfl-predict
   ```

3. Restart Heroku dyno:
   ```bash
   heroku restart -a nfl-predict
   ```

4. Clear browser cache and reload

### Issue: API Returns 500 - Dataset Not Found

**Error:**
```json
{"detail": "Dataset not found: backend/data/merged_game_features.csv"}
```

**Solution:**

Generate the dataset:
```bash
# On Heroku (if you have enough dyno hours)
heroku run python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data -a nfl-predict

# OR locally and commit (if dataset is small enough)
python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data
git add backend/data/merged_game_features.csv -f  # Force add despite .gitignore
git commit -m "Add dataset for deployment"
git push heroku main
```

### Issue: Frontend Shows Wrong API URL

**Solution:**

1. Check Vercel environment variables:
   - Go to Vercel dashboard → Project → Settings → Environment Variables
   - Verify `VITE_API_BASE_URL` = `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`

2. Rebuild frontend:
   ```bash
   cd frontend
   npm run build
   vercel --prod
   ```

---

## Success Indicators

✅ **Backend Health:**
- `/health` endpoint returns 200 OK
- Response: `{"status":"healthy"}`
- Logs show: "CORS Origins configured: [...]"

✅ **CORS Working:**
- OPTIONS preflight requests return CORS headers
- No CORS errors in browser console
- Fetch requests succeed from Vercel frontend

✅ **Predictions Working:**
- `/predict` endpoint returns 200 OK
- Response contains all required fields
- Frontend displays predictions

✅ **Frontend-Backend Communication:**
- Network tab shows requests to Heroku backend
- Responses are JSON with expected data
- No authentication or authorization errors

---

## Next Steps

1. **Generate Dataset** (if not done):
   ```bash
   python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data
   ```

2. **Deploy Changes**:
   ```bash
   # Backend
   git push heroku main
   
   # Frontend
   vercel --prod
   ```

3. **Monitor**:
   ```bash
   # Backend logs
   heroku logs --tail -a nfl-predict
   
   # Frontend logs
   vercel logs nfl-ml-predictions
   ```

4. **Test Continuously**:
   - Run `python scripts/verify_api_cors.py` after each deployment
   - Check browser console for errors
   - Monitor API response times

---

## Documentation References

- **CORS Configuration Guide:** [docs/CORS_API_CONFIGURATION.md](CORS_API_CONFIGURATION.md)
- **Deployment Guide:** [DEPLOYMENT_FIXED.md](../DEPLOYMENT_FIXED.md)
- **Change Log:** [docs/report.md](report.md)
- **Main README:** [README.md](../README.md)

---

**Verification Completed:** 2025-10-13  
**Next Review:** After next deployment
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
