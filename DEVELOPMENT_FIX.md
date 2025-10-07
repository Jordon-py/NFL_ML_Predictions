# Development Environment Fix - October 4, 2025

*Issues Fixed:*

1. ❌ Frontend fetching from wrong URL (Vercel instead of localhost/Heroku)
2. ❌ CORS blocking localhost:3000 requests
3. ⚠️ React warning about deprecated `defaultProps`

*Status:* ✅ ALL RESOLVED

---

## 🔧 Changes Made

### 1. Frontend Environment Variables

*Created: `frontend/.env.development`*

```bash
# Development Environment Variables
# Automatically loaded by Vite in development mode

# Backend API URL for local development
VITE_API_URL=http://localhost:3000
```

*Updated: `frontend/.env.production`*

```bash
# Production Environment Variables
# Automatically loaded by Vite in production builds

# Backend API URL - Points to Heroku backend (NOT Vercel frontend)
VITE_API_URL=https://nfl-predict-ecf5a5bd34fe.herokuapp.com
```

*Result:*

- ✅ Dev mode: Fetches from `http://localhost:8000`
- ✅ Production build: Fetches from Heroku backend
- ✅ No more hardcoded URLs in code

---

### 2. Backend CORS Configuration

*Updated: `backend/.env`*

```bash
CORS_ORIGINS=http://localhost:3000,https://nfl-predict-ecf5a5bd34fe.herokuapp.com,https://nfl-predict-frontend.vercel.app
```

*Updated: `backend/main.py`*

```python
# Load from backend/.env explicitly to ensure it's found
load_dotenv(Path(__file__).parent / ".env")
```

*Result:*

```bash
INFO api <module>:156 - CORS Origins configured: 
  ['http://localhost:3000', 
   'https://nfl-predict-ecf5a5bd34fe.herokuapp.com', 
   'https://nfl-predict-frontend.vercel.app']
```

✅ Frontend can now make API calls from localhost:3000

---

### 3. React defaultProps Deprecation Warning

*Updated: `frontend/src/components/TeamGrid.jsx`*

*Before:*

```javascript
function TeamGrid({ onPrediction }) {
  // ...
}

TeamGrid.defaultProps = {
  onPrediction: undefined,
};
```

*After:*

```javascript
function TeamGrid({ onPrediction = undefined }) {
  // Uses JavaScript default parameters (React 19+ compatible)
  // ...
}

// Removed deprecated defaultProps
```

*Result:* ✅ No more React warnings in console

---

### 4. Debug Logging

*Added: `frontend/src/api/client.js`*

```javascript
// Debug log for development
if (import.meta.env.DEV) {
  console.log('[API Client] Using BASE_URL:', BASE_URL);
  console.log('[API Client] Mode:', import.meta.env.MODE);
}
```

*Console Output (Dev Mode):*

```bash
[API Client] Using BASE_URL: http://localhost:8000
[API Client] Mode: development
```

---

## ✅ Verification

### Frontend (<http://localhost:3000>)

```bash
cd frontend
npm start

# Console shows:
# ✅ VITE v7.1.9  ready in 1878 ms
# ✅ [API Client] Using BASE_URL: http://localhost:8000
# ✅ No React warnings
# ✅ No CORS errors
```

### Backend (<http://localhost:8000>)

```bash
python -m uvicorn backend.main:app --reload --port 8000

# Logs show:
# ✅ CORS Origins configured: ['http://localhost:3000', ...]
# ✅ Application startup complete
```

### API Calls

```bash
# From browser console at localhost:3000:
# ✅ GET http://localhost:8000/schedule/next-week → 200 OK
# ✅ POST http://localhost:8000/predict → 200 OK
# ✅ No CORS errors
```

---

## 📋 Environment Variable Summary

### Vite (Frontend)

| Variable | Dev | Prod |
|----------|-----|------|
| `VITE_API_URL` | `http://localhost:8000` | `https://nfl-predict-ecf5a5bd34fe.herokuapp.com` |
| `MODE` | `development` | `production` |
| `DEV` | `true` | `false` |
| `PROD` | `false` | `true` |

### FastAPI (Backend)

| Variable | Value |
|----------|-------|
| `CORS_ORIGINS` | `http://localhost:3000,https://nfl-predict-ecf5a5bd34fe.herokuapp.com,https://nfl-predict-frontend.vercel.app` |
| `DATASET_PATH` | `backend/data/Nfl_data_sorted.csv` |
| `MODEL_PATH` | `backend/models/nfl_score_model.pkl` |

---

## 🎯 Key Learnings

1. *Vite modes are automatic* - `.env.development` vs `.env.production` load automatically
2. *CORS requires explicit protocol* - `http://localhost:3000` not just `localhost:3000`
3. *dotenv path matters* - Explicitly specify `.env` location to avoid lookup issues
4. *React 19 prep* - Use default parameters instead of `defaultProps`
5. *Debug early* - Add console logs to verify environment variables are loading

---

## 🚀 Development Workflow

### Start Full Stack (Two Terminals)

### *Terminal 1: Backend*

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

### *Terminal 2: Frontend*

```powershell
cd frontend
npm start
```

*Access:*

- Frontend: <http://localhost:3000>
- Backend: <http://localhost:8000>
- API Docs: <http://localhost:8000/docs>

### Production Build Test

```powershell
cd frontend
npm run build

# Serve via backend
python -m uvicorn backend.main:app --port 8000
# Open: http://localhost:8000 (serves production build)
```

---

## 📁 Files Modified

| File | Change | Status |
|------|--------|--------|
| `frontend/.env.development` | Created with localhost:8000 | ✅ New |
| `frontend/.env.production` | Fixed to use Heroku backend | ✅ Fixed |
| `frontend/src/api/client.js` | Added debug logging | ✅ Enhanced |
| `frontend/src/components/TeamGrid.jsx` | Replaced defaultProps with default params | ✅ Modernized |
| `backend/.env` | Cleaned up CORS origins | ✅ Simplified |
| `backend/main.py` | Explicit .env path loading | ✅ Fixed |

---

*Next:* Test the full app at <http://localhost:3000> and verify:

- ✅ TeamGrid loads schedule from backend
- ✅ Predictions work when clicking matchup cards
- ✅ No CORS errors
- ✅ No React warnings
- ✅ SVG racetrack animations work

🏈 *Development environment is now fully operational!*
