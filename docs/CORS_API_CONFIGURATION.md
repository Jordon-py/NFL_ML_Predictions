# CORS and API Configuration Guide

## Overview

This document explains the CORS (Cross-Origin Resource Sharing) configuration between the NFL ML Predictions frontend and backend, ensuring proper API communication across different deployment environments.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    CLIENT (Frontend)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Vercel (Production)                                 │  │
│  │  https://nfl-ml-predictions.vercel.app              │  │
│  │  https://nfl-predict-frontend.vercel.app            │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Localhost (Development)                             │  │
│  │  http://localhost:3000                               │  │
│  │  https://localhost:3000                              │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP Requests
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    SERVER (Backend)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Heroku                                              │  │
│  │  https://nfl-predict-ecf5a5bd34fe.herokuapp.com     │  │
│  │                                                      │  │
│  │  FastAPI + CORSMiddleware                           │  │
│  │  Allowed Origins: CORS_ORIGINS env variable         │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

## CORS Configuration

### Backend (FastAPI)

**File:** `backend/main.py`

```python
# Primary production mode: prefer ALLOWED_ORIGINS when RESTRICT_CORS is true
# - ALLOWED_ORIGINS: comma-separated list of origins to allow (Heroku config)
# - CORS_ORIGINS: legacy/compat fallback used in earlier deployments
# - CORS_ORIGINS_REGEX: optional regex to match allowed origins (used if provided)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")
CORS_ORIGINS = os.getenv("CORS_ORIGINS")
CORS_ORIGINS_REGEX = os.getenv("CORS_ORIGINS_REGEX")
RESTRICT_CORS = os.getenv("RESTRICT_CORS", "true").lower() in ("1", "true", "yes")

def _parse_origins(value: Optional[str]) -> List[str]:
  if not value:
    return []
  return [o.strip().rstrip('/') for o in value.replace(';', ',').split(',') if o.strip()]

allowed = _parse_origins(ALLOWED_ORIGINS) if ALLOWED_ORIGINS else _parse_origins(CORS_ORIGINS)

app = FastAPI(title="NFL Game Prediction API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
  CORSMiddleware,
  allow_origins=allowed,
  allow_origin_regex=CORS_ORIGINS_REGEX if CORS_ORIGINS_REGEX else None,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
```

**Configuration Details:**

- Reads from `CORS_ORIGINS` environment variable
- Splits on comma to support multiple origins
- Defaults to `http://localhost:3000` if not set
- Allows credentials (cookies, authorization headers)
- Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
- Allows all headers

### Environment Variables

#### Root `.env` (Deployed to Heroku)

```bash
# Production should set RESTRICT_CORS=true and ALLOWED_ORIGINS to a comma-separated list.
RESTRICT_CORS=true
ALLOWED_ORIGINS=http://localhost:3000,https://localhost:3000,https://nfl-ml-predictions.vercel.app,https://nfl-predict-frontend.vercel.app
```

**Purpose:** Production configuration for Heroku backend

**Allowed Origins:**

- `http://localhost:3000` - Local development HTTP
- `https://localhost:3000` - Local development HTTPS
- `https://nfl-ml-predictions.vercel.app` - Primary production frontend
- `https://nfl-predict-frontend.vercel.app` - Alternative production frontend

#### `backend/.env` (Local Development)

```bash
CORS_ORIGINS=http://localhost:3000,https://localhost:3000,https://nfl-ml-predictions.vercel.app,https://nfl-predict-frontend.vercel.app
```

**Purpose:** Local backend development configuration (matches production)

### Frontend Configuration

#### `frontend/.env` (Local Development)

```bash
VITE_API_URL=http://127.0.0.1:8000
```

**Purpose:** Points local frontend to local backend

#### `frontend/.env.production` (Vercel Deployment)

```bash
VITE_API_URL=https://nfl-predict-ecf5a5bd34fe.herokuapp.com
```

**Purpose:** Points production frontend to Heroku backend

#### `vercel.json` (Vercel Build Configuration)

```json
{
  "env": {
    "VITE_API_URL": "https://nfl-predict-ecf5a5bd34fe.herokuapp.com"
  }
}
```

**Purpose:** Ensures VITE_API_URL is set during Vercel build

#### `frontend/vite.config.js` (Development Proxy)

```javascript
server: {
  port: 3000,
  open: true,
  proxy: {
    '/api': { target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com', changeOrigin: true },
    '/schedule': { target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com', changeOrigin: true },
    '/predict': { target: 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com', changeOrigin: true },
  },
}
```

**Purpose:** Proxies API requests during local development to avoid CORS issues

## API Client

**File:** `frontend/src/api/client.js`

```javascript
const BASE_URL = import.meta.env.VITE_API_URL || 
                 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';

async function api(path, opts = {}) {
  const url = buildUrl(path);
  const res = await fetch(url, {
    headers: {'Content-Type': 'application/json'},
    ...opts,
  });
  // ... error handling
}
```

**Key Features:**

- Reads `VITE_API_URL` from environment
- Falls back to Heroku URL if not set
- Sets JSON content type by default
- Provides error handling and logging

## Testing CORS Configuration

### 1. Test Backend Health Endpoint

```bash
curl -X GET https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health
```

**Expected Response:**

```json
{
  "status": "healthy",
  "mode": "production",
  "reason": "models loaded"
}
```

### 2. Test CORS Headers

```bash
curl -X OPTIONS https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health \
  -H "Origin: https://nfl-ml-predictions.vercel.app" \
  -H "Access-Control-Request-Method: GET" \
  -v
```

**Expected Headers:**

```
Access-Control-Allow-Origin: https://nfl-ml-predictions.vercel.app
Access-Control-Allow-Methods: *
Access-Control-Allow-Headers: *
Access-Control-Allow-Credentials: true
```

### 3. Test Prediction Endpoint

```bash
curl -X POST https://nfl-predict-ecf5a5bd34fe.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -H "Origin: https://nfl-ml-predictions.vercel.app" \
  -d '{
    "home_team": "KC",
    "away_team": "BUF",
    "season": 2025,
    "week": 10
  }'
```

**Expected Response:**

```json
{
  "home_score": 24.5,
  "away_score": 23.2,
  "home_win_probability": 0.543,
  "away_win_probability": 0.457,
  "point_diff": 1.3,
  "mode": "production"
}
```

## Troubleshooting

### Issue: CORS Error in Browser Console

**Error Message:**

```
Access to fetch at 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com/predict' 
from origin 'https://nfl-ml-predictions.vercel.app' has been blocked by CORS policy
```

**Solutions:**

1. **Verify Heroku CORS_ORIGINS:**

   ```bash
   heroku config:get CORS_ORIGINS -a nfl-predict
   ```

2. **Update CORS_ORIGINS on Heroku:**

   ```bash
   heroku config:set CORS_ORIGINS="http://localhost:3000,https://localhost:3000,https://nfl-ml-predictions.vercel.app,https://nfl-predict-frontend.vercel.app" -a nfl-predict
   ```

3. **Restart Heroku Dyno:**

   ```bash
   heroku restart -a nfl-predict
   ```

4. **Check Backend Logs:**

   ```bash
   heroku logs --tail -a nfl-predict
   ```

   Look for: `CORS Origins configured: [...]`

### Issue: API Request to Wrong URL

**Error Message:**

```
Failed to fetch
```

**Solutions:**

1. **Check Frontend Environment Variable:**
   - In Vercel dashboard → Project Settings → Environment Variables
   - Verify `VITE_API_URL` is set correctly

2. **Check Browser Console:**

   ```javascript
   console.log('[API Client] Using BASE_URL:', BASE_URL);
   ```

3. **Rebuild Frontend:**

   ```bash
   npm run build --prefix frontend
   vercel --prod
   ```

### Issue: Missing Dataset Error

**Error Message:**

```
500 Internal Server Error: Dataset not found
```

**Solution:**

Generate the dataset on Heroku:

```bash
heroku run python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data -a nfl-predict
```

Or include dataset in git (if small enough):

```bash
# Remove *.csv from .gitignore temporarily
git add backend/data/merged_game_features.csv
git commit -m "Add dataset for deployment"
git push heroku main
```

## Deployment Checklist

### Before Deploying Backend to Heroku

- [ ] Set `CORS_ORIGINS` environment variable in Heroku
- [ ] Verify dataset exists or can be generated
- [ ] Test locally with `uvicorn backend.main:app --reload`
- [ ] Check `.env` file has correct CORS origins

### Before Deploying Frontend to Vercel

- [ ] Verify `VITE_API_URL` in `frontend/.env.production`
- [ ] Set `VITE_API_URL` in Vercel project settings
- [ ] Test locally with `npm run dev --prefix frontend`
- [ ] Verify API calls work from localhost

### After Deployment

- [ ] Test backend `/health` endpoint
- [ ] Test backend `/predict` endpoint with curl
- [ ] Test frontend loads from Vercel URL
- [ ] Test frontend can make API calls to backend
- [ ] Check Heroku logs for CORS configuration
- [ ] Verify no CORS errors in browser console

## Security Considerations

1. **Production CORS Origins:** Only include trusted domains in CORS_ORIGINS
2. **Environment Variables:** Never commit sensitive data to `.env` files
3. **HTTPS:** Always use HTTPS in production
4. **Credentials:** Only enable `allow_credentials=True` if needed for authentication

## References

- FastAPI CORS Documentation: <https://fastapi.tiangolo.com/tutorial/cors/>
- Vite Environment Variables: <https://vitejs.dev/guide/env-and-mode.html>
- Heroku Config Vars: <https://devcenter.heroku.com/articles/config-vars>
- Vercel Environment Variables: <https://vercel.com/docs/environment-variables>
