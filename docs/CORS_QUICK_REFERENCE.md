# CORS Quick Reference Card

## 🎯 Quick Overview

**Backend:** Heroku @ `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`  
**Frontend:** Vercel @ `https://nfl-ml-predictions.vercel.app`  
**Tech Stack:** FastAPI (backend) + React/Vite (frontend)

---

## 📋 Environment Variables

### Backend (Heroku)

```bash
CORS_ORIGINS="http://localhost:3000,https://localhost:3000,https://nfl-ml-predictions.vercel.app,https://nfl-predict-frontend.vercel.app"
```

**Set on Heroku:**
```bash
heroku config:set CORS_ORIGINS="..." -a nfl-predict
```

### Frontend (Vercel)

**Production:**
```bash
VITE_API_URL=https://nfl-predict-ecf5a5bd34fe.herokuapp.com
```

**Development:**
```bash
VITE_API_URL=http://127.0.0.1:8000
```

---

## 🔧 Configuration Files

| File | Purpose | Key Setting |
|------|---------|-------------|
| `.env` (root) | Backend env vars | `CORS_ORIGINS` |
| `backend/.env` | Local backend dev | `CORS_ORIGINS` |
| `backend/main.py` | CORS middleware | Lines 265-278 |
| `frontend/.env` | Local frontend dev | `VITE_API_URL=http://127.0.0.1:8000` |
| `frontend/.env.production` | Production frontend | `VITE_API_URL=https://nfl-predict-...` |
| `frontend/vite.config.js` | Dev proxy | Proxies `/api`, `/schedule`, `/predict` |
| `vercel.json` | Vercel build | Sets `VITE_API_URL` |

---

## ✅ Quick Test Commands

### Test Backend Health
```bash
curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health
# Expected: {"status":"healthy",...}
```

### Test CORS Headers
```bash
curl -X OPTIONS https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health \
  -H "Origin: https://nfl-ml-predictions.vercel.app" \
  -H "Access-Control-Request-Method: GET" -v
# Look for: Access-Control-Allow-Origin header
```

### Run Verification Script
```bash
python scripts/verify_api_cors.py
```

---

## 🚀 Quick Deploy

### Backend
```bash
git push heroku main
heroku logs --tail -a nfl-predict
```

### Frontend
```bash
cd frontend && npm run build && vercel --prod
```

---

## 🐛 Common Issues

### CORS Error in Browser?
1. Check: `heroku config:get CORS_ORIGINS -a nfl-predict`
2. Update: `heroku config:set CORS_ORIGINS="..." -a nfl-predict`
3. Restart: `heroku restart -a nfl-predict`

### API 500 Error (Dataset Missing)?
```bash
python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data
```

### Wrong API URL in Frontend?
- Vercel: Settings → Environment Variables → Check `VITE_API_URL`
- Rebuild: `vercel --prod`

---

## 📚 Full Documentation

- **Detailed Guide:** [docs/CORS_API_CONFIGURATION.md](CORS_API_CONFIGURATION.md)
- **Checklist:** [docs/API_CORS_CHECKLIST.md](API_CORS_CHECKLIST.md)
- **Deployment:** [DEPLOYMENT_FIXED.md](../DEPLOYMENT_FIXED.md)

---

**Last Updated:** 2025-10-13
