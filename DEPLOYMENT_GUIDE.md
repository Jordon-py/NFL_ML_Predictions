# Prediction Fix - Deployment Guide

## Current Situation

Your **frontend is pointing to Heroku** (`https://nfl-predict-ecf5a5bd34fe.herokuapp.com`), but your **backend code updates are only on your local machine**.

**Test Results**:

- ✅ Local backend working: Returns varying predictions (not uniform 21-23)
- ✅ Models loaded correctly from `backend/data/prod-models/models`
- ✅ Dataset loaded: `game_features_20251213.csv` with 214 columns
- ❌ Frontend still showing old predictions: Because it's using Heroku, not localhost

## Solution: Choose One

### Option 1: Deploy to Heroku (Production Fix) ⭐RECOMMENDED⭐

This will fix the live dashboard for all users.

```bash
# 1. Make sure all changes are committed
git status
git add backend/.env backend/main.py alfred.log.md
git commit -m "fix: correct dataset path, models dir, and add smart stat roll-forward for predictions"

# 2. Push to Heroku
git push heroku main

# 3. Verify deployment
curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health

# 4. Test a prediction
curl -X POST https://nfl-predict-ecf5a5bd34fe.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "KC", "away_team": "LAC", "season": 2025, "week": 15}'

# 5. Check the live dashboard
# Visit: https://nfl-ml-predictions.vercel.app
```

**Important Heroku Notes**:

- Heroku will use the `.env` values as defaults but **environment variables** override them
- Make sure Heroku environment has:
  - `MODELS_DIR=backend/data/prod-models/models`
  - `DATASET_PATH=backend/data/game_features_20251213.csv`

To set Heroku environment variables:

```bash
heroku config:set MODELS_DIR="backend/data/prod-models/models" -a nfl-predict
heroku config:set DATASET_PATH="backend/data/game_features_20251213.csv" -a nfl-predict
```

---

### Option 2: Test Locally (Development Testing)

This changes the frontend to use your local backend for testing.

#### Step 1: Update Frontend `.env`

```bash
# frontend/.env
VITE_API_BASE_URL='http://localhost:8000'
VITE_DEV_ENV=development
```

#### Step 2: Start Local Backend

```bash
cd backend
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

#### Step 3: Start Local Frontend

```bash
cd frontend
npm run dev
```

#### Step 4: Open Browser

```
http://localhost:5173
```

Now predictions should work with your local backend!

#### Step 5: Reset to Production When Done

```bash
# frontend/.env (put back)
VITE_API_BASE_URL='https://nfl-predict-ecf5a5bd34fe.herokuapp.com'
VITE_DEV_ENV=production
```

---

## Quick Test Without Changing Frontend

You can also test if Heroku backend is working by calling it directly:

```bash
# Test Heroku health
curl https://nfl-predict-ecf5a5bd34fe.herokuapp.com/health

# Test Heroku prediction
curl -X POST https://nfl-predict-ecf5a5bd34fe.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "TB", "away_team": "ATL", "season": 2025, "week": 15}'
```

If Heroku returns the same old predictions (21-23), then you need to deploy the fix to Heroku.

---

## What Should You Do?

**Recommended**: Deploy to Heroku first (Option 1) so the production dashboard works.

Then if you want to develop/test locally in the future, use Option 2.

---

## Files Changed

All changes are committed locally. Just need to push to Heroku:

### Backend

- `backend/.env` - Updated MODELS_DIR and DATASET_PATH
- `backend/main.py` - Added `_roll_forward_last_game_stats()` function

### Documentation

- `alfred.log.md` - Added fix documentation
- `PREDICTION_FIX_SUMMARY.md` - Technical summary
- `verify_prediction_fix.py` - Verification script
- `test_predictions.py` - Test script
