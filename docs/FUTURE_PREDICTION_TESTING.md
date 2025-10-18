# Future Game Prediction - Testing Guide

## 📋 Overview

**Commit**: `10f0b9801` - "feat: implement dynamic feature building for future game predictions"

This implementation enables the NFL prediction API to make predictions for **future/scheduled games** that don't exist in the historical dataset yet.

## 🎯 What Was Implemented

### 1. Enhanced `_build_future_row()` Function

**Location**: `backend/main.py` lines 354-471

**Purpose**: Dynamically computes engineered features for future games using historical team performance data.

**Key Features**:
- ✅ Computes **rolling averages** (3-game and 5-game windows)
- ✅ Extracts **team-specific stats** from historical games
- ✅ Handles **home/away context** correctly
- ✅ Calculates **differential features** (home_prior_X - away_prior_X)
- ✅ Fills **betting/rest features** with neutral defaults
- ✅ Validates **sufficient historical data** exists

**Logic Flow**:
```python
1. Filter historical games before target date (time_key < cutoff)
2. For each team:
   - Find last 3 completed games → compute 3-game averages
   - Find last 5 completed games → compute 5-game averages
   - Extract advanced stats from most recent game
3. Compute home_minus_away differentials for all stats
4. Fill betting lines with neutral values (0.5 prob, 0 spread)
5. Return pd.Series with ~85 engineered features
```

### 2. Updated `predict_game()` Endpoint

**Location**: `backend/main.py` lines 610-658

**Changes**:
- ✅ First tries to find game in existing dataset
- ✅ If not found, calls `_build_future_row()` to generate features
- ✅ Extracts features using column prefix matching
- ✅ Improved error handling with specific messages

**Before**:
```python
if rows_any.empty:
    raise HTTPException(400, "No data found for {h} vs {a}...")
```

**After**:
```python
if rows.empty:
    log.info("Building features for future game...")
    row = _build_future_row(dataset_df, h, a, season, week)
```

## 🧪 Testing Instructions

### Step 1: Fix Python Environment

Your current Python environment has a broken `click` module. Fix it:

```powershell
# Option A: Reinstall click
pip uninstall click -y
pip install click

# Option B: Reinstall uvicorn
pip uninstall uvicorn -y
pip install uvicorn

# Option C: Full reset (if needed)
pip install --force-reinstall -r requirements.txt
```

### Step 2: Start Backend Server

```powershell
cd C:\Users\iProg\OneDrive\Documents\Football_predict\nfl_prediction_system\NFL_ML_Predictions
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Expected Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     api Startup: loading models and dataset
INFO:     api Loaded dataset rows=3282 cols=97
INFO:     Application startup complete.
```

### Step 3: Test Health Check

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Method GET
```

**Expected Response**:
```json
{
  "status": "healthy",
  "mode": "production",
  "reason": "models loaded"
}
```

### Step 4: Test Historical Game (Should Reject)

```powershell
$body = @{home_team='KC'; away_team='TEN'; season=2014; week=1} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method POST -Body $body -ContentType 'application/json'
```

**Expected Response**:
```json
{
  "detail": "Game completed; no prediction needed."
}
```

### Step 5: Test Future Game (NEW FEATURE!)

```powershell
# Get next week's schedule first
$schedule = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/schedule/next-week' -Method GET
$game = $schedule[0]  # Pick first game

# Make prediction
$body = @{
    home_team=$game.home_abbr
    away_team=$game.away_abbr
    season=$game.season
    week=$game.week
} | ConvertTo-Json

Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method POST -Body $body -ContentType 'application/json'
```

**Expected Response** (example):
```json
{
  "home_score": 24.3,
  "away_score": 21.7,
  "home_win_probability": 0.623,
  "away_win_probability": 0.377,
  "point_diff": 2.6,
  "mode": "models"
}
```

### Step 6: Test Batch Predictions

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict/next-week' -Method GET
```

**Expected Response**:
```json
{
  "context": {
    "current_season": 2025,
    "last_completed_season": 2024,
    "last_completed_week": 6,
    "next_prediction_season": 2025,
    "next_prediction_week": 7,
    "status": "nfl_season_active"
  },
  "games": [
    {
      "game_id": "2025_07_PIT_CIN",
      "season": 2025,
      "week": 7,
      "home_team": "CIN",
      "away_team": "PIT",
      "kickoff": "2025-10-16",
      "prediction": {
        "home_score": 27.1,
        "away_score": 23.4,
        ...
      }
    },
    ...
  ],
  "total_games": 15,
  "successful_predictions": 15
}
```

## 🔍 Verification Checklist

- [ ] Server starts without errors
- [ ] Health endpoint returns "healthy"
- [ ] Historical games return "Game completed" error
- [ ] Future games return valid predictions with:
  - [ ] home_score and away_score (0-70 range)
  - [ ] home_win_probability + away_win_probability = 1.0
  - [ ] point_diff = home_score - away_score
  - [ ] mode = "models"
- [ ] Batch predictions work for all scheduled games
- [ ] Logs show "Building features for future game" messages

## 📊 Expected Log Output

When predicting a future game, check `backend/logs/api.log`:

```
2025-10-17 19:00:00,123 INFO api Building features for future game: KC vs LV (2025 Week 7)
2025-10-17 19:00:00,456 DEBUG api Built future row for KC vs LV: 85 features
```

## ⚠️ Known Limitations

1. **Requires Historical Data**: Teams must have at least 1 prior completed game
   - Early season Week 1 predictions may fail for new teams
   - Solution: Use league-average defaults (future enhancement)

2. **Betting Lines**: Currently uses neutral defaults (0.5 prob, 0 spread)
   - Could integrate live betting data API (future enhancement)

3. **Advanced Stats**: Copies last game's EPA/success rate values
   - More sophisticated recalculation could improve accuracy

4. **Rest Days**: Defaults to 7 days for all games
   - Could calculate from actual schedule dates (future enhancement)

## 🎓 How It Works (Technical Deep Dive)

### Feature Engineering Process

For a game **KC @ DEN (2025 Week 7)**:

1. **Find Historical Games**:
   ```python
   # KC's last 5 games before 2025 Week 7
   KC_games = dataset[
       (team == 'KC') & 
       (time_key < 202507) & 
       (scores_not_null)
   ].tail(5)
   ```

2. **Compute Rolling Averages**:
   ```python
   home_prior_pf_avg_3 = mean([KC_game5.pf, KC_game4.pf, KC_game3.pf])
   home_prior_pa_avg_3 = mean([KC_game5.pa, KC_game4.pa, KC_game3.pa])
   home_prior_win_pct_3 = mean([KC_game5.win, KC_game4.win, KC_game3.win])
   ```

3. **Extract Advanced Stats** from most recent game:
   ```python
   home_prior_off_epa_per_play_3 = KC_game5.home_prior_off_epa_per_play_3
   home_prior_def_explosive_rate_3 = KC_game5.home_prior_def_explosive_rate_3
   # ... (copy all 20 advanced metrics)
   ```

4. **Compute Differentials**:
   ```python
   home_minus_away_pf_avg_3 = home_prior_pf_avg_3 - away_prior_pf_avg_3
   home_minus_away_win_pct_5 = home_prior_win_pct_5 - away_prior_win_pct_5
   # ... (compute all 26 differentials)
   ```

5. **Create Feature Vector** → Pass to ML models → Return predictions

## 📝 Testing Commands Summary

```powershell
# 1. Health check
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Method GET

# 2. Get schedule
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/schedule/next-week' -Method GET

# 3. Single prediction
$body = @{home_team='KC'; away_team='LV'; season=2025; week=7} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method POST -Body $body -ContentType 'application/json'

# 4. Batch predictions
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict/next-week' -Method GET

# 5. Check logs
Get-Content backend\logs\api.log -Tail 20
```

## ✅ Success Criteria

The implementation is successful if:

1. ✅ **Code compiles** without syntax errors (verified with `python -m py_compile`)
2. ✅ **Server starts** and loads 3282 rows × 97 columns dataset
3. ✅ **Future games return predictions** instead of "No data found" errors
4. ✅ **Predictions are reasonable** (scores 0-70, probabilities 0-1)
5. ✅ **Logs show feature building** for games not in dataset

## 🚀 Next Steps

Once testing is complete:

1. Monitor prediction accuracy for Week 7 games
2. Compare with actual results after games complete
3. Consider enhancements:
   - Integrate live betting data API
   - Add team strength ratings
   - Implement home field advantage adjustments
   - Calculate actual rest days from schedule

---

**Implementation Date**: October 17, 2025  
**Commit**: 10f0b9801  
**Status**: ✅ Code Complete, ⏳ Testing Pending (Python env issues)
