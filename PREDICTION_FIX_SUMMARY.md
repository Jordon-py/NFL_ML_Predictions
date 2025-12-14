# Prediction Display Fix - Technical Summary

## 🎯 Objective

Fix dashboard displaying incorrect predictions (away_score showing wrong values) and ensure backend uses correct dataset and models for production inference.

## 🔧 Changes Made

### 1. Backend Configuration (.env)

**File**: `backend/.env`

```diff
- MODELS_DIR="backend/models"
- DATASET_PATH="backend/data/production_inference.csv"
+ MODELS_DIR="backend/data/prod-models/models"
+ DATASET_PATH="backend/data/game_features_20251213.csv"
```

**Impact**:

- Backend now loads models from `backend/data/prod-models/models/` (trained 2025-12-10)
- Uses latest engineered dataset with complete feature set (2,149 rows, 200+ features)

### 2. Smart Stats Roll-Forward Function

**File**: `backend/main.py`

Added `_roll_forward_last_game_stats()` function (line ~890):

**Purpose**: When predicting future games where stats haven't been calculated yet (because the game hasn't been played), this function intelligently copies the team's most recent game statistics.

**How it Works**:

1. Finds the team's most recent completed game
2. Extracts rolling averages (3-game, 5-game, 10-game windows)
3. Maps stats correctly from home/away context
4. Returns stats for THIS prediction only (not saved to dataset)

**Example**:

```python
# Predicting KC vs LAC Week 15 (not yet played)
# KC's last game was Week 14

home_rolled = _roll_forward_last_game_stats(df, "KC", 2025, 15, "home")
# Returns:
{
  "home_rolling_pf_3": 28.3,    # From KC Week 14
  "home_rolling_pa_3": 21.7,
  "home_rolling_win_pct_3": 0.667,
  # ... more stat values
}
```

### 3. Integration into Feature Building

**File**: `backend/main.py` (line ~1104)

Modified `_build_future_row()` to use rolled-forward stats:

```python
# Compute priors from team history
home_feats = compute_priors(home, "home_")
away_feats = compute_priors(away, "away_")

# NEW: Roll forward last game's stats if needed
home_rolled = _roll_forward_last_game_stats(local, home, season, week, "home")
away_rolled = _roll_forward_last_game_stats(local, away, season, week, "away")

# Merge (don't overwrite existing computed values)
for k, v in home_rolled.items():
    if k not in home_feats or pd.isna(home_feats.get(k)):
        home_feats[k] = v
```

## 📊 Before vs After

### Before Fix

```json
{
  "home_score": 23.1,
  "away_score": 20.7,  // ← Always same values
  "home_win_probability": 0.65,  // ← Heuristic fallback
  "prediction_source": "feature_fallback+win_fallback"
}
```

**Problem**: Rolling stats were 0/NaN for future games → models fell back to heuristics

### After Fix

```json
{
  "home_score": 25.4,  // ← Varies by matchup
  "away_score": 22.3,  // ← Uses team-specific stats
  "home_win_probability": 0.547,  // ← From ML model
  "prediction_source": "model",
  "win_classifier_used": true
}
```

**Solution**: Uses real team stats from last played game → full ML prediction pipeline

## 🔍 Key Technical Details

### Why Roll-Forward Instead of Zero-Filling?

**Zero-filling** would mean:

- Rolling averages = 0
- Win percentages = 0
- Model sees unrealistic inputs → produces garbage predictions

**Roll-forward** means:

- Use KC's Week 14 rolling averages for Week 15 prediction
- Realistic stat values → model produces meaningful predictions
- When Week 15 actually happens, real stats automatically used next time

### Dynamic vs Static

**This is dynamic** (per-prediction):

- Function runs every time `/predict` is called
- Stats NOT saved to dataset
- When real game data becomes available, it's automatically used

**Not static** (pre-computed):

- We don't modify `game_features_20251213.csv`
- Future games in CSV can have NaN/0 for unplayed stats
- Roll-forward happens at prediction time only

## 🧪 Testing

### Verify Backend Configuration

```bash
# From project root
cd backend
python -c "
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env')
print('MODELS_DIR:', os.getenv('MODELS_DIR'))
print('DATASET_PATH:', os.getenv('DATASET_PATH'))
print('Models exist:', Path(os.getenv('MODELS_DIR', '')).exists())
print('Dataset exists:', Path(os.getenv('DATASET_PATH', '')).exists())
"
```

### Test Prediction

```bash
# Start backend
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# In another terminal, test prediction:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "KC", "away_team": "LAC", "season": 2025, "week": 15}'
```

Expected response:

- `prediction_source: "model"` (not fallback)
- `away_score` varies by matchup
- Logs show: `✓ Rolled forward N stats for KC from 2025 W14`

## 📝 Frontend Impact

### No Changes Needed

The frontend already correctly handles the prediction response structure:

```javascript
// Dashboard.jsx (line 267)
const awayScore = rawPrediction?.away_score ?? rawPrediction?.away_score_pred ?? null;
```

This code already looks for `away_score` in the response, which the backend now provides with correct values.

## 🚀 Deployment

### Backend

1. Ensure `.env` has correct paths:
   - `DATASET_PATH=backend/data/game_features_20251213.csv`
   - `MODELS_DIR=backend/data/prod-models/models`

2. Restart backend service:

   ```bash
   # Local
   python -m uvicorn backend.main:app --reload

   # Heroku
   git push heroku main
   ```

### Frontend

No changes needed - already compatible!

## 🎓 Educational Notes

### Why This Pattern?

This "roll-forward" pattern is common in time-series prediction when:

- Future events haven't occurred yet
- Need to make predictions with latest available data
- Don't want to retrain models constantly

### Alternative Approaches

1. **Exponential Smoothing**: Weight recent games more heavily
2. **Seasonal Averages**: Use same-week stats from previous season
3. **League Averages**: Use league-wide stats as baseline

We chose **roll-forward** because:

- Simple and transparent
- Uses team-specific recent performance
- Automatically updates when new data arrives
- No model retraining required

---

**Bottom Line**: The dashboard will now display varying, realistic predictions based on each team's most recent performance, rather than falling back to generic heuristics.
