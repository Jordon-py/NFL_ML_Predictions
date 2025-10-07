# NFL Prediction System - AI Agent Instructions

*Concise guide for AI coding agents working in this NFL game prediction system.*

## 🏈 Architecture Overview

**Data Flow**: `nfl_data_py` API → CSV Processing → LightGBM Training → FastAPI → React

**System Purpose**: Predict NFL game scores and win probabilities using leak-free rolling statistics.

## 🔧 Essential Components

### 1. Data Pipeline (`backend/build_csv_datasets.py`)
**Critical Command** (MUST use these exact args):
```bash
python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data
```

**Anti-Leakage Pattern** (THE most critical code pattern):
```python
# CORRECT: shift(1) BEFORE rolling() - only prior games used
shifted = s.shift(1)
return shifted.rolling(window=N, min_periods=1).mean()
```

**Team Normalization** (prevents join failures):
```python
ABBR_FIX = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS"}
```

**Output**: `backend/data/Nfl_data_sorted.csv` with 18 base features + 6 differentials

### 2. Model Training (`backend/train_models.py`)
**Time-Series CV** (not regular KFold):
```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)  # Respects temporal order
```

**Fail-Fast Validation**:
- If R² < -0.2 → raises ValueError (no silent failures)
- Uses RandomizedSearchCV for speed (not full GridSearchCV)
- Outputs: `home_model.joblib`, `away_model.joblib`, `win_clf_calibrated.joblib`, `metadata.json`

**Features**: 3/5-game rolling averages for home/away (pf_avg, pa_avg, win_pct) + differentials

### 3. FastAPI Backend (`backend/main.py`)
**Startup Pattern** (fail-fast):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_objects
    model_objects = _load_models()  # Fails entire app if models missing
    yield
```

**Key Endpoints**:
- `GET /health` - Check model load status
- `POST /predict` - Score prediction for home/away teams
- `GET /schedule/next-week` - Upcoming games from CSV schedule

**Run Command** (note the path - VS Code task uses `backend.app.main:app` but file is `backend/main.py`):
```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. React Frontend State (`frontend/src/App.jsx`)
**Three-State Architecture** (managed via PredictionContext):
1. `result` - Current prediction output
2. `history` - Archived predictions array
3. `currentPrediction` - TeamGrid selection (separate from form)

**Dual Prediction Workflows**:
- **Manual Form**: User input → API call → sets `result` + pushes to `history`
- **TeamGrid**: Game card click → fetches schedule → `onPrediction()` callback → sets `currentPrediction`

**Independence**: TeamGrid does NOT share state with PredictionForm

## 🔄 Complete Setup Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Build dataset (REQUIRED before training)
python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data

# 3. Train models
python backend/train_models.py

# 4. Start backend (new terminal)
uvicorn backend.main:app --reload --port 8000

# 5. Start frontend (new terminal)
cd frontend && npm start
```

## 🎯 Critical Patterns

### Data Leakage Prevention
**Always use `.shift(1)` before `.rolling()` when creating features**:
```python
# BAD - future leakage
df.groupby('team')['score'].rolling(3).mean()

# GOOD - leak-free
df.groupby('team')['score'].shift(1).rolling(3, min_periods=1).mean()
```

### Team Code Normalization
**Apply ABBR_FIX in all data loading** (`build_csv_datasets.py`, `main.py`):
- Relocations: STL→LAR, SD→LAC, OAK→LV
- Legacy: WSH→WAS, LA→LAR

### Fail-Fast Error Handling
**Never use fallback predictions or silent failures**:
```python
# At startup: if models can't load, fail immediately
if not (MODELS_DIR / "home_model.joblib").exists():
    raise FileNotFoundError("home_model.joblib not found")
```

### Model Validation Checks
**Always inspect `backend/models/metadata.json`**:
```json
{
  "model_scores": {"home_r2_cv": 9.85, "win_auc_cv": 0.636},
  "production_ready_win_model": false  // <-- Check this flag
}
```

## 📁 Key Files Reference

| File | Purpose | Critical Details |
|------|---------|------------------|
| `backend/build_csv_datasets.py` | Dataset creation | Uses `shift(1).rolling()`, ABBR_FIX normalization |
| `backend/train_models.py` | Model training | TimeSeriesSplit, RandomizedSearchCV, fail-fast validation |
| `backend/main.py` | FastAPI server | Lifespan model loading, no fallbacks |
| `backend/models/metadata.json` | Training metrics | Check `production_ready_win_model` flag |
| `frontend/src/App.jsx` | State management | PredictionContext with 3-state architecture |
| `frontend/src/components/TeamGrid.jsx` | Interactive schedule | Independent prediction flow via `onPrediction()` |

## 🚫 Common Pitfalls

1. **Running training before building dataset** → Train fails with missing features
2. **Using regular KFold instead of TimeSeriesSplit** → Time leakage in validation
3. **Forgetting team normalization** → Join failures, missing predictions
4. **Assuming graceful degradation in API** → Models fail-fast by design
5. **Mixing TeamGrid and Form state** → They operate independently

## 🔍 Debugging Checklist

- [ ] Dataset exists: `backend/data/Nfl_data_sorted.csv`
- [ ] Models exist: `backend/models/*.joblib` (4 files)
- [ ] Metadata valid: Check `production_ready_win_model` flag
- [ ] Backend healthy: `curl http://localhost:8000/health`
- [ ] Features match: 18 base + 6 differential columns
- [ ] Team codes normalized: Check ABBR_FIX applied correctly