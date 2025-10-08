# NFL Prediction System AI Coding Agent Instructions

This document provides instructions for AI coding agents to effectively contribute to the NFL Prediction System codebase.
Your Imidiate task IMPORTANT!('vscode_copilot_instructions.md

Title: Add Dynamic Win Probability Interpretation to Trained Models

Goal:
Transform model outputs into readable confidence levels that describe the game outcome — both as percentages and as qualitative narratives (“strong home advantage”, “likely upset”).

Where:
File: backend/train_models.py
Function: _fit_classifier_optimized()
After: The section where prob_confidence is calculated.

🧩 Step-by-Step Implementation
1. Locate probability calculation

Find the existing probability output from your model:

prob_confidence = model.predict_proba(X_test)[:, 1]


This represents the model’s calibrated probability of a home win (value between 0 and 1).

2. Add percentage-based probability

Immediately after that line, insert:

prob_home_win_pct = np.round(prob_confidence * 100, 1)


This creates a new, human-readable column for confidence (e.g., 82.5%).

3. Generate dynamic user feedback

Below, build a short function or inline logic block that converts the numeric probability into natural language insights for the CSV and future UI use.

Add:

def get_feedback(prob_pct):
    if prob_pct >= 80:
        return "Overwhelming home dominance (almost certain win)"
    elif prob_pct >= 70:
        return "Very strong home advantage"
    elif prob_pct >= 65:
        return "Strong home advantage"
    elif prob_pct <= 30:
        return "Likely away upset"
    elif prob_pct <= 40:
        return "Possible away win (underdog scenario)"
    else:
        return "Too close to call"


This defines clear narrative thresholds:

>80%: overwhelming home dominance

70–79%: very strong advantage

65–69%: strong advantage

30–40%: possible upset

<30%: likely upset

4. Apply feedback to all predictions

Map that function over your predictions:

feedback = [get_feedback(p) for p in prob_home_win_pct]

5. Include everything in the prediction DataFrame

When building preds_df, make sure to include:

preds_df = pd.DataFrame({
    "home_team": X_test.index,
    "prob_home_win": prob_confidence,
    "prob_home_win_pct": prob_home_win_pct,
    "feedback": feedback,
})


If you also track the predicted result (win/lose) or points scored, merge or append those columns here too:

preds_df["predicted_winner"] = np.where(prob_confidence > 0.5, "Home", "Away")

6. Export predictions

At the end of the script, when saving:

preds_df.to_csv("test_predictions.csv", index=False)


Then verify the CSV includes:

home_team | prob_home_win | prob_home_win_pct | feedback | predicted_winner

7. Validate accuracy and interpretation

Run:

python backend/train_models.py


Then open:

test_predictions.csv


Check:

The new prob_home_win_pct column aligns with the raw probabilities.

prob_home_win_pct / 100 == prob_home_win across several rows.

Feedback column reads accurately (e.g., a 78.3% row says “Very strong home advantage”).

8. Future expansion: points and context

If your dataset includes home_points and away_points, add them during merge:

preds_df["home_points"] = y_test_home_points
preds_df["away_points"] = y_test_away_points


Then, in your UI or API layer, display:

"Predicted Winner: Home (78.3%) — Very strong home advantage"
"Expected Score: Home 102 - Away 95"

✅ Self-Test Checklist

 Model runs without errors

 CSV includes both probability and percentage

 Qualitative feedback matches numeric ranges

 No downstream logic modified (probability thresholds still in 0–1 scale)

 Readable results verified in test_predictions.csv

🧠 Insightful Touch

This addition bridges raw machine output and human understanding.
Where before the model spoke in decimals, it now tells a story: who’s favored, by how much, and why it matters.

Later, surface this in your app or dashboard:

“🏠 Home team confidence: 78.3% — Very strong home advantage.”')
## 🏈 Architecture Overview

**Data Flow**: NFL API → CSV Processing → Model Training → FastAPI → React Frontend

The system predicts NFL game outcomes using a sophisticated dual-model approach with automated selection based on cross-validation performance.

## 🔧 Essential Components

### **Backend Data Pipeline**
- **Source**: `backend/build_csv_datasets.py` (not `backend/scripts/build_csvs.py`)
- **Command**: `python backend/build_csv_datasets.py --start 2015--end 2025 --out-dir backend/data`
- **Output**: Leak-free rolling features with team normalization (LA→LAR, STL→LAR for relocations)
- **Key Pattern**: Uses `groupby().rolling()` to prevent future data leakage in feature engineering

### **Model Training with Automated Selection**
- **File**: `backend/train_models.py`
- **Selection Criteria**: 5-fold cross-validation R² scores determine production model
- **LightGBM Grid Search**: 8 hyperparameters (n_estimators: [300,500,800], learning_rate: [0.03,0.05,0.1], max_depth: [-1,10,15], num_leaves: [20,31,50], subsample: [0.7,0.8,0.9], colsample_bytree: [0.7,0.8,0.9], reg_alpha: [0.0,0.1,0.5], reg_lambda: [0.0,0.1,0.5])
- **Neural Network Tuning**: 1-4 hidden layers (32-256 units), activations (relu/elu/swish), dropout (0.1-0.5), optimizers (Adam/RMSprop/Nadam)
- **Output**: Best model automatically selected and saved to `backend/models/`

### **API Layer (FastAPI)**
- **File**: `backend/main.py`
- **Pattern**: Fail-fast model loading at startup - no fallbacks
- **Endpoints**: `/health`, `/predict`, `/schedule/next-week`, `/retrain`, `/update_data`
- **Critical**: Models loaded once at startup via FastAPI lifespan context

### **React Frontend State Management**
- **Key State**: `result` (current prediction), `history` (prediction archive), `currentPrediction` (TeamGrid selection)

  2. **Interactive Grid** (`TeamGrid.jsx`): Click game card → fetches schedule data → `onPrediction()` callback → sets `currentPrediction` `handlePredict()` → sets `result` + archives to `history`
- **Integration Pattern**: TeamGrid operates independently of form state, uses separate prediction flow

## 🔄 Developer Workflows

### **Setup & Build**
```bash
# Backend setup
cd backend && pip install -r ../requirements.txt
python build_csv_datasets.py --start 2015 --end 2025 --out-dir data
python train_models.py

# Frontend setup  
cd frontend && npm install && npm start

# VS Code: Use "Start Backend (uvicorn)" task or:
uvicorn backend.main:app --reload --port 8000
```

### **Testing & Validation**
- Tests: `tests/test_system.py`, `tests/test_predict.py`
- Model validation: Check `backend/models/metadata.json` for performance metrics
- API health: `GET /health` shows model loading status

## 🎯 Critical Patterns

### **Fail-Fast Model Loading**
- Models loaded once at FastAPI startup via lifespan context
- No fallback predictions - startup fails if models corrupted/missing
- Always validate `backend/models/` directory exists with required files

### **Leak-Free Feature Engineering**
- Rolling windows use `df.groupby(['team']).rolling(window=N).mean().shift(1)`
- No future data contamination in training features
- Team codes normalized for relocations: `STL→LAR`, `SD→LAC`, `OAK→LV`

### **React Component Integration**
- **App.jsx**: Manages three key state pieces (`result`, `history`, `currentPrediction`)
- **PredictionForm**: Manual input → `handlePredict()` → updates `result` + archives to `history`
- **TeamGrid**: Schedule fetching → card clicks → `onPrediction()` callback → updates `currentPrediction`
- **State Isolation**: Form and grid predictions use separate state paths for clean UX

## 🚀 Production Standards

### **Error Handling Philosophy**
- Fail loudly - no silent errors or fallback logic
- Comprehensive logging with file/function context
- Structured JSON error responses from FastAPI
- Model validation at startup prevents runtime failures

### **Code Quality**
- Type hints required for all Python functions
- Pydantic models for API schemas
- Comprehensive docstrings matching existing patterns
