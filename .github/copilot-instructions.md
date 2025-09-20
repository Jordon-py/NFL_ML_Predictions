# NFL Prediction System AI Coding Agent Instructions

This document provides instructions for AI coding agents to effectively contribute to the NFL Prediction System codebase.

## 🏈 Architecture Overview

**Data Flow**: NFL API → CSV Processing → Model Training → FastAPI → React Frontend

The system predicts NFL game outcomes using a sophisticated dual-model approach with automated selection based on cross-validation performance.

## 🔧 Essential Components

### **Backend Data Pipeline**
- **Source**: `backend/build_csv_datasets.py` (not `backend/scripts/build_csvs.py`)
- **Command**: `python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data`
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
- **Dual Workflows**:
  1. **Manual Form** (`PredictionForm.jsx`): User enters stats → `handlePredict()` → sets `result` + archives to `history`
  2. **Interactive Grid** (`TeamGrid.jsx`): Click game card → fetches schedule data → `onPrediction()` callback → sets `currentPrediction`
- **Integration Pattern**: TeamGrid operates independently of form state, uses separate prediction flow

## 🔄 Developer Workflows

### **Setup & Build**
```bash
# Backend setup
cd backend && pip install -r ../requirements.txt
python build_csv_datasets.py --start 2010 --end 2025 --out-dir data
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
- Pre-commit hooks: black, isort, flake8, bandit