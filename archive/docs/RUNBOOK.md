# Application Runbook

This document provides the essential commands to build, run, and test the application.

## 1. Backend

All commands should be run from the repository root (`NFL_ML_Predictions/`).

### Setup

```powershell
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Deactivate virtual environment when done
deactivate
```

### Running the Server

```powershell
# From the repository root, with the venv active:
cd c:\Users\iProg\OneDrive\Documents\Football_predict\nfl_prediction_system\NFL_ML_Predictions
.\backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Running Tests

```powershell
# From the repository root, with the venv active:
.\backend\.venv\Scripts\python.exe -m pytest backend/
```

### Retraining Models

```powershell
# From the repository root, with the venv active:
.\backend\.venv\Scripts\python.exe backend/train_models.py
```

## 2. Frontend

All commands should be run from the `frontend/` directory.

### Frontend Setup

```powershell
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install
```

### Running the Dev Server

The dev server is configured to proxy API requests to `http://127.0.0.1:8000`.

```powershell
# From the frontend/ directory:
npm run dev
```

### Building for Production

```powershell
# From the frontend/ directory:
npm run build
```
