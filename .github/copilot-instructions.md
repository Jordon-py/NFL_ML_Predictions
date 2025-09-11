# NFL Prediction System AI Coding Agent Instructions

This document provides instructions for AI coding agents to effectively contribute to the NFL Prediction System codebase.

# 🏈 NFL Prediction App — Copilot Agent Finalization & Hardening Directive

## 🎯 Objective
You are acting as a **production-ready AI coding assistant** (e.g., VSCode Copilot) embedded into this repo. Your goal is to:

- Complete all unfinished parts of the codebase.
- Add thorough, centralized logging throughout backend & model code.
- **Remove all fallback logic** — fail loudly so all edge cases and bugs are exposed during development.
- Ensure that the application runs fully end-to-end **with zero silent errors**.
- Surface **all exceptions**, trace logs, and potential failures at runtime.
- Help make the system production-grade in error handling, code clarity, and architecture.

---

## ⚙️ System Architecture Summary

### 📦 Backend (Python + FastAPI)

- **Data pipeline**: `backend/scripts/build_csvs.py`  
  - Fetches data via `nfl-data-py`, generates datasets into `backend/data/`
  - Outputs multiple CSVs for each processing iteration

- **Model training**: `backend/train_models.py`  
  - Trains a Neural Network and a Gradient Boosting Model
  - Saves models + preprocessor to `backend/models/`

- **API**: `backend/main.py`  
  - `/health` - Check service availability  
  - `/predict` - Return prediction results  
  - `/predict_raw` - Return unprocessed model output  
  - `/retrain` - Rebuild models from updated datasets  

> All API endpoints should include proper error handling, try/except with logging, and must not fallback to defaults.

### 💻 Frontend (React)

- React-based interface located in `frontend/`
- Consumes API endpoints and visualizes predictions

---

## 🔄 Developer Workflow (Maintain & Expand)

### 🧪 Dataset Generation
```bash
python backend/scripts/build_csvs.py --start 2014 --end 2025 --out-dir backend/data

🤖 Model Training
python backend/train_models.py

🚀 Run Server (Development)
uvicorn backend.main:app --reload --port 8000

🧩 Enhancement Requirements for Copilot

Logging Standardization

Inject logging or loguru with INFO, WARNING, ERROR levels across all Python files.

No print statements. Logs must include file, function, and message context.

Remove Silent Failures

Remove any except: pass or generic error swallowing.

Replace with raise or detailed logging (with stack trace).

Typed Function Signatures

Use Python type hints for every function and class.

Add docstrings to describe purpose and params clearly.

Validation + Error Tracing

Add strict input validation for API inputs and model predictions.

Return standardized JSON error responses from FastAPI.

Track errors with timestamp, traceback, and correlation ID.

Model File Existence

Ensure startup fails (with logs) if model files are missing or corrupt.

Production-Ready Patterns

Suggest or implement Pydantic models for input/output schemas.

Central error middleware for FastAPI.

Logging middleware for incoming requests.

Catch TODOs & Stubs

Scan for any TODO, # pending, or unimplemented functions.

Suggest complete implementation or prompt developer to resolve.

🧠 Agent Behavior Guidelines

Assume ownership of the backend hardening.

Propose complete code blocks, not just suggestions.

You are responsible for final polish, not scaffolding.

Act like a senior backend dev preparing this for production launch.

🔍 Targets of Attention

backend/main.py

backend/train_models.py

backend/scripts/build_csvs.py

backend/models/ (presence, schema, and handling)

Logging & validation inside backend/

Input schema validation for all FastAPI endpoints

🧠 Begin reviewing files now. Proactively complete, correct, and upgrade the codebase for production.