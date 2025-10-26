"""Lightweight fallback app used only for deployment troubleshooting.

This module intentionally avoids heavy imports (pandas, joblib, nfl-data-py) so
the Heroku web process can boot even if the main backend has import-time issues.

Endpoints:
  - / : basic welcome
  - /health : quick status JSON
  - /debug : minimal diagnostic info

Remove or revert this change once the main app is confirmed healthy.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="NFL Prediction Stub", version="0.0.1")


class Health(BaseModel):
    status: str
    timestamp: str


@app.get("/", response_model=dict)
def root():
    return {"message": "stub app running", "time": datetime.utcnow().isoformat()}


@app.get("/health", response_model=Health)
def health():
    return Health(status="healthy", timestamp=datetime.utcnow().isoformat())


@app.get("/debug", response_model=dict)
def debug():
    return {"stub": True, "note": "This is a lightweight stub app for deploy troubleshooting."}
