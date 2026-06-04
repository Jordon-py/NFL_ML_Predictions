# ==========================================
# File: backend/main.py
# Role: FastAPI application bootstrap for the NFL prediction dashboard.
# Input Data: HTTP requests, process environment, app startup lifecycle.
# Output Data: Mounted FastAPI application at ``backend.main:app``.
# Dependencies: fastapi, backend.routes, backend.services.api_runtime
# Notes: Route declarations live in backend/routes; business workflows live in
# backend/services so this file stays small and deployment-focused.
# ==========================================

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import api_router
from backend.services import api_runtime as runtime


app = FastAPI(lifespan=runtime.lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=runtime.ALLOWED_ORIGINS,
    allow_origin_regex=runtime.ALLOW_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(HTTPException, runtime._http_exception_handler)
app.add_exception_handler(RequestValidationError, runtime._validation_exception_handler)
app.include_router(api_router)


def create_app() -> FastAPI:
    """Return the configured FastAPI application used by tests and ASGI servers."""
    return app


def __getattr__(name: str) -> Any:
    """Backward-compatible access to runtime helpers formerly defined here."""
    return getattr(runtime, name)


logging.info("[App] Routes mounted from backend.routes.api")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
