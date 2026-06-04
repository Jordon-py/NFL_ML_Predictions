"""HTTP route package for the FastAPI backend.

Route modules own URL registration only. Request handling logic lives in
``backend.services`` so the API surface is easy to scan and business workflows
stay reusable from tests, scripts, and future background jobs.
"""

from backend.routes.api import router as api_router

__all__ = ["api_router"]
