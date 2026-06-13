"""Compatibility ASGI entrypoint for repo-root launches.

The FastAPI application lives in ``backend.main``. This module re-exports the
app so commands like ``python -m uvicorn main:app --reload`` work when run from
the repository root.
"""

from backend.main import app, create_app

__all__ = ["app", "create_app", "main"]


def main() -> None:
    """Run the FastAPI app directly with sensible local defaults."""
    import os

    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
