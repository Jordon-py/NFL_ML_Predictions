TestClient usage and app startup notes

This project uses FastAPI with an explicit lifespan manager that initializes application state (datasets, models, history) at startup. When writing tests or using TestClient locally, please be aware of the following:

1) The FastAPI lifespan is only executed when the ASGI server runs startup/shutdown, or when TestClient is used as a context manager.

Recommended test patterns

- Preferred: use TestClient as a context manager (runs the app lifespan automatically):

```python
from fastapi.testclient import TestClient
from backend.main import create_app

app = create_app()
with TestClient(app) as client:
    resp = client.get("/schedule/next-week")
    assert resp.status_code == 200
```

- Alternative: initialize the app_state manually (useful in lightweight tests where you prefer not to manage TestClient context):

```python
from fastapi.testclient import TestClient
from backend.main import create_app

app = create_app()
# Manually initialize dataset/models/history before issuing requests
app.state.app_state.initialize()
client = TestClient(app)
resp = client.get("/schedule/next-week")
assert resp.status_code == 200
```

Notes

- The lifespan manager in `backend.main.create_app()` now checks whether the application was already initialized and will skip initialization if so. This prevents double-loading when tests or helpers bootstrap `app.state.app_state` manually.

- If you observe a 503 Service Unavailable from `/schedule/next-week` in tests, it's usually because the app_state initialization has not run. Either use the context manager pattern above or call `app.state.app_state.initialize()` prior to making requests.

- For integration tests that run the real ASGI server (uvicorn), the lifespan runs automatically and no action is required.

Short checklist for CI/test maintainers

- Use `with TestClient(app) as client:` for full-lifecycle tests.
- For lightweight unit tests that don't need startup/shutdown, call `app.state.app_state.initialize()` explicitly.
- If you need auto-initialize behavior in `create_app()`, consider using an environment variable (e.g. AUTO_INITIALIZE) in CI to opt into that behavior.
