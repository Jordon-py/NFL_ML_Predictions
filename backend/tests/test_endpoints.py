import json
import pytest
from fastapi.testclient import TestClient

from backend.main import app


def test_status_overview_and_history():
    with TestClient(app) as client:
        r = client.get("/status/overview")
        assert r.status_code == 200
        j = r.json()
        assert "health" in j and "dataset" in j and "history" in j
        assert "status" in j["health"] and "mode" in j["health"] and "reason" in j["health"]

        r2 = client.get("/history?limit=5")
        assert r2.status_code == 200
        payload = r2.json()
        assert "entries" in payload and "total" in payload
        assert isinstance(payload["entries"], list)


@pytest.mark.parametrize("payload", [
    {"home_team": "BUF", "away_team": "KC", "season": 2025, "week": 1},
    {"home_team": "MIA", "away_team": "BUF", "season": 2025, "week": 7},
])
def test_predict_endpoint(payload):
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        # Accept either a successful prediction or a 503 when models are not loaded
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            j = r.json()
            # basic shape checks
            for k in [
                "home_score",
                "away_score",
                "point_diff",
                "home_win_probability",
                "away_win_probability",
            ]:
                assert k in j
                assert isinstance(j[k], (int, float))


def test_predict_next_week_or_service_unavailable():
    with TestClient(app) as client:
        r = client.get("/predict/next-week")
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            j = r.json()
            assert "games" in j and isinstance(j["games"], list)
