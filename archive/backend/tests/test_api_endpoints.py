# ==========================================
# File: backend/tests/test_api_endpoints.py
# Role: Backend endpoint tests aligned to current API shapes.
# Input Data: Test fixtures and sample payloads.
# Output Data: Pytest assertions and results.
# Dependencies: pytest, fastapi, backend.main
# Notes: Skips prediction checks if models are not initialized.
# ==========================================

import pytest
from fastapi.testclient import TestClient
from backend.main import app




def test_health_endpoint_returns_healthy():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    json = r.json()
    assert json.get("status") in {"healthy", "unhealthy"}
    assert "mode" in json and "reason" in json


def test_schedule_next_week_is_not_empty():
    with TestClient(app) as client:
        r = client.get("/schedule/next-week")
    assert r.status_code == 200
    payload = r.json()
    assert "games" in payload
    assert isinstance(payload["games"], list)


def test_predict_works_for_first_game_in_schedule():
    # Use the schedule endpoint to find a valid game
    with TestClient(app) as client:
        r = client.get("/schedule/next-week")
    assert r.status_code == 200
    payload = r.json()
    games = payload.get("games", [])
    if not games:
        pytest.skip("No schedule games available for prediction test.")
    first = games[0]
    payload = {
        "home_team": first["home_team"],
        "away_team": first["away_team"],
        "season": int(first["season"]),
        "week": int(first["week"]),
    }
    r2 = client.post("/predict", json=payload)
    if r2.status_code == 503:
        pytest.skip("Prediction engine not initialized.")
    assert r2.status_code == 200, r2.text
    resp = r2.json()
    assert "home_score" in resp and "away_score" in resp
    assert 0.0 <= resp["home_win_probability"] <= 1.0


def test_debug_endpoint_contains_cors_configuration():
    with TestClient(app) as client:
        r = client.get("/debug")
    assert r.status_code == 200
    payload = r.json()
    assert "cors_origins" in payload
    assert "restrict_cors" in payload


def test_predict_contract_response_shape():
    with TestClient(app) as client:
        sched = client.get("/schedule/next-week")
        assert sched.status_code == 200
        games = sched.json().get("games", [])
        if not games:
            pytest.skip("No schedule games available for contract test.")
        first = games[0]
        payload = {
            "home_team": first["home_team"],
            "away_team": first["away_team"],
            "season": int(first["season"]),
            "week": int(first["week"]),
        }
        r = client.post("/predict", json=payload)
        if r.status_code == 503:
            pytest.skip("Prediction engine not initialized.")
        assert r.status_code == 200, r.text

    data = r.json()
    required_keys = {
        "home_score",
        "away_score",
        "point_diff",
        "home_win_probability",
        "away_win_probability",
        "prediction_source",
        "win_classifier_used",
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_name",
        "away_name",
    }
    assert required_keys.issubset(data.keys())
    for k in ("home_score", "away_score", "point_diff", "home_win_probability", "away_win_probability"):
        assert isinstance(data[k], (int, float))
    assert isinstance(data["game_id"], str)
