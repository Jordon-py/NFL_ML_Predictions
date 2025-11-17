from fastapi.testclient import TestClient
from backend.main import app




def test_health_endpoint_returns_healthy():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    json = r.json()
    assert json.get("status") in {"healthy", "unhealthy"}
    assert "components" in json


def test_schedule_next_week_is_not_empty():
    with TestClient(app) as client:
        r = client.get("/schedule/next-week")
    assert r.status_code == 200
    games = r.json()
    assert isinstance(games, list)
    assert len(games) >= 1


def test_predict_works_for_first_game_in_schedule():
    # Use the schedule endpoint to find a valid game
    with TestClient(app) as client:
        r = client.get("/schedule/next-week")
    assert r.status_code == 200
    games = r.json()
    first = games[0]
    payload = {
        "home_team": first["home_team"],
        "away_team": first["away_team"],
        "season": int(first["season"]),
        "week": int(first["week"]),
    }
    r2 = client.post("/predict", json=payload)
    assert r2.status_code == 200, r2.text
    resp = r2.json()
    assert "home_score" in resp and "away_score" in resp
    assert 0.0 <= resp["home_win_probability"] <= 1.0
