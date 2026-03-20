from fastapi.testclient import TestClient
from backend.main import app
from backend import prediction_store, sqlite_store




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


def test_debug_endpoint_contains_cors_configuration():
    with TestClient(app) as client:
        r = client.get("/debug")
    assert r.status_code == 200
    payload = r.json()
    assert "cors_origins" in payload
    assert "restrict_cors" in payload


def test_team_logos_endpoint_returns_branding_metadata():
    with TestClient(app) as client:
        r = client.get("/teams/logos")

    assert r.status_code == 200
    payload = r.json()
    assert "teams" in payload
    assert payload["teams"]["BUF"]["name"] == "Buffalo Bills"
    assert payload["teams"]["BUF"]["logoUrl"]


def test_predict_persists_user_scoped_history_and_status_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_store, "PREDICTION_STORE_ROOT", tmp_path / "Predictions" / "users")
    monkeypatch.setattr(sqlite_store, "DB_PATH", tmp_path / "predictions.db")

    with TestClient(app) as client:
        schedule = client.get("/schedule/next-week")
        assert schedule.status_code == 200
        game = schedule.json()[0]
        payload = {
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "season": int(game["season"]),
            "week": int(game["week"]),
        }

        headers_a = {"X-User-Id": "analyst@example.com"}
        headers_b = {"X-User-Id": "scout@example.com"}

        prediction = client.post("/predict", json=payload, headers=headers_a)
        assert prediction.status_code == 200, prediction.text

        history_a = client.get("/history?limit=5", headers=headers_a)
        history_b = client.get("/history?limit=5", headers=headers_b)
        assert history_a.status_code == 200
        assert history_b.status_code == 200

        entries_a = history_a.json()
        entries_b = history_b.json()
        assert len(entries_a) == 1
        assert entries_b == []
        assert entries_a[0]["user_id"] == "analyst@example.com"
        assert entries_a[0]["prediction_source"]
        assert entries_a[0]["ts"]

        overview_a = client.get("/status/overview", headers=headers_a)
        overview_b = client.get("/status/overview", headers=headers_b)
        assert overview_a.status_code == 200
        assert overview_b.status_code == 200
        assert overview_a.json()["history"]["metrics"]["total_predictions"] == 1
        assert overview_b.json()["history"]["metrics"]["total_predictions"] == 0
