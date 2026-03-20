import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backend.main import app


def _first_game_payload(client: TestClient):
    sched = client.get("/schedule/next-week")
    assert sched.status_code == 200, sched.text
    games = sched.json()
    assert isinstance(games, list)
    if not games:
        return None
    first = games[0]
    return {
        "home_team": first.get("home_abbr") or first.get("home_team"),
        "away_team": first.get("away_abbr") or first.get("away_team"),
        "season": int(first["season"]),
        "week": int(first["week"]),
    }


def test_public_routes_smoke():
    with TestClient(app) as client:
        for path in (
            "/health",
            "/status",
            "/status/overview",
            "/status/runtime",
            "/status/performance-drift?limit=5",
            "/offseason/status",
            "/schedule/next-week",
            "/history?limit=5",
            "/debug",
            "/debug/dataset?limit=3",
            "/api/debug/dataset?limit=3",
        ):
            resp = client.get(path)
            assert resp.status_code in (200, 503), f"{path} -> {resp.status_code}: {resp.text}"

        payload = _first_game_payload(client)
        if payload is None:
            return

        for path in (
            "/predict",
            "/api/predict",
            "/debug/predict-input",
            "/api/debug/predict-input",
        ):
            resp = client.post(path, json=payload)
            assert resp.status_code in (200, 503), f"{path} -> {resp.status_code}: {resp.text}"

