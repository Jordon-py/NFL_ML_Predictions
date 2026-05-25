"""
File: backend/tests/test_api_endpoints.py

What it does:
    Exercises the public FastAPI endpoints and focused schedule-selection
    helpers used by the NFL prediction dashboard.

Data shapes:
    - Schedule tests use list/dict rows and pandas DataFrames with
      season/week/home_team/away_team/dt columns.
    - API tests assert JSON arrays or Pydantic-shaped response dictionaries.

Syntax notes:
    - FastAPI TestClient wraps backend.main.app for route-level checks.
    - pytest monkeypatch swaps filesystem paths and offline nflreadpy behavior.

Important tests (line numbers last refreshed 2026-04-30):
    - test_schedule_query_returns_requested_week: around line 104
    - test_default_schedule_prefers_upcoming_season_during_offseason: around line 120
    - test_schedule_loader_reads_newer_packaged_schedule_when_default_is_stale: around line 191

Possible bugs:
    - Tests that use the real app can be slower if startup artifacts are large.
    - nflreadpy network behavior must be monkeypatched for deterministic fallback tests.

Enhancement ideas:
    - Split pure schedule-policy tests into a dedicated test_schedule_policy.py.
    - Add contract tests for frontend-facing schedule metadata if the response grows.
"""

import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from backend import main as main_module
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
        assert r2.status_code in {200, 503}, r2.text
        if r2.status_code == 200:
            resp = r2.json()
            assert "home_score" in resp and "away_score" in resp
            assert 0.0 <= resp["home_win_probability"] <= 1.0
        else:
            detail = r2.json()["detail"]
            assert isinstance(detail, dict)
            assert detail["message"] == "Prediction service unavailable."
            assert detail["blockers"]


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


def test_schedule_query_returns_requested_week():
    with TestClient(app) as client:
        next_week = client.get("/schedule/next-week")
        assert next_week.status_code == 200
        games = next_week.json()
        assert games
        sample = games[0]

        queried = client.get(f"/schedule?season={sample['season']}&week={sample['week']}")
        assert queried.status_code == 200
        queried_games = queried.json()
        assert queried_games
        assert all(int(game["season"]) == int(sample["season"]) for game in queried_games)
        assert all(int(game["week"]) == int(sample["week"]) for game in queried_games)


def test_default_schedule_prefers_upcoming_season_during_offseason():
    now = pd.Timestamp("2026-04-25T12:00:00Z")
    schedule_df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 22,
                "home_team": "PHI",
                "away_team": "KC",
                "dt": pd.Timestamp("2026-02-08T23:30:00Z"),
            },
            {
                "season": 2026,
                "week": 1,
                "home_team": "BUF",
                "away_team": "MIA",
                "dt": pd.NaT,
            },
            {
                "season": 2026,
                "week": 2,
                "home_team": "DAL",
                "away_team": "PHI",
                "dt": pd.NaT,
            },
        ]
    )

    week_df, target_season, target_week = main_module._select_schedule_slice(
        schedule_df,
        now_utc=now,
    )

    assert target_season == 2026
    assert target_week == 1
    assert len(week_df) == 1
    assert week_df.iloc[0]["home_team"] == "BUF"


def test_default_schedule_keeps_future_postseason_before_offseason():
    now = pd.Timestamp("2026-01-10T12:00:00Z")
    schedule_df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 20,
                "home_team": "BAL",
                "away_team": "PIT",
                "dt": pd.Timestamp("2026-01-11T18:00:00Z"),
            },
            {
                "season": 2026,
                "week": 1,
                "home_team": "BUF",
                "away_team": "MIA",
                "dt": pd.Timestamp("2026-09-10T00:20:00Z"),
            },
        ]
    )

    week_df, target_season, target_week = main_module._select_schedule_slice(
        schedule_df,
        now_utc=now,
    )

    assert target_season == 2025
    assert target_week == 20
    assert len(week_df) == 1
    assert week_df.iloc[0]["home_team"] == "BAL"


def test_schedule_loader_reads_newer_packaged_schedule_when_default_is_stale(monkeypatch):
    def offline_load_schedules(*args, **kwargs):
        raise RuntimeError("offline")

    def fake_schedule_paths(requested_season=None):
        return [
            main_module.Path("Nfl_schedule_2025.csv"),
            main_module.Path("Nfl_schedule_2026.csv"),
        ]

    def fake_read_csv(path):
        if str(path).endswith("Nfl_schedule_2025.csv"):
            return pd.DataFrame(
                [
                    {
                        "season": 2025,
                        "week": 22,
                        "gameday": "2026-02-08",
                        "gametime": "18:30",
                        "home_team": "PHI",
                        "away_team": "KC",
                    }
                ]
            )
        if str(path).endswith("Nfl_schedule_2026.csv"):
            return pd.DataFrame(
                [
                    {
                        "season": 2026,
                        "week": 1,
                        "gameday": "2026-09-10",
                        "gametime": "20:20",
                        "home_team": "BUF",
                        "away_team": "MIA",
                    }
                ]
            )
        raise FileNotFoundError(path)

    monkeypatch.setattr(main_module, "_find_schedule_paths", fake_schedule_paths)
    monkeypatch.setattr(main_module.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(main_module.nfl, "load_schedules", offline_load_schedules)

    schedule_df = main_module._load_schedule_dataframe()

    assert set(schedule_df["season"]) == {2025, 2026}

    week_df, target_season, target_week = main_module._select_schedule_slice(
        schedule_df,
        now_utc=pd.Timestamp("2026-04-25T12:00:00Z"),
    )

    assert target_season == 2026
    assert target_week == 1
    assert len(week_df) == 1
    assert week_df.iloc[0]["home_team"] == "BUF"


def test_offseason_status_treats_stale_slates_as_next_season(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "get_schedule",
        lambda: [
            {
                "season": 2025,
                "week": 22,
                "kickoff": "2026-02-08T23:30:00Z",
            }
        ],
    )

    payload = main_module.offseason_status().model_dump()

    assert payload["offseason_mode"] is True
    assert payload["current_season"] == 2025
    assert payload["current_week"] == 22
    assert payload["next_known_schedule_date"] is None
    assert payload["days_until_next_game"] is None


def test_predict_persists_user_scoped_history_and_status_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_store, "PREDICTION_STORE_ROOT", tmp_path / "Predictions" / "users")
    monkeypatch.setattr(sqlite_store, "DB_PATH", tmp_path / "predictions.db")

    with TestClient(app) as client:
        payload = {
            "home_team": "PHI",
            "away_team": "DAL",
            "season": 2025,
            "week": 1,
        }

        headers_a = {"X-User-Id": "analyst@example.com"}
        headers_b = {"X-User-Id": "scout@example.com"}

        monkeypatch.setattr(main_module.state, "model_load_errors", {})
        monkeypatch.setattr(main_module.state, "production_warnings", [])
        monkeypatch.setattr(main_module.state, "production_blockers", [])

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

        summary_a = client.get("/history/summary", headers=headers_a)
        summary_b = client.get("/history/summary", headers=headers_b)
        assert summary_a.status_code == 200
        assert summary_b.status_code == 200
        assert summary_a.json()["total_predictions"] == 1
        assert summary_b.json()["total_predictions"] == 0


def test_history_backfills_final_scores_for_completed_games(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_store, "PREDICTION_STORE_ROOT", tmp_path / "Predictions" / "users")
    monkeypatch.setattr(sqlite_store, "DB_PATH", tmp_path / "predictions.db")

    with TestClient(app) as client:
        payload = {
            "home_team": "PHI",
            "away_team": "DAL",
            "season": 2025,
            "week": 1,
        }
        headers = {"X-User-Id": "historian@example.com"}

        monkeypatch.setattr(main_module.state, "model_load_errors", {})
        monkeypatch.setattr(main_module.state, "production_warnings", [])
        monkeypatch.setattr(main_module.state, "production_blockers", [])

        prediction = client.post("/predict", json=payload, headers=headers)
        assert prediction.status_code == 200, prediction.text

        sqlite_store.upsert_game_scores(
            [
                {
                    # Deliberately use the older away-home shape to prove
                    # backfill works by season/week/team identity too.
                    "game_id": f"{payload['season']}_{payload['week']:02d}_{payload['away_team']}_{payload['home_team']}",
                    "season": payload["season"],
                    "week": payload["week"],
                    "home_team": payload["home_team"],
                    "away_team": payload["away_team"],
                    "home_score": 27,
                    "away_score": 20,
                    "status": "final",
                    "updated_at": "2026-03-25T12:00:00+00:00",
                }
            ]
        )

        history = client.get("/history?limit=5", headers=headers)
        assert history.status_code == 200
        entries = history.json()
        assert len(entries) == 1
        assert entries[0]["final_home_score"] == 27
        assert entries[0]["final_away_score"] == 20
        assert entries[0]["game_status"] == "final"
        assert entries[0]["score_updated_at"] == "2026-03-25T12:00:00+00:00"

        summary = client.get("/history/summary", headers=headers)
        assert summary.status_code == 200
        payload = summary.json()
        assert payload["resolved_games"] == 1
        assert payload["last_score_sync_at"] == "2026-03-25T12:00:00+00:00"
