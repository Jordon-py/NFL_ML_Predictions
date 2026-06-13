from pathlib import Path

import pandas as pd

from backend.services import api_runtime as svc


def _write_schedule_csv(path: Path, season: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "season,week,home_team,away_team\n"
        f"{season},1,SEA,NE\n",
        encoding="utf-8",
    )
    return path


def test_find_schedule_paths_returns_list_for_explicit_schedule_path(monkeypatch, tmp_path):
    explicit_path = _write_schedule_csv(tmp_path / "configured" / "Nfl_schedule_2025.csv", 2025)
    packaged_path = _write_schedule_csv(tmp_path / "data" / "Nfl_schedule_2026.csv", 2026)

    monkeypatch.setattr(svc, "SCHEDULE_PATH", explicit_path)
    monkeypatch.setattr(svc, "DATA_DIR", packaged_path.parent)
    monkeypatch.setattr(svc, "BASE_DIR", tmp_path / "backend")

    paths = svc._find_schedule_paths(requested_season=2026)

    assert isinstance(paths, list)
    assert explicit_path in paths
    assert packaged_path in paths
    assert all(isinstance(path, Path) for path in paths)


def test_find_schedule_paths_returns_empty_list_when_no_csvs(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "SCHEDULE_PATH", tmp_path / "missing" / "Nfl_schedule_2026.csv")
    monkeypatch.setattr(svc, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(svc, "BASE_DIR", tmp_path / "backend")

    assert svc._find_schedule_paths() == []


def test_load_schedule_dataframe_returns_full_dataframe_from_fallback(monkeypatch, tmp_path):
    schedule_path = _write_schedule_csv(tmp_path / "data" / "Nfl_schedule_2026.csv", 2026)

    monkeypatch.setattr(svc.nfl, "load_schedules", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(svc, "SCHEDULE_PATH", schedule_path)
    monkeypatch.setattr(svc, "DATA_DIR", schedule_path.parent)
    monkeypatch.setattr(svc, "BASE_DIR", tmp_path / "backend")

    df = svc._load_schedule_dataframe(requested_season=2026)

    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df, tuple)
    assert len(df) == 1


def test_select_schedule_slice_returns_requested_week():
    schedule = pd.DataFrame(
        [
            {"season": 2026, "week": 1, "home_team": "SEA", "away_team": "NE"},
            {"season": 2026, "week": 2, "home_team": "BUF", "away_team": "KC"},
        ]
    )

    week_df, target_season, target_week = svc._select_schedule_slice(schedule, season=2026, week=2)

    assert target_season == 2026
    assert target_week == 2
    assert week_df.iloc[0]["home_team"] == "BUF"
