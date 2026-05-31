import pandas as pd

from backend import build_csv_datasets_v3 as builder


class _PartialStatsBackend:
    def load_player_stats(self, seasons, summary_level="week"):
        if 2026 in seasons:
            raise RuntimeError("stats_player_week_2026.parquet not found")
        rows = []
        for season in seasons:
            rows.append(
                {
                    "season": season,
                    "week": 1,
                    "recent_team": "KC",
                    "position": "QB",
                    "passing_yards": 280,
                    "passing_tds": 2,
                    "interceptions": 0,
                    "sacks": 1,
                    "completions": 24,
                    "attempts": 32,
                }
            )
        return pd.DataFrame(rows)

    def load_team_stats(self, seasons, summary_level="week"):
        if 2026 in seasons:
            raise RuntimeError("stats_team_week_2026.parquet not found")
        return pd.DataFrame(
            [
                {
                    "season": season,
                    "week": 1,
                    "team": "KC",
                    "points_scored": 24,
                    "points_allowed": 17,
                    "total_yards": 360,
                    "total_yards_allowed": 290,
                    "turnovers": 1,
                    "turnovers_forced": 2,
                }
                for season in seasons
            ]
        )


def test_player_stats_preserve_available_seasons_when_future_season_missing(monkeypatch):
    monkeypatch.setattr(builder, "nfl", _PartialStatsBackend())
    monkeypatch.setattr(builder, "NFL_BACKEND", "nflreadpy")

    out = builder.load_player_game_stats([2025, 2026])

    assert out["season"].tolist() == [2025]
    assert out.loc[0, "team"] == "KC"
    assert out.loc[0, "team_qb_completion_pct"] == 24 / 32


def test_team_stats_preserve_available_seasons_when_future_season_missing(monkeypatch):
    monkeypatch.setattr(builder, "nfl", _PartialStatsBackend())
    monkeypatch.setattr(builder, "NFL_BACKEND", "nflreadpy")

    out = builder.load_team_weekly_stats([2025, 2026])

    assert out["season"].tolist() == [2025]
    assert out.loc[0, "team"] == "KC"
    assert out.loc[0, "points_scored"] == 24
