# README (CSV-Only Expert Pipeline)

## Goal

Produce **clean, leakage-safe CSVs** that encode strong predictive signals for NFL team-game modeling. No scraping. Data source is nflverse via `nfl-data-py`.

## Outputs

- `team_game_base.csv` — core aggregates + scores and targets.
- `team_game_iter1.csv` — + pace-normalized rates and EPA/TO margins.
- `team_game_iter2.csv` — + rolling 3-game form and league-relative pass rate.
- `team_game_iter3.csv` — + opponent matchup differentials. Use this as the “final” dataset.

## Run

```bash
pip install nfl-data-py pandas numpy
python build_csvs.py --start 2014 --end 2023 --out-dir data
```

Data model
Keys: season, week, team, opponent, game_id, home.

Offense aggregates: plays, yards, pass/rush attempts, TD, INT, fumbles, penalties, EPA sum/mean.

Defense allowed mirrors.

Targets: points_for, points_allowed, point_diff, win.

Feature iterations
Iter 1: normalize for pace and create margins:

off_ypp, off_pr, off_td_rate, off_pen_ypp

def_ypp_allowed, def_td_rate_allowed

to_margin, epa_margin, epa_play_margin

Iter 2: rolling form with shift(1).rolling(3) per team-season and league-relative pass rate:

*_r3, off_pr_over_lg, home_pass_interact

Iter 3: matchup differentials by merging opponent rolling defense:

m_diff_ypp, m_diff_td, m_diff_epa, script_pass_edge

Leakage control
Rolling windows use shift(1) so each game’s features exclude itself.

Time ordering is preserved by team, season, week before transforms.

Extend (optional)
Add early-down splits, 3rd/4th down, red-zone via down/distance fields if retained.

Merge schedule/weather/market lines to inject context features.

Persist a compact data dictionary from the final columns for documentation.

Education notes
GroupBy→Agg converts plays to team-game stats.

Rate features remove pace bias.

Rolling means capture form without peeking.

Opponent diffs model the matchup, which lifts predictive power.
