# Team-Game Dataset Schema

| Column | Dtype | Non-null % | Min | Max | Example | Description |
|---|---|---:|---:|---:|---|---|
| `season` | `int64` | 1.000 | 2014 | 2025 | 2014 |  |
| `week` | `int64` | 1.000 | 1 | 22 | 1 |  |
| `team` | `object` | 1.000 |  |  | ARI |  |
| `opponent` | `object` | 1.000 |  |  | LAC |  |
| `home` | `int64` | 1.000 | 0 | 1 | 1 |  |
| `game_id` | `object` | 1.000 |  |  | 2014_01_SD_ARI |  |
| `off_plays` | `int64` | 1.000 | 50 | 122 | 88 |  |
| `off_yards` | `float64` | 1.000 | 47 | 726 | 403.0 |  |
| `off_pass_att` | `float64` | 1.000 | 3 | 72 | 40.0 |  |
| `off_rush_att` | `float64` | 1.000 | 5 | 54 | 27.0 |  |
| `off_tds` | `float64` | 1.000 | 0 | 10 | 2.0 |  |
| `off_int` | `float64` | 1.000 | 0 | 6 | 0.0 |  |
| `off_fumbles_lost` | `float64` | 1.000 | 0 | 4 | 2.0 |  |
| `off_penalties` | `float64` | 1.000 | 0 | 20 | 8.0 |  |
| `off_qb_spike` | `float64` | 1.000 | 0 | 4 | 0.0 |  |
| `off_qb_kneel` | `float64` | 1.000 | 0 | 6 | 1.0 |  |
| `off_epa_sum` | `float64` | 1.000 | -49.8755 | 47.6611 | -9.108023 |  |
| `off_epa_mean` | `float64` | 1.000 | -0.642944 | 0.529671 | -0.10468992 |  |
| `def_plays_allowed` | `int64` | 1.000 | 50 | 122 | 76 |  |
| `def_yards_allowed` | `float64` | 1.000 | 47 | 726 | 290.0 |  |
| `def_pass_att_allowed` | `float64` | 1.000 | 3 | 72 | 36.0 |  |
| `def_rush_att_allowed` | `float64` | 1.000 | 5 | 54 | 24.0 |  |
| `def_tds_allowed` | `float64` | 1.000 | 0 | 10 | 2.0 |  |
| `def_int_made` | `float64` | 1.000 | 0 | 6 | 1.0 |  |
| `def_fumbles_gained` | `float64` | 1.000 | 0 | 4 | 0.0 |  |
| `def_penalties_committed` | `float64` | 1.000 | 0 | 20 | 3.0 |  |
| `def_epa_allowed_sum` | `float64` | 1.000 | -49.8755 | 47.6611 | -8.843307 |  |
| `def_epa_allowed_mean` | `float64` | 1.000 | -0.642944 | 0.529671 | -0.11635929 |  |
| `home_team` | `object` | 1.000 |  |  | ARI |  |
| `away_team` | `object` | 1.000 |  |  | LAC |  |
| `points_for` | `float64` | 1.000 | 0 | 70 | 18.0 |  |
| `points_allowed` | `float64` | 1.000 | 0 | 70 | 17.0 |  |
| `point_diff` | `float64` | 1.000 | -52 | 50 | 1.0 |  |
| `win` | `int64` | 1.000 | 0 | 1 | 1 |  |
| `team_name` | `object` | 1.000 |  |  | Arizona Cardinals |  |
| `opponent_name` | `object` | 0.967 |  |  | Los Angeles Chargers |  |
| `off_ypp` | `float64` | 1.000 | 0.758065 | 7.80645 | 4.579545454545454 |  |
| `off_pr` | `float64` | 1.000 | 0.0454545 | 0.75 | 0.4545454545454545 |  |
| `off_rr` | `float64` | 1.000 | 0.0740741 | 0.712121 | 0.3068181818181818 |  |
| `off_td_rate` | `float64` | 1.000 | 0 | 0.107527 | 0.0227272727272727 |  |
| `off_pen_ypp` | `float64` | 1.000 | 0 | 0.215054 | 0.0909090909090909 |  |
| `def_ypp_allowed` | `float64` | 1.000 | 0.758065 | 7.80645 | 3.8157894736842106 |  |
| `def_pass_share` | `float64` | 1.000 | 0.0454545 | 0.75 | 0.4736842105263157 |  |
| `def_rush_share` | `float64` | 1.000 | 0.0740741 | 0.712121 | 0.3157894736842105 |  |
| `def_td_rate_allowed` | `float64` | 1.000 | 0 | 0.107527 | 0.0263157894736842 |  |
| `def_pen_ypp` | `float64` | 1.000 | 0 | 0.215054 | 0.0394736842105263 |  |
| `turnovers` | `float64` | 1.000 | 0 | 8 | 2.0 |  |
| `takeaways` | `float64` | 1.000 | 0 | 8 | 1.0 |  |
| `to_margin` | `float64` | 1.000 | -7 | 7 | -1.0 |  |
| `epa_margin` | `float64` | 1.000 | -58.9286 | 58.9286 | -0.26471615 |  |
| `epa_play_margin` | `float64` | 1.000 | -0.721663 | 0.721663 | 0.011669375 |  |
| `off_ypp_r3` | `float64` | 0.937 | 1.7191 | 6.91398 | 4.579545454545454 |  |
| `off_pr_r3` | `float64` | 0.937 | 0.157143 | 0.64557 | 0.4545454545454545 |  |
| `off_td_rate_r3` | `float64` | 0.937 | 0 | 0.0860215 | 0.0227272727272727 |  |
| `off_epa_mean_r3` | `float64` | 0.937 | -0.501044 | 0.518055 | -0.1046899184584617 |  |
| `def_ypp_allowed_r3` | `float64` | 0.937 | 1.7191 | 6.91398 | 3.8157894736842106 |  |
| `def_td_rate_allowed_r3` | `float64` | 0.937 | 0 | 0.0860215 | 0.0263157894736842 |  |
| `def_epa_allowed_mean_r3` | `float64` | 0.937 | -0.501044 | 0.518055 | -0.1163592934608459 |  |
| `to_margin_r3` | `float64` | 0.937 | -5 | 5 | -1.0 |  |
| `epa_play_margin_r3` | `float64` | 0.937 | -0.681352 | 0.681352 | 0.0116693750023841 |  |
| `league_off_pr` | `float64` | 1.000 | 0.387654 | 0.543695 | 0.4445036962121266 |  |
| `off_pr_over_lg` | `float64` | 1.000 | -0.396134 | 0.298785 | 0.0100417583333279 |  |
| `home_pass_interact` | `float64` | 1.000 | 0 | 0.75 | 0.4545454545454545 |  |
| `opp_def_ypp_allowed_r3` | `float64` | 0.906 | 1.7191 | 6.91398 | 5.2375 |  |
| `opp_def_td_rate_allowed_r3` | `float64` | 0.906 | 0 | 0.0860215 | 0.05 |  |
| `opp_def_epa_allowed_mean_r3` | `float64` | 0.906 | -0.501044 | 0.518055 | 0.0809374004602432 |  |
| `opp_off_pr_r3` | `float64` | 0.906 | 0.157143 | 0.64557 | 0.4430379746835443 |  |
| `m_diff_ypp` | `float64` | 0.906 | -3.28274 | 3.39286 | -0.6579545454545457 |  |
| `m_diff_td` | `float64` | 0.906 | -0.0677966 | 0.0595238 | -0.0272727272727272 |  |
| `m_diff_epa` | `float64` | 0.906 | -0.746765 | 0.606495 | -0.185627318918705 |  |
| `script_pass_edge` | `float64` | 0.906 | -0.323333 | 0.323333 | 0.0115074798619102 |  |

> Tip: Add human descriptions over time, then commit.
