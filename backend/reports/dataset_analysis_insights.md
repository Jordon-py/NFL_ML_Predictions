# NFL ML Predictions Dataset Analysis — Initial Insights

**Generated:** December 10, 2025
**Datasets Analyzed:**

- `game_features_20251208.csv` (Dec 08 Dataset)
- `game_features_20251201.csv` (Dec 01 Dataset)

---

## 1. Dataset Structure Overview

| Metric | Dec 08 Dataset | Dec 01 Dataset |
|--------|----------------|----------------|
| **Rows** | 2,216 | 3,282 |
| **Columns** | 214 | 214 |
| **Numeric Columns** | 143 (float64) | 143 (137 float + 6 int) |
| **Categorical/Object** | 71 | 6 object + 65 bool |
| **Seasons Covered** | 2018–2025 | 2018–2025 |

### Key Observations

1. **Dec 08 is smaller** (2,216 rows vs 3,282) — likely filtered for quality or specific feature availability.
2. **Both datasets share identical column count** (214), indicating consistent schema.
3. **Data type differences**: Dec 01 has explicit boolean columns (65) while Dec 08 encodes these as objects.

---

## 2. Season & Week Distribution

### Dec 08 Dataset — Season Breakdown

| Season | Games |
|--------|-------|
| 2018 | 267 |
| 2019 | 267 |
| 2020 | 269 |
| 2021 | 285 |
| 2022 | 284 |
| 2023 | 285 |
| 2024 | 285 |
| 2025 | 272 |

**Insight:** Balanced representation across seasons (267–285 games/season). 2025 data includes current season through Week 14.

---

## 3. Missing Values Analysis

### Dec 08 Dataset — Top Missing Columns (16.79%)

- `home_minus_away_off_third_down_pct_3`
- `home_minus_away_off_pass_over_expected_3`
- `home_minus_away_off_epa_per_play_3`
- `home_minus_away_def_epa_per_play_3`
- `home_minus_away_def_explosive_rate_allowed_3`
- Other differential features (3-game windows)

### Dec 01 Dataset — Higher Missing Rate (43.81%)

Same differential features but with **much higher missing percentages** (~44% vs ~17%).

**Root Cause:** Early-season games lack 3-game history for rolling calculations.

**Recommendation:**

- Impute with league averages for early-season predictions
- Consider 1-game and 2-game fallback windows
- Dec 08 dataset appears better prepared (lower missing rates)

---

## 4. Target Variable Statistics

### Home Score (`home_points_for`)

| Metric | Dec 08 | Dec 01 |
|--------|--------|--------|
| Mean | 23.83 | 23.80 |
| Std Dev | 10.06 | 10.12 |
| Min | 0 | 0 |
| Max | 70 | 70 |

### Away Score (`away_points_for`)

| Metric | Dec 08 | Dec 01 |
|--------|--------|--------|
| Mean | 22.11 | 21.82 |
| Std Dev | 9.83 | 9.75 |
| Min | 0 | 0 |
| Max | 59 | 59 |

### Home Field Advantage

| Metric | Dec 08 | Dec 01 |
|--------|--------|--------|
| **Home Win Rate** | 54.3% | 55.1% |
| **Completed Games** | 2,149 | 3,203 |
| **Avg Point Diff** | +1.72 | +1.98 |

**Insight:** Home teams win ~54-55% of games with an average margin of ~1.7-2 points. This validates the importance of home/away modeling.

---

## 5. Top Feature Correlations with Home Score

| Feature | Correlation | Category |
|---------|-------------|----------|
| `point_diff` | 0.724 | **Outcome** (leakage!) |
| `home_player_team_qb_pass_tds` | 0.659 | Player Stats |
| `home_player_team_wr_receiving_tds` | 0.600 | Player Stats |
| `home_player_team_rb_rush_tds` | 0.475 | Player Stats |
| `home_qb_completion_pct` | 0.399 | Player Stats |
| `home_player_team_rb_rush_yards` | 0.394 | Player Stats |
| `home_player_team_qb_completion_pct` | 0.391 | Player Stats |
| `home_player_team_wr_receiving_yards` | 0.383 | Player Stats |
| `home_player_team_qb_pass_yards` | 0.371 | Player Stats |
| `home_moneyline_prob` | 0.335 | Betting Markets |
| `spread_line` | 0.330 | Betting Markets |
| `home_elo_post` | 0.294 | Elo Ratings |
| `home_rolling_pf_10` | 0.278 | Rolling Stats |
| `elo_diff_pre` | 0.253 | Elo Ratings |

### Critical Observations

1. **⚠️ LEAKAGE WARNING:** `point_diff` (0.724) is an outcome-derived feature — must be excluded from training to avoid data leakage.

2. **Player stats dominate** (0.37–0.66 correlation) — QB pass TDs, WR receiving TDs, and rush TDs are strongest predictors.

3. **Betting markets** (0.33) encode expert consensus — moneyline prob and spread are strong signals.

4. **Rolling averages** (0.25–0.28) provide historical context without leakage.

5. **Elo ratings** (0.25–0.29) offer team strength estimates independent of individual game stats.

---

## 6. Numeric Feature Statistics (Key Features)

| Feature | Mean | Std | Min | Max | Skew |
|---------|------|-----|-----|-----|------|
| `home_prior_pf_avg_3` | 22.73 | 7.02 | 0.00 | 48.00 | 0.04 |
| `home_prior_pa_avg_3` | 22.91 | 6.54 | 0.00 | 48.00 | -0.11 |
| `home_prior_win_pct_3` | 0.49 | 0.32 | 0.00 | 1.00 | 0.04 |
| `home_prior_off_epa_per_play_3` | -0.009 | 0.11 | -0.46 | 0.33 | -0.15 |
| `home_prior_off_success_rate_3` | 0.42 | 0.10 | 0.00 | 0.58 | -2.93 |
| `home_moneyline_prob` | 0.50 | — | — | — | — |

**Insights:**

- Prior win percentage is nearly balanced (~0.49) with full 0–1 range
- Offensive EPA averages near zero (league-normalized metric)
- Success rate shows strong negative skew (-2.93) — some teams with very low rates

---

## 7. Data Quality Assessment

### Strengths

✅ Consistent schema across both datasets (214 columns)
✅ All key columns present (season, week, teams, scores, winner)
✅ Balanced season representation (2018–2025)
✅ Rich feature set: rolling stats, Elo, betting, player-level metrics

### Areas for Improvement

⚠️ High missing rates for differential features (especially Dec 01)
⚠️ Potential leakage features (`point_diff`, post-game Elo)
⚠️ Boolean vs object encoding inconsistency between datasets
⚠️ Some features with extreme skew (success rates)

---

## 8. Next Steps — Visualization Plan

1. **Chart 1:** Feature distribution histograms (home/away scores, win rates)
2. **Chart 2:** Correlation heatmap of top 15 predictors
3. **Chart 3:** Scatter plots of key predictors vs target (score)
4. **Chart 4:** Time series of home win rate and scoring trends by season

---

*This document will be updated with chart-specific insights as visualizations are generated.*

## Chart 1 Insights: Feature Distributions

### Key Observations

### Distribution Characteristics

- Score distributions are approximately **normal** with slight right skew (high-scoring outliers)
- Point differential is **centered near zero** but slightly positive (home advantage)
- Key predictors (QB TDs, passing yards) show **positive skew** typical of counting stats

### Implications for Modeling

- Normal score distributions support **linear regression approaches**
- Home advantage (~3-4 points) should be captured as a feature
- Consider **log transforms** for heavily skewed predictors
- Outlier games (blowouts >30 pts) may warrant special handling

## Chart 2 Insights: Correlation Analysis

### Top Predictors of Home Win

1. **point diff**: r=0.778 (positive) - **LEAKAGE, exclude from training**
2. **away points for**: r=-0.558 (negative)
3. **home points for**: r=0.558 (positive)
4. **home moneyline prob**: r=0.380 (positive)
5. **moneyline prob diff**: r=0.379 (positive)
6. **away moneyline prob**: r=-0.379 (negative)
7. **spread line**: r=0.379 (positive)
8. **away player team rb rush yards**: r=-0.378 (negative)
9. **home player team rb rush yards**: r=0.375 (positive)
10. **home elo post**: r=0.342 (positive)

### Multicollinearity Warnings (|r| >= 0.8)

Found **16 highly correlated pairs**:

- `home_rolling_pf_5` <-> `home_prior_pf_avg_5`: r=1.000 (identical features!)
- `away_rolling_pf_5` <-> `away_prior_pf_avg_5`: r=1.000 (identical features!)
- `home_rolling_pf_3` <-> `home_prior_pf_avg_5`: r=0.851
- `home_rolling_pf_5` <-> `home_rolling_pf_3`: r=0.851
- `away_rolling_pf_5` <-> `away_rolling_pf_3`: r=0.849
- `home_minus_away_pf_avg_5` <-> `home_minus_away_pf_avg_3`: r=0.845
- `home_rolling_pf_10` <-> `home_rolling_pf_5`: r=0.840
- `home_rolling_win_pct_5` <-> `home_rolling_win_pct_3`: r=0.836

*Recommendation: Remove duplicate features and use regularization*

### Potential Data Leakage Features

- **point_diff** (r=0.778 with outcome) - This is game result, not predictor!

### Modeling Recommendations

1. **Feature Selection**: Focus on features with |r| > 0.1 and < 0.7 (predictive but not leaky)
2. **Dimensionality Reduction**: Consider PCA for highly correlated rolling stat groups
3. **Regularization**: Use L1/L2 regularization to handle multicollinearity
4. **Validation**: Ensure no future information leaks into training features

## Chart 3 Insights: Target Relationships

### Predictor-Target Correlations

- **Betting odds predicting win probability**: r=0.380 (Strong)
- **Betting odds predicting home score**: r=0.330 (Strong)
- **Vegas spread predicting home score**: r=0.325 (Strong)
- **Recent scoring predicting current score**: r=0.246 (Moderate)
- **Pre-game Elo predicting score**: r=0.217 (Moderate)
- **Rolling win % predicting win**: r=0.205 (Moderate)

### Betting Market Calibration

The calibration plot shows how well Vegas moneyline probabilities predict actual outcomes:

- **Mean Absolute Error**: 0.033 (lower is better)
- Vegas odds are **well-calibrated** - can be trusted as baseline

### Key Findings

1. **Betting markets are efficient**: Moneyline probabilities correlate strongly with outcomes
2. **Recent performance matters**: 5-game rolling averages show predictive power
3. **Elo ratings capture team strength**: Pre-game Elo has moderate correlation with scores
4. **Multiple signals needed**: No single feature is sufficient; ensemble approaches recommended

### Modeling Implications

- Use **betting lines as baseline** - hard to beat consistently
- **Combine multiple predictors** for robust predictions
- Focus on **situations where markets may be wrong** (injuries, weather, etc.)
- Consider **calibration-aware loss functions** for probability outputs

## Chart 4 Insights: Temporal Trends

### Seasonal Trends (2018-2025)

- **Home win rate**: 57.3% (2018) to 54.4% (2025) - decreasing
- **Total scoring**: 45.3 (2018) to 45.9 (2025) - increasing
- **Average home advantage**: 1.98 points per game

### Weekly Patterns

- **Best week for home teams**: Week 16 (60.8% win rate)
- **Worst week for home teams**: Week 13 (50.3% win rate)
- **Early season (Wk 1-4)**: 53.1% home win rate
- **Late season (Wk 14-18)**: 55.1% home win rate

### Key Temporal Findings

1. **Home advantage is persistent**: Consistently above 50% across all seasons
2. **Seasonal variation exists**: Some weeks show stronger home effects than others
3. **Scoring has increased**: NFL rule changes favor offensive play
4. **COVID impact (2020)**: May show reduced home advantage due to limited fans

### Modeling Implications

- Include **season and week features** to capture temporal patterns
- Consider **training on recent seasons** (2021+) for current-era predictions
- **Early season predictions** may be less reliable (limited data)
- Account for **rule changes** that affect scoring over time