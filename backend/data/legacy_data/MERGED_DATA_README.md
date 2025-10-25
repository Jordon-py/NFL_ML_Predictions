# NFL Merged Dataset Documentation

## Overview
This dataset combines team-level statistics with aggregated player statistics for enhanced predictive modeling.

**Generated:** 2025-10-12 23:41:42

## Dataset Specifications
- **Rows:** 14,143
- **Columns:** 128
- **Date Range:** 1999 - 2025
- **Weeks:** 1 - 22

## Key Features

### Offensive Metrics
- Passing: yards, TDs, interceptions, EPA
- Rushing: yards, TDs, EPA
- Receiving: yards, TDs, receptions

### Defensive Metrics
- Sacks, interceptions, tackles for loss
- QB hits, fumbles forced

### Special Teams
- Field goals (made/attempted/percentage)
- PATs, returns

### Engineered Features
- `yards_per_attempt`: Passing efficiency
- `yards_per_carry`: Rushing efficiency
- `turnover_differential`: INT differential
- `total_offensive_tds`: Combined TD scoring

## Usage Example
```python
import pandas as pd

# Load merged dataset
df = pd.read_csv('merged_nfl_data.csv')

# Basic filtering
season_2023 = df[df['season'] == 2023]
playoffs = df[df['season_type'] == 'POST']

# Feature selection for ML
predictive_features = [
    'passing_epa_team', 'rushing_epa_team',
    'def_sacks_team', 'turnover_differential'
]
```

## Data Quality
- Completeness: 97.84%
- Missing values handled via aggregation and left join

## Notes
- Player stats aggregated to team-week level
- Team stats represent official team totals
- EPA = Expected Points Added (advanced metric)
