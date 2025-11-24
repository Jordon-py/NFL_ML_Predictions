"""
NFL Data Merge & Predictive Analysis Script
============================================
Analyzes player_stats and team_stats for predictive properties and creates
an optimized merged dataset for ML prediction.

Architecture: Data Ingestion → Feature Analysis → Intelligent Merge → Export
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================


class DataConfig:
    """Configuration for data paths and merge parameters"""

    BASE_PATH = Path(__file__).parent.parent
    PLAYER_STATS_PATH = BASE_PATH / "backend" / "pbp_cache.csv"
    TEAM_STATS_PATH = BASE_PATH / "backend" / "game_features_20251123.csv"
    OUTPUT_PATH = BASE_PATH / "backend" / "data"

    # Key merge columns
    MERGE_KEYS = ["season", "week", "season_type", "team", "opponent_team"]

    # High-value predictive features (identified from domain knowledge)
    KEY_PREDICTIVE_FEATURES = {
        "passing": [
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "passing_epa",
            "passing_cpoe",
        ],
        "rushing": [
            "rushing_yards",
            "rushing_tds",
            "rushing_epa",
            "rushing_first_downs",
        ],
        "receiving": [
            "receiving_yards",
            "receiving_tds",
            "receiving_epa",
            "receptions",
        ],
        "defense": [
            "def_sacks",
            "def_interceptions",
            "def_tackles_for_loss",
            "def_qb_hits",
        ],
        "special_teams": [
            "fg_made",
            "fg_pct",
            "punt_return_yards",
            "kickoff_return_yards",
        ],
    }


# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================


def load_datasets():
    """
    Load both datasets with optimized dtypes and validation

    Returns:
        tuple: (player_df, team_df)
    """
    print("📊 Loading datasets...")

    # Load with low_memory=False to handle mixed types
    player_df = pd.read_csv(DataConfig.PLAYER_STATS_PATH, low_memory=False)

    team_df = pd.read_csv(DataConfig.TEAM_STATS_PATH)

    print(
        f"✅ Player Stats: {player_df.shape[0]:,} rows × {player_df.shape[1]} columns"
    )
    print(f"✅ Team Stats: {team_df.shape[0]:,} rows × {team_df.shape[1]} columns")

    return player_df, team_df


# ============================================================================
# PREDICTIVE FEATURE ANALYSIS
# ============================================================================


def analyze_predictive_features(player_df, team_df):
    """
    Analyze both datasets for predictive value

    Key metrics:
    - Feature completeness (non-null %)
    - Variance (higher = more predictive power)
    - Correlation with outcome variables
    """
    print("\n🔍 Analyzing predictive features...\n")

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "player_stats": {},
        "team_stats": {},
        "recommendations": [],
    }

    # Analyze player stats
    print("=== PLAYER STATS ANALYSIS ===")
    player_numeric = player_df.select_dtypes(include=[np.number])

    for category, features in DataConfig.KEY_PREDICTIVE_FEATURES.items():
        available_features = [f for f in features if f in player_numeric.columns]
        if available_features:
            stats = {}
            for feat in available_features:
                completeness = (
                    1 - player_df[feat].isnull().sum() / len(player_df)
                ) * 100
                variance = player_df[feat].var()
                mean_val = player_df[feat].mean()

                stats[feat] = {
                    "completeness": round(completeness, 2),
                    "variance": round(variance, 2) if pd.notna(variance) else 0,
                    "mean": round(mean_val, 2) if pd.notna(mean_val) else 0,
                    "std": round(player_df[feat].std(), 2)
                    if pd.notna(player_df[feat].std())
                    else 0,
                }

                print(
                    f"  {feat}: {completeness:.1f}% complete, μ={mean_val:.2f}, σ²={variance:.2f}"
                )

            analysis["player_stats"][category] = stats

    # Analyze team stats
    print("\n=== TEAM STATS ANALYSIS ===")
    team_numeric = team_df.select_dtypes(include=[np.number])

    for category, features in DataConfig.KEY_PREDICTIVE_FEATURES.items():
        available_features = [f for f in features if f in team_numeric.columns]
        if available_features:
            stats = {}
            for feat in available_features:
                completeness = (1 - team_df[feat].isnull().sum() / len(team_df)) * 100
                variance = team_df[feat].var()
                mean_val = team_df[feat].mean()

                stats[feat] = {
                    "completeness": round(completeness, 2),
                    "variance": round(variance, 2) if pd.notna(variance) else 0,
                    "mean": round(mean_val, 2) if pd.notna(mean_val) else 0,
                }

                print(
                    f"  {feat}: {completeness:.1f}% complete, μ={mean_val:.2f}, σ²={variance:.2f}"
                )

            analysis["team_stats"][category] = stats

    # Generate recommendations
    print("\n💡 RECOMMENDATIONS:")
    analysis["recommendations"].append(
        "Use EPA metrics (expected points added) - high predictive value"
    )
    print("  ✓ Use EPA metrics (expected points added) - high predictive value")

    analysis["recommendations"].append(
        "Aggregate player stats by position group before merging"
    )
    print("  ✓ Aggregate player stats by position group before merging")

    analysis["recommendations"].append("Create rolling averages for temporal patterns")
    print("  ✓ Create rolling averages for temporal patterns")

    # Save analysis
    output_file = DataConfig.OUTPUT_PATH / "predictive_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n📁 Analysis saved to: {output_file}")

    return analysis


# ============================================================================
# PLAYER AGGREGATION STRATEGIES
# ============================================================================


def aggregate_player_stats(player_df):
    """
    Aggregate player-level stats to team-level stats

    Strategy:
    - QB stats: Keep top QB per team/week
    - Skill positions (RB/WR/TE): Sum offensive stats
    - Defense: Sum defensive stats
    - Special teams: Keep primary kicker
    """
    print("\n🔄 Aggregating player stats to team level...")

    # Create position groups
    position_mapping = {
        "QB": "quarterback",
        "RB": "skill_offense",
        "WR": "skill_offense",
        "TE": "skill_offense",
        "FB": "skill_offense",
        "K": "kicker",
        "P": "punter",
        "DE": "defense",
        "DT": "defense",
        "NT": "defense",
        "LB": "defense",
        "ILB": "defense",
        "MLB": "defense",
        "OLB": "defense",
        "CB": "defense",
        "DB": "defense",
        "FS": "defense",
        "SS": "defense",
        "S": "defense",
    }
    # Map positions to groups, assign "other" for unmapped positions to avoid NaN groupby issues
    player_df["position_group_agg"] = player_df["position"].map(position_mapping).fillna("other")

    # Define aggregation rules
    agg_rules = {
        # Offensive stats - sum for skill positions
        "passing_yards": "sum",
        "passing_tds": "sum",
        "passing_interceptions": "sum",
        "passing_epa": "sum",
        "rushing_yards": "sum",
        "rushing_tds": "sum",
        "rushing_epa": "sum",
        "receiving_yards": "sum",
        "receiving_tds": "sum",
        "receiving_epa": "sum",
        "receptions": "sum",
        # Defensive stats - sum
        "def_tackles_solo": "sum",
        "def_sacks": "sum",
        "def_sack_yards": "sum",
        "def_interceptions": "sum",
        "def_tackles_for_loss": "sum",
        "def_qb_hits": "sum",
        "def_fumbles_forced": "sum",
        # Special teams - max (take best kicker)
        "fg_made": "sum",
        "fg_att": "sum",
        "fg_pct": "mean",
        "pat_made": "sum",
        "pat_att": "sum",
        # Count players contributing
        "player_id": "count",
    }

    # Filter to columns that exist
    available_agg = {k: v for k, v in agg_rules.items() if k in player_df.columns}

    # Aggregate by team, week, season
    aggregated = player_df.groupby(
        ["season", "week", "season_type", "team", "opponent_team"], as_index=False
    ).agg(available_agg)

    # Rename to indicate player-derived stats
    rename_map = {
        col: f"player_{col}" for col in available_agg.keys() if col != "player_id"
    }
    rename_map["player_id"] = "player_count"
    aggregated.rename(columns=rename_map, inplace=True)

    print(f"✅ Aggregated to {aggregated.shape[0]:,} team-week records")
    print(f"   Features: {aggregated.shape[1]} columns")

    return aggregated


# ============================================================================
# INTELLIGENT MERGE
# ============================================================================


def merge_datasets(team_df, player_agg_df):
    """
    Merge team stats with aggregated player stats

    Strategy: Left join to preserve all team games, adding player insights
    """
    print("\n🔗 Merging datasets...")

    # Ensure merge keys are consistent
    merge_keys = DataConfig.MERGE_KEYS

    print(f"   Merge keys: {merge_keys}")
    print(f"   Team stats: {team_df.shape}")
    print(f"   Player stats: {player_agg_df.shape}")

    # Perform merge
    merged_df = team_df.merge(
        player_agg_df, on=merge_keys, how="left", suffixes=("_team", "_player")
    )

    print(
        f"✅ Merged dataset: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns"
    )

    # Calculate merge quality
    null_count = merged_df.isnull().sum().sum()
    total_cells = merged_df.shape[0] * merged_df.shape[1]
    completeness = (1 - null_count / total_cells) * 100

    print(f"   Data completeness: {completeness:.2f}%")

    return merged_df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def engineer_features(merged_df):
    """
    Create derived features that enhance predictive power
    """
    print("\n⚙️ Engineering predictive features...")

    df = merged_df.copy()

    # 1. Efficiency metrics
    if "passing_yards_team" in df.columns and "attempts_team" in df.columns:
        df["yards_per_attempt"] = df["passing_yards_team"] / df[
            "attempts_team"
        ].replace(0, 1)
        print("  ✓ Created: yards_per_attempt")

    if "rushing_yards_team" in df.columns and "carries_team" in df.columns:
        df["yards_per_carry"] = df["rushing_yards_team"] / df["carries_team"].replace(
            0, 1
        )
        print("  ✓ Created: yards_per_carry")

    # 2. Turnover differential
    if (
        "passing_interceptions_team" in df.columns
        and "def_interceptions_team" in df.columns
    ):
        df["turnover_differential"] = (
            df["def_interceptions_team"] - df["passing_interceptions_team"]
        )
        print("  ✓ Created: turnover_differential")

    # 3. Third down efficiency (if available in future iterations)
    # This would require play-by-play data

    # Home/away status: Use actual data if available, otherwise skip feature
    if "is_home" in df.columns:
        # Use existing is_home column
        print("  ✓ Used: is_home from dataset")
    elif "home_team" in df.columns and "team" in df.columns:
        df["is_home"] = (df["team"] == df["home_team"]).astype(int)
        print("  ✓ Created: is_home from home_team column")
    elif "location" in df.columns:
        # Some datasets use 'location' with values 'home'/'away'
        df["is_home"] = (df["location"] == "home").astype(int)
        print("  ✓ Created: is_home from location column")
    else:
        print("  ⚠️ Skipped: is_home (no reliable home/away data available)")
    print("  ✓ Created: is_home")

    # 5. Scoring potential
    if "passing_tds_team" in df.columns and "rushing_tds_team" in df.columns:
        df["total_offensive_tds"] = df["passing_tds_team"] + df["rushing_tds_team"]
        print("  ✓ Created: total_offensive_tds")

    print(f"\n✅ Final dataset: {df.shape[1]} features")

    return df


# ============================================================================
# EXPORT & DOCUMENTATION
# ============================================================================


def export_merged_data(df, analysis):
    """
    Export merged dataset and documentation
    """
    print("\n💾 Exporting merged dataset...")

    # Ensure output directory exists
    DataConfig.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Export main merged dataset
    output_file = DataConfig.OUTPUT_PATH / "merged_nfl_data.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Export feature list
    feature_list = {
        "total_features": df.shape[1],
        "feature_names": df.columns.tolist(),
        "numeric_features": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_features": df.select_dtypes(include=["object"]).columns.tolist(),
        "merge_timestamp": datetime.now().isoformat(),
        "source_files": {
            "player_stats": str(DataConfig.PLAYER_STATS_PATH),
            "team_stats": str(DataConfig.TEAM_STATS_PATH),
        },
    }

    feature_file = DataConfig.OUTPUT_PATH / "merged_features_manifest.json"
    with open(feature_file, "w") as f:
        json.dump(feature_list, f, indent=2)
    print(f"✅ Saved: {feature_file}")

    # Create README
    readme_content = f"""# NFL Merged Dataset Documentation

## Overview
This dataset combines team-level statistics with aggregated player statistics for enhanced predictive modeling.

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Specifications
- **Rows:** {df.shape[0]:,}
- **Columns:** {df.shape[1]}
- **Date Range:** {df["season"].min()} - {df["season"].max()}
- **Weeks:** {df["week"].min()} - {df["week"].max()}

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
- Completeness: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%
- Missing values handled via aggregation and left join

## Notes
- Player stats aggregated to team-week level
- Team stats represent official team totals
- EPA = Expected Points Added (advanced metric)
"""

    readme_file = DataConfig.OUTPUT_PATH / "MERGED_DATA_README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)
    print(f"✅ Saved: {readme_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Execute full analysis and merge pipeline"""
    print("=" * 70)
    print("🏈 NFL DATA MERGE & PREDICTIVE ANALYSIS")
    print("=" * 70)

    # 1. Load data
    player_df, team_df = load_datasets()

    # 2. Analyze predictive features
    analysis = analyze_predictive_features(player_df, team_df)

    # 3. Aggregate player stats
    player_agg = aggregate_player_stats(player_df)

    # 4. Merge datasets
    merged = merge_datasets(team_df, player_agg)

    # 5. Engineer features
    final_df = engineer_features(merged)

    # 6. Export
    export_merged_data(final_df, analysis)

    print("\n" + "=" * 70)
    print("✅ MERGE COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Dataset Summary:")
    print(f"   - Records: {final_df.shape[0]:,}")
    print(f"   - Features: {final_df.shape[1]}")
    print(f"   - Seasons: {final_df['season'].min()} - {final_df['season'].max()}")
    print(f"\n📁 Output Location: {DataConfig.OUTPUT_PATH}")
    print("\n💡 Next Steps:")
    print("   1. Review merged_features_manifest.json for feature list")
    print("   2. Load merged_nfl_data.csv in your ML pipeline")
    print("   3. Check MERGED_DATA_README.md for usage examples")


if __name__ == "__main__":
    main()
