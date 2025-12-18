"""
NFL Dataset Deep Analysis Script
Analyzes game_features_20251208.csv and game_features_20251201.csv
Generates insights, metrics, and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
BACKEND_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BACKEND_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_datasets():
    """Load the two latest datasets"""
    df_dec08 = pd.read_csv(BACKEND_DIR / "game_features_20251208.csv")
    df_dec01 = pd.read_csv(BACKEND_DIR / "game_features_20251201.csv")
    return df_dec08, df_dec01

def analyze_dataset_structure(df, name):
    """Analyze basic structure of a dataset"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nData Types:")
    print(df.dtypes.value_counts())

    # Key columns
    key_cols = ['season', 'week', 'home_team', 'away_team', 'home_points_for', 'away_points_for', 'winner']
    print(f"\nKey Columns Present:")
    for col in key_cols:
        status = "✓" if col in df.columns else "✗"
        print(f"  {status} {col}")

    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'object_cols': len(df.select_dtypes(include=['object']).columns),
    }

def analyze_missing_values(df):
    """Analyze missing values in dataset"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    return missing_df

def analyze_numeric_stats(df):
    """Generate statistics for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats['skew'] = numeric_df.skew()
    stats['kurtosis'] = numeric_df.kurtosis()
    return stats

def analyze_target_variables(df):
    """Analyze target variables (scores, winner)"""
    results = {}

    if 'home_points_for' in df.columns and 'away_points_for' in df.columns:
        home_scores = df['home_points_for'].dropna()
        away_scores = df['away_points_for'].dropna()

        results['home_score_mean'] = home_scores.mean()
        results['home_score_std'] = home_scores.std()
        results['away_score_mean'] = away_scores.mean()
        results['away_score_std'] = away_scores.std()

        # Win rates
        if 'winner' in df.columns and 'home_team' in df.columns:
            valid_games = df[(df['home_points_for'].notna()) & (df['away_points_for'].notna())]
            home_wins = (valid_games['winner'] == valid_games['home_team']).sum()
            results['home_win_rate'] = home_wins / len(valid_games) if len(valid_games) > 0 else 0
            results['total_completed_games'] = len(valid_games)

    return results

def analyze_feature_correlations(df, target_col='home_points_for', top_n=20):
    """Find top correlated features with target"""
    numeric_df = df.select_dtypes(include=[np.number])

    if target_col not in numeric_df.columns:
        return pd.DataFrame()

    correlations = numeric_df.corr()[target_col].drop(target_col, errors='ignore')
    correlations = correlations.abs().sort_values(ascending=False)

    return correlations.head(top_n)

def run_full_analysis():
    """Run complete analysis and return insights"""
    print("Loading datasets...")
    df_dec08, df_dec01 = load_datasets()

    insights = {
        'dec08': {},
        'dec01': {},
        'comparison': {}
    }

    # Structure analysis
    insights['dec08']['structure'] = analyze_dataset_structure(df_dec08, "game_features_20251208")
    insights['dec01']['structure'] = analyze_dataset_structure(df_dec01, "game_features_20251201")

    # Missing values
    print("\n--- Missing Values Analysis ---")
    missing_08 = analyze_missing_values(df_dec08)
    missing_01 = analyze_missing_values(df_dec01)
    print(f"\nDec 08 - Columns with missing values: {len(missing_08)}")
    if not missing_08.empty:
        print(missing_08.head(15))
    print(f"\nDec 01 - Columns with missing values: {len(missing_01)}")
    if not missing_01.empty:
        print(missing_01.head(15))

    # Numeric statistics
    print("\n--- Numeric Statistics (Dec 08) ---")
    stats_08 = analyze_numeric_stats(df_dec08)
    print(stats_08[['mean', 'std', 'min', 'max', 'skew']].head(20).round(3))

    # Target analysis
    print("\n--- Target Variable Analysis ---")
    targets_08 = analyze_target_variables(df_dec08)
    targets_01 = analyze_target_variables(df_dec01)
    print(f"\nDec 08 Dataset:")
    for k, v in targets_08.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nDec 01 Dataset:")
    for k, v in targets_01.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    insights['dec08']['targets'] = targets_08
    insights['dec01']['targets'] = targets_01

    # Feature correlations
    print("\n--- Top Feature Correlations with Home Score (Dec 08) ---")
    corr_08 = analyze_feature_correlations(df_dec08)
    print(corr_08.round(3))

    # Season distribution
    if 'season' in df_dec08.columns:
        print("\n--- Season Distribution (Dec 08) ---")
        print(df_dec08['season'].value_counts().sort_index())

    return df_dec08, df_dec01, insights

if __name__ == "__main__":
    df_dec08, df_dec01, insights = run_full_analysis()
    print("\n" + "="*60)
    print("Analysis complete! Ready for visualization.")
    print("="*60)
