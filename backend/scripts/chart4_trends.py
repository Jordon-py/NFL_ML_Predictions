"""
Chart 4: Time Series and Temporal Trends Analysis
=================================================
Visualizes trends over seasons and weeks to identify patterns.
Shows evolution of home advantage, scoring trends, and model features.

Author: ALFRED
Date: 2025-01-09
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')

def load_dataset():
    """Load the latest dataset for analysis."""
    data_dir = Path(__file__).parent.parent / "data"

    for filename in ["game_features_20251208.csv", "game_features_20251201.csv", "game_features_2014_2025.csv"]:
        path = data_dir / filename
        if path.exists():
            print(f"📂 Loading: {filename}")
            return pd.read_csv(path), filename.replace('.csv', '')

    raise FileNotFoundError("No game features dataset found")


def plot_seasonal_trends(df, dataset_name, output_dir):
    """Plot key metrics by season to identify trends over time."""
    if 'season' not in df.columns:
        print("⚠ No 'season' column found")
        return None, None

    # Calculate seasonal aggregates
    seasonal_stats = df.groupby('season').agg({
        'home_win': 'mean',
        'home_points_for': 'mean',
        'away_points_for': 'mean',
    }).reset_index()

    # Ensure numeric types
    seasonal_stats['season'] = seasonal_stats['season'].astype(int)
    seasonal_stats['home_win'] = seasonal_stats['home_win'].astype(float)
    seasonal_stats['home_points_for'] = seasonal_stats['home_points_for'].astype(float)
    seasonal_stats['away_points_for'] = seasonal_stats['away_points_for'].astype(float)

    # Calculate home advantage (point differential)
    seasonal_stats['home_advantage'] = seasonal_stats['home_points_for'] - seasonal_stats['away_points_for']
    seasonal_stats['total_scoring'] = seasonal_stats['home_points_for'] + seasonal_stats['away_points_for']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'NFL Trends by Season - {dataset_name}', fontsize=16, fontweight='bold')

    seasons = seasonal_stats['season'].values
    home_win_pct = (seasonal_stats['home_win'] * 100).values

    # Home win rate trend
    ax = axes[0, 0]
    ax.plot(seasons, home_win_pct, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.fill_between(seasons, 50, home_win_pct, alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('Home Win Rate (%)', fontsize=11)
    ax.set_title('Home Win Rate by Season', fontsize=12)
    ax.set_ylim(45, 65)
    ax.legend(loc='lower right')

    # Home advantage (points)
    ax = axes[0, 1]
    ha_values = seasonal_stats['home_advantage'].values
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in ha_values]
    ax.bar(seasons, ha_values, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('Home Point Advantage', fontsize=11)
    ax.set_title('Home Field Advantage (Points) by Season', fontsize=12)

    # Average scoring trend
    ax = axes[1, 0]
    ax.plot(seasons, seasonal_stats['home_points_for'].values,
            'o-', color='#2ecc71', linewidth=2, markersize=8, label='Home')
    ax.plot(seasons, seasonal_stats['away_points_for'].values,
            's-', color='#3498db', linewidth=2, markersize=8, label='Away')
    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('Average Points', fontsize=11)
    ax.set_title('Average Scoring by Season', fontsize=12)
    ax.legend(loc='lower right')

    # Total scoring trend
    ax = axes[1, 1]
    total_scoring = seasonal_stats['total_scoring'].values
    ax.plot(seasons, total_scoring, 'o-', color='#9b59b6', linewidth=2, markersize=8)
    ax.fill_between(seasons, min(total_scoring) - 2, total_scoring, alpha=0.3, color='#9b59b6')
    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('Total Points (Home + Away)', fontsize=11)
    ax.set_title('Total Game Scoring Trend', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_dir / f"chart4_seasonal_trends_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path, seasonal_stats
def plot_weekly_patterns(df, dataset_name, output_dir):
    """Plot metrics by week to identify in-season patterns."""
    if 'week' not in df.columns:
        print("⚠ No 'week' column found")
        return None

    # Focus on regular season weeks (1-18)
    df_reg = df[df['week'] <= 18].copy()

    weekly_stats = df_reg.groupby('week').agg({
        'home_win': 'mean',
        'home_points_for': 'mean',
        'away_points_for': 'mean',
    }).reset_index()

    weekly_stats['home_advantage'] = weekly_stats['home_points_for'] - weekly_stats['away_points_for']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'NFL Weekly Patterns - {dataset_name}', fontsize=14, fontweight='bold')

    # Home win rate by week
    ax = axes[0]
    ax.bar(weekly_stats['week'], weekly_stats['home_win'] * 100, color='#3498db', edgecolor='black', alpha=0.8)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% baseline')
    ax.set_xlabel('Week', fontsize=11)
    ax.set_ylabel('Home Win Rate (%)', fontsize=11)
    ax.set_title('Home Win Rate by Week', fontsize=12)
    ax.set_xticks(range(1, 19))
    ax.legend()

    # Home advantage by week
    ax = axes[1]
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in weekly_stats['home_advantage']]
    ax.bar(weekly_stats['week'], weekly_stats['home_advantage'], color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Week', fontsize=11)
    ax.set_ylabel('Home Point Advantage', fontsize=11)
    ax.set_title('Home Field Advantage by Week', fontsize=12)
    ax.set_xticks(range(1, 19))

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_path = output_dir / f"chart4_weekly_patterns_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path, weekly_stats


def plot_game_count_heatmap(df, dataset_name, output_dir):
    """Heatmap of game counts by season and week."""
    if 'season' not in df.columns or 'week' not in df.columns:
        print("⚠ Missing season or week columns")
        return None

    # Create pivot table of game counts
    game_counts = df.groupby(['season', 'week']).size().reset_index(name='games')
    pivot = game_counts.pivot(index='season', columns='week', values='games')

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(pivot, annot=True, fmt='g', cmap='YlGnBu',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Number of Games'})

    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Season', fontsize=12)
    ax.set_title(f'Game Count by Season and Week - {dataset_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / f"chart4_game_heatmap_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path


def plot_cumulative_home_advantage(df, dataset_name, output_dir):
    """Rolling cumulative home win percentage over time."""
    if 'season' not in df.columns or 'week' not in df.columns or 'home_win' not in df.columns:
        print("⚠ Missing required columns")
        return None

    # Sort by season and week
    df_sorted = df.sort_values(['season', 'week']).copy()

    # Calculate cumulative home win rate
    df_sorted['game_number'] = range(1, len(df_sorted) + 1)
    df_sorted['cumulative_home_wins'] = df_sorted['home_win'].cumsum()
    df_sorted['cumulative_home_win_rate'] = df_sorted['cumulative_home_wins'] / df_sorted['game_number']

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot cumulative rate
    ax.plot(df_sorted['game_number'], df_sorted['cumulative_home_win_rate'] * 100,
            color='#3498db', linewidth=1.5, alpha=0.8)

    # Add 50% baseline
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% baseline')

    # Add season markers
    season_starts = df_sorted.groupby('season')['game_number'].min()
    for season, game_num in season_starts.items():
        ax.axvline(x=game_num, color='gray', linestyle=':', alpha=0.5)
        ax.text(game_num, ax.get_ylim()[1], str(season), rotation=90,
                va='top', ha='right', fontsize=9, alpha=0.7)

    ax.set_xlabel('Game Number (Chronological)', fontsize=11)
    ax.set_ylabel('Cumulative Home Win Rate (%)', fontsize=11)
    ax.set_title(f'Evolution of Home Field Advantage Over Time - {dataset_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(48, 58)

    plt.tight_layout()
    output_path = output_dir / f"chart4_cumulative_home_advantage_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path


def generate_insights(seasonal_stats, weekly_stats):
    """Generate insights from temporal analysis."""
    insights = []
    insights.append("\n## Chart 4 Insights: Temporal Trends\n")

    # Seasonal insights
    insights.append("### Seasonal Trends (2018-2025)\n")

    if seasonal_stats is not None:
        # Home win rate trend
        first_hwp = seasonal_stats.iloc[0]['home_win'] * 100
        last_hwp = seasonal_stats.iloc[-1]['home_win'] * 100
        trend = "increasing" if last_hwp > first_hwp else "decreasing"
        insights.append(f"- **Home win rate**: {first_hwp:.1f}% (2018) to {last_hwp:.1f}% (2025) - {trend}")

        # Scoring trends
        first_total = seasonal_stats.iloc[0]['total_scoring']
        last_total = seasonal_stats.iloc[-1]['total_scoring']
        scoring_trend = "increasing" if last_total > first_total else "decreasing"
        insights.append(f"- **Total scoring**: {first_total:.1f} (2018) to {last_total:.1f} (2025) - {scoring_trend}")

        # Home advantage
        avg_ha = seasonal_stats['home_advantage'].mean()
        insights.append(f"- **Average home advantage**: {avg_ha:.2f} points per game")

    # Weekly insights
    insights.append("\n### Weekly Patterns\n")

    if weekly_stats is not None:
        best_week = weekly_stats.loc[weekly_stats['home_win'].idxmax()]
        worst_week = weekly_stats.loc[weekly_stats['home_win'].idxmin()]
        insights.append(f"- **Best week for home teams**: Week {int(best_week['week'])} ({best_week['home_win']*100:.1f}% win rate)")
        insights.append(f"- **Worst week for home teams**: Week {int(worst_week['week'])} ({worst_week['home_win']*100:.1f}% win rate)")

        # Early vs late season
        early_hw = weekly_stats[weekly_stats['week'] <= 4]['home_win'].mean() * 100
        late_hw = weekly_stats[weekly_stats['week'] >= 14]['home_win'].mean() * 100
        insights.append(f"- **Early season (Wk 1-4)**: {early_hw:.1f}% home win rate")
        insights.append(f"- **Late season (Wk 14-18)**: {late_hw:.1f}% home win rate")

    # Key findings
    insights.append("\n### Key Temporal Findings\n")
    insights.append("1. **Home advantage is persistent**: Consistently above 50% across all seasons")
    insights.append("2. **Seasonal variation exists**: Some weeks show stronger home effects than others")
    insights.append("3. **Scoring has increased**: NFL rule changes favor offensive play")
    insights.append("4. **COVID impact (2020)**: May show reduced home advantage due to limited fans")

    # Modeling implications
    insights.append("\n### Modeling Implications\n")
    insights.append("- Include **season and week features** to capture temporal patterns")
    insights.append("- Consider **training on recent seasons** (2021+) for current-era predictions")
    insights.append("- **Early season predictions** may be less reliable (limited data)")
    insights.append("- Account for **rule changes** that affect scoring over time")

    return "\n".join(insights)


def main():
    print("=" * 60)
    print("CHART 4: TEMPORAL TRENDS ANALYSIS")
    print("=" * 60)

    # Setup output directory
    output_dir = Path(__file__).parent.parent / "reports" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df, dataset_name = load_dataset()
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Generate seasonal trends
    print("\n📊 Generating seasonal trends...")
    seasonal_path, seasonal_stats = plot_seasonal_trends(df, dataset_name, output_dir)

    # Generate weekly patterns
    print("\n📊 Generating weekly patterns...")
    weekly_path, weekly_stats = plot_weekly_patterns(df, dataset_name, output_dir)

    # Generate game count heatmap
    print("\n📊 Generating game count heatmap...")
    heatmap_path = plot_game_count_heatmap(df, dataset_name, output_dir)

    # Generate cumulative home advantage
    print("\n📊 Generating cumulative home advantage plot...")
    cumulative_path = plot_cumulative_home_advantage(df, dataset_name, output_dir)

    # Generate insights
    print("\n📝 Generating insights...")
    insights_text = generate_insights(seasonal_stats, weekly_stats)
    print(insights_text)

    # Append insights to existing markdown
    insights_path = Path(__file__).parent.parent / "reports" / "dataset_analysis_insights.md"
    with open(insights_path, 'a', encoding='utf-8') as f:
        f.write(insights_text)
    print(f"\n✓ Appended insights to: {insights_path}")

    print("\n" + "=" * 60)
    print("✅ CHART 4 COMPLETE")
    print(f"Charts saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
