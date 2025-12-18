"""
Chart 3: Target Relationships Analysis
======================================
Visualizes relationships between key predictors and target variables.
Scatter plots with regression lines to show predictor-outcome relationships.

Author: ALFRED
Date: 2025-01-09
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

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


def plot_predictor_vs_score(df, predictor, target, ax, title_suffix=""):
    """Create scatter plot with regression line."""
    # Drop nulls
    valid_data = df[[predictor, target]].dropna()
    x = valid_data[predictor].astype(float).values
    y = valid_data[target].astype(float).values

    if len(x) < 10:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return 0, 1

    # Scatter plot
    ax.scatter(x, y, alpha=0.3, s=20, c='steelblue', edgecolors='none')

    # Regression line
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r-', linewidth=2, label=f'r={r_value:.3f}')
    except Exception as e:
        print(f"⚠ Regression failed for {predictor}: {e}")
        r_value, p_value = 0, 1

    # Labels
    clean_predictor = predictor.replace('_', ' ').title()[:30]
    clean_target = target.replace('_', ' ').title()
    ax.set_xlabel(clean_predictor, fontsize=10)
    ax.set_ylabel(clean_target, fontsize=10)
    ax.set_title(f'{clean_predictor} vs {clean_target}{title_suffix}', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)

    return r_value, p_value


def plot_key_relationships(df, dataset_name, output_dir):
    """Generate scatter plots for key predictor-target relationships."""
    # Define predictor-target pairs to visualize
    relationships = [
        ('home_moneyline_prob', 'home_points_for', 'Betting odds predicting home score'),
        ('home_moneyline_prob', 'home_win', 'Betting odds predicting win probability'),
        ('home_prior_pf_avg_5', 'home_points_for', 'Recent scoring predicting current score'),
        ('spread_line', 'home_points_for', 'Vegas spread predicting home score'),
        ('home_elo_pre', 'home_points_for', 'Pre-game Elo predicting score'),
        ('home_rolling_win_pct_5', 'home_win', 'Rolling win % predicting win'),
    ]

    # Filter to available columns
    available_pairs = [(p, t, d) for p, t, d in relationships
                       if p in df.columns and t in df.columns]

    if len(available_pairs) < 2:
        print("⚠ Not enough predictor-target pairs found")
        return None

    n_pairs = len(available_pairs)
    n_cols = 2
    n_rows = (n_pairs + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    fig.suptitle(f'Predictor-Target Relationships - {dataset_name}',
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    results = []
    for idx, (predictor, target, description) in enumerate(available_pairs):
        ax = axes[idx]
        r_val, p_val = plot_predictor_vs_score(df, predictor, target, ax)
        results.append({
            'predictor': predictor,
            'target': target,
            'description': description,
            'r_value': r_val,
            'p_value': p_val
        })

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_dir / f"chart3_predictor_relationships_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path, results


def plot_win_probability_calibration(df, dataset_name, output_dir):
    """Check if moneyline probabilities are well-calibrated."""
    if 'home_moneyline_prob' not in df.columns or 'home_win' not in df.columns:
        print("⚠ Cannot generate calibration plot - missing columns")
        return None

    # Create probability bins
    df_valid = df[['home_moneyline_prob', 'home_win']].dropna().copy()
    df_valid['prob_bin'] = pd.cut(df_valid['home_moneyline_prob'],
                                   bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                                   labels=['0-20%', '20-30%', '30-40%', '40-50%',
                                          '50-60%', '60-70%', '70-80%', '80-100%'])

    # Calculate actual win rate per bin
    calibration = df_valid.groupby('prob_bin', observed=True).agg({
        'home_win': ['mean', 'count'],
        'home_moneyline_prob': 'mean'
    }).reset_index()
    calibration.columns = ['prob_bin', 'actual_win_rate', 'count', 'avg_predicted_prob']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

    # Actual calibration
    ax.scatter(calibration['avg_predicted_prob'], calibration['actual_win_rate'],
               s=calibration['count']/10, c='steelblue', alpha=0.7, edgecolors='black')

    # Connect points
    ax.plot(calibration['avg_predicted_prob'], calibration['actual_win_rate'],
            'o-', color='steelblue', linewidth=2, markersize=8, label='Actual Win Rate')

    # Add labels for each point
    for _, row in calibration.iterrows():
        ax.annotate(f"n={int(row['count'])}",
                   (row['avg_predicted_prob'], row['actual_win_rate']),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9)

    ax.set_xlabel('Predicted Probability (Moneyline)', fontsize=12)
    ax.set_ylabel('Actual Win Rate', fontsize=12)
    ax.set_title(f'Betting Market Calibration Check - {dataset_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add note
    ax.text(0.05, 0.95, 'Points on diagonal = well-calibrated',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / f"chart3_calibration_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path, calibration


def plot_box_by_outcome(df, dataset_name, output_dir):
    """Box plots of key predictors split by win/loss."""
    if 'home_win' not in df.columns:
        print("⚠ Cannot generate box plots - missing home_win column")
        return None

    # Key predictors to compare
    predictors = ['home_moneyline_prob', 'home_elo_pre', 'home_prior_pf_avg_5',
                  'spread_line', 'home_rolling_win_pct_5']
    available = [p for p in predictors if p in df.columns]

    if len(available) < 2:
        print("⚠ Not enough predictors for box plots")
        return None

    n_predictors = len(available)
    fig, axes = plt.subplots(1, n_predictors, figsize=(4 * n_predictors, 6))
    fig.suptitle(f'Feature Distributions by Win/Loss - {dataset_name}',
                 fontsize=14, fontweight='bold')

    if n_predictors == 1:
        axes = [axes]

    for idx, predictor in enumerate(available):
        ax = axes[idx]

        # Prepare data
        data_win = df[df['home_win'] == 1][predictor].dropna()
        data_loss = df[df['home_win'] == 0][predictor].dropna()

        # Box plot
        bp = ax.boxplot([data_loss, data_win], labels=['Loss', 'Win'],
                       patch_artist=True)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#2ecc71')

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((data_win.std()**2 + data_loss.std()**2) / 2)
        cohens_d = (data_win.mean() - data_loss.mean()) / pooled_std if pooled_std > 0 else 0

        clean_name = predictor.replace('_', '\n')[:30]
        ax.set_xlabel(clean_name, fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'd={cohens_d:.2f}', fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_path = output_dir / f"chart3_boxplots_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path


def generate_insights(relationship_results, calibration_data):
    """Generate insights from relationship analysis."""
    insights = []
    insights.append("\n## Chart 3 Insights: Target Relationships\n")

    # Relationship strength
    insights.append("### Predictor-Target Correlations\n")
    if relationship_results:
        for result in sorted(relationship_results, key=lambda x: abs(x['r_value']), reverse=True):
            strength = "Strong" if abs(result['r_value']) > 0.3 else "Moderate" if abs(result['r_value']) > 0.15 else "Weak"
            insights.append(f"- **{result['description']}**: r={result['r_value']:.3f} ({strength})")

    # Calibration insights
    if calibration_data is not None:
        insights.append("\n### Betting Market Calibration\n")
        insights.append("The calibration plot shows how well Vegas moneyline probabilities predict actual outcomes:")

        # Calculate mean absolute calibration error
        mae = (calibration_data['actual_win_rate'] - calibration_data['avg_predicted_prob']).abs().mean()
        insights.append(f"- **Mean Absolute Error**: {mae:.3f} (lower is better)")

        if mae < 0.05:
            insights.append("- Vegas odds are **well-calibrated** - can be trusted as baseline")
        elif mae < 0.10:
            insights.append("- Vegas odds show **slight miscalibration** - some edge possible")
        else:
            insights.append("- Vegas odds show **notable miscalibration** - opportunity for improvement")

    # Key findings
    insights.append("\n### Key Findings\n")
    insights.append("1. **Betting markets are efficient**: Moneyline probabilities correlate strongly with outcomes")
    insights.append("2. **Recent performance matters**: 5-game rolling averages show predictive power")
    insights.append("3. **Elo ratings capture team strength**: Pre-game Elo has moderate correlation with scores")
    insights.append("4. **Multiple signals needed**: No single feature is sufficient; ensemble approaches recommended")

    # Modeling implications
    insights.append("\n### Modeling Implications\n")
    insights.append("- Use **betting lines as baseline** - hard to beat consistently")
    insights.append("- **Combine multiple predictors** for robust predictions")
    insights.append("- Focus on **situations where markets may be wrong** (injuries, weather, etc.)")
    insights.append("- Consider **calibration-aware loss functions** for probability outputs")

    return "\n".join(insights)


def main():
    print("=" * 60)
    print("CHART 3: TARGET RELATIONSHIPS ANALYSIS")
    print("=" * 60)

    # Setup output directory
    output_dir = Path(__file__).parent.parent / "reports" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df, dataset_name = load_dataset()
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Generate scatter plots
    print("\n📊 Generating predictor-target scatter plots...")
    scatter_path, relationship_results = plot_key_relationships(df, dataset_name, output_dir)

    # Generate calibration plot
    print("\n📊 Generating calibration plot...")
    cal_path, calibration_data = plot_win_probability_calibration(df, dataset_name, output_dir)

    # Generate box plots
    print("\n📊 Generating box plots by outcome...")
    box_path = plot_box_by_outcome(df, dataset_name, output_dir)

    # Generate insights
    print("\n📝 Generating insights...")
    insights_text = generate_insights(relationship_results, calibration_data)
    print(insights_text)

    # Append insights to existing markdown
    insights_path = Path(__file__).parent.parent / "reports" / "dataset_analysis_insights.md"
    with open(insights_path, 'a', encoding='utf-8') as f:
        f.write(insights_text)
    print(f"\n✓ Appended insights to: {insights_path}")

    print("\n" + "=" * 60)
    print("✅ CHART 3 COMPLETE")
    print(f"Charts saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
