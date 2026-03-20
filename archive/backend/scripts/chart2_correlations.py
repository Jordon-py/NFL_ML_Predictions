# ==========================================
# File: backend/scripts/chart2_correlations.py
# Role: Backend utility script.
# Input Data: CLI args and input files.
# Output Data: Reports, charts, or artifacts.
# Dependencies: pandas, matplotlib, seaborn, numpy
# Notes: Standalone execution.
# ==========================================

"""
Chart 2: Correlation Heatmap Analysis
=====================================
Visualizes correlations between key features and target variables.
Identifies multicollinearity and feature importance signals.

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

    # Find available dataset
    for filename in ["game_features_20251208.csv", "game_features_20251201.csv", "game_features_2014_2025.csv"]:
        path = data_dir / filename
        if path.exists():
            print(f"📂 Loading: {filename}")
            return pd.read_csv(path), filename.replace('.csv', '')

    raise FileNotFoundError("No game features dataset found")


def get_key_features(df, target_col='home_win'):
    """Select key features for correlation analysis."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Prioritize features by category
    priority_patterns = [
        # Target and direct outcomes (potential leakage)
        'home_win', 'home_points', 'away_points', 'point_diff', 'total_points',
        # Rolling performance features
        'rolling', 'avg', 'mean',
        # Player stats
        'qb_', 'rb_', 'wr_', 'te_', 'def_',
        # Team performance
        'wins', 'losses', 'streak',
        # Efficiency metrics
        'efficiency', 'rate', 'pct', 'percentage'
    ]

    # Score each feature
    scored_features = []
    for col in numeric_cols:
        score = 0
        col_lower = col.lower()
        for i, pattern in enumerate(priority_patterns):
            if pattern in col_lower:
                score = len(priority_patterns) - i
                break

        # Calculate correlation with target if exists
        if target_col in df.columns and col != target_col:
            try:
                corr = abs(df[col].corr(df[target_col]))
                if not np.isnan(corr):
                    score += corr * 10  # Weight correlation highly
            except:
                pass

        scored_features.append((col, score))

    # Sort by score and take top features
    scored_features.sort(key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in scored_features[:25]]

    return top_features


def plot_correlation_heatmap(df, features, dataset_name, output_dir):
    """Generate correlation heatmap for selected features."""
    # Compute correlation matrix
    corr_matrix = df[features].corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Generate heatmap with annotations
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Upper triangle mask

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
        annot_kws={"size": 8}
    )

    # Clean up feature names for display
    clean_labels = [name.replace('_', '\n').replace('player team', '').strip()[:25] for name in features]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(clean_labels, rotation=0, fontsize=9)

    ax.set_title(f'Feature Correlation Matrix - {dataset_name}\n(Lower Triangle)',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = output_dir / f"chart2_correlation_heatmap_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path, corr_matrix


def plot_target_correlations(df, target_col, dataset_name, output_dir):
    """Bar chart of top correlations with target variable."""
    if target_col not in df.columns:
        print(f"⚠ Target column '{target_col}' not found")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    correlations = {}
    for col in numeric_cols:
        try:
            corr = df[col].corr(df[target_col])
            if not np.isnan(corr):
                correlations[col] = corr
        except:
            pass

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(12, 10))

    features = [item[0] for item in sorted_corrs]
    values = [item[1] for item in sorted_corrs]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

    bars = ax.barh(range(len(features)), values, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(range(len(features)))
    # Clean feature names
    clean_names = [f.replace('_', ' ')[:40] for f in features]
    ax.set_yticklabels(clean_names, fontsize=10)
    ax.set_xlabel('Correlation with Home Win', fontsize=12)
    ax.set_title(f'Top 20 Features Correlated with Home Win - {dataset_name}',
                 fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.01 if width > 0 else width - 0.05,
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlim(-1.0, 1.0)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Positive Correlation'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Negative Correlation')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / f"chart2_target_correlations_{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path, sorted_corrs


def identify_multicollinearity(corr_matrix, threshold=0.8):
    """Find highly correlated feature pairs (potential multicollinearity)."""
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })

    return sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)


def generate_insights(corr_matrix, target_correlations, multicollinear_pairs):
    """Generate insights from correlation analysis."""
    insights = []
    insights.append("\n## Chart 2 Insights: Correlation Analysis\n")

    # Top correlations with target
    insights.append("### Top Predictors of Home Win\n")
    for i, (feature, corr) in enumerate(target_correlations[:10], 1):
        direction = "↑ positive" if corr > 0 else "↓ negative"
        clean_name = feature.replace('_', ' ')
        insights.append(f"{i}. **{clean_name}**: r={corr:.3f} ({direction})")

    # Multicollinearity warnings
    if multicollinear_pairs:
        insights.append("\n### ⚠️ Multicollinearity Warnings (|r| ≥ 0.8)\n")
        for pair in multicollinear_pairs[:10]:
            insights.append(f"- {pair['feature1']} ↔ {pair['feature2']}: r={pair['correlation']:.3f}")
        insights.append("\n*Consider removing or combining these highly correlated features*")

    # Data leakage check
    insights.append("\n### 🚨 Potential Data Leakage Features\n")
    leakage_features = [f for f, c in target_correlations if abs(c) > 0.7 and 'point' in f.lower()]
    if leakage_features:
        for f in leakage_features:
            insights.append(f"- **{f}** (correlation > 0.7 with outcome - likely leakage)")
        insights.append("\n*These features should be excluded from training as they're derived from game outcomes*")
    else:
        insights.append("- No obvious point-based leakage features detected in top correlations")

    # Recommendations
    insights.append("\n### Modeling Recommendations\n")
    insights.append("1. **Feature Selection**: Focus on features with |r| > 0.1 and < 0.7 (predictive but not leaky)")
    insights.append("2. **Dimensionality Reduction**: Consider PCA for highly correlated feature groups")
    insights.append("3. **Regularization**: Use L1/L2 regularization to handle multicollinearity")
    insights.append("4. **Validation**: Ensure no future information leaks into training features")

    return "\n".join(insights)


def main():
    print("=" * 60)
    print("CHART 2: CORRELATION HEATMAP ANALYSIS")
    print("=" * 60)

    # Setup output directory
    output_dir = Path(__file__).parent.parent / "reports" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df, dataset_name = load_dataset()
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Get key features
    print("\n🔍 Selecting key features for correlation analysis...")
    key_features = get_key_features(df)
    print(f"✓ Selected {len(key_features)} features")

    # Generate heatmap
    print("\n📊 Generating correlation heatmap...")
    heatmap_path, corr_matrix = plot_correlation_heatmap(df, key_features, dataset_name, output_dir)

    # Generate target correlations bar chart
    print("\n📊 Generating target correlation chart...")
    target_path, target_correlations = plot_target_correlations(df, 'home_win', dataset_name, output_dir)

    # Identify multicollinearity
    print("\n🔍 Checking for multicollinearity...")
    multicollinear_pairs = identify_multicollinearity(corr_matrix)
    print(f"✓ Found {len(multicollinear_pairs)} highly correlated pairs (|r| ≥ 0.8)")

    # Generate insights
    print("\n📝 Generating insights...")
    insights_text = generate_insights(corr_matrix, target_correlations, multicollinear_pairs)
    print(insights_text)

    # Append insights to existing markdown
    insights_path = Path(__file__).parent.parent / "reports" / "dataset_analysis_insights.md"
    with open(insights_path, 'a', encoding='utf-8') as f:
        f.write(insights_text)
    print(f"\n✓ Appended insights to: {insights_path}")

    print("\n" + "=" * 60)
    print("✅ CHART 2 COMPLETE")
    print(f"Charts saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
