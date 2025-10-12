"""
BONUS VISUALIZATION: Team Performance Analysis
Hypothesis: Some teams are consistently harder to predict than others
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / 'models/validation_errors.csv')
output_dir = script_dir / 'models/validation_analysis'

# Add correct_prediction column
df['predicted_winner'] = (df['prob_home_win'] > 0.5).astype(int)
df['correct_prediction'] = (df['predicted_winner'] == df['home_win'])

# ============================================================================
# BONUS PLOT: Team-Level Analysis - Which teams are hardest to predict?
# ============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

# Calculate metrics per team (both home and away)
home_stats = df.groupby('home_team').agg({
    'abs_error': 'mean',
    'correct_prediction': 'mean',
    'home_win': 'count'
}).rename(columns={'home_win': 'games_as_home'})

away_stats = df.groupby('away_team').agg({
    'abs_error': 'mean',
    'correct_prediction': 'mean',
    'home_win': 'count'
}).rename(columns={'home_win': 'games_as_away'})

# Combine
team_stats = pd.DataFrame({
    'team': home_stats.index,
    'home_error': home_stats['abs_error'].values,
    'away_error': away_stats['abs_error'].values,
    'home_accuracy': home_stats['correct_prediction'].values * 100,
    'away_accuracy': away_stats['correct_prediction'].values * 100,
    'total_games': home_stats['games_as_home'].values + away_stats['games_as_away'].values
})

team_stats['avg_error'] = (team_stats['home_error'] + team_stats['away_error']) / 2
team_stats['avg_accuracy'] = (team_stats['home_accuracy'] + team_stats['away_accuracy']) / 2
team_stats['error_variance'] = np.abs(team_stats['home_error'] - team_stats['away_error'])

# Sort by average error
team_stats_sorted = team_stats.sort_values('avg_error', ascending=False)

print("\n" + "="*80)
print("TEAM-LEVEL ANALYSIS")
print("="*80)

print("\nMOST DIFFICULT TO PREDICT (Highest Average Error):")
print(team_stats_sorted[['team', 'avg_error', 'avg_accuracy', 'total_games']].head(10).to_string())

print("\nEASIEST TO PREDICT (Lowest Average Error):")
print(team_stats_sorted[['team', 'avg_error', 'avg_accuracy', 'total_games']].tail(10).to_string())

print("\nMOST INCONSISTENT (Home vs Away Error Variance):")
variance_sorted = team_stats.sort_values('error_variance', ascending=False)
print(variance_sorted[['team', 'home_error', 'away_error', 'error_variance']].head(10).to_string())

# Plot 1: Top 15 Hardest to Predict Teams
top_15 = team_stats_sorted.head(15)
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top_15)))
bars = ax1.barh(range(len(top_15)), top_15['avg_error'], color=colors, alpha=0.8, edgecolor='black')
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['team'], fontsize=10, fontweight='bold')
ax1.set_xlabel('Average Prediction Error', fontsize=12, fontweight='bold')
ax1.set_title('15 Hardest Teams to Predict\n(Higher error = more unpredictable)', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# Add accuracy labels
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax1.text(row['avg_error'] + 0.01, i, f"{row['avg_accuracy']:.0f}% acc", 
            va='center', fontsize=9, fontweight='bold')

# Plot 2: Home vs Away Error Comparison
ax2.scatter(team_stats['home_error'], team_stats['away_error'], 
           s=team_stats['total_games']*3, alpha=0.6, c=team_stats['avg_error'],
           cmap='coolwarm', edgecolors='black', linewidth=1)
ax2.plot([0, 0.6], [0, 0.6], 'r--', linewidth=2, label='Equal Error (Home = Away)')
ax2.set_xlabel('Home Error', fontsize=12, fontweight='bold')
ax2.set_ylabel('Away Error', fontsize=12, fontweight='bold')
ax2.set_title('Home vs Away Prediction Error by Team\n(Points above line: worse as away team)', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Annotate outliers
outliers = team_stats.nlargest(5, 'error_variance')
for _, row in outliers.iterrows():
    ax2.annotate(row['team'], xy=(row['home_error'], row['away_error']),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Plot 3: Accuracy Distribution by Team
ax3.hist([team_stats['home_accuracy'], team_stats['away_accuracy']], 
         bins=15, alpha=0.6, label=['Home Accuracy', 'Away Accuracy'],
         color=['#3498DB', '#E74C3C'], edgecolor='black')
ax3.axvline(x=team_stats['home_accuracy'].mean(), color='#3498DB', linestyle='--', 
           linewidth=2, label=f"Mean Home: {team_stats['home_accuracy'].mean():.1f}%")
ax3.axvline(x=team_stats['away_accuracy'].mean(), color='#E74C3C', linestyle='--', 
           linewidth=2, label=f"Mean Away: {team_stats['away_accuracy'].mean():.1f}%")
ax3.set_xlabel('Prediction Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Teams', fontsize=12, fontweight='bold')
ax3.set_title('Team Accuracy Distribution\nAre predictions better for home or away teams?', 
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Error vs Sample Size (Do we predict better with more data?)
ax4.scatter(team_stats['total_games'], team_stats['avg_error'], 
           s=200, alpha=0.6, c=team_stats['avg_accuracy'], cmap='viridis',
           edgecolors='black', linewidth=1)
ax4.set_xlabel('Total Games in Validation Set', fontsize=12, fontweight='bold')
ax4.set_ylabel('Average Prediction Error', fontsize=12, fontweight='bold')
ax4.set_title('Sample Size vs Prediction Quality\n(Are frequently seen teams easier to predict?)', 
              fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add trend line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(team_stats['total_games'], 
                                                          team_stats['avg_error'])
x_trend = np.array([team_stats['total_games'].min(), team_stats['total_games'].max()])
y_trend = slope * x_trend + intercept
ax4.plot(x_trend, y_trend, 'r--', linewidth=2, alpha=0.7,
        label=f'Trend (R²={r_value**2:.3f}, p={p_value:.3f})')
ax4.legend(fontsize=10)

# Add colorbar
cbar = plt.colorbar(ax4.collections[0], ax=ax4)
cbar.set_label('Average Accuracy (%)', fontsize=10)

# Annotate extreme teams
extreme_teams = pd.concat([
    team_stats.nlargest(3, 'avg_error'),
    team_stats.nsmallest(3, 'avg_error')
])
for _, row in extreme_teams.iterrows():
    ax4.annotate(row['team'], xy=(row['total_games'], row['avg_error']),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / 'plot5_team_level_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'plot5_team_level_analysis.png'}")

# ============================================================================
# Additional Team Insights
# ============================================================================
print("\n" + "="*80)
print("TEAM INSIGHTS")
print("="*80)

# Home field advantage by team
home_win_by_team = df.groupby('home_team')['home_win'].mean().sort_values(ascending=False)
print("\nSTRONGEST HOME FIELD ADVANTAGE (Top 10):")
print(home_win_by_team.head(10).apply(lambda x: f"{x:.1%}"))

print("\nWEAKEST HOME FIELD ADVANTAGE (Bottom 10):")
print(home_win_by_team.tail(10).apply(lambda x: f"{x:.1%}"))

# Teams we overestimate vs underestimate
df['prediction_bias'] = df['prob_home_win'] - df['home_win']
bias_home = df.groupby('home_team')['prediction_bias'].mean().sort_values(ascending=False)
print("\nMOST OVERESTIMATED TEAMS (We predict them too high):")
print(bias_home.head(5).apply(lambda x: f"{x:+.3f}"))

print("\nMOST UNDERESTIMATED TEAMS (We predict them too low):")
print(bias_home.tail(5).apply(lambda x: f"{x:+.3f}"))

print("\n" + "="*80)
print("BONUS VISUALIZATION COMPLETE!")
print("="*80)
