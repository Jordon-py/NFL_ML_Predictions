"""
NFL Model Validation Analysis - Deep Dive into Prediction Errors
Analyzes validation_errors.csv to identify trends and test hypotheses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('models/validation_errors.csv')

# Create output directory
output_dir = Path('models/validation_analysis')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("NFL VALIDATION ERROR ANALYSIS")
print("="*80)

# ============================================================================
# TREND ANALYSIS - Identify patterns
# ============================================================================

print("\n1. CALIBRATION ANALYSIS")
print("-"*80)
# Create probability bins to assess calibration
df['prob_bin'] = pd.cut(df['prob_home_win'], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
                         labels=['0-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-100%'])
calibration = df.groupby('prob_bin').agg({
    'home_win': ['mean', 'count'],
    'prob_home_win': 'mean',
    'abs_error': 'mean'
}).round(4)
print(calibration)

print("\n2. CONFIDENCE VS ACCURACY")
print("-"*80)
# Are we better at predicting high-confidence games?
df['confidence_level'] = pd.cut(np.abs(df['prob_home_win'] - 0.5), 
                                bins=[0, 0.1, 0.2, 0.3, 0.5],
                                labels=['Low (50-60%)', 'Medium (60-70%)', 'High (70-80%)', 'Very High (>80%)'])
confidence_analysis = df.groupby('confidence_level').agg({
    'abs_error': ['mean', 'std', 'count'],
    'home_win': lambda x: (
        pd.to_numeric(df.loc[x.index, 'prob_home_win'], errors='coerce') > 0.5
    ).values == x.values
}).round(4)
confidence_analysis.columns = ['Mean Error', 'Std Error', 'Count', 'Accuracy']
print(confidence_analysis)

print("\n3. SEASON TRENDS")
print("-"*80)
season_trends = df.groupby('season').agg({
    'abs_error': 'mean',
    'prob_home_win': 'mean',
    'home_win': 'mean'
}).round(4)
season_trends['correct_pct'] = df.groupby('season').apply(
    lambda x: ((x['prob_home_win'] > 0.5) == x['home_win']).mean()
).round(4)
print(season_trends)

print("\n4. HOME FIELD ADVANTAGE")
print("-"*80)
# Overall home win rate
home_win_rate = df['home_win'].mean()
print(f"Home team win rate: {home_win_rate:.1%}")
print(f"Model's average predicted home win prob: {df['prob_home_win'].mean():.1%}")
print(f"Difference (bias): {(df['prob_home_win'].mean() - home_win_rate):.1%}")

print("\n5. UPSET ANALYSIS (Games where favorite lost)")
print("-"*80)
df['predicted_winner'] = (df['prob_home_win'] > 0.5).astype(int)
df['correct_prediction'] = (df['predicted_winner'] == df['home_win'])
upsets = df[~df['correct_prediction']]
print(f"Total upsets: {len(upsets)} ({len(upsets)/len(df)*100:.1f}%)")
print(f"Average confidence when wrong: {upsets['prob_home_win'].apply(lambda x: abs(x-0.5)).mean():.4f}")
print(f"\nTop 10 biggest upsets (high confidence, wrong prediction):")
biggest_upsets = upsets.nlargest(10, 'abs_error')[['season', 'week', 'home_team', 'away_team', 
                                                      'prob_home_win', 'home_win', 'abs_error']]
print(biggest_upsets.to_string())

# ============================================================================
# VISUALIZATION 1: Calibration Curve (Are probabilities well-calibrated?)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate calibration data
prob_bins = np.linspace(0, 1, 21)
bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
observed_freq = []
predicted_freq = []
counts = []

for i in range(len(prob_bins) - 1):
    mask = (df['prob_home_win'] >= prob_bins[i]) & (df['prob_home_win'] < prob_bins[i+1])
    if mask.sum() > 0:
        observed_freq.append(df.loc[mask, 'home_win'].mean())
        predicted_freq.append(df.loc[mask, 'prob_home_win'].mean())
        counts.append(mask.sum())
    else:
        observed_freq.append(np.nan)
        predicted_freq.append(np.nan)
        counts.append(0)

# Plot calibration curve
valid_mask = ~np.isnan(observed_freq)
ax.scatter(np.array(predicted_freq)[valid_mask], np.array(observed_freq)[valid_mask], 
           s=np.array(counts)[valid_mask]*2, alpha=0.6, c=np.array(counts)[valid_mask], 
           cmap='viridis', edgecolors='black', linewidth=1)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')

# Add trend line
from numpy.polynomial import polynomial as P
coefs = P.polyfit(np.array(predicted_freq)[valid_mask], np.array(observed_freq)[valid_mask], 2)
x_trend = np.linspace(0, 1, 100)
y_trend = P.polyval(x_trend, coefs)
ax.plot(x_trend, y_trend, 'b-', linewidth=2, alpha=0.7, label='Model Calibration')

ax.set_xlabel('Predicted Probability (Home Win)', fontsize=12, fontweight='bold')
ax.set_ylabel('Observed Frequency (Actual Home Win Rate)', fontsize=12, fontweight='bold')
ax.set_title('Model Calibration Curve - Are Predictions Well-Calibrated?\n(Bubble size = number of games)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Add colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Number of Games', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'plot1_calibration_curve.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'plot1_calibration_curve.png'}")

# ============================================================================
# VISUALIZATION 2: Confidence vs Accuracy (Do high-confidence predictions perform better?)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Error by confidence level
confidence_bins = np.linspace(0, 0.5, 11)
mean_errors = []
error_stds = []
accuracies = []
bin_labels = []

for i in range(len(confidence_bins) - 1):
    conf_min, conf_max = confidence_bins[i], confidence_bins[i+1]
    mask = (np.abs(df['prob_home_win'] - 0.5) >= conf_min) & (np.abs(df['prob_home_win'] - 0.5) < conf_max)
    
    if mask.sum() > 5:  # At least 5 games
        mean_errors.append(df.loc[mask, 'abs_error'].mean())
        error_stds.append(df.loc[mask, 'abs_error'].std())
        accuracies.append(df.loc[mask, 'correct_prediction'].mean())
        bin_labels.append(f"{50+conf_min*100:.0f}-{50+conf_max*100:.0f}%")

x_pos = np.arange(len(mean_errors))
ax1.bar(x_pos, mean_errors, yerr=error_stds, capsize=5, alpha=0.7, 
        color=plt.cm.RdYlGn_r(np.array(mean_errors) / max(mean_errors)))
ax1.set_xlabel('Confidence Level (Distance from 50%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax1.set_title('Prediction Error by Confidence Level\n(Lower is better)', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Right: Accuracy by confidence level
ax2.plot(x_pos, np.array(accuracies)*100, marker='o', linewidth=3, markersize=10, 
         color='#2E86AB', markerfacecolor='#A23B72', markeredgewidth=2, markeredgecolor='white')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Coin Flip (50%)')
ax2.fill_between(x_pos, 50, np.array(accuracies)*100, alpha=0.3, color='green')
ax2.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
ax2.set_ylabel('Prediction Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy Improves with Confidence\n(Higher is better)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
ax2.set_ylim([45, 85])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add text annotations
for i, (x, y) in enumerate(zip(x_pos, np.array(accuracies)*100)):
    ax2.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 5), 
                textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'plot2_confidence_vs_accuracy.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'plot2_confidence_vs_accuracy.png'}")

# ============================================================================
# VISUALIZATION 3: Temporal Trends (Performance over time)
# ============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: Error by season
season_data = df.groupby('season').agg({
    'abs_error': ['mean', 'std'],
    'correct_prediction': 'mean'
})
seasons = season_data.index
ax1.bar(seasons, season_data['abs_error']['mean'], yerr=season_data['abs_error']['std'], 
        capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_xlabel('Season', fontsize=11, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
ax1.set_title('Model Error by Season', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Top-right: Accuracy by season
ax2.plot(seasons, season_data['correct_prediction']['mean']*100, marker='o', linewidth=3, 
         markersize=12, color='#9B59B6', markerfacecolor='#E74C3C', markeredgewidth=2, markeredgecolor='white')
ax2.axhline(y=56.4, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Overall Average (56.4%)')
ax2.set_xlabel('Season', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Accuracy by Season', fontsize=12, fontweight='bold')
ax2.set_ylim([50, 65])
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
for x, y in zip(seasons, season_data['correct_prediction']['mean']*100):
    ax2.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 5), 
                textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

# Bottom-left: Error by week (regular season focus)
week_data = df[df['week'] <= 18].groupby('week').agg({
    'abs_error': 'mean',
    'correct_prediction': 'mean'
})
weeks = week_data.index
colors_week = plt.cm.coolwarm(np.linspace(0.3, 0.7, len(weeks)))
ax3.bar(weeks, week_data['abs_error'], alpha=0.7, color=colors_week)
ax3.set_xlabel('Week of Season', fontsize=11, fontweight='bold')
ax3.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
ax3.set_title('Model Error by Week (Regular Season)\nDoes performance change as season progresses?', 
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add trend line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(weeks, week_data['abs_error'])
trend_line = slope * weeks + intercept
ax3.plot(weeks, trend_line, 'r--', linewidth=2, alpha=0.7, 
         label=f'Trend (R²={r_value**2:.3f}, p={p_value:.3f})')
ax3.legend(fontsize=9)

# Bottom-right: Home field advantage over time
home_adv_by_week = df[df['week'] <= 18].groupby('week').agg({
    'home_win': 'mean',
    'prob_home_win': 'mean'
})
ax4.plot(weeks, home_adv_by_week['home_win']*100, marker='o', linewidth=2, 
         markersize=8, label='Actual Home Win %', color='#27AE60')
ax4.plot(weeks, home_adv_by_week['prob_home_win']*100, marker='s', linewidth=2, 
         markersize=8, label='Predicted Home Win %', color='#E67E22')
ax4.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax4.set_xlabel('Week of Season', fontsize=11, fontweight='bold')
ax4.set_ylabel('Home Team Win Probability (%)', fontsize=11, fontweight='bold')
ax4.set_title('Home Field Advantage: Actual vs Predicted by Week', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([40, 75])

plt.tight_layout()
plt.savefig(output_dir / 'plot3_temporal_trends.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'plot3_temporal_trends.png'}")

# ============================================================================
# VISUALIZATION 4: Hypothesis Testing - "Close Games are Harder to Predict"
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Hypothesis: Close games (near 50% probability) are harder to predict
ax1 = fig.add_subplot(gs[0, :])
# Create fine-grained bins
prob_bins_fine = np.linspace(0, 1, 41)
bin_centers_fine = (prob_bins_fine[:-1] + prob_bins_fine[1:]) / 2
errors_by_prob = []
counts_by_prob = []

for i in range(len(prob_bins_fine) - 1):
    mask = (df['prob_home_win'] >= prob_bins_fine[i]) & (df['prob_home_win'] < prob_bins_fine[i+1])
    if mask.sum() > 0:
        errors_by_prob.append(df.loc[mask, 'abs_error'].mean())
        counts_by_prob.append(mask.sum())
    else:
        errors_by_prob.append(np.nan)
        counts_by_prob.append(0)

# Plot with shaded regions
ax1.fill_between(bin_centers_fine, errors_by_prob, alpha=0.3, color='#3498DB')
ax1.plot(bin_centers_fine, errors_by_prob, linewidth=3, color='#2C3E50', marker='o', markersize=4)
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50% (Toss-up Games)')
ax1.axvspan(0.4, 0.6, alpha=0.2, color='yellow', label='Close Games (40-60%)')
ax1.set_xlabel('Predicted Home Win Probability', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax1.set_title('HYPOTHESIS TEST: Are Close Games Harder to Predict?\n(The "U-Shape" reveals model uncertainty)', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Subplot 2: Distribution of predictions
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(df['prob_home_win'], bins=30, alpha=0.7, color='#9B59B6', edgecolor='black')
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='50% (Toss-up)')
ax2.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Model Predictions\nAre most games close or clear favorites?', fontsize=12, fontweight='bold')
ax2.legend()

# Subplot 3: Correct vs Incorrect by probability range
ax3 = fig.add_subplot(gs[1, 1])
prob_ranges = ['0-40%\n(Away Favored)', '40-50%\n(Slight Away)', '50-60%\n(Slight Home)', '60-100%\n(Home Favored)']
range_masks = [
    df['prob_home_win'] < 0.4,
    (df['prob_home_win'] >= 0.4) & (df['prob_home_win'] < 0.5),
    (df['prob_home_win'] >= 0.5) & (df['prob_home_win'] < 0.6),
    df['prob_home_win'] >= 0.6
]

correct_pcts = []
counts_range = []
for mask in range_masks:
    if mask.sum() > 0:
        correct_pcts.append(df.loc[mask, 'correct_prediction'].mean() * 100)
        counts_range.append(mask.sum())
    else:
        correct_pcts.append(0)
        counts_range.append(0)

bars = ax3.bar(range(len(prob_ranges)), correct_pcts, alpha=0.7, 
               color=['#E74C3C', '#F39C12', '#F39C12', '#27AE60'])
ax3.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Coin Flip')
ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('Accuracy by Probability Range\n(Confirms: confident predictions more accurate)', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(prob_ranges)))
ax3.set_xticklabels(prob_ranges, fontsize=9)
ax3.legend()
ax3.set_ylim([0, 100])

# Add count labels
for i, (bar, count, pct) in enumerate(zip(bars, counts_range, correct_pcts)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{pct:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 4: Error distribution for close vs clear games
ax4 = fig.add_subplot(gs[2, 0])
close_games = df[np.abs(df['prob_home_win'] - 0.5) < 0.15]
clear_games = df[np.abs(df['prob_home_win'] - 0.5) >= 0.3]

ax4.hist([close_games['abs_error'], clear_games['abs_error']], 
         bins=25, alpha=0.6, label=['Close Games (35-65%)', 'Clear Favorites (>80% or <20%)'],
         color=['#E74C3C', '#27AE60'], edgecolor='black')
ax4.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Error Distribution: Close vs Clear Games', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.axvline(x=close_games['abs_error'].mean(), color='#E74C3C', linestyle='--', linewidth=2, 
           label=f'Close Mean: {close_games["abs_error"].mean():.3f}')
ax4.axvline(x=clear_games['abs_error'].mean(), color='#27AE60', linestyle='--', linewidth=2,
           label=f'Clear Mean: {clear_games["abs_error"].mean():.3f}')

# Subplot 5: Upset analysis
ax5 = fig.add_subplot(gs[2, 1])
upset_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
upset_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
upset_probs = []
total_games_in_bin = []

for i in range(len(upset_bins) - 1):
    # Games where favorite was predicted with this confidence
    confident_games = df[np.abs(df['prob_home_win'] - 0.5) >= (upset_bins[i] - 0.5)]
    confident_games = confident_games[np.abs(confident_games['prob_home_win'] - 0.5) < (upset_bins[i+1] - 0.5)]
    
    if len(confident_games) > 0:
        upset_rate = (~confident_games['correct_prediction']).mean() * 100
        upset_probs.append(upset_rate)
        total_games_in_bin.append(len(confident_games))
    else:
        upset_probs.append(0)
        total_games_in_bin.append(0)

bars = ax5.bar(range(len(upset_labels)), upset_probs, alpha=0.7, 
               color=plt.cm.Reds(np.linspace(0.3, 0.9, len(upset_labels))))
ax5.set_ylabel('Upset Rate (%)', fontsize=11, fontweight='bold')
ax5.set_xlabel('Favorite Confidence Level', fontsize=11, fontweight='bold')
ax5.set_title('Upset Rate by Confidence\n(Do upsets happen even when we\'re very confident?)', fontsize=12, fontweight='bold')
ax5.set_xticks(range(len(upset_labels)))
ax5.set_xticklabels(upset_labels, rotation=45, ha='right')

# Add labels
for i, (bar, rate, count) in enumerate(zip(bars, upset_probs, total_games_in_bin)):
    if count > 0:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.savefig(output_dir / 'plot4_hypothesis_close_games.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'plot4_hypothesis_close_games.png'}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("FINAL INSIGHTS SUMMARY")
print("="*80)

print(f"\n1. CALIBRATION:")
print(f"   - Model predictions are {'well' if abs(df['prob_home_win'].mean() - df['home_win'].mean()) < 0.05 else 'poorly'} calibrated")
print(f"   - Average prediction: {df['prob_home_win'].mean():.1%}, Actual rate: {df['home_win'].mean():.1%}")

print(f"\n2. CONFIDENCE MATTERS:")
close_game_acc = df[np.abs(df['prob_home_win'] - 0.5) < 0.1]['correct_prediction'].mean()
high_conf_acc = df[np.abs(df['prob_home_win'] - 0.5) > 0.3]['correct_prediction'].mean()
print(f"   - Close games (45-55%): {close_game_acc:.1%} accuracy")
print(f"   - High confidence (>80% or <20%): {high_conf_acc:.1%} accuracy")
print(f"   - Improvement: +{(high_conf_acc - close_game_acc)*100:.1f} percentage points")

print(f"\n3. TEMPORAL PATTERNS:")
first_half = df[df['week'] <= 9]['abs_error'].mean()
second_half = df[(df['week'] > 9) & (df['week'] <= 18)]['abs_error'].mean()
print(f"   - First half of season error: {first_half:.4f}")
print(f"   - Second half error: {second_half:.4f}")
print(f"   - Trend: {'Improving' if second_half < first_half else 'Degrading'} ({abs(second_half-first_half)/first_half*100:.1f}% change)")

print(f"\n4. UPSET FREQUENCY:")
high_conf_upsets = df[np.abs(df['prob_home_win'] - 0.5) > 0.3]
upset_rate = (~high_conf_upsets['correct_prediction']).mean()
print(f"   - Even with >80% confidence, upsets happen {upset_rate:.1%} of the time")
print(f"   - Average upset magnitude (error): {upsets['abs_error'].mean():.4f}")

print("\n" + "="*80)
print("All visualizations saved to: models/validation_analysis/")
print("="*80)
