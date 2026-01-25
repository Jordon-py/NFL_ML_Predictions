# NFLEX v6 Predictive Pipeline Report

This report summarises the performance of base models and a convex blend on NFL game data up to 2025.

## Cross-validated results (training seasons)

| Model | Brier | Brier CI | Log-loss | LL CI | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic | 0.1769 | [0.1665, 0.1911] | 0.5473 | [0.5112, 0.5967] | 0.8102 | 0.7197 | 0.285 |
| SVM | 0.1803 | [0.1702, 0.1919] | 0.5663 | [0.5303, 0.6141] | 0.8115 | 0.7145 | 0.271 |
| GradientBoosting | 0.1755 | [0.1653, 0.1875] | 0.5048 | [0.4770, 0.5383] | 0.8108 | 0.7215 | 0.290 |
| MonotonicHGB | 0.1725 | [0.1627, 0.1856] | 0.5011 | [0.4734, 0.5358] | 0.8176 | 0.7268 | 0.302 |

## Hold-out season results ("never_seen" season)

| Model | Brier | Log-loss | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- |
| Logistic | 0.1997 | 0.5824 | 0.7592 | 0.8156 | 0.186 |
| SVM | 0.2253 | 0.6760 | 0.7145 | 0.7420 | 0.082 |
| GradientBoosting | 0.2237 | 0.6422 | 0.6984 | 0.7586 | 0.088 |
| MonotonicHGB | 0.2224 | 0.6405 | 0.7096 | 0.7693 | 0.094 |
| Blend(Logit,GB) w=0.00 | 0.2237 | 0.6422 | 0.6984 | 0.7586 | 0.088 |

## Brier decomposition (hold-out season)

| Model | Brier | Reliability | Resolution | Uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.1997 | 0.0108 | 0.0544 | 0.2451 |
| SVM | 0.2253 | 0.0210 | 0.0400 | 0.2451 |
| GradientBoosting | 0.2237 | 0.0241 | 0.0472 | 0.2451 |
| MonotonicHGB | 0.2224 | 0.0216 | 0.0438 | 0.2451 |
| Blend(Logit,GB) w=0.00 | 0.2237 | 0.0241 | 0.0472 | 0.2451 |

**Notes**:
- Purged walk-forward CV uses one-group embargo and five folds.
- Hold-out season models are trained strictly on prior seasons.
- Brier Skill Score baseline = weighted mean home-win rate on train.
- Blend = convex log-loss-minimizing weight over Logistic and GB.
- Monotonic constraints assume increasing diffs → higher home-win probability.