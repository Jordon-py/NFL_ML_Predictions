# NFLEX v6 Predictive Pipeline Report

This report summarises the performance of base models and a convex blend on NFL game data up to 2025.

## Cross-validated results (training seasons)

| Model | Brier | Brier CI | Log-loss | LL CI | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic | 0.0000 | [0.0000, 0.0000] | 0.0001 | [0.0000, 0.0001] | 1.0000 | 1.0000 | 1.000 |
| SVM | 0.0044 | [0.0031, 0.0056] | 0.0225 | [0.0136, 0.0290] | 0.9995 | 0.9994 | 0.982 |
| GradientBoosting | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0001] | 1.0000 | 1.0000 | 1.000 |
| MonotonicHGB | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0001] | 1.0000 | 1.0000 | 1.000 |

## Hold-out season results ("never_seen" season)

| Model | Brier | Log-loss | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- |
| Logistic | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.000 |
| SVM | 0.0001 | 0.0010 | 1.0000 | 1.0000 | 1.000 |
| GradientBoosting | 0.0000 | 0.0001 | 1.0000 | 1.0000 | 1.000 |
| MonotonicHGB | 0.0000 | 0.0001 | 1.0000 | 1.0000 | 1.000 |
| Blend(Logit,GB) w=1.00 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.000 |

## Brier decomposition (hold-out season)

| Model | Brier | Reliability | Resolution | Uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.0000 | 0.0000 | 0.2451 | 0.2451 |
| SVM | 0.0001 | 0.0000 | 0.2451 | 0.2451 |
| GradientBoosting | 0.0000 | 0.0000 | 0.2451 | 0.2451 |
| MonotonicHGB | 0.0000 | 0.0000 | 0.2451 | 0.2451 |
| Blend(Logit,GB) w=1.00 | 0.0000 | 0.0000 | 0.2451 | 0.2451 |

**Notes**:
- Purged walk-forward CV uses one-group embargo and five folds.
- Hold-out season models are trained strictly on prior seasons.
- Brier Skill Score baseline = weighted mean home-win rate on train.
- Blend = convex log-loss-minimizing weight over Logistic and GB.
- Monotonic constraints assume increasing diffs → higher home-win probability.