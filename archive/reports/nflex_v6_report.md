# NFLEX v6 Predictive Pipeline Report

This report summarises the performance of base models and a convex blend on NFL game data up to 2024.

## Cross-validated results (training seasons)

| Model | Brier | Brier CI | Log-loss | LL CI | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic | 0.2039 | [0.1896, 0.2198] | 0.8305 | [0.7473, 0.9210] | 0.7743 | 0.6633 | 0.175 |
| SVM | 0.1959 | [0.1795, 0.2114] | 0.7901 | [0.7175, 0.8841] | 0.7962 | 0.6934 | 0.207 |
| GradientBoosting | 0.1793 | [0.1677, 0.1914] | 0.5137 | [0.4825, 0.5481] | 0.7996 | 0.7016 | 0.274 |
| MonotonicHGB | 0.1793 | [0.1671, 0.1938] | 0.5146 | [0.4803, 0.5558] | 0.8006 | 0.7124 | 0.274 |

## Hold-out season results ("never_seen" season)

| Model | Brier | Log-loss | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- |
| Logistic | 0.2164 | 0.6315 | 0.7149 | 0.7194 | 0.127 |
| SVM | 0.2242 | 0.9950 | 0.7245 | 0.7294 | 0.095 |
| GradientBoosting | 0.2075 | 0.6026 | 0.7337 | 0.7593 | 0.163 |
| MonotonicHGB | 0.2067 | 0.6045 | 0.7406 | 0.7517 | 0.166 |
| Blend(Logit,GB) w=0.00 | 0.2075 | 0.6026 | 0.7337 | 0.7593 | 0.163 |

## Brier decomposition (hold-out season)

| Model | Brier | Reliability | Resolution | Uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.2164 | 0.0060 | 0.0375 | 0.2478 |
| SVM | 0.2242 | 0.0246 | 0.0509 | 0.2478 |
| GradientBoosting | 0.2075 | 0.0017 | 0.0413 | 0.2478 |
| MonotonicHGB | 0.2067 | 0.0054 | 0.0472 | 0.2478 |
| Blend(Logit,GB) w=0.00 | 0.2075 | 0.0017 | 0.0413 | 0.2478 |

**Notes**:
- Purged walk-forward CV uses one-group embargo and five folds.
- Hold-out season models are trained strictly on prior seasons.
- Brier Skill Score baseline = weighted mean home-win rate on train.
- Blend = convex log-loss-minimizing weight over Logistic and GB.
- Monotonic constraints assume increasing diffs → higher home-win probability.