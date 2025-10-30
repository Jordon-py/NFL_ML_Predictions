# NFLEX v6 Predictive Pipeline Report

This report summarises the performance of four base models and a convex blend on NFL game data from 2014–2025.

## Cross‑validated results (training seasons)

| Model | Brier | Brier CI | Log‑loss | LL CI | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic | 0.0008 | [0.0002, 0.0012] | 0.0060 | [0.0008, 0.0097] | 0.9996 | 0.9996 | 0.997 |
| SVM | 0.0392 | [0.0333, 0.0436] | 0.1777 | [0.1492, 0.2077] | 0.9879 | 0.9815 | 0.842 |
| GradientBoosting | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 | 1.000 |
| MonotonicHGB | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 | 1.000 |

## Hold‑out season results ("never_seen" season)

| Model | Brier | Log‑loss | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- |
| Logistic | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.000 |
| SVM | 0.1287 | 0.3572 | 0.9908 | 0.9933 | 0.476 |
| GradientBoosting | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.000 |
| MonotonicHGB | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.000 |
| Blend(Logit,GB) w=0.98 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.000 |

## Brier decomposition (hold‑out season)

| Model | Brier | Reliability | Resolution | Uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.0000 | 0.0000 | 0.2451 | 0.2451 |
| SVM | 0.1287 | 0.0626 | 0.1765 | 0.2451 |
| GradientBoosting | 0.0000 | 0.0000 | 0.2451 | 0.2451 |
| MonotonicHGB | 0.0000 | 0.0000 | 0.2451 | 0.2451 |
| Blend(Logit,GB) w=0.98 | 0.0000 | 0.0000 | 0.2451 | 0.2451 |

**Notes**:
- Cross‑validated results use a purged walk‑forward splitter with one‑week embargo and five folds.
- The hold‑out season is 2025; models were trained exclusively on prior seasons.
- Brier Skill Score is relative to the mean home‑win rate in the training set.
- The convex blend combines Logistic and GradientBoosting predictions using a weight that minimises log‑loss on the training set.
- Monotonic constraints assume that increasing differential statistics generally increase the probability of a home win.