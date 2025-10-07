# NFLEX v6 Predictive Pipeline Report

This report summarises the performance of four base models and a convex blend on NFL game data from 2014–2025.

## Cross‑validated results (training seasons)

| Model | Brier | Brier CI | Log‑loss | LL CI | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic | 0.0007 | [0.0002, 0.0012] | 0.0075 | [0.0005, 0.0121] | 0.9996 | 0.9989 | 0.997 |
| SVM | 0.0116 | [0.0089, 0.0139] | 0.0955 | [0.0682, 0.1192] | 0.9944 | 0.9885 | 0.953 |
| GradientBoosting | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 | 1.000 |
| MonotonicHGB | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 | 1.000 |

## Hold‑out season results ("never‑seen" season)

| Model | Brier | Log‑loss | ROC AUC | PR AUC | Brier Skill |
| --- | --- | --- | --- | --- | --- |
| Logistic | 0.7132 | 9.8537 | 0.5764 | 0.1814 | -1.440 |
| SVM | 0.6823 | 9.1174 | 0.6019 | 0.1908 | -1.334 |
| GradientBoosting | 0.7132 | 7.7239 | 0.5764 | 0.1814 | -1.440 |
| MonotonicHGB | 0.7132 | 7.7239 | 0.5764 | 0.1814 | -1.440 |
| Blend(Logit,GB) w=0.98 | 0.7132 | 9.8537 | 0.5764 | 0.1814 | -1.440 |

## Brier decomposition (hold‑out season)

| Model | Brier | Reliability | Resolution | Uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.7132 | 0.5838 | 0.0037 | 0.1331 |
| SVM | 0.6823 | 0.5541 | 0.0049 | 0.1331 |
| GradientBoosting | 0.7132 | 0.5838 | 0.0037 | 0.1331 |
| MonotonicHGB | 0.7132 | 0.5838 | 0.0037 | 0.1331 |
| Blend(Logit,GB) w=0.98 | 0.7132 | 0.5838 | 0.0037 | 0.1331 |

**Notes**:
- Cross‑validated results use a purged walk‑forward splitter with one‑week embargo and five folds.
- The hold‑out season is 2025; models were trained exclusively on prior seasons.
- Brier Skill Score is relative to the mean home‑win rate in the training set.
- The convex blend combines Logistic and GradientBoosting predictions using a weight that minimises log‑loss on the training set.
- Monotonic constraints assume that increasing differential statistics generally increase the probability of a home win.