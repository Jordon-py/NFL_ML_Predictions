# NFL Model Evaluation Report

- **Dataset**: `game_features_20251208.csv`
- **Train end boundary**: season `2023`, week `18`

## Home score regression performance
- **Average MAE**: 4.37 points. On average, the model's score prediction is this many points off.
- **Average RMSE**: 5.64 points. Large mistakes are amplified here, so a big gap vs MAE indicates occasional blowouts.
- **Average R²**: 0.674. R² is moderate; the model captures a meaningful share of score variability.

Per-season breakdown:

         model  season  n_games      MAE     RMSE       R2
home_score_reg    2023       13 4.486774 5.393120 0.522025
home_score_reg    2024      285 4.161958 5.510025 0.706188
home_score_reg    2025      207 4.643121 5.841513 0.639977

## Away score regression performance
- **Average MAE**: 4.27 points. On average, the model's score prediction is this many points off.
- **Average RMSE**: 5.44 points. Large mistakes are amplified here, so a big gap vs MAE indicates occasional blowouts.
- **Average R²**: 0.674. R² is moderate; the model captures a meaningful share of score variability.

Per-season breakdown:

         model  season  n_games      MAE     RMSE       R2
away_score_reg    2023       13 4.275674 5.079705 0.762898
away_score_reg    2024      285 4.202611 5.400410 0.668033
away_score_reg    2025      207 4.356633 5.505208 0.677643

## Win classifier vs baselines
- **Brier score vs always-home**: better by ~48.1% (0.130 vs 0.250)
- **LogLoss vs always-home**: better by ~41.2% (0.407 vs 0.693)
- **Accuracy vs always-home**: better by ~50.5% (0.826 vs 0.549)
- **Brier score vs moneyline**: better by ~36.8% (0.130 vs 0.205)
- **LogLoss vs moneyline**: better by ~31.8% (0.407 vs 0.597)
- **AUC vs moneyline**: better by ~21.3% (0.900 vs 0.742)

Overall classifier metrics table:

               model   scope    Brier  LogLoss      AUC  Accuracy
      win_classifier overall 0.129686 0.407291 0.899550  0.825743
baseline_always_home overall 0.250000 0.693147      NaN  0.548515
  baseline_moneyline overall 0.205138 0.597471 0.741624  0.699010

Per-season classifier metrics:

 season                model  n_games    Brier  LogLoss      AUC  Accuracy
   2023 baseline_always_home       13 0.250000 0.693147      NaN  0.769231
   2023   baseline_moneyline       13 0.212538 0.607755 0.533333  0.615385
   2023       win_classifier       13 0.036084 0.171631 1.000000  1.000000
   2024 baseline_always_home      285 0.250000 0.693147      NaN  0.547368
   2024   baseline_moneyline      285 0.200824 0.588641 0.756758  0.726316
   2024       win_classifier      285 0.131549 0.409268 0.899324  0.824561
   2025 baseline_always_home      207 0.250000 0.693147      NaN  0.536232
   2025   baseline_moneyline      207 0.210611 0.608981 0.726821  0.666667
   2025       win_classifier      207 0.133000 0.419369 0.892173  0.816425

## Threshold diagnostics at 0.5
- **Accuracy**: 0.826 → overall fraction of correctly classified games.
- **Precision (home win)**: 0.865 → among predicted home wins, this fraction were actually wins.
- **Recall (home win)**: 0.809 → among all true home wins, this fraction were correctly predicted.
- **F1 score (home win)**: 0.836 → balance between precision and recall for predicting home wins.

Confusion matrix (rows = actual, columns = predicted):

                    Pred_0_home_loss  Pred_1_home_win
Actual_0_home_loss               193               35
Actual_1_home_win                 53              224
- The model commits more **false negatives** (predicted home loss, actually win): about 10.5% of all games.

_This report was generated automatically by eval_models.py to provide both raw metrics and an interpretation layer that is readable by humans working with the model._
