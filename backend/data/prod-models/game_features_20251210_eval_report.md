# NFL Model Evaluation Report

- **Dataset**: `game_features_20251210.csv`
- **Train end boundary**: season `2023`, week `18`

## Home score regression performance
- **Average MAE**: 4.06 points. On average, the model's score prediction is this many points off.
- **Average RMSE**: 5.22 points. Large mistakes are amplified here, so a big gap vs MAE indicates occasional blowouts.
- **Average R²**: 0.721. R² is moderate; the model captures a meaningful share of score variability.

Per-season breakdown:

         model  season  n_games      MAE     RMSE       R2
home_score_reg    2023       13 3.128090 3.694464 0.775701
home_score_reg    2024      285 3.828606 4.998387 0.758219
home_score_reg    2025      208 4.427963 5.612642 0.666109

## Away score regression performance
- **Average MAE**: 3.97 points. On average, the model's score prediction is this many points off.
- **Average RMSE**: 5.03 points. Large mistakes are amplified here, so a big gap vs MAE indicates occasional blowouts.
- **Average R²**: 0.720. R² is moderate; the model captures a meaningful share of score variability.

Per-season breakdown:

         model  season  n_games      MAE     RMSE       R2
away_score_reg    2023       13 3.366182 3.765576 0.869707
away_score_reg    2024      285 3.861008 4.907321 0.725886
away_score_reg    2025      208 4.163037 5.278689 0.702334

## Win classifier vs baselines
- **Brier score vs always-home**: better by ~55.5% (0.111 vs 0.250)
- **LogLoss vs always-home**: better by ~48.7% (0.355 vs 0.693)
- **Accuracy vs always-home**: better by ~52.9% (0.840 vs 0.549)
- **Brier score vs moneyline**: better by ~45.8% (0.111 vs 0.205)
- **LogLoss vs moneyline**: better by ~40.5% (0.355 vs 0.598)
- **AUC vs moneyline**: better by ~24.5% (0.923 vs 0.741)

Overall classifier metrics table:

               model   scope    Brier  LogLoss      AUC  Accuracy
      win_classifier overall 0.111179 0.355330 0.922678  0.839921
baseline_always_home overall 0.250000 0.693147      NaN  0.549407
  baseline_moneyline overall 0.205226 0.597660 0.741023  0.699605

Per-season classifier metrics:

 season                model  n_games    Brier  LogLoss      AUC  Accuracy
   2023 baseline_always_home       13 0.250000 0.693147      NaN  0.769231
   2023   baseline_moneyline       13 0.212538 0.607755 0.533333  0.615385
   2023       win_classifier       13 0.015543 0.112689 1.000000  1.000000
   2024 baseline_always_home      285 0.250000 0.693147      NaN  0.547368
   2024   baseline_moneyline      285 0.200824 0.588641 0.756758  0.726316
   2024       win_classifier      285 0.100079 0.316577 0.939699  0.852632
   2025 baseline_always_home      208 0.250000 0.693147      NaN  0.538462
   2025   baseline_moneyline      208 0.210801 0.609386 0.725307  0.668269
   2025       win_classifier      208 0.132364 0.423595 0.890439  0.812500

## Threshold diagnostics at 0.5
- **Accuracy**: 0.840 → overall fraction of correctly classified games.
- **Precision (home win)**: 0.869 → among predicted home wins, this fraction were actually wins.
- **Recall (home win)**: 0.835 → among all true home wins, this fraction were correctly predicted.
- **F1 score (home win)**: 0.851 → balance between precision and recall for predicting home wins.

Confusion matrix (rows = actual, columns = predicted):

                    Pred_0_home_loss  Pred_1_home_win
Actual_0_home_loss               193               35
Actual_1_home_win                 46              232
- The model commits more **false negatives** (predicted home loss, actually win): about 9.1% of all games.

_This report was generated automatically by eval_models.py to provide both raw metrics and an interpretation layer that is readable by humans working with the model._
