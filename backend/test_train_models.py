import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_test_specific_week():
    """Train score regressors on all past data and test on 2025 week 1.

    Saves a CSV with columns: season,week,home_team,away_team,
    actual_home_points,actual_away_points,actual_point_diff,
    pred_home_points,pred_away_points,pred_point_diff.
    """
    # Load data
    df = pd.read_csv('Nfl_data_sorted.csv')
    df.columns = [c.strip() for c in df.columns]
    logging.info("Loaded %d rows", len(df))

    # Features
    features = [
        'home_prior_pa_avg_3', 'home_prior_pa_avg_5', 'home_prior_pf_avg_3',
        'home_prior_pf_avg_5', 'home_prior_win_pct_3', 'home_prior_win_pct_5',
        'away_prior_pa_avg_3', 'away_prior_pa_avg_5', 'away_prior_pf_avg_3',
        'away_prior_pf_avg_5', 'away_prior_win_pct_3', 'away_prior_win_pct_5'
    ]
    for c in features + ['season','week','home_points_for','away_points_for']:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Time-aware split: train on all games prior to 2025 week 1
    train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < 1))
    # Since week cannot be <1, effectively seasons < 2025
    train_df = df[train_mask]
    test_df = df[(df['season'] == 2025) & (df['week'] == 1)]

    if test_df.empty:
        raise RuntimeError("No rows found for 2025 week 1 in dataset.")

    X_train = train_df[features]
    X_test = test_df[features]
    y_home = train_df['home_points_for']
    y_away = train_df['away_points_for']

    logging.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), features)
        ],
        remainder='drop'
    )
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Models
    home_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42)
    away_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42)
    home_model.fit(X_train_proc, y_home)
    away_model.fit(X_train_proc, y_away)

    # Predict
    pred_home = home_model.predict(X_test_proc)
    pred_away = away_model.predict(X_test_proc)

    # Build results
    results = test_df[['season','week','home_team','away_team','home_points_for','away_points_for']].copy()
    results.rename(columns={'home_points_for':'actual_home_points','away_points_for':'actual_away_points'}, inplace=True)
    results['actual_point_diff'] = results['actual_home_points'] - results['actual_away_points']
    results['pred_home_points'] = pred_home
    results['pred_away_points'] = pred_away
    results['pred_point_diff'] = results['pred_home_points'] - results['pred_away_points']

    # Optional: clip to plausible range
    for c in ['pred_home_points','pred_away_points']:
        results[c] = results[c].clip(lower=0, upper=70).round(1)
    results['pred_point_diff'] = (results['pred_home_points'] - results['pred_away_points']).round(1)

    # Save the results to a CSV file
    out_path = 'test_results_2025_week_1.csv'
    results.to_csv(out_path, index=False)
    logging.info("Saved: %s", out_path)

if __name__ == '__main__':
    train_and_test_specific_week()
