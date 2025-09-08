"""
train_models.py
================

Train models using the provided historical NFL team statistics dataset
(`data/nfl_team_stats_2002-2024.csv`). This refactor consumes the real
dataset and uses (almost) all of its informative features after light
preprocessing:

- Target: `home_win = (score_home > score_away)`
- Drop direct outcome leakage features: `score_home`, `score_away`
- Parse possession time strings (MM:SS) into seconds for both teams
- Keep numeric/boolean columns as-is, one-hot encode team names (`home`, `away`)
- Drop non-predictive text columns: `date`, `time_et`

We build a scikit-learn ColumnTransformer that standardises numeric features
and one‑hot encodes team names, then train an MLPClassifier (as a lightweight
neural net) and a LightGBM model. Artefacts (preprocessor, models, metadata)
are saved under `models/` for the API to load.

Run:
    python train_models.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas.api.types as pdt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier


# Directory configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'Nfl_data.csv'  # Fixed path - data is in project root/data/
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SimpleNN = MLPClassifier  # lightweight feedforward neural net


def _parse_possession_to_seconds(col: pd.Series) -> pd.Series:
    """Convert 'MM:SS' strings to total seconds (int). Handles NaNs gracefully."""
    def to_sec(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return int(x)
        try:
            s = str(x)
            parts = s.split(':')
            if len(parts) == 2:
                m, s = parts
                return int(m) * 60 + int(s)
        except Exception:
            pass
        return np.nan
    return col.apply(to_sec).astype('float64')


def build_prepared_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from raw team-game data.
    
    Expected columns in df:
    - 'points_for': points scored by the team
    - 'points_allowed': points allowed by the team  
    - 'win': binary target (1 if team won, 0 if lost)
    - Various other game statistics columns
    """
    df = df.copy()
    
    # Validate required columns exist
    required_cols = ['points_for', 'points_allowed', 'win']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        available_cols = list(df.columns)
        raise ValueError(f"Dataset must contain {required_cols} columns. Missing: {missing_cols}. Available columns: {available_cols}")
    
    # Target variable
    y = df['win'].astype(int)
    
    # Remove target and direct outcome leakage columns
    drop_cols = ['win', 'points_for', 'points_allowed', 'point_diff']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Features are remaining columns
    X = df
    return X, y


def load_raw_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split the raw dataset into train/val/test by season."""
    df = pd.read_csv(DATA_PATH)

    df.columns = [c.strip() for c in df.columns]        # Normalise column names: strip surrounding whitespace introduced in CSV formatting
    if 'season' in df.columns:                              # Clean and coerce season/week numeric if present
        df['season'] = pd.to_numeric(df['season'], errors='coerce')
    
    if 'week' in df.columns:
        df['week'] = df['week'].astype(str).str.strip()         # Strip strings then coerce
        df['week'] = pd.to_numeric(df['week'], errors='coerce')
    
    ''' Desired time-aware split when a week column exists:
    #   Train: seasons 2002–2023
    #   Validation: first 4 weeks of 2024
    #   Test: weeks 5+ of 2024
    # Some historical exports may lack a 'week' column. In that case we
    # fallback to a deterministic index-based split of the 2024 season:
    #   Validation = first 25% of 2024 rows, Test = remaining 75%.'''

    train_df = df[df['season'].between(2002, 2023)].reset_index(drop=True)

    if 'week' in df.columns:
        season_2024 = df[df['season'] == 2024]
        val_df = season_2024[season_2024['week'] <= 4].reset_index(drop=True)
        test_df = season_2024[season_2024['week'] >= 5].reset_index(drop=True)
    else:
        # Fallback strategy
        season_2024 = df[df['season'] == 2024].reset_index(drop=True)
        if not season_2024.empty:
            split_idx = max(1, int(0.25 * len(season_2024)))  # at least 1 row in val if possible
            val_df = season_2024.iloc[:split_idx].reset_index(drop=True)
            test_df = season_2024.iloc[split_idx:].reset_index(drop=True)
        else:
            # No 2024 data; create empty frames for val/test
            val_df = season_2024.copy()
            test_df = season_2024.copy()
    return train_df, val_df, test_df


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                         num_epochs: int = 40, batch_size: int = 128,
                         lr: float = 0.001) -> SimpleNN:
    """Train a multilayer perceptron using scikit‑learn."""
    hidden_layer_sizes = (128, 64)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        activation='relu', solver='adam',
                        learning_rate_init=lr, max_iter=num_epochs,
                        batch_size=batch_size, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp


def evaluate_predictions(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute accuracy, log loss and Brier score for predicted probabilities."""
    # Convert probabilities to binary predictions using 0.5 threshold
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba)
    }


def _dense2d(a: object) -> np.ndarray:
    """Convert predict_proba output to a 2D dense numpy array."""
    if isinstance(a, list):
        return np.hstack([
            (np.asarray(p.toarray()) if callable(getattr(p, "toarray", None)) else np.asarray(p))
            for p in a
        ])
    toarray = getattr(a, "toarray", None)
    return np.asarray(toarray()) if callable(toarray) else np.asarray(a)


def main() -> None:
    # Load raw splits
    train_raw, val_raw, test_raw = load_raw_splits()

    # Prepare frames (target and features)
    X_train_df, y_train = build_prepared_frames(train_raw)
    X_val_df, y_val = build_prepared_frames(val_raw)
    X_test_df, y_test = build_prepared_frames(test_raw)

    # Identify column types - categorical are object type columns, but exclude non-predictive ones
    all_categorical_cols = X_train_df.select_dtypes(include=['object']).columns.tolist()
    # Filter out non-predictive categorical columns
    exclude_categorical = ['game_id', 'team_name', 'opponent_name']  # These are too specific or redundant
    categorical_cols = [c for c in all_categorical_cols if c not in exclude_categorical]
    
    # All remaining numeric/boolean columns (float/int); exclude categoricals
    numeric_cols = [c for c in X_train_df.columns
                    if c not in categorical_cols and pdt.is_numeric_dtype(X_train_df[c])]

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numeric columns: {len(numeric_cols)} columns")
    print(f"Total features: {len(categorical_cols) + len(numeric_cols)}")
    
    # ColumnTransformer: scale numeric, one-hot teams (dense output)
    transformers = []
    
    if numeric_cols:
        transformers.append(('num',
                           Pipeline([
                               ('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler())
                           ]), numeric_cols))
    
    if categorical_cols:
        transformers.append(('cat',
                           Pipeline([
                               ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
                               ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                           ]), categorical_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    # Fit on train, transform all splits
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # Ensure dense numpy arrays (cast to np.array in case of unions)
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    X_test = np.asarray(X_test)

    # Ensure y are numpy arrays
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    # Train models
    nn_model = train_neural_network(X_train, y_train)

    # Predict probabilities using scikit‑learn's predict_proba
    nn_val_proba = nn_model.predict_proba(X_val)[:, 1]
    nn_test_proba = nn_model.predict_proba(X_test)[:, 1]

    gbm_model = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               max_depth=-1, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42)
    gbm_model.fit(X_train, y_train)

    probs_val = _dense2d(gbm_model.predict_proba(X_val))
    gbm_val_proba = probs_val[:, 1]
    probs_test = _dense2d(gbm_model.predict_proba(X_test))
    gbm_test_proba = probs_test[:, 1]

    # Ensemble
    ensemble_val_proba = (nn_val_proba + gbm_val_proba) / 2
    ensemble_test_proba = (nn_test_proba + gbm_test_proba) / 2

    # Evaluate
    print("Validation metrics:")
    metrics = {
        'neural_network': evaluate_predictions(y_val, nn_val_proba),
        'gradient_boosting': evaluate_predictions(y_val, gbm_val_proba),
        'ensemble': evaluate_predictions(y_val, ensemble_val_proba)
    }
    for model_name, m in metrics.items():
        print(f" {model_name}: accuracy={m['accuracy']:.3f}, log_loss={m['log_loss']:.3f}, brier_score={m['brier_score']:.3f}")

    print("\nTest metrics:")
    test_metrics = {
        'neural_network': evaluate_predictions(y_test, nn_test_proba),
        'gradient_boosting': evaluate_predictions(y_test, gbm_test_proba),
        'ensemble': evaluate_predictions(y_test, ensemble_test_proba)
    }
    for model_name, m in test_metrics.items():
        print(f" {model_name}: accuracy={m['accuracy']:.3f}, log_loss={m['log_loss']:.3f}, brier_score={m['brier_score']:.3f}")

    # Persist artefacts
    joblib.dump(preprocessor, MODELS_DIR / 'preprocessor.joblib')
    joblib.dump(nn_model, MODELS_DIR / 'nn_model.joblib')
    gbm_model.booster_.save_model(str(MODELS_DIR / 'gbm_model.txt'))

    # Save metadata
    feature_names = []
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        pass
    meta = {
        'raw_feature_columns': {
            'numeric': numeric_cols,
            'categorical': categorical_cols
        },
        'transformed_feature_names': feature_names,
        'models': {
            'nn_model': 'nn_model.joblib',
            'gbm_model': 'gbm_model.txt'
        },
        'preprocessor': 'preprocessor.joblib'
    }
    with open(MODELS_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()