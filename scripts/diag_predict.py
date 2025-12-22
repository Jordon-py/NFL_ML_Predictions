import pandas as pd
import numpy as np
import joblib
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / 'backend' / 'data'
MODELS_DIR = BASE / 'backend' / 'models'

print('DATA_DIR', DATA_DIR)
df = pd.read_csv(DATA_DIR / 'game_features.csv')
# normalize columns same as server
for col in ('home_team','away_team','home_abbr','away_abbr'):
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
for col in ('season','season_num'):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
for col in ('week','week_num'):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

print('Columns:', df.columns[:50])
# target game
season = 2025
week = 12
home = 'HOU'
away = 'BUF'

matches = df[(df['season']==season) & (df['week']==week) & (df['home_team']==home) & (df['away_team']==away)]
print('Matched rows:', len(matches))
if matches.empty:
    print('No matching row')
    exit(0)
row = matches.iloc[0:1]
print('Row index', row.index.tolist())
num_cols = row.select_dtypes(include=[np.number]).columns.tolist()
print('Numeric columns count:', len(num_cols))
print('Numeric columns sample:', num_cols[:40])
X = row.select_dtypes(include=[np.number])
print('X shape:', X.shape)

# Load models
home_model = joblib.load(MODELS_DIR / 'home_model.joblib')
away_model = joblib.load(MODELS_DIR / 'away_model.joblib')
win_model = joblib.load(MODELS_DIR / 'win_clf_calibrated.joblib')

print('Home model type:', type(home_model))
print('Away model type:', type(away_model))
print('Attempting predictions...')
try:
    h = home_model.predict(X)
    a = away_model.predict(X)
    print('Predicted', h, a)
except Exception as e:
    print('Prediction error:')
    traceback.print_exc()
    print('Home model has attributes:', [attr for attr in dir(home_model) if not attr.startswith('_')][:40])
    # Try to inspect home_model if it's a pipeline
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(home_model, Pipeline):
            print('Home model pipeline steps:', [name for name,_ in home_model.steps])
    except Exception:
        pass

# Try predict_proba on win_model if available
try:
    print('Win model predict_proba available?', hasattr(win_model,'predict_proba'))
    if hasattr(win_model, 'predict_proba'):
        probs = win_model.predict_proba(X)
        print('Probs shape:', getattr(probs, 'shape', None))
except Exception:
    print('Win model predict_proba failed')
    traceback.print_exc()
