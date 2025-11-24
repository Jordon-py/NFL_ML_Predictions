import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import backend package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import main as m

# Ensure models and dataset are loaded
m.state.load()
print('Models loaded:', list(m.state.models.keys()))
print('Preprocessor loaded:', bool(m.state.preprocessor))

if m.state.dataset is None or len(m.state.dataset) == 0:
    print('No dataset loaded; aborting')
    raise SystemExit(1)

# Pick the first game from the dataset that has season/week/home/away
row = m.state.dataset.iloc[0]
home = row.get('home_team')
away = row.get('away_team')
season = int(row.get('season') if row.get('season') is not None else row.get('season_num', 2025))
week = int(row.get('week') if row.get('week') is not None else row.get('week_num', 1))
print('Testing predict for:', home, 'vs', away, 'season', season, 'week', week)
req = m.PredictRequest(home_team=home, away_team=away, season=season, week=week)

res = asyncio.run(m.predict(req))
print('Prediction result:')
print(res)
