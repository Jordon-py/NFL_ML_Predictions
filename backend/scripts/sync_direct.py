"""Fetch nflverse schedule data and write backend schedule CSVs.

Data shape:
- Input: nflverse `games.csv` DataFrame with one row per game and columns such
  as `season`, `week`, teams, dates, and score/status fields.
- Output: yearly CSV files under `backend/data/` plus compatibility copies under
  `backend/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sync_schedules_direct():
    """Download nflverse games and write selected seasons to backend CSVs."""
    print("Fetching schedule from nflverse...")
    # NFLverse provides the games data here
    url = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"Downloaded {len(df)} games from {url}")
        
        data_dir = BACKEND_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for year in [2024, 2025]:
            season_df = df[df["season"] == year]
            if not season_df.empty:
                path1 = data_dir / f"Nfl_schedule_{year}.csv"
                season_df.to_csv(path1, index=False)
                print(f"Saved {year} schedule ({len(season_df)} games) to {path1}")
                
                path2 = BACKEND_DIR / f"Nfl_schedule_{year}.csv"
                season_df.to_csv(path2, index=False)
                print(f"Saved {year} schedule to {path2}")
            else:
                print(f"No games found for {year}")
    except Exception as e:
        print(f"Failed to fetch games via pandas: {e}")

if __name__ == "__main__":
    sync_schedules_direct()
