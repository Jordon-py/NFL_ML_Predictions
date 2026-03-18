import pandas as pd
import sys
import os
from pathlib import Path

def sync_schedules_direct():
    print("Fetching schedule from nflverse...")
    # NFLverse provides the games data here
    url = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"Downloaded {len(df)} games from {url}")
        
        backend_dir = Path(__file__).parent / "backend"
        data_dir = backend_dir / "data"
        
        for year in [2024, 2025]:
            season_df = df[df["season"] == year]
            if not season_df.empty:
                path1 = data_dir / f"Nfl_schedule_{year}.csv"
                season_df.to_csv(path1, index=False)
                print(f"Saved {year} schedule ({len(season_df)} games) to {path1}")
                
                path2 = backend_dir / f"Nfl_schedule_{year}.csv"
                season_df.to_csv(path2, index=False)
                print(f"Saved {year} schedule to {path2}")
            else:
                print(f"No games found for {year}")
    except Exception as e:
        print(f"Failed to fetch games via pandas: {e}")

if __name__ == "__main__":
    sync_schedules_direct()
