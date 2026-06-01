import sys
import os
import asyncio
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from backend.config import load_schedule_data_safe
from backend.main import _sync_scores_job
from backend.sqlite_store import upsert_game_scores
from backend.main import _fetch_remote_scores_for_date, _load_local_scores_for_date, _score_sync_dates

def sync_schedules():
    print("Syncing schedules...")
    backend_dir = Path(__file__).parent / "backend"
    data_dir = backend_dir / "data"
    
    for year in [2024, 2025, 2026]:
        print(f"Fetching schedule for {year}...")
        try:
            df = load_schedule_data_safe(year)
            if df is not None and not df.empty:
                # Save to backend/data/
                path1 = data_dir / f"Nfl_schedule_{year}.csv"
                df.to_csv(path1, index=False)
                print(f"Saved {year} schedule to {path1}")
                
                # Save to backend/ (just in case)
                path2 = backend_dir / f"Nfl_schedule_{year}.csv"
                df.to_csv(path2, index=False)
                print(f"Saved {year} schedule to {path2}")
            else:
                print(f"Failed to fetch or empty dataframe for {year}.")
        except Exception as e:
            print(f"Error fetching schedule for {year}: {e}")

async def run_sync_scores():
    print("Syncing scores via job...")
    await _sync_scores_job()
    print("Score sync complete.")

if __name__ == "__main__":
    sync_schedules()
    asyncio.run(run_sync_scores())
