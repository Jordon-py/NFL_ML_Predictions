"""Sync schedule CSVs and completed-game scores into backend storage.

Data shape:
- Schedule input: yearly pandas DataFrames from `load_schedule_data_safe`, one
  row per game with season/week/team/date/score-style columns when available.
- Schedule output: CSV files under `backend/data/Nfl_schedule_<year>.csv`.
- Score output: SQLite rows written by `_sync_scores_job` using canonical game
  IDs and completed score fields.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import load_schedule_data_safe
from backend.main import _sync_scores_job


def sync_schedules():
    """Write yearly schedule DataFrames into backend schedule CSV files."""
    print("Syncing schedules...")
    data_dir = BACKEND_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for year in [2024, 2025, 2026]:
        print(f"Fetching schedule for {year}...")
        try:
            df = load_schedule_data_safe(year)
            if df is not None and not df.empty:
                # Save to backend/data/
                path1 = data_dir / f"Nfl_schedule_{year}.csv"
                df.to_csv(path1, index=False)
                print(f"Saved {year} schedule to {path1}")
                
                # Save to backend/ for compatibility with older fallback loaders.
                path2 = BACKEND_DIR / f"Nfl_schedule_{year}.csv"
                df.to_csv(path2, index=False)
                print(f"Saved {year} schedule to {path2}")
            else:
                print(f"Failed to fetch or empty dataframe for {year}.")
        except Exception as e:
            print(f"Error fetching schedule for {year}: {e}")

async def run_sync_scores():
    """Run the backend score-sync job and persist completed score entries."""
    print("Syncing scores via job...")
    await _sync_scores_job()
    print("Score sync complete.")

if __name__ == "__main__":
    sync_schedules()
    asyncio.run(run_sync_scores())
