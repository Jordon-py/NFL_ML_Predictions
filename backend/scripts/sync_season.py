"""Backfill completed game scores for a season window.

Data shape:
- Input: date strings shaped as `YYYYMMDD` for ESPN scoreboard requests.
- Intermediate: lists of score-entry dictionaries from
  `_fetch_remote_scores_for_date`.
- Output: SQLite score rows written through `upsert_game_scores`.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import _fetch_remote_scores_for_date
from backend.sqlite_store import upsert_game_scores


async def fetch_season():
    """Fetch completed scores over the configured season date range."""
    start_date = date(2025, 9, 1)
    end_date = date(2026, 2, 28)
    
    current = start_date
    dates = []
    while current <= end_date:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
        
    print(f"Fetching scores for {len(dates)} days...")
    
    # We will process in batches of 10 to be nice to the API
    all_entries = []
    
    for i in range(0, len(dates), 10):
        batch = dates[i:i+10]
        tasks = [_fetch_remote_scores_for_date(d) for d in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list) and res:
                all_entries.extend(res)
        print(f"Processed batch {i//10 + 1}/{(len(dates)-1)//10 + 1}... Found {len(all_entries)} scores so far.")
        await asyncio.sleep(0.5)
        
    if all_entries:
        upsert_game_scores(all_entries)
        print(f"Upserted {len(all_entries)} game scores.")
    else:
        print("No scores found.")

if __name__ == "__main__":
    asyncio.run(fetch_season())
