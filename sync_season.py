import sys
import asyncio
from datetime import date, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backend.main import _fetch_remote_scores_for_date
from backend.sqlite_store import upsert_game_scores

async def fetch_season():
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
