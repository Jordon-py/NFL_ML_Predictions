import nflreadpy as nfl
import pandas as pd

def check_live_schedule():
    try:
        sch = nfl.load_schedules(seasons=[2025])
        if hasattr(sch, "to_pandas"):
            df = sch.to_pandas()
            # Filter for games with no score (future)
            future = df[df["home_score"].isna()]
            if not future.empty:
                first = future.iloc[0]
                print("Columns:", future.columns.tolist())
                print("Sample 'kickoff':", repr(first.get("kickoff")), type(first.get("kickoff")))
                print("Sample 'gameday':", repr(first.get("gameday")), type(first.get("gameday")))
                print("Sample 'gametime':", repr(first.get("gametime")), type(first.get("gametime")))
    except Exception as e:
        print(f"Error check_live: {e}")

if __name__ == "__main__":
    check_live_schedule()
