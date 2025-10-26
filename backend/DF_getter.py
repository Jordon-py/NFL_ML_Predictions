import pandas as pd
import nflreadpy as nfl

# Load current season play-by-play data
pbp = nfl.load_pbp()

# Load player game-level stats for multiple seasons
player_stats = nfl.load_player_stats(seasons=True)

# Load all available team level stats
team_stats = nfl.load_team_stats(seasons=True)

# nflreadpy uses Polars instead of pandas. Convert to pandas if needed:
pbp = pbp.to_pandas()
team_stats = team_stats.to_pandas()
player_stats = player_stats.to_pandas()

print(f"PBP: {pbp}, TEAM_STATS: {team_stats}, PLAYER_STATS: {player_stats} ")

pbp.to_csv("pbp_data.csv")
team_stats.to_csv("team_stats.csv")
player_stats.to_csv("player_stats.csv")

print("saved dfs")
