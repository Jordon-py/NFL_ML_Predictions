import pandas as pd
import nflreadpy as nfl
import logging



pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.describe_option



df = pd.read_csv('backend\\data\\datasets\\game_features_20260121.csv', low_memory=False)

teams = pd.unique(
    pd.concat([df["home_team"], df["away_team"]]).dropna()
)

teams = sorted(teams)  # alphabetical

print("Team count:", len(teams))
print(teams)

for t in teams:
    if t in df['home_team']:
        mask = t
        data = df.values.iloc(t)
        print(data) 
