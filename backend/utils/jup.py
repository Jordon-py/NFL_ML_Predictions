import pandas as pd 
import nflreadpy as nfl 

pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 130)


load_schedules = nfl.load_schedules()
# print(f'load_schedules : {load_schedules}')
schedules_df = load_schedules.to_pandas()
columns  = schedules_df.columns
print(f'columns : {columns.tolist()}')
schedule = schedules_df.sort_values(by=['season', 'week'], ascending=[False, False])
schedule_h = schedule['home_score'] == None
schedule_a = schedule['away_score'] == None
print(schedule_h)
print(schedule_a)

teams = schedules_df['home_team'].unique()
print(f'teams : {teams}')

team_dict = dict([])

for team in teams:
    mask_home = schedules_df['home_team'] == team
    mask_away = schedules_df['away_team'] == team
    team_dict[team] = schedules_df[mask_home].filter(items=str(columns).startswith('home_'))
    team_dict[team] = schedules_df[mask_away].filter(items=str(columns).startswith('away_'))
    team_dict[team].to_csv(f'{team}.csv')

print(team_dict.to_csv(f'{team} : {team_dict[team]}'))