import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("Current working directory:", os.getcwd())
#laster inn data
print("loading dataset...")
df = pd.read_csv("data/results.csv")
print("dataset loaded successfully!")
#Håndterer data som ikke er fylt inn, erstatter data med medianen for kolonnen. 

print("Filling missing values...")
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
print("Missing values filled!")

#gjør om hjemme og bortelag til numeriske verdier
new_columns = pd.DataFrame({
    'GoalDifference': df['FTHG'] - df['FTAG'],
    'ShotsEffiency_Home': df['HST'] / df['HS'],
    'ShotsEffiency_Away': df['AST'] / df['AS']
})

# Concatenate new columns to the existing DataFrame
df = pd.concat([df, new_columns], axis=1)

#konvertere data til numeriske verdier H = hjemmeseier, D = uavgjort, A = borteseier
# 1 = seier, 0 = uavgjort, -1 = borteseier /tap
df['FTR'] = df['FTR'].map({'H':1, 'D': 0, 'A': -1 })


#dropp rader uten data
df.dropna(inplace=True)

#filtrer slik vi kun bruker sist sesong
latest_season = df['Season'].max()
df = df[df['Season']== latest_season]

#sporer hvordan et lag gjør det gjennom hele sesongen
teams = df['HomeTeam'].unique()
team_points = {team: 0 for team in teams}
team_goal_diff = {team: 0 for team in teams}

df['HomeTeamPoints'] = 0
df['AwayTeamPoints'] = 0
df['HomeTeamGoalDifference'] = 0
df['HomeTeamGoalDifference'] = 0

for index, row in df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

     #oppdater poeng basert på fulltid resultatet
    if row['FTR'] == 1:
          team_points[home_team] += 3
    elif row['FTR'] == 0:
         team_points[home_team] += 1
         team_points[away_team] += 1
    elif ['FTR'] == -1:
         team_points[away_team] += 3

    team_goal_diff[home_team] += row['FTHG'] - row['FTAG']
    team_goal_diff[away_team] += row['FTAG'] - row['FTHG']

    # Store cumulative statistics in the DataFrame
    df.at[index, 'HomeTeamPoints'] = team_points[home_team]
    df.at[index, 'AwayTeamPoints'] = team_points[away_team]
    df.at[index, 'HomeTeamGoalDifference'] = team_goal_diff[home_team]
    df.at[index, 'AwayTeamGoalDifference'] = team_goal_diff[away_team]





#lagre den nye prosseserte dataen 
df.to_csv("data/processed_league_data.csv", index = False)
print("Data prepocessing fullført og lagret som '../data/processed_league_data.csv'")





