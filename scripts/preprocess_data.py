import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("Current working directory:", os.getcwd())

# Load data
print("Loading dataset...")
df = pd.read_csv("data/PL_202425.csv")
print("Dataset loaded successfully!")

# Load upcoming fixtures to get all teams
print("Loading upcoming fixtures...")
upcoming_fixtures = pd.read_csv("data/PL_202425_fixtures.csv")
print("Fixtures loaded successfully!")

# Handle missing values
print("Filling missing values...")
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
print("Missing values filled!")

# Create new columns for goal difference and shot efficiency
print("Creating new features...")
new_columns = pd.DataFrame({
    'GoalDifference': df['FTHG'] - df['FTAG'],
    'ShotsEfficiency_Home': df['HST'] / df['HS'],
    'ShotsEfficiency_Away': df['AST'] / df['AS']
})

# Concatenate new columns to the existing DataFrame
df = pd.concat([df, new_columns], axis=1)

# Defragment the DataFrame
df = df.copy()
print("New features added successfully!")

# Convert result to numerical values
df['FTR'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})

# Drop rows with missing data
df.dropna(inplace=True)

# Get all team names from both datasets
teams_in_df = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
teams_in_fixtures = pd.concat([upcoming_fixtures['home'], upcoming_fixtures['away']]).unique()

all_teams = pd.Series(list(set(teams_in_df).union(set(teams_in_fixtures))))

# Fit LabelEncoder on all teams
le = LabelEncoder()
le.fit(all_teams)

# Encode team names in df
df['HomeTeam'] = le.transform(df['HomeTeam'])
df['AwayTeam'] = le.transform(df['AwayTeam'])

# Save the LabelEncoder
joblib.dump(le, 'models/team_label_encoder.pkl')
print("Team LabelEncoder saved successfully!")

# Track team points and goal difference
teams = df['HomeTeam'].unique()
team_points = {team: 0 for team in teams}
team_goal_diff = {team: 0 for team in teams}

df['HomeTeamPoints'] = 0
df['AwayTeamPoints'] = 0
df['HomeTeamGoalDifference'] = 0
df['AwayTeamGoalDifference'] = 0

print("Calculating cumulative statistics...")
for index, row in df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    # Update points based on full-time result
    if row['FTR'] == 1:
        team_points[home_team] += 3
    elif row['FTR'] == 0:
        team_points[home_team] += 1
        team_points[away_team] += 1
    elif row['FTR'] == -1:
        team_points[away_team] += 3

    # Update goal difference
    team_goal_diff[home_team] += row['FTHG'] - row['FTAG']
    team_goal_diff[away_team] += row['FTAG'] - row['FTHG']

    # Store cumulative statistics in the DataFrame
    df.at[index, 'HomeTeamPoints'] = team_points[home_team]
    df.at[index, 'AwayTeamPoints'] = team_points[away_team]
    df.at[index, 'HomeTeamGoalDifference'] = team_goal_diff[home_team]
    df.at[index, 'AwayTeamGoalDifference'] = team_goal_diff[away_team]
print("Cumulative statistics calculated successfully!")

# Standardize numeric features
print("Standardizing numeric features...")
scaler = StandardScaler()
numeric_features = [
    'HS', 'AS', 'HST', 'AST', 'GoalDifference',
    'HomeTeamPoints', 'AwayTeamPoints',
    'HomeTeamGoalDifference', 'AwayTeamGoalDifference'
]
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Save the scaler
joblib.dump(scaler, 'models/feature_scaler.pkl')
print("Feature scaler saved successfully!")

# Save the processed data
df.to_csv("data/processed_league_data.csv", index=False)
print("Data preprocessing complete and saved as 'data/processed_league_data.csv'")
