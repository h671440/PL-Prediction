import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

print("Loading trained model...")
model = joblib.load("models/pl_table_predictor.pkl")
print("Model loaded successfully!")

# Load the upcoming fixtures
print("Loading upcoming fixtures...")
upcoming_fixtures = pd.read_csv("data/premier_league_fixtures.csv")
print("Fixtures loaded successfully!")

#endre navn på rader:
upcoming_fixtures.rename(columns={"home": "HomeTeam", "away": "AwayTeam"}, inplace=True)

print("Loading dataset for league simulation...")
df = pd.read_csv("data/processed_league_data.csv")
print("Dataset loaded successfully!")

existing_teams = set(df['HomeTeam'].unique())
upcoming_teams = set(upcoming_fixtures['HomeTeam'].unique()).union(set(upcoming_fixtures['AwayTeam'].unique()))

# Identify newly promoted teams
new_teams = upcoming_teams - existing_teams

# Initialisere tabellen
teams = df['HomeTeam'].unique()
team_points = {team: 0 for team in teams}
team_goal_diff = {team: 0 for team in teams}


# Prepare the upcoming fixtures
le = LabelEncoder()
upcoming_fixtures['HomeTeam'] = le.fit_transform(upcoming_fixtures['HomeTeam'])
upcoming_fixtures['AwayTeam'] = le.fit_transform(upcoming_fixtures['AwayTeam'])

print("Columns in processed dataset:", df.columns)

# Initialize standings from the recent season data
for index, row in upcoming_fixtures.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    team_points[home_team] = row.get('HomeTeamPoints')
    team_points[away_team] = row.get('AwayTeamPoints')
    team_goal_diff[home_team] = row.get('HomeTeamGoalDifference')
    team_goal_diff[away_team] = row.get('AwayTeamGoalDifference')


# Initialize standings
standings = {team: {'points': team_points[team], 'goal_difference': team_goal_diff[team]} for team in teams}

for _, match in df.iterrows():
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']


     # Predict matchene
    input_features ={
        'HomeTeamPoints': standings.get(home_team, {}).get('points', 0),
        'AwayTeamPoints': standings.get(away_team, {}).get('points', 0),
        'HomeTeamGoalDifference': standings.get(home_team, {}).get('goal_difference', 0),
        'AwayTeamGoalDifference': standings.get(away_team, {}).get('goal_difference', 0),
        'HS': 10,  # Assuming average shot statistics since we don’t have this data for upcoming games
        'HST': 5,  # Assuming average shot on target statistics
        'AS': 8,   # Assuming average shot statistics for away team
        'AST': 4   # Assuming average shot on target statistics for away team
    }

    input_df = pd.DataFrame([input_features])

    prediction = model.predict(input_df)[0]

    # oppdater poeng 
    if prediction == 1:  # Home win
        standings[home_team]['points'] += 3
    elif prediction == 0:  # Draw
        standings[home_team]['points'] += 1
        standings[away_team]['points'] += 1
    else:  # Away win
        standings[away_team]['points'] += 3

    # oppdater målforskjell
    standings[home_team]['goal_difference'] += match['GoalDifference']
    standings[away_team]['goal_difference'] -= match['GoalDifference']

# Convert kampene til DataFrame og sorter etter poeng og målforskjell 
standings_df = pd.DataFrame.from_dict(standings, orient='index').reset_index().rename(columns={'index': 'Team'})
standings_df = standings_df.sort_values(by=['points', 'goal_difference'], ascending=[False, False])

standings_df.insert(0, 'Position', range(1, len(standings_df) + 1))
# Save the final league table
standings_df.to_csv("data/league_table_simulation.csv", index=False)
print("League table simulation saved as 'data/league_table_simulation.csv'")

