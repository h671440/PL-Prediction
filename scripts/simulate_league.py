import pandas as pd
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
print("Loading trained model...")
model = joblib.load("models/pl_table_predictor.pkl")
print("Model loaded successfully!")

# Load the upcoming fixtures
print("Loading upcoming fixtures...")
upcoming_fixtures = pd.read_csv("data/PL_202425_fixtures.csv")
print("Fixtures loaded successfully!")

# Load processed dataset
print("Loading dataset for league simulation...")
df = pd.read_csv("data/processed_league_data.csv")
print("Dataset loaded successfully!")

# Load the saved LabelEncoder and StandardScaler
le = joblib.load('models/team_label_encoder.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
print("LabelEncoder and StandardScaler loaded successfully!")

# Load saved feature names
features = joblib.load('models/feature_names.pkl')
print("Feature names loaded successfully!")

# Define the team name mapping dictionary and function
team_name_mapping = {
    'Man United': 'Manchester Utd',
    'Ipswich': 'Ipswich Town',
    'Newcastle': 'Newcastle Utd',
    'Wolves': 'Wolverhampton',
    'Nott\'m Forest': 'Nottingham Forest',
    'West Ham': 'West Ham United',
    'Brighton': 'Brighton & Hove Albion',
    'Tottenham': 'Tottenham Hotspur',
    'Spurs': 'Tottenham Hotspur',
    'Leicester': 'Leicester City',
    'Man City': 'Manchester City',
    'Crystal Palace': 'Crystal Palace',
    'Aston Villa': 'Aston Villa',
    'Liverpool': 'Liverpool',
    'Chelsea': 'Chelsea',
    'Everton': 'Everton',
    'Brentford': 'Brentford',
    'Southampton': 'Southampton',
    'Arsenal': 'Arsenal',
    'Bournemouth': 'AFC Bournemouth',
}

def map_team_names(name):
    return team_name_mapping.get(name, name)

# Apply the mapping to current season results
current_season_results = pd.read_csv("data/PL_202425.csv")
current_season_results['HomeTeam'] = current_season_results['HomeTeam'].apply(map_team_names)
current_season_results['AwayTeam'] = current_season_results['AwayTeam'].apply(map_team_names)

# Apply the mapping to fixtures
upcoming_fixtures['home'] = upcoming_fixtures['home'].apply(map_team_names)
upcoming_fixtures['away'] = upcoming_fixtures['away'].apply(map_team_names)

# Encode team names in upcoming fixtures using the saved LabelEncoder
upcoming_fixtures['HomeTeam'] = le.transform(upcoming_fixtures['home'])
upcoming_fixtures['AwayTeam'] = le.transform(upcoming_fixtures['away'])

 # Check for unknown teams
unknown_teams = set(upcoming_fixtures['home']).union(set(upcoming_fixtures['away'])) - set(le.classes_)
if unknown_teams:
    print(f"Unknown teams found: {unknown_teams}")
    # Handle unknown teams by adding them to the LabelEncoder
    le.classes_ = np.append(le.classes_, list(unknown_teams))

# Encode team names in upcoming fixtures using the saved LabelEncoder
try:
    upcoming_fixtures['HomeTeam'] = le.transform(upcoming_fixtures['home'])
    upcoming_fixtures['AwayTeam'] = le.transform(upcoming_fixtures['away'])
except ValueError as e:
    logging.error(f"Error during team name encoding: {e}")
    exit()

# Filter necessary columns and encode team names
current_season_results = current_season_results[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'AS', 'AST']]
current_season_results['HomeTeam'] = le.transform(current_season_results['HomeTeam'])
current_season_results['AwayTeam'] = le.transform(current_season_results['AwayTeam'])


expected_features = features

# Initialize standings based on actual results
teams = le.transform(le.classes_)
standings = {team_id: {'points': 0, 'goal_difference': 0} for team_id in teams}

for _, match in current_season_results.iterrows():
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']
    home_goals = match['FTHG']
    away_goals = match['FTAG']

    # Update goal difference
    standings[home_team]['goal_difference'] += home_goals - away_goals
    standings[away_team]['goal_difference'] += away_goals - home_goals

    # Update points
    if home_goals > away_goals:
        standings[home_team]['points'] += 3
    elif home_goals < away_goals:
        standings[away_team]['points'] += 3
    else:
        standings[home_team]['points'] += 1
        standings[away_team]['points'] += 1

# Identify played and remaining fixtures
played_matches = current_season_results[['HomeTeam', 'AwayTeam']]
played_matches = played_matches.copy()  # Avoid SettingWithCopyWarning
played_matches['played'] = True

fixtures = upcoming_fixtures.merge(played_matches, on=['HomeTeam', 'AwayTeam'], how='left')
remaining_fixtures = fixtures[fixtures['played'].isna()].drop(columns=['played'])

print(f"Number of remaining fixtures to simulate: {len(remaining_fixtures)}")

# Calculate team statistics based on matches played so far
team_stats_home = current_season_results.groupby('HomeTeam').agg({
    'HS': 'mean',
    'HST': 'mean'
}).rename_axis('Team')

team_stats_away = current_season_results.groupby('AwayTeam').agg({
    'AS': 'mean',
    'AST': 'mean'
}).rename_axis('Team')

# Iterate over remaining fixtures to predict outcomes
for _, match in remaining_fixtures.iterrows():
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']

   # Use overall averages if team stats are missing
    hs_avg = team_stats_home['HS'].mean() if home_team not in team_stats_home.index else team_stats_home.loc[home_team]['HS']
    hst_avg = team_stats_home['HST'].mean() if home_team not in team_stats_home.index else team_stats_home.loc[home_team]['HST']
    as_avg = team_stats_away['AS'].mean() if away_team not in team_stats_away.index else team_stats_away.loc[away_team]['AS']
    ast_avg = team_stats_away['AST'].mean() if away_team not in team_stats_away.index else team_stats_away.loc[away_team]['AST']

    # Use current standings for goal differences
    home_team_goal_diff = standings[home_team]['goal_difference']
    away_team_goal_diff = standings[away_team]['goal_difference']

    # Prepare input features
    input_features_dict = {
        'HS': hs_avg,
        'HST': hst_avg,
        'AS': as_avg,
        'AST': ast_avg,
        'GoalDifference': home_team_goal_diff - away_team_goal_diff,
        'ShotsEfficiency_Home': hst_avg / hs_avg if hs_avg != 0 else 0,
        'ShotsEfficiency_Away': ast_avg / as_avg if as_avg != 0 else 0
    }

   # Build the input DataFrame with the expected features
    input_df = pd.DataFrame([input_features_dict], columns=features)

    # Ensure features are in the correct order
    input_df = input_df[features]

    # Standardize the input features using the saved StandardScaler
    input_df_scaled = scaler.transform(input_df)

    # Reconstruct a DataFrame with the feature names
    input_df_scaled = pd.DataFrame(input_df_scaled, columns=features)

    # Predict the match outcome
    prediction = model.predict(input_df_scaled)[0]
    prediction_proba = model.predict_proba(input_df_scaled)[0]

    
   # Adjust goal margin based on prediction probabilities
    if prediction == 2:  # Home win
        mean_goals = 1 + (prediction_proba[2] * 2)
    elif prediction == 1:  # Draw
        mean_goals = 1
    else:  # Away win
        mean_goals = 1 + (prediction_proba[0] * 2)
    goal_margin = max(1, int(np.round(mean_goals)))

    # Update standings based on prediction
    if prediction == 2:  # Home win
        standings[home_team]['points'] += 3
        standings[home_team]['goal_difference'] += goal_margin
        standings[away_team]['goal_difference'] -= goal_margin
    elif prediction == 1:  # Draw
        standings[home_team]['points'] += 1
        standings[away_team]['points'] += 1
    elif prediction == 0:  # Away win
        standings[away_team]['points'] += 3
        standings[away_team]['goal_difference'] += goal_margin
        standings[home_team]['goal_difference'] -= goal_margin

# Generate the final league table
standings_df = pd.DataFrame.from_dict(standings, orient='index').reset_index().rename(columns={'index': 'Team'})
standings_df['Team'] = le.inverse_transform(standings_df['Team'])
standings_df['points'] = standings_df['points'].astype(int)
standings_df['goal_difference'] = standings_df['goal_difference'].astype(int)
standings_df = standings_df.sort_values(by=['points', 'goal_difference'], ascending=[False, False])
standings_df.insert(0, 'Position', range(1, len(standings_df) + 1))

# Save the final league table
standings_df.to_csv("data/league_table_prediction.csv", index=False)
print("Final league table prediction saved as 'data/league_table_prediction.csv'")
