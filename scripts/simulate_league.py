import pandas as pd
import joblib
import numpy as np

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

# Encode team names in upcoming fixtures using the saved LabelEncoder
upcoming_fixtures['HomeTeam'] = le.transform(upcoming_fixtures['home'])
upcoming_fixtures['AwayTeam'] = le.transform(upcoming_fixtures['away'])

# Load the actual results of the current season
print("Loading current season's results...")
current_season_results = pd.read_csv("data/PL_202425.csv")
print("Current season results loaded successfully!")

# Filter necessary columns and encode team names
current_season_results = current_season_results[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'AS', 'AST']]
current_season_results['HomeTeam'] = le.transform(current_season_results['HomeTeam'])
current_season_results['AwayTeam'] = le.transform(current_season_results['AwayTeam'])


expected_features = scaler.feature_names_in_

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

    # Get team statistics, with fallbacks
    hs_avg = team_stats_home.loc[home_team]['HS'] if home_team in team_stats_home.index else df['HS'].mean()
    hst_avg = team_stats_home.loc[home_team]['HST'] if home_team in team_stats_home.index else df['HST'].mean()
    as_avg = team_stats_away.loc[away_team]['AS'] if away_team in team_stats_away.index else df['AS'].mean()
    ast_avg = team_stats_away.loc[away_team]['AST'] if away_team in team_stats_away.index else df['AST'].mean()

    # Use current standings for goal differences
    home_team_goal_diff = standings[home_team]['goal_difference']
    away_team_goal_diff = standings[away_team]['goal_difference']

    # Prepare input features
    input_features_dict = {
        'HS': hs_avg,
        'AS': as_avg,
        'HST': hst_avg,
        'AST': ast_avg,
        'GoalDifference': home_team_goal_diff - away_team_goal_diff,
        'ShotsEfficiency_Home': hst_avg / hs_avg if hs_avg != 0 else 0,
        'ShotsEfficiency_Away': ast_avg / as_avg if as_avg != 0 else 0
    }

   # Build the input DataFrame with the expected features
    input_df = pd.DataFrame([input_features_dict], columns=expected_features)

    # Check for missing features and handle them
    missing_features = set(expected_features) - set(input_df.columns)
    if missing_features:
        print(f"Missing features: {missing_features}")
        for feature in missing_features:
            input_df[feature] = df[feature].mean()  # or another appropriate default value

    # Ensure features are in the correct order
    input_df = input_df[expected_features]

    # Standardize the input features using the saved StandardScaler
    input_df_scaled = scaler.transform(input_df)

    # Predict the match outcome
    prediction = model.predict(input_df_scaled)[0]

    # Simulate goal margin
    goal_margin = np.random.poisson(1)

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
