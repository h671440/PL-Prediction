import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
print("Loading trained model...")
model = joblib.load("models/pl_table_predictor.pkl")
print("Model loaded successfully!")

# Load the upcoming fixtures
print("Loading upcoming fixtures...")
upcoming_fixtures = pd.read_csv("data/PL_202425_fixtures.csv")
print("Fixtures loaded successfully!")

# Rename columns for consistency
upcoming_fixtures.rename(columns={"home": "HomeTeam", "away": "AwayTeam"}, inplace=True)

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
print("Encoding team names in upcoming fixtures...")
upcoming_fixtures['HomeTeam'] = le.transform(upcoming_fixtures['HomeTeam'])
upcoming_fixtures['AwayTeam'] = le.transform(upcoming_fixtures['AwayTeam'])
print("Team names encoded successfully!")

# Initialize standings with 0 points and goal difference for each team
teams = le.classes_
team_ids = le.transform(teams)
standings = {team_id: {'points': 0, 'goal_difference': 0} for team_id in team_ids}

# Update standings based on historical matches in the dataset
print("Updating standings based on historical data...")
for index, row in df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    # Update points based on full-time result
    if row['FTR'] == 1:  # Home Win
        standings[home_team]['points'] += 3
    elif row['FTR'] == 0:  # Draw
        standings[home_team]['points'] += 1
        standings[away_team]['points'] += 1
    elif row['FTR'] == -1:  # Away Win
        standings[away_team]['points'] += 3

    # Update goal difference
    standings[home_team]['goal_difference'] += row['GoalDifference']
    standings[away_team]['goal_difference'] -= row['GoalDifference']
print("Standings updated successfully!")

print("Preparing data for upcoming fixtures prediction...")

# Use team averages for shots statistics for prediction
team_stats_home = df.groupby('HomeTeam').agg({
    'HS': 'mean',
    'HST': 'mean',
    'GoalDifference': 'mean',
    'HomeTeamPoints': 'mean',
    'HomeTeamGoalDifference': 'mean'
}).rename_axis('Team')

team_stats_away = df.groupby('AwayTeam').agg({
    'AS': 'mean',
    'AST': 'mean',
    'AwayTeamPoints': 'mean',
    'AwayTeamGoalDifference': 'mean'
}).rename_axis('Team')

# Iterate over upcoming fixtures to predict outcomes
for _, match in upcoming_fixtures.iterrows():
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']

    # Use historical averages if available, otherwise fallback to general average values
    hs_avg = team_stats_home.loc[home_team]['HS'] if home_team in team_stats_home.index else df['HS'].mean()
    hst_avg = team_stats_home.loc[home_team]['HST'] if home_team in team_stats_home.index else df['HST'].mean()
    as_avg = team_stats_away.loc[away_team]['AS'] if away_team in team_stats_away.index else df['AS'].mean()
    ast_avg = team_stats_away.loc[away_team]['AST'] if away_team in team_stats_away.index else df['AST'].mean()
    goal_diff_avg = team_stats_home.loc[home_team]['GoalDifference'] if home_team in team_stats_home.index else 0

    ## Prepare input features for prediction using the correct feature order
    input_features_dict = {
        'HomeTeamPoints': standings.get(home_team, {}).get('points', 0),
        'AwayTeamPoints': standings.get(away_team, {}).get('points', 0),
        'HomeTeamGoalDifference': standings.get(home_team, {}).get('goal_difference', 0),
        'AwayTeamGoalDifference': standings.get(away_team, {}).get('goal_difference', 0),
        'HS': hs_avg,
        'HST': hst_avg,
        'AS': as_avg,
        'AST': ast_avg,
        'GoalDifference': goal_diff_avg
    }

    input_features = [input_features_dict[feature] for feature in features]


# Convert input to DataFrame format for model prediction with correct feature order
   
    input_df = pd.DataFrame([input_features], columns=features)

# Standardize the input features using the saved StandardScaler
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=features)

# Predict the match outcome
    prediction = model.predict(input_df_scaled)[0]

    # Update points based on prediction
    if prediction == 1:  # Home win
        standings[home_team]['points'] += 3
    elif prediction == 0:  # Draw
        standings[home_team]['points'] += 1
        standings[away_team]['points'] += 1
    else:  # Away win
        standings[away_team]['points'] += 3

    # Update goal difference (based on simple assumptions)
    standings[home_team]['goal_difference'] += 1 if prediction == 1 else -1 if prediction == -1 else 0
    standings[away_team]['goal_difference'] += -1 if prediction == 1 else 1 if prediction == -1 else 0

# Convert standings to DataFrame and sort by points and goal difference
standings_df = pd.DataFrame.from_dict(standings, orient='index').reset_index().rename(columns={'index': 'Team'})
standings_df['Team'] = standings_df['Team'].astype(int)
standings_df['Team'] = le.inverse_transform(standings_df['Team'])
standings_df = standings_df.sort_values(by=['points', 'goal_difference'], ascending=[False, False])

# Add position column
standings_df.insert(0, 'Position', range(1, len(standings_df) + 1))

# Save the final league table to CSV
standings_df.to_csv("data/league_table_simulation.csv", index=False)
print("League table simulation saved as 'data/league_table_simulation.csv'")
