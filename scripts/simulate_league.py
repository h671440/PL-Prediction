import random
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

# Ensure 'GameWeek' column exists
if 'GameWeek' not in current_season_results.columns:
    if 'Date' in current_season_results.columns:
        try:
            current_season_results['Date'] = pd.to_datetime(current_season_results['Date'], format='%d/%m/%Y')
        except Exception as e:
            print(f"Error while parsing dates: {e}")    
        
        # Create a date-to-GameWeek mapping based on the actual schedule
        date_to_gameweek = {
            # Add actual date-to-gameweek mappings here
            
    "2024-08-16": 1,
    "2024-08-17": 1,
    "2024-08-18": 1,
    "2024-08-19": 1,
    "2024-08-24": 2,
    "2024-08-25": 2,
    "2024-08-31": 3,
    "2024-09-01": 3,
    "2024-09-14": 4,
    "2024-09-15": 4,
    "2024-09-21": 5,
    "2024-09-22": 5,
    "2024-09-28": 6,
    "2024-09-29": 6,
    "2024-09-30": 6,
    "2024-10-05": 7,
    "2024-10-06": 7,
    "2024-10-19": 8,
    "2024-10-20": 8,
    "2024-10-21": 8,
    "2024-10-26": 9,
    "2024-10-25": 9,
    "2024-10-27": 9,
    "2024-11-02": 10,
    "2024-11-03": 10,
    "2024-11-04": 10,
    "2024-11-09": 11,
    "2024-11-23": 12,
    "2024-11-30": 13,
    "2024-12-03": 14,
    "2024-12-04": 14,
    "2024-12-07": 15,
    "2024-12-14": 16,
    "2024-12-21": 17,
    "2024-12-26": 18,
    "2024-12-29": 19,
    "2025-01-04": 20,
    "2025-01-14": 21,
    "2025-01-15": 21,
    "2025-01-18": 22,
    "2025-01-25": 23,
    "2025-02-01": 24,
    "2025-02-15": 25,
    "2025-02-22": 26,
    "2025-02-25": 27,
    "2025-02-26": 27,
    "2025-03-08": 28,
    "2025-03-15": 29,
    "2025-04-01": 30,
    "2025-04-02": 30,
    "2025-04-05": 31,
    "2025-04-12": 32,
    "2025-04-19": 33,
    "2025-04-26": 34,
    "2025-05-03": 35,
    "2025-05-10": 36,
    "2025-05-18": 37,
    "2025-05-25": 38
   
   
   


        }
        
         # Apply the date to GameWeek mapping
        
        current_season_results['GameWeek'] = current_season_results['Date'].dt.strftime('%Y-%m-%d').map(date_to_gameweek)
        
      
        missing_dates = current_season_results[current_season_results['GameWeek'].isna()]['Date'].unique()
        if len(missing_dates) > 0:
            print(f"Missing mappings for dates: {missing_dates}")
            exit("Error: Some dates are missing in the date_to_gameweek mapping.")
    else:
        exit("Error: Neither 'GameWeek' nor 'Date' column found in PL_202425.csv")



# Filter to include only games up to game week 10
current_season_results = current_season_results[current_season_results['GameWeek'] <= 10]

current_season_results['HomeTeam'] = current_season_results['HomeTeam'].apply(map_team_names)
current_season_results['AwayTeam'] = current_season_results['AwayTeam'].apply(map_team_names)

# Apply the mapping to fixtures
upcoming_fixtures['home'] = upcoming_fixtures['home'].apply(map_team_names)
upcoming_fixtures['away'] = upcoming_fixtures['away'].apply(map_team_names)

# Check for unknown teams
unknown_teams = set(upcoming_fixtures['home']).union(set(upcoming_fixtures['away'])) - set(le.classes_)
if unknown_teams:
    print(f"Unknown teams found: {unknown_teams}")
    # Update the LabelEncoder to include unknown teams
    le.classes_ = np.concatenate([le.classes_, np.array(list(unknown_teams))])
    le.classes_.sort()

# Encode team names in upcoming fixtures using the saved LabelEncoder
upcoming_fixtures['HomeTeam'] = le.transform(upcoming_fixtures['home'])
upcoming_fixtures['AwayTeam'] = le.transform(upcoming_fixtures['away'])

# Filter necessary columns and encode team names
current_season_results = current_season_results[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'AS', 'AST']]
current_season_results['HomeTeam'] = le.transform(current_season_results['HomeTeam'])
current_season_results['AwayTeam'] = le.transform(current_season_results['AwayTeam'])

# Identify played and remaining fixtures
played_matches = current_season_results[['HomeTeam', 'AwayTeam']]
played_matches = played_matches.copy()  # Avoid SettingWithCopyWarning
played_matches['played'] = True

fixtures = upcoming_fixtures.merge(played_matches, on=['HomeTeam', 'AwayTeam'], how='left')
remaining_fixtures = fixtures[fixtures['played'].isna()].drop(columns=['played'])

print(f"Number of remaining fixtures to simulate: {len(remaining_fixtures)}")
total_matches = len(current_season_results) + len(remaining_fixtures)
print(f"Total matches accounted for: {total_matches}")

# Calculate team statistics based on matches played so far
team_stats_home = current_season_results.groupby('HomeTeam').agg({
    'HS': 'mean',
    'HST': 'mean'
}).rename_axis('Team')

team_stats_away = current_season_results.groupby('AwayTeam').agg({
    'AS': 'mean',
    'AST': 'mean'
}).rename_axis('Team')

# Calculate league averages
league_hs_avg = team_stats_home['HS'].mean()
league_hst_avg = team_stats_home['HST'].mean()
league_as_avg = team_stats_away['AS'].mean()
league_ast_avg = team_stats_away['AST'].mean()

def blended_average(team_avg, league_avg, weight=0.7):
    return weight * team_avg + (1 - weight) * league_avg

# Define caps for shots and shots on target
HS_CAP = team_stats_home['HS'].quantile(0.95)
HST_CAP = team_stats_home['HST'].quantile(0.95)
AS_CAP = team_stats_away['AS'].quantile(0.95)
AST_CAP = team_stats_away['AST'].quantile(0.95)

# Number of simulations
num_simulations = 3
standings_list = []

for simulation in range(num_simulations):
    print(f"Starting simulation {simulation + 1}...")

    # Initialize standings for this simulation
    teams_in_fixtures = le.transform(le.classes_)
    standings = {team_id: {'points': 0, 'goal_difference': 0} for team_id in teams_in_fixtures}

    # Update standings based on actual results
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

    if simulation == 0:
        print("Standings after actual results (first 10 games):")
        standings_df_actual = pd.DataFrame.from_dict(standings, orient='index').reset_index().rename(columns={'index': 'Team'})
        standings_df_actual['Team'] = le.inverse_transform(standings_df_actual['Team'])
        standings_df_actual['points'] = standings_df_actual['points'].astype(int)
        standings_df_actual['goal_difference'] = standings_df_actual['goal_difference'].astype(int)
        standings_df_actual = standings_df_actual.sort_values(by=['points', 'goal_difference'], ascending=[False, False])
        print(standings_df_actual[['Team', 'points', 'goal_difference']])

    # Simulate remaining fixtures
    for _, match in remaining_fixtures.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']

        # Blend team averages with league averages
        if home_team in team_stats_home.index:
            hs_avg = blended_average(team_stats_home.loc[home_team]['HS'], league_hs_avg)
            hst_avg = blended_average(team_stats_home.loc[home_team]['HST'], league_hst_avg)
        else:
            hs_avg = league_hs_avg
            hst_avg = league_hst_avg

        if away_team in team_stats_away.index:
            as_avg = blended_average(team_stats_away.loc[away_team]['AS'], league_as_avg)
            ast_avg = blended_average(team_stats_away.loc[away_team]['AST'], league_ast_avg)
        else:
            as_avg = league_as_avg
            ast_avg = league_ast_avg

        # Cap extreme values
        hs_avg = min(hs_avg, HS_CAP)
        hst_avg = min(hst_avg, HST_CAP)
        as_avg = min(as_avg, AS_CAP)
        ast_avg = min(ast_avg, AST_CAP)

        # Prepare input features
        input_features_dict = {
            'HS': hs_avg,
            'HST': hst_avg,
            'AS': as_avg,
            'AST': ast_avg,
            'ShotsEfficiency_Home': hst_avg / hs_avg if hs_avg != 0 else 0,
            'ShotsEfficiency_Away': ast_avg / as_avg if as_avg != 0 else 0,
        }

        # Add team strength features if present in the saved features list
        if 'HomeTeamStrength' in features:
            input_features_dict['HomeTeamStrength'] = df[df['HomeTeam'] == home_team]['HomeTeamStrength'].mean()
        if 'AwayTeamStrength' in features:
            input_features_dict['AwayTeamStrength'] = df[df['AwayTeam'] == away_team]['AwayTeamStrength'].mean()

        # Build the input DataFrame with the expected features in the correct order
        input_df = pd.DataFrame([input_features_dict])

        # Reorder columns to match the saved features list
        input_df = input_df[features]

        # Predict the match outcome
        prediction_proba = model.predict_proba(input_df)[0]

        # Sample the outcome based on probabilities
        outcome = np.random.choice(model.classes_, p=prediction_proba)
        goal_margin = random.choice([1, 2, 3])

        # Update standings based on sampled outcome
        if outcome == 2:  # Home win
            standings[home_team]['points'] += 3
            standings[home_team]['goal_difference'] += goal_margin
            standings[away_team]['goal_difference'] -= goal_margin
        elif outcome == 1:  # Draw
            standings[home_team]['points'] += 1
            standings[away_team]['points'] += 1
            # Goal difference remains the same
        elif outcome == 0:  # Away win
            standings[away_team]['points'] += 3
            standings[away_team]['goal_difference'] += goal_margin
            standings[home_team]['goal_difference'] -= goal_margin

    # Convert standings to DataFrame for this simulation
    standings_df = pd.DataFrame.from_dict(standings, orient='index').reset_index().rename(columns={'index': 'Team'})
    standings_df['Team'] = le.inverse_transform(standings_df['Team'])
    standings_df['points'] = standings_df['points'].astype(int)
    standings_df['goal_difference'] = standings_df['goal_difference'].astype(int)
    standings_df['Simulation'] = simulation + 1

    # Add the standings to the list
    standings_list.append(standings_df)

# After all simulations, average the standings
# Concatenate all standings DataFrames
all_standings = pd.concat(standings_list)

# Group by team and calculate average points and goal difference
average_standings = all_standings.groupby('Team').agg({
    'points': 'mean',
    'goal_difference': 'mean'
}).reset_index()

# Round the averaged points and goal differences to integers
average_standings['points'] = average_standings['points'].round().astype(int)
average_standings['goal_difference'] = average_standings['goal_difference'].round().astype(int)

# Sort the standings
average_standings = average_standings.sort_values(by=['points', 'goal_difference'], ascending=[False, False])
average_standings.insert(0, 'Position', range(1, len(average_standings) + 1))

# Save the final league table
average_standings.to_csv("data/league_table_prediction.csv", index=False)
print("Final league table prediction saved as 'data/league_table_prediction.csv'")
