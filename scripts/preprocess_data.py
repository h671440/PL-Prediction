import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("Current working directory:", os.getcwd())

data_files = [ "data/PL_202324.csv", "data/PL_202223.csv", "data/PL_202122.csv", "data/PL_202021.csv"] ## ekskludert årets sesongs tidligere resultater "data/PL_202425.csv"

print("Loading datasets...")

data_frames = []

for file in data_files:
    df_temp = pd.read_csv(file)
    data_frames.append(df_temp)
    print(f"Loaded {file} with shape {df_temp.shape}")

df= pd.concat(data_frames, ignore_index = True)
print("combined dataset shape: " , df.shape)


# Load upcoming fixtures to get all teams
print("Loading upcoming fixtures...")
upcoming_fixtures = pd.read_csv("data/PL_202425_fixtures.csv")


# Define a mapping dictionary to standardize team names
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


# Function to map team names
def map_team_names(name):
    return team_name_mapping.get(name, name)

# Apply the mapping to historical data
df['HomeTeam'] = df['HomeTeam'].apply(map_team_names)
df['AwayTeam'] = df['AwayTeam'].apply(map_team_names)

# Apply the mapping to fixtures
upcoming_fixtures['home'] = upcoming_fixtures['home'].apply(map_team_names)
upcoming_fixtures['away'] = upcoming_fixtures['away'].apply(map_team_names)

current_teams = pd.unique(upcoming_fixtures[['home', 'away']].values.ravel('K'))
print("print current teams", current_teams)

# Filter historical data for current teams
print("Filtering historical data for current season teams...")
df = df[df['HomeTeam'].isin(current_teams) & df['AwayTeam'].isin(current_teams)]
print("Filtered dataset shape:", df.shape)


# Step 4: Select Necessary Columns
columns_needed = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'HS', 'AS', 'HST', 'AST'
]
df = df[columns_needed]
print("DataFrame shape after selecting necessary columns:", df.shape)


# Handle missing values
print("Filling missing values...")

numeric_columns = ['FTHG', 'FTAG', 'HST', 'AST', 'HS', 'AS']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
print("DataFrame shape after filling missing values:", df.shape)

# Create new columns for goal difference and shot efficiency
print("Creating new features...")
# df['GoalDifference'] = df['FTHG'] - df['FTAG']
df['ShotsEfficiency_Home'] = df['HST'] / df['HS'].replace(0, 1)
df['ShotsEfficiency_Away'] = df['AST'] / df['AS'].replace(0, 1)
print("DataFrame shape after creating new features:", df.shape)

# Create a new feature for team strength

team_strengths = {
    'Manchester City': 92,
    'Liverpool': 89,
    'Manchester Utd': 80,
    'Chelsea': 82,
    'Tottenham Hotspur': 82,
    'Arsenal': 88,
    'Newcastle Utd': 75,
    'Aston Villa': 73,
    'Brighton & Hove Albion': 72,
    'West Ham United': 70,
    'Wolverhampton': 68,
    'Leicester City': 64,
    'Everton': 62,
    'Brentford': 61,
    'Crystal Palace': 61,
    'Fulham': 60,
    'Southampton': 58,
    'AFC Bournemouth': 58,
    'Nottingham Forest': 59,
    'Ipswich Town': 50
}


# Assign strength values to each team in the dataset
df['HomeTeamStrength'] = df['HomeTeam'].map(team_strengths)
df['AwayTeamStrength'] = df['AwayTeam'].map(team_strengths)

# Convert result to numerical values
print("Unique values in 'FTR' before mapping:", df['FTR'].unique())
df['FTR'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0}).fillna(-1).astype(int)
print("Unique values in 'FTR' after mapping:", df['FTR'].unique())
print("Number of NaN values in 'FTR' after mapping:", df['FTR'].isna().sum())


# Drop rows with missing data in critical columns
critical_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
df.dropna(subset=critical_columns, inplace=True)
print("DataFrame shape after dropping missing data in critical columns:", df.shape)


# Get all teams from historical data and upcoming fixtures
teams_in_df = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
teams_in_fixtures = pd.unique(upcoming_fixtures[['home', 'away']].values.ravel('K'))
all_teams = pd.Series(list(set(teams_in_df).union(set(teams_in_fixtures))))
print("All teams:", all_teams.tolist())

#filtrer historiske data for nåverende lag..
print("Filtering historical data for current season teams...")
df = df[df['HomeTeam'].isin(all_teams) & df['AwayTeam'].isin(all_teams)]
print("Filtered dataset shape:", df.shape)
print("DataFrame shape after filtering:", df.shape)

# Fit LabelEncoder on all teams
le = LabelEncoder()
le.fit(all_teams)

# Encode team names in df
df['HomeTeam'] = le.transform(df['HomeTeam'])
df['AwayTeam'] = le.transform(df['AwayTeam'])

# Save the LabelEncoder
joblib.dump(le, 'models/team_label_encoder.pkl')
print("Team LabelEncoder saved successfully!")

# Calculate Cumulative Statistics by Season
print("Calculating cumulative statistics...")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Season'] = df['Date'].dt.year.astype(str)

# Standardize numeric features
print("Standardizing numeric features...")
scaler = StandardScaler()
numeric_features = ['HS', 'AS', 'HST', 'AST', 'ShotsEfficiency_Home',
                     'ShotsEfficiency_Away', 'HomeTeamStrength', 'AwayTeamStrength']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

#  Save the scaler and feature names
joblib.dump(scaler, 'models/feature_scaler.pkl')
joblib.dump(numeric_features, 'models/feature_names.pkl')
print("Feature scaler and feature names saved successfully!")

# Save the processed data
df.to_csv("data/processed_league_data.csv", index=False)
print("Data preprocessing complete and saved as 'data/processed_league_data.csv'")
