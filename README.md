Premier League Prediction Project - README

Overview

This project simulates the Premier League table based on historical data, upcoming fixtures, and a trained machine learning model. It processes match statistics from previous seasons, trains a predictive model, and simulates league standings for the current season.

The project includes the following components:

Data preprocessing
Model training
League table simulation
Visualization using a web interface
Project Files

preprocess_data.py:

Purpose: Prepares and cleans historical match data for model training.

Functionality:
Combines match data from previous seasons (e.g., 2020-2024) into a single dataset.
Maps team names to standardized formats for consistency.
Calculates new features, such as shot efficiency and team strength.
Encodes team names using a LabelEncoder and scales numerical features using a StandardScaler.
Saves the processed dataset and encoders for later use.

Output: A cleaned dataset (processed_league_data.csv) and models (team_label_encoder.pkl, feature_scaler.pkl).
train_model.py

Purpose: Trains a Random Forest classifier to predict match outcomes (Home Win, Draw, Away Win).

Functionality:
Splits the data into training and test sets with stratified sampling to handle class imbalances.
Handles imbalanced classes using undersampling techniques.
Performs hyperparameter tuning with GridSearchCV for optimal model performance.
Evaluates the model using cross-validation and outputs metrics such as accuracy.
Saves the trained model (pl_table_predictor.pkl) for simulation.

Performance:
Training Accuracy: ~95%
Test Accuracy: ~54-56%
Cross-Validation Accuracy: ~55-57%
simulate_league.py

Purpose: Simulates the Premier League table based on the trained model.

Functionality:
Predicts outcomes for upcoming fixtures using historical statistics and the trained model.
Updates league standings by incorporating both historical match results and predicted results for remaining fixtures.
Runs multiple simulations (e.g., 25 iterations) to calculate average points and goal differences for each team.
Saves the simulated league table (league_table_prediction.csv).

Output:
Simulated league table showing positions, points, and goal differences after the first 10 rounds or the entire season.

app.py:
Purpose: Provides a web interface to display the simulated Premier League table.

Functionality:
Loads the simulated league table.
Dynamically adds team logos to the table for improved visualization.
Renders the league table using a Bootstrap-styled HTML table.
Runs a Flask web server to serve the application.
Data Used

Historical Premier League match data from the 2020/21 to 2023/24 seasons (PL_202021.csv, PL_202122.csv, etc.).
Upcoming fixtures for the current season (PL_202425_fixtures.csv).

Key Features in the Dataset:
Match statistics such as goals scored, shots, shots on target.
New features such as shot efficiency and team strength.
Encoded match outcomes (FTR):
2 = Home Win,
1 = Draw,
0 = Away Win.
How It Works

Preprocessing: Combines and cleans historical data, calculates features, and encodes categorical variables.
Model Training: Trains a Random Forest classifier to predict match outcomes based on historical data.
Simulation: Predicts outcomes for upcoming matches and updates the league table.
Web Visualization: Displays the simulated league table with team logos.

Key Insights:

The model is trained using historical match data and team-specific statistics.
Cross-validation ensures the model performs consistently across different data splits.
Simulation results depend on:
Team strengths (predefined values in preprocess_data.py).
Historical and predicted performance of teams.
How to Run

Preprocess Data:
python preprocess_data.py
Train Model:
python train_model.py
Simulate League:
python simulate_league.py
Run Web App:
python app.py
Open the app in your browser at http://127.0.0.1:5000/.
Outputs

processed_league_data.csv: Preprocessed dataset.
pl_table_predictor.pkl: Trained model for match prediction.
league_table_prediction.csv: Simulated league table.
Web Interface: Displays the league table with team logos.
Accuracy and Limitations

Accuracy:
Training: ~95%
Test: ~54-56%
Cross-Validation: ~55-57%

Limitations:
Predictions rely on historical data, which may not fully represent current team performance.
Team strength values are static and do not account for dynamic changes during the season.
Specially Ipswich Town is no data about since they have not been in the premier league for a while.
