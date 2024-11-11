import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

print("Loading processed dataset...")
df = pd.read_csv("data/processed_league_data.csv")
print("Dataset loaded successfully!")

# Load the saved LabelEncoder and StandardScaler (if needed)
le = joblib.load('models/team_label_encoder.pkl')
scaler = joblib.load('models/feature_scaler.pkl')



# Definer features and target
features = [
    'HS', 'HST', 'AS', 'AST', 'GoalDifference',
    'ShotsEfficiency_Home', 'ShotsEfficiency_Away'
]
target = 'FTR'

# Dropp rader som mangler verdier fra heler databildet
print("Dropping rows with missing values...")
df.dropna(subset=features + [target], inplace=True)

# Split data into features and target
X = df[features]
y = df[target]
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

#split data inn i trening og test set
# Correct split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#trene modellen
print("training model..")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#evaluer modellen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"model accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))

#lagre den trenede modellen
joblib.dump(model, "models/pl_table_predictor.pkl")
print("model saved as: models/pl_table_predictor.pkl")

# Save the feature names
joblib.dump(features, "models/feature_names.pkl")
print("Feature names saved as 'models/feature_names.pkl'")


