import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading processed dataset...")
df = pd.read_csv("data/processed_league_data.csv")
print("Dataset loaded successfully!")

# Load the saved LabelEncoder and StandardScaler (if needed)
le = joblib.load('models/team_label_encoder.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Load saved feature names
features = joblib.load('models/feature_names.pkl')
print("Feature names loaded successfully!")

target = 'FTR'


# Fill missing values instead of dropping them
print("Filling missing values in features and target...")
df[features] = df[features].fillna(df[features].mean())
# df[target] = df[target].fillna(df[target].mode()[0])

# Analyze class distribution
class_counts = df[target].value_counts()
print("Class distribution in target variable:")
print(class_counts)

# Ensure that all classes are represented
if class_counts.min() == 0:
    print("Warning: Some classes are not represented in the data.")
    


# Split data into features and target
X = df[features]
y = df[target]
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

#split data inn i trening og test set
# Correct split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


#trene modellen
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

#evaluer modellen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"model accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Perform cross-validation
print("Performing cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")


#lagre den trenede modellen
joblib.dump(model, "models/pl_table_predictor.pkl")
print("model saved as: models/pl_table_predictor.pkl")

# Save the feature names
joblib.dump(features, "models/feature_names.pkl")
print("Feature names saved as 'models/feature_names.pkl'")


