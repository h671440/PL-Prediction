import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



print("Loading processed dataset...")
df = pd.read_csv("data/processed_league_data.csv")
print("Dataset loaded successfully!")

# Load the saved LabelEncoder and StandardScaler (if needed)
le = joblib.load('models/team_label_encoder.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
features = joblib.load('models/feature_names.pkl')
print("Feature names loaded successfully!")

target = 'FTR'

if target in features:
    features.remove(target) 

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

# Handle class imbalance with undersampling
undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Include scaling
    ('classifier', rf)
])


# Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 15],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 5],
    'classifier__max_features': ['sqrt']
}
# Set up GridSearchCV
print("Starting hyperparameter tuning using GridSearchCV...")
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model
best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the best model
print("Evaluating the best model...")
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Generate classification report
y_pred = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Perform cross-validation
print("Performing cross-validation...")
cv_scores = cross_val_score(best_model, X, y, cv=tscv, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")



#lagre den trenede modellen
joblib.dump(best_model, "models/pl_table_predictor.pkl")
print("model saved as: models/pl_table_predictor.pkl")

# Save the feature names
joblib.dump(features, "models/feature_names.pkl")
print("Feature names saved as 'models/feature_names.pkl'")


