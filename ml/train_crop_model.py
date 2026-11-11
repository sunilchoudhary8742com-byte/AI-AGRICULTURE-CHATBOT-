import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
base_path = Path(__file__).resolve().parent.parent
df = pd.read_csv(base_path / "data" / "Crop_recommendation.csv")

# Separate features and target
X = df.drop(columns=["label"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("MODEL TRAINING SUCCESSFUL!")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
model_path = base_path / "ml" / "crop_recommendation_model.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved at: {model_path}")