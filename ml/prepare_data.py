import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load dataset
dataset_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Crop_recommendation.csv"
)

df = pd.read_csv(dataset_path)

# Input features (X)
X = df.drop(columns=["label"])

# Target output (y)
y = df["label"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display results
print("DATA PREPARATION SUCCESSFUL!")
print("-" * 40)

print("Total dataset:", X.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining records:", len(y_train))
print("Testing records:", len(y_test))

print("\nTraining crop distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting crop distribution:")
print(y_test.value_counts().sort_index())

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget column:", y.name)