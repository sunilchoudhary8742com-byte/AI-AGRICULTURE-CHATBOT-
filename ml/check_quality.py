import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parent.parent / "data" / "Crop_recommendation.csv"
df = pd.read_csv(file_path)

print("TOTAL ROWS AND COLUMNS:", df.shape)
print("TOTAL MISSING VALUES:", df.isnull().sum().sum())

print("DUPLICATE ROWS:", df.duplicated().sum())

print("\nRECORDS PER CROP:")
print(df["label"].value_counts().sort_index())
