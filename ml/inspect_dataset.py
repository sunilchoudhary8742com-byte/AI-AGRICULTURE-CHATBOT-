import pandas as pd
from pathlib import Path

dataset_path = Path(__file__).resolve().parent.parent / "data" / "Crop_recommendation.csv"
df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print("Dataset shape:" ,df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values :")
print(df.isnull().sum())

print("\nNumber of crop categories:", df["label"].nunique())
print("\nCrop categories:")
print(sorted(df["label"].unique()))

print("\nDataset information:")
print(df.info())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nRecords per crop:")
print(df["label"].value_counts().sort_index())

print("\nStatistical summary:")
print(df.describe().round(2))