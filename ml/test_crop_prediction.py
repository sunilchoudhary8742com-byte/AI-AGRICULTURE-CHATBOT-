import joblib
import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
model_path = base_path / "ml" / "crop_recommendation_model.joblib"

model = joblib.load(model_path)

# Sample input: N, P, K, temperature, humidity, pH, rainfall
sample = pd.DataFrame(
    [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]],
    columns=[
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ]
)

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
confidence = probabilities.max() * 100

print("CROP PREDICTION SUCCESSFUL!")
print("Recommended crop:", prediction)
print(f"Model confidence: {confidence:.2f}%")