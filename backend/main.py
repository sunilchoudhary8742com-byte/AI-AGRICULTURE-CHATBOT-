
from pathlib import Path
import io
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import requests

from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    Depends
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy.orm import Session
from sqlalchemy import text, desc

from backend.auth_routes import router as auth_router
from backend.database.connection import get_db
from backend.database.models import Farmer, PredictionHistory
from backend.dependencies import get_current_farmer


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Agriculture Assistant",
    description="An AI-powered assistant to help farmers",
    version="1.3.0"
)

app.include_router(auth_router)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================
# LOAD CROP RECOMMENDATION MODEL
# ==========================================

MODEL_PATH = BASE_DIR / "ml" / "crop_recommendation_model.joblib"
model = joblib.load(MODEL_PATH)


# ==========================================
# LOAD PLANT DISEASE MODEL
# ==========================================

DISEASE_MODEL_PATH = BASE_DIR / "ml" / "plant_disease_model.keras"
DISEASE_CLASSES_PATH = BASE_DIR / "ml" / "plant_disease_classes.json"

disease_model = tf.keras.models.load_model(
    DISEASE_MODEL_PATH
)

with open(
    DISEASE_CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    disease_classes = json.load(f)


# ==========================================
# LOAD WEATHER API KEY
# ==========================================

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# ==========================================
# CROP INPUT VALIDATION
# ==========================================

class CropInput(BaseModel):
    N: float = Field(ge=0, le=300)
    P: float = Field(ge=0, le=300)
    K: float = Field(ge=0, le=300)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=5000)
    crop: str | None = None


# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Welcome to AI Agriculture Assistant API",
        "status": "running"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "crop_model_loaded": True,
        "disease_model_loaded": True
    }


# ==========================================
# DATABASE CONNECTION TEST
# ==========================================

@app.get("/db-health")
def database_health(
    db: Session = Depends(get_db)
):
    try:
        db.execute(text("SELECT 1"))

        return {
            "status": "connected",
            "database": "SQLite",
            "message": "Database connection successful!"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )


# ==========================================
# CROP RECOMMENDATION + SAVE HISTORY
# ==========================================

@app.post("/predict")
def predict_crop(
    data: CropInput,
    db: Session = Depends(get_db),
    current_farmer: Farmer = Depends(get_current_farmer)
):

    input_data = pd.DataFrame(
        [[
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall
        ]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    top_indices = probabilities.argsort()[::-1][:3]

    top_3_crops = [
        {
            "crop": str(classes[i]),
            "model_confidence_percent": round(
                float(probabilities[i]) * 100,
                2
            )
        }
        for i in top_indices
    ]

    result = {
        "recommended_crop": str(prediction),
        "top_3_crops": top_3_crops,
        "note": (
            "Recommendations are based on the trained model. "
            "Verify results with local agricultural experts."
        )
    }

    # Save prediction in database
    try:
        history = PredictionHistory(
            farmer_id=current_farmer.id,
            prediction_type="crop_recommendation",
            input_data=data.model_dump(),
            result=result
        )

        db.add(history)
        db.commit()
        db.refresh(history)

    except Exception as e:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Could not save prediction history: {str(e)}"
        )

    return {
        **result,
        "history_id": history.id,
        "history_saved": True
    }


# ==========================================
# GET FARMER'S CROP PREDICTION HISTORY
# ==========================================

@app.get("/prediction-history")
def get_prediction_history(
    db: Session = Depends(get_db),
    current_farmer: Farmer = Depends(get_current_farmer),
    limit: int = Query(default=20, ge=1, le=100)
):

    history_records = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.farmer_id == current_farmer.id,
            PredictionHistory.prediction_type == "crop_recommendation"
        )
        .order_by(desc(PredictionHistory.created_at))
        .limit(limit)
        .all()
    )

    return {
        "farmer_id": current_farmer.id,
        "total_records": len(history_records),
        "history": [
            {
                "id": record.id,
                "prediction_type": record.prediction_type,
                "input_data": record.input_data,
                "result": record.result,
                "created_at": (
                    record.created_at.isoformat()
                    if record.created_at else None
                )
            }
            for record in history_records
        ]
    }


# ==========================================
# FERTILIZER GUIDANCE
# ==========================================

@app.post("/fertilizer-guidance")
def fertilizer_guidance(data: CropInput):

    crop_name = data.crop

    input_data = pd.DataFrame(
        [[
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall
        ]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    predicted_crop = str(
        model.predict(input_data)[0]
    )

    selected_crop = (
        crop_name.strip()
        if isinstance(crop_name, str) and crop_name.strip()
        else predicted_crop
    )

    return {
        "selected_crop": selected_crop,
        "model_predicted_crop": predicted_crop,
        "soil_nutrient_values": {
            "nitrogen_N": data.N,
            "phosphorus_P": data.P,
            "potassium_K": data.K
        },
        "fertilizer_guidance": (
            f"Selected crop: {selected_crop}. "
            "Fertilizer type and quantity should be determined "
            "using a soil test, crop growth stage, and local "
            "agricultural recommendations. Do not determine a "
            "fertilizer dose from NPK inputs alone."
        ),
        "next_step": (
            "Consult your local Krishi Vigyan Kendra (KVK) "
            "or agriculture officer for a crop-specific fertilizer plan."
        )
    }


# ==========================================
# PLANT DISEASE DETECTION
# ==========================================

@app.post("/predict-disease")
async def predict_disease(
    file: UploadFile = File(...)
):

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image."
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB."
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize((224, 224))

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        predictions = disease_model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_disease = disease_classes[predicted_index]

        confidence = (
            float(predictions[predicted_index]) * 100
        )

        return {
            "predicted_disease": predicted_disease,
            "confidence_percent": round(
                confidence,
                2
            ),
            "note": (
                "This is an AI prediction, not a confirmed diagnosis. "
                "Check with an agricultural expert before treatment."
            )
        }

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image. Please upload a valid leaf photo."
        )

    finally:
        await file.close()


# ==========================================
# DISEASE TREATMENT AND PREVENTION GUIDANCE
# ==========================================

DISEASE_GUIDANCE = {

    "Tomato_Early_blight": {
        "symptoms": (
            "Older leaves may develop brown spots "
            "with concentric rings and yellowing."
        ),
        "prevention": (
            "Avoid overhead watering, improve airflow, "
            "remove infected plant debris, and rotate crops."
        ),
        "treatment": (
            "Remove badly affected leaves when practical. "
            "Ask a local agricultural expert whether a locally "
            "registered fungicide is appropriate."
        )
    },

    "Tomato_Late_blight": {
        "symptoms": (
            "Dark, water-soaked leaf patches may spread quickly; "
            "white growth can appear in humid conditions."
        ),
        "prevention": (
            "Avoid wetting foliage, provide good airflow, "
            "inspect plants frequently, and remove infected debris safely."
        ),
        "treatment": (
            "Seek prompt advice from a local agricultural expert "
            "because this disease can spread rapidly."
        )
    },

    "Tomato_Leaf_Mold": {
        "symptoms": (
            "Yellow patches may appear on upper leaf surfaces, "
            "with olive or greyish growth underneath."
        ),
        "prevention": (
            "Improve ventilation, reduce leaf wetness, "
            "avoid overcrowding, and remove infected plant debris."
        ),
        "treatment": (
            "Improve humidity management and confirm the diagnosis "
            "with an agricultural expert."
        )
    },

    "Tomato_Septoria_leaf_spot": {
        "symptoms": (
            "Small circular spots with pale centres "
            "and dark edges can develop on older leaves."
        ),
        "prevention": (
            "Avoid overhead irrigation, rotate crops, "
            "remove infected debris, and keep foliage dry."
        ),
        "treatment": (
            "Remove severely affected leaves when practical "
            "and ask an agricultural expert about suitable treatment."
        )
    },

    "Tomato_Bacterial_spot": {
        "symptoms": (
            "Small dark leaf spots may develop; "
            "fruit can also show raised or rough spots."
        ),
        "prevention": (
            "Use healthy seed or seedlings, avoid handling wet plants, "
            "sanitize tools, and rotate crops."
        ),
        "treatment": (
            "Confirm the disease before treatment. "
            "Consult a local agricultural expert."
        )
    },

    "Tomato_healthy": {
        "symptoms": (
            "The model classified the uploaded image as healthy."
        ),
        "prevention": (
            "Continue regular inspection, balanced watering, "
            "good airflow, and appropriate field hygiene."
        ),
        "treatment": (
            "No disease treatment is indicated by this prediction. "
            "Monitor the plant for new symptoms."
        )
    }

}


@app.get("/disease-guidance/{disease_name}")
def get_disease_guidance(disease_name: str):

    guidance = DISEASE_GUIDANCE.get(disease_name)

    if guidance is None:
        return {
            "disease": disease_name,
            "message": (
                "Specific guidance is not yet available for this class. "
                "Confirm the diagnosis with a local agriculture expert "
                "or Krishi Vigyan Kendra (KVK)."
            ),
            "symptoms": (
                "Compare the plant with trusted agricultural references."
            ),
            "prevention": (
                "Keep foliage dry when possible, maintain airflow, "
                "and remove plant debris safely."
            ),
            "treatment": (
                "Do not apply pesticides based only on an AI prediction. "
                "Seek crop-specific local advice."
            )
        }

    return {
        "disease": disease_name,
        **guidance,
        "note": (
            "AI predictions are not confirmed diagnoses. "
            "Verify the disease before taking treatment action."
        )
    }


# ==========================================
# WEATHER AND 5-DAY FORECAST
# ==========================================

@app.get("/weather")
def get_weather(
    city: str = Query(..., min_length=2)
):

    if not OPENWEATHER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenWeather API key is missing. Check the .env file."
        )

    try:
        # Current weather
        current_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=10
        )

        if current_response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="City not found. Please check the city name."
            )

        if current_response.status_code == 401:
            raise HTTPException(
                status_code=502,
                detail="Weather API key is invalid or not active yet."
            )

        current_response.raise_for_status()
        current = current_response.json()

        # 5-day forecast
        forecast_response = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=10
        )

        if forecast_response.status_code == 401:
            raise HTTPException(
                status_code=502,
                detail="Weather API key is invalid or not active yet."
            )

        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        forecasts = []

        for item in forecast_data.get("list", []):
            forecasts.append({
                "datetime": item["dt_txt"],
                "temperature_c": item["main"]["temp"],
                "feels_like_c": item["main"]["feels_like"],
                "humidity_percent": item["main"]["humidity"],
                "weather": item["weather"][0]["description"],
                "wind_speed_m_s": item["wind"]["speed"],
                "rain_mm": item.get("rain", {}).get("3h", 0)
            })

        return {
            "city": current["name"],
            "country": current["sys"]["country"],
            "temperature_c": current["main"]["temp"],
            "feels_like_c": current["main"]["feels_like"],
            "humidity_percent": current["main"]["humidity"],
            "weather": current["weather"][0]["description"],
            "wind_speed_m_s": current["wind"]["speed"],
            "forecast_count": len(forecasts),
            "forecasts": forecasts
        }

    except HTTPException:
        raise

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve weather or forecast. Please try again later."
        )
