import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "house_price_model.joblib")

app = FastAPI(title="House Price Predictor API")

_model = None


def get_model():
    """Lazy-load so the API can still start (and /health can report the
    real status) even before a model has been trained."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"No trained model found at {MODEL_PATH}. Run `python src/train_mlflow.py` first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class HouseFeatures(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float


@app.get("/health")
def health():
    return {"model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predict_price(features: HouseFeatures):
    model = get_model()
    data = np.array([[
        features.feature_1, features.feature_2, features.feature_3,
        features.feature_4, features.feature_5,
    ]])
    predicted_price = float(model.predict(data)[0])
    return {"predicted_price_usd": round(predicted_price, 2)}
