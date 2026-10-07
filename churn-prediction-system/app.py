from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

class CustomerData(BaseModel):
    gender: str
    contract_type: str
    payment_method: str
    tenure: int
    monthly_charges: float
    total_charges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])
    processed_data = preprocessor.transform(input_df)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]
    return {"churn_prediction": int(prediction), "churn_probability": float(probability)}
