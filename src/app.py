from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Credit Default Prediction API")

# Загрузка модели (предполагаем, что она сохранена как credit_default_model.pkl)
model_path = "models/credit_default_model.pkl"
model = joblib.load(model_path)

class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    BILL_AMT1: float
    PAY_AMT1: float

@app.post("/predict")
def predict(data: ClientData):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    return {
        "default_prediction": int(prediction[0]),
        "default_probability": float(probability)
    }

@app.get("/")
def read_root():
    return {"message": "Credit Default Prediction API is alive!"}