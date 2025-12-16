from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import onnxruntime as rt
import numpy as np

app = FastAPI(title="Credit Scoring API (ONNX)", version="1.0")

# Загружаем ONNX модель
session = rt.InferenceSession("models/model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Все 23 фичи в правильном порядке (из Taiwanese Credit Dataset)
class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.get("/")
def read_root():
    return {"message": "Credit Scoring API with ONNX model is running!"}

@app.post("/predict")
def predict(data: ClientData):
    # Преобразуем в numpy array (1 объект, 23 фичи)
    features = list(data.dict().values())
    input_data = np.array(features, dtype=np.float32).reshape(1, -1)

    # Inference через ONNX
    pred = session.run([output_name], {input_name: input_data})[0]

    probability = float(pred[0][1])  # вероятность дефолта
    prediction = 1 if probability > 0.5 else 0

    return {
        "default_prediction": prediction,
        "default_probability": round(probability, 4)
    }

@app.get("/health")
def health():
    return {"status": "healthy"}