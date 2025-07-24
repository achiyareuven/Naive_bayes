from typing import Dict , Any
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException
from model.predictor import NaiveBayesPredictor
import dill
import os
import requests



app = FastAPI()

MODEL_PATH = "trained_model.pkl"
MODEL_URL = "http://train_container:8000/get_model"

model = None
predictor =None
accuracy = None


class PredictionRequest(BaseModel):
    sample: Dict[str, Any]


@app.on_event("startup")
def load_model():
    global model, predictor, accuracy
    try:
        response = requests.get(MODEL_URL)
        if response.status_code != 200:
            raise RuntimeError ("failed to get model")

        with open(MODEL_PATH,"wb") as f:
            f.write(response.content)

        with open(MODEL_PATH, "rb") as f:
                model, accuracy = dill.load(f)
                predictor = NaiveBayesPredictor(model)

    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}")


@app.get("/")
def home():
    return { " Hello to my model project. model loaded"}

@app.get("/evaluation")
def get_accuracy():
    if accuracy is None:
        raise  HTTPException(status_code=500, detail="Accuracy not available. Model may not be loaded.")
    return {"accuracy":  (accuracy)}

@app.post("/predict")
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:

        prediction = predictor.predict(request.sample)
        probabilities = predictor.predict_proba(request.sample)

        prediction_clean = int(prediction) if hasattr(prediction, "item") else prediction
        probabilities_clean = {
            str(int(k) if hasattr(k, "item") else k): float(v) if hasattr(v, "item") else v
            for k, v in probabilities.items()}

        return {
            "prediction": prediction_clean,
            "probabilities": probabilities_clean
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


