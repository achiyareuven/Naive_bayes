from typing import Dict , Any
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException
import uvicorn
from model.predictor import NaiveBayesPredictor
import dill
import os




app = FastAPI()

model_app = None
predictor_app = None
accuracy =None

class PredictionRequest(BaseModel):
    sample: Dict[str, Any]

class LoadModelRequest(BaseModel):
    path: str

@app.on_event("startup")
def load_default_model():
    global model_app, predictor_app, accuracy
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_model.pkl")
    if not os.path.isfile(model_path):
        print("No default model found. Skipping model load.")
        return
    try:
        with open(model_path,"rb") as file:
            model_app ,accuracy = dill.load(file)
        predictor_app = NaiveBayesPredictor(model_app)
    except Exception as e:
        raise RuntimeError(f"failed to load model on startup: {e}")



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
    if predictor_app is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:

        prediction = predictor_app.predict(request.sample)
        probabilities = predictor_app.predict_proba(request.sample)

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

@app.post("/load_model")
def load_new_model(request: LoadModelRequest):
    global model_app, predictor_app, accuracy
    try:
        if not request.path.strip():
            raise HTTPException(status_code=400, detail="Path not provided.")
        if not os.path.isfile(request.path):
            raise HTTPException(status_code=400,detail="model file is invalid format ")

        with open(request.path, "rb") as file:
            model_app, accuracy = dill.load(file)
        predictor_app = NaiveBayesPredictor(model_app)
        return {"message": f"Model loaded from {request.path}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")


if __name__ == "__main__":
    uvicorn.run("server.server_app:app",host="127.0.0.1",port= 8000,reload=True)