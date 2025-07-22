from typing import Dict , Any
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException
import uvicorn
from model.predictor import NaiveBayesPredictor
from model.Naive_Bayes_Model import NaiveBayesClassifier
from evaluation.evaluator import Evaluator
from utils.data_utils import split_train_test,split_feature_target
from data_cleaner.clean_data import DataCleaner
from data_loader.load_data_from_csv import CSVLoader
import dill
import os




app = FastAPI()
cleaner = DataCleaner()
load_data_csv  =CSVLoader(r"C:\Users\achiy\PycharmProjects\Naive_bayes\data\phishing.csv")
model_app =  None
predictor_app = None
evaluator = None
accuracy =None

class PredictionRequest(BaseModel):
    sample: Dict[str, Any]


@app.on_event("startup")
def started():
    global model_app, predictor_app, accuracy
    try:
        #load
        df = load_data_csv.load_data()

        #clean
        df = cleaner.clean_data(df)

        # split_feature_target
        X,y =  split_feature_target(df)

        #split_train_test
        X_train,X_test,y_train,y_test =split_train_test(X,y)

        #training model
        model_app =NaiveBayesClassifier()
        model_app.fit(X_train, y_train)

        #prdict
        predictor_app = NaiveBayesPredictor(model_app)

        # eavalution
        evaluator = Evaluator(predictor_app)
        accuracy = evaluator.evaluate_accuracy(X_test, y_test)
    except Exception as e:
        raise RuntimeError(f"startup failed: {e}")


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
                "probabilities": probabilities_clean}

    except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    uvicorn.run("server.server_app:app",host="127.0.0.1",port= 8000,reload=True)