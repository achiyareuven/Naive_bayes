from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import dill
import os
from data_loader.load_data_from_csv import CSVLoader
from data_cleaner.clean_data import DataCleaner
from model.Naive_Bayes_Model import NaiveBayesClassifier
from model.predictor import NaiveBayesPredictor
from evaluation.evaluator import Evaluator
from utils.data_utils import split_feature_target,split_train_test


app = FastAPI()
data_cleaner = DataCleaner()

MODEL_PATH = "trained_model.pkl"


@app.on_event("startup")
def started():
    try:
        csv_loader = CSVLoader("data/phishing.csv")
        df = csv_loader.load_data()
        df = data_cleaner.clean_data(df)

        X, y = split_feature_target(df)
        X_train, X_test, y_train, y_test = split_train_test(X, y)

        model = NaiveBayesClassifier()
        model.fit(X_train, y_train)

        #prdict
        predictor = NaiveBayesPredictor(model)

        evaluator = Evaluator(predictor)
        accuracy = evaluator.evaluate_accuracy(X_test, y_test)

        with open("trained_model.pkl","wb") as f:
            dill.dump((model,accuracy),f)

    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}")

@app.get("/")
def home():
    return { "hello world"}

@app.get("/get_model")
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model file not found.")

    return FileResponse(
        path=MODEL_PATH,
        media_type="application/octet-stream",
        filename="trained_model.pkl"
    )




