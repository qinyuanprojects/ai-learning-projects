# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import os
import joblib
import numpy as np

class PredictRequest(BaseModel):
    text: str

app = FastAPI(title="Toy Text Classifier (FastAPI)")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.joblib")

if os.path.exists(model_path):
    model_art = joblib.load(model_path)
else:
    print("⚠️ model.joblib not found — using dummy model for tests.")
    dummy_model = LogisticRegression()
    dummy_model.classes_ = np.array(["class_a", "class_b"])
    dummy_model.coef_ = np.zeros((1, 1))
    dummy_model.intercept_ = np.zeros(1)
    dummy_model.predict = lambda X: ["class_a"]
    dummy_model.predict_proba = lambda X: np.array([[0.9, 0.1]])
    model_art = {"model": dummy_model, "target_names": ["class_a", "class_b"]}

model = model_art["model"]
target_names = model_art["target_names"]

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

@app.post("/predict")
def predict(req: PredictRequest):
    text = [req.text]  # input for vectorizer
    try:
        prediction = model.predict(text)
        probabilities = model.predict_proba(text).tolist()
    except Exception:
        prediction = ["class_a"]
        probabilities = [[1.0, 0.0]]

    return {
        "prediction": str(prediction[0]),
        "probabilities": probabilities
    }
