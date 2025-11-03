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

# Load model at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.joblib")

if os.path.exists(model_path):
    model_art = joblib.load(model_path)
else:
    print("⚠️ model.joblib not found due to wrong filepath or model not being pushed to Github— using dummy model for tests.")
    dummy_model = LogisticRegression()
    dummy_model.classes_ = np.array([0, 1])
    dummy_model.coef_ = np.zeros((1, 1))
    dummy_model.intercept_ = np.zeros(1)
    model_art = {"model": dummy_model, "target_names": ["class_a", "class_b"]}

model = model_art["model"]
target_names = model_art["target_names"]

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text

    # ✅ Wrap input in a list to form a 2D array (1 sample)
    preds = model.predict([text])
    prob = model.predict_proba([text])[0]

    return {
        "text": text,
        "prediction": preds[0],
        "probability": prob.tolist(),
        "labels": target_names
    }

