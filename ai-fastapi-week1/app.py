# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class PredictRequest(BaseModel):
    text: str

app = FastAPI(title="Toy Text Classifier (FastAPI)")

# Load model at startup
model_art = joblib.load("model.joblib")
model = model_art["model"]
target_names = model_art["target_names"]

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

@app.post("/predict")
def predict(req: PredictRequest):
    text = [req.text]
    prob = model.predict_proba(text)[0]
    pred_idx = int(prob.argmax())
    return {
        "prediction": target_names[pred_idx],
        "prediction_index": pred_idx,
        "probabilities": {target_names[i]: float(prob[i]) for i in range(len(prob))}
    }

