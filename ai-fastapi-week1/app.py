# app.py (clean + correct)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

class PredictRequest(BaseModel):
    text: str

app = FastAPI(title="Text Classifier API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.joblib")

# Load model on startup
try:
    logger.info(f"Attempting to load model from: {model_path}")
    model_art = joblib.load(model_path)
    model = model_art["model"]
    target_names = model_art["target_names"]
    logger.info("Model loaded successfully!")
except Exception:
    logger.error("Model failed to load:\n" + traceback.format_exc())
    model = None
    target_names = None

@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        text = [req.text]
        pred = model.predict(text)[0]
        proba = model.predict_proba(text)[0]

        # If pred is int, map using class names
        if isinstance(pred, (int, float)):
            pred_name = target_names[pred]
        else:
            pred_name = str(pred)

        proba_map = {
            target_names[i]: float(proba[i])
            for i in range(len(target_names))
        }

        return {
            "prediction": pred_name,
            "probabilities": proba_map
        }

    except Exception:
        logger.error("Exception during prediction:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed; check logs.")
