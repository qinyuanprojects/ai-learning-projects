Week 1–3 — FastAPI + Machine Learning Text Classifier 🚗🔌

Toy NLP model: rec.autos vs sci.electronics

🎯 Project Goal

Build a complete ML deployment workflow:

Train a text classification model using scikit-learn

Serve predictions using FastAPI

Deploy to the cloud (Render)

This demonstrates both backend API skills & ML engineering skills.

🧠 Machine Learning Overview
Component	Tool	Purpose
Dataset	scikit-learn 20 Newsgroups	Real-world text categories
Feature extraction	TF-IDF Vectorizer	Convert raw text → numeric vectors
Classifier	Logistic Regression	Predict topic + probabilities
Persistence	joblib	Save + load model for deployment
Workflow

Load filtered dataset (autos vs electronics)

Split into train/test

Train TF-IDF + Logistic Regression pipeline

Evaluate accuracy + classification report

Save model (model.joblib) for API

📌 Predictions include:

Predicted category name

Confidence probability for each class

Example:

{
  "prediction": "rec.autos",
  "probabilities": {
    "rec.autos": 0.84,
    "sci.electronics": 0.16
  }
}

🖥 Local Development
1️⃣ Create & activate environment

Mac/Linux:

python3 -m venv .venv && source .venv/bin/activate


Windows (PowerShell):

python -m venv .venv
.venv\Scripts\Activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model
python train_model.py


→ outputs accuracy & saves model.joblib

4️⃣ Start the API server
uvicorn app:app --reload --port 8000

5️⃣ Try it in browser

➡ http://127.0.0.1:8000/docs

(Interactive Swagger UI)

☁️ Deployment (Render)

A production version is deployed to Render using:

FastAPI app

Gunicorn + Uvicorn workers

runtime.txt to enforce correct Python version

The deployed API matches local behavior, though small probability differences can occur due to:

Different OS-level dependencies (SciPy/BLAS)

Updated model joblib in deployment environment

🔍 API Endpoints
Method	Endpoint	Description
GET	/	Health/status check
POST	/predict	Text classification

Prediction request format:

{
  "text": "I need help with my engine noise"
}

🏆 What This Demonstrates

✔ End-to-end ML product lifecycle
✔ Deployment-ready API engineering
✔ Data preprocessing + model training
✔ Reproducible environment setup
✔ Cloud hosting + real HTTP requests

📌 Future Enhancements (Next milestones)

Add more categories from 20 Newsgroups

Model versioning + continuous redeploy

Add confidence thresholds + error handling UI

Expand to a full NLP microservice portfolio
