# Week 1–3: FastAPI + Machine Learning + Cloud Deployment 🚀

A practical mini-project applying **text classification with scikit-learn**, wrapped in **FastAPI**, and deployed to the **cloud via Render**.

---

## 📌 What’s Inside

| Week | Focus Area | Key Skills Learned |
|------|------------|------------------|
| Week 1 | FastAPI Basics | API routing, request/response models |
| Week 2 | ML Integration | scikit-learn training + model serving |
| Week 3 | Cloud Deployment | Render deployment, runtime config, debugging |

Classifier predicts between:
- `rec.autos` 🚗 (car discussions)
- `sci.electronics` 🔌 (electronics discussions)

---

## 🧠 Machine Learning Pipeline (scikit-learn)

- Dataset: **20 Newsgroups** (subset: 2 categories)
- Preprocessing: **TfidfVectorizer**
- Model: **LogisticRegression**
- Metrics: **Accuracy, Classification Report**
- Model saved using **joblib** → `model.joblib`

Run model training locally:

```sh
python train_model.py
```

This generates:
- model.joblib (trained model)
- vectorizer.joblib (TF-IDF vocabulary)

---

## 🖥️ Local Development Setup

```sh
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Visit interactive API docs:
👉 http://127.0.0.1:8000/docs

---

## 🔌 API Endpoints

- **GET** `/`  
  Health check → returns the status of the API.

- **POST** `/predict`  
  Request JSON: 
```json
{
    "text": "Your input text here"
}
```
Response JSON:
```json
{
  "prediction": "predicted_class",
  "probabilities": {
    "rec.autos": 0.85,
    "sci.electronics": 0.15
    }
}
```

---

## ☁️ Week 3 — Deploying to Render
### Key Deployment Tasks Completed
✔ `requirements.txt` cleanup for compatibility.  
✔ Debugged SciPy and Pydantic dependency issues.  
✔ Confirmed predictions match local testing.

Deployed service:
🔗 https://ai-fastapi-week1.onrender.com/docs

Test with PowerShell:
```powershell
$headers = @{ "Content-Type" = "application/json" }
$body = '{ "text": "I need advice on car maintenance and engine noise" }'
Invoke-WebRequest -Uri "https://ai-fastapi-week1.onrender.com/predict" -Method POST -Headers $headers -Body $body
```

---
### 🧑‍💻 Author

Project by **Qin Yuan**  
Learning & building through real shipped AI ✨

---
*Thanks for checking out this project!* 🚀