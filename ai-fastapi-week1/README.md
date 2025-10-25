# Week1 - FastAPI + ML Demo

**What:** Toy text classifier (rec.autos vs sci.electronics) wrapped with FastAPI.

**How to run**
1. python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. python train_model.py
4. uvicorn app:app --reload --port 8000
5. Open http://127.0.0.1:8000/docs

**API endpoints**
- GET `/` health check
- POST `/predict` { "text": "..." } → returns prediction + probabilities
