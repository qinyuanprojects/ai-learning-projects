# test_predict.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict():
    resp = client.post("/predict", json={"text": "My car's brakes make a weird noise."})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "probabilities" in data


