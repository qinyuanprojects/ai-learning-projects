# AI Learning Projects

[![CI Pipeline](https://github.com/qinyuanprojects/ai-learning-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/qinyuanprojects/ai-learning-projects/actions/workflows/ci.yml)

A collection of AI backend learning projects built with FastAPI, ML, and Docker.


Python version:
![Python](https://img.shields.io/badge/python-3.11-blue.svg)


License:
![License](https://img.shields.io/github/license/qinyuanprojects/ai-learning-projects)


## Run locally (Docker)
docker build -t fastapi-ml:latest .
docker run -p 8000:8000 fastapi-ml:latest

# Test endpoint
- $headers = @{ "Content-Type" = "application/json" }
- $body = '{ "text": "I need advice on car maintenance and engine noise" }'
- Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method POST -Headers $headers -Body $body

- (Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method POST -Headers $headers -Body $body).Content