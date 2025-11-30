from joblib import load

saved = load("model.joblib")
pipeline = saved["model"]
target_names = saved["target_names"]

while True:
    text = input("Enter text: ")
    if text.lower() in ['quit','exit']:
        break
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    print(f"Prediction: {target_names[pred]}")
    print(f"Probabilities: {proba}")