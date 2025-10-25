# train_model.py
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    # Toy binary classification: pick two categories
    cats = ['rec.autos', 'sci.electronics']
    data = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers','quotes'))
    X, y = data.data, data.target

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline: TF-IDF + logistic regression
    pipeline = make_pipeline(TfidfVectorizer(max_features=10000, ngram_range=(1,2)), LogisticRegression(max_iter=1000))

    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds, target_names=cats))

    # Save model
    joblib.dump({"model": pipeline, "target_names": cats}, "model.joblib")
    print("Saved model.joblib")

if __name__ == "__main__":
    main()
