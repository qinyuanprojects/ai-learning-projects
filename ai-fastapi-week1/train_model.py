# train_model.py (clean + compatible)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("📥 Loading dataset...")
    categories = ["rec.autos", "sci.electronics"]
    data = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes")
    )
    X, y = data.data, data.target

    print("🔄 Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🤖 Training model pipeline...")
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=2000)
    )
    pipeline.fit(X_train, y_train)

    print("📊 Evaluating model...")
    preds = pipeline.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds, target_names=categories))

    print("💾 Saving full pipeline to model.joblib")
    joblib.dump({
        "model": pipeline,
        "target_names": categories
    }, "model.joblib")

    print("🎯 Done! model.joblib ready for deployment")

if __name__ == "__main__":
    main()
