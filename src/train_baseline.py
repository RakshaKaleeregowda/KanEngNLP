from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the datasets
# --------------------------------------------------

train_path = DATA_DIR / "train_clean.csv"
dev_path = DATA_DIR / "dev_clean.csv"

train_df = pd.read_csv(train_path, encoding="utf-8-sig")
dev_df = pd.read_csv(dev_path, encoding="utf-8-sig")

print("=" * 60)
print("KANNADA-ENGLISH SENTIMENT ANALYSIS")
print("Baseline: Word TF-IDF + Logistic Regression")
print("=" * 60)

print("\nTraining examples:", len(train_df))
print("Validation examples:", len(dev_df))

print("\nTraining class distribution:")
print(train_df["sentiment"].value_counts())

print("\nValidation class distribution:")
print(dev_df["sentiment"].value_counts())


# --------------------------------------------------
# 3. Separate text and labels
# --------------------------------------------------

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["sentiment"]

X_dev = dev_df["text"].fillna("").astype(str)
y_dev = dev_df["sentiment"]


# --------------------------------------------------
# 4. Build the model
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])


# --------------------------------------------------
# 5. Train
# --------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 6. Evaluate on validation data
# --------------------------------------------------

print("\nEvaluating on validation data...")

y_pred = model.predict(X_dev)

accuracy = accuracy_score(y_dev, y_pred)
macro_f1 = f1_score(
    y_dev,
    y_pred,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_dev,
        y_pred,
        labels=["positive", "negative", "mixed"],
        zero_division=0,
    )
)

print("\nConfusion matrix:")

cm = confusion_matrix(
    y_dev,
    y_pred,
    labels=["positive", "negative", "mixed"],
)

print(
    pd.DataFrame(
        cm,
        index=["Actual positive", "Actual negative", "Actual mixed"],
        columns=["Predicted positive", "Predicted negative", "Predicted mixed"],
    )
)


# --------------------------------------------------
# 7. Save the model and validation results
# --------------------------------------------------

model_path = MODEL_DIR / "sentiment_tfidf_logreg.joblib"

joblib.dump(model, model_path)

results_path = RESULTS_DIR / "baseline_validation_predictions.csv"

pd.DataFrame({
    "text": X_dev,
    "actual_sentiment": y_dev,
    "predicted_sentiment": y_pred,
}).to_csv(results_path, index=False, encoding="utf-8-sig")

print("\nModel saved to:", model_path)
print("Predictions saved to:", results_path)


# --------------------------------------------------
# 8. Try a few example sentences
# --------------------------------------------------

examples = [
    "Movie tumba chennagide",
    "Movie tumba chennagilla",
    "Story chennagide but climax worst agittu",
]

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for sentence in examples:
    prediction = model.predict([sentence])[0]
    probabilities = model.predict_proba([sentence])[0]

    print(f"\nSentence: {sentence}")
    print(f"Predicted sentiment: {prediction}")

    for label, probability in zip(model.classes_, probabilities):
        print(f"  {label}: {probability:.3f}")