
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "deduplicated"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load the deduplicated training and validation datasets
train_df = pd.read_csv(
    DATA_DIR / "train_deduplicated.csv",
    encoding="utf-8-sig",
)

dev_df = pd.read_csv(
    DATA_DIR / "dev_deduplicated.csv",
    encoding="utf-8-sig",
)

# Basic validation
required_columns = {"text", "sentiment"}

for name, df in [("train", train_df), ("dev", dev_df)]:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )

    df["text"] = df["text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].astype(str)

# Remove empty text rows, if any
train_df = train_df[train_df["text"].str.strip() != ""].copy()
dev_df = dev_df[dev_df["text"].str.strip() != ""].copy()

X_train = train_df["text"]
y_train = train_df["sentiment"]

X_dev = dev_df["text"]
y_dev = dev_df["sentiment"]

print("CHARACTER TF-IDF BASELINE")
print("=" * 50)
print("Training rows:", len(train_df))
print("Validation rows:", len(dev_df))
print("\nTraining class distribution:")
print(y_train.value_counts().to_dict())


# Character-level TF-IDF + Logistic Regression
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=200000,
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

print("\nTraining model...")
model.fit(X_train, y_train)

print("Training completed.")

# Evaluate on validation data only
y_pred = model.predict(X_dev)

accuracy = accuracy_score(y_dev, y_pred)
macro_f1 = f1_score(
    y_dev,
    y_pred,
    average="macro",
    zero_division=0,
)

print("\nVALIDATION RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_dev,
        y_pred,
        labels=["negative", "positive", "mixed"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    pd.DataFrame(
        confusion_matrix(
            y_dev,
            y_pred,
            labels=["negative", "positive", "mixed"],
        ),
        index=["actual_negative", "actual_positive", "actual_mixed"],
        columns=["pred_negative", "pred_positive", "pred_mixed"],
    )
)

# Save the trained model
model_path = MODEL_DIR / "char_tfidf_logreg.joblib"
joblib.dump(model, model_path)

# Save validation predictions for later comparison
predictions = dev_df[["text", "sentiment"]].copy()
predictions["predicted_sentiment"] = y_pred

predictions.to_csv(
    RESULTS_DIR / "char_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

# Save a compact metrics file
metrics = pd.DataFrame([{
    "model": "Character TF-IDF + Logistic Regression",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "training_rows": len(train_df),
    "validation_rows": len(dev_df),
}])

metrics.to_csv(
    RESULTS_DIR / "char_baseline_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved model:", model_path)
print(
    "Saved predictions:",
    RESULTS_DIR / "char_validation_predictions.csv",
)
print(
    "Saved metrics:",
    RESULTS_DIR / "char_baseline_metrics.csv",
)
