
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "deduplicated"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load deduplicated training and validation datasets
train_df = pd.read_csv(
    DATA_DIR / "train_deduplicated.csv",
    encoding="utf-8-sig",
)
dev_df = pd.read_csv(
    DATA_DIR / "dev_deduplicated.csv",
    encoding="utf-8-sig",
)

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["sentiment"]

X_dev = dev_df["text"].fillna("").astype(str)
y_dev = dev_df["sentiment"]


# Combine word-level and character-level features
combined_features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=200000,
            sublinear_tf=True,
        ),
    ),
])

model = Pipeline([
    ("features", combined_features),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])


print("COMBINED WORD + CHARACTER TF-IDF")
print("=" * 50)
print("Training rows:", len(train_df))
print("Validation rows:", len(dev_df))

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed.")

# Evaluate on validation data
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

# Save the trained model
model_path = MODEL_DIR / "combined_word_char_tfidf.joblib"
joblib.dump(model, model_path)

# Save validation predictions
predictions = dev_df[["text", "sentiment"]].copy()
predictions["predicted_sentiment"] = y_pred

predictions.to_csv(
    RESULTS_DIR / "combined_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

# Save metrics
pd.DataFrame([{
    "model": "Combined Word + Character TF-IDF",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "training_rows": len(train_df),
    "validation_rows": len(dev_df),
}]).to_csv(
    RESULTS_DIR / "combined_baseline_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved model:", model_path)
print("Saved predictions:", RESULTS_DIR / "combined_validation_predictions.csv")
print("Saved metrics:", RESULTS_DIR / "combined_baseline_metrics.csv")
