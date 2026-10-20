
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

predictions_path = (
    RESULTS_DIR / "combined_validation_predictions.csv"
)

df = pd.read_csv(predictions_path, encoding="utf-8-sig")

required_columns = {
    "text",
    "sentiment",
    "predicted_sentiment",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["text"] = df["text"].fillna("").astype(str)
df["sentiment"] = df["sentiment"].astype(str)
df["predicted_sentiment"] = (
    df["predicted_sentiment"].astype(str)
)

# Keep only incorrectly classified examples
errors = df[
    df["sentiment"] != df["predicted_sentiment"]
].copy()

errors["error_type"] = (
    errors["sentiment"]
    + " -> "
    + errors["predicted_sentiment"]
)

print("COMBINED MODEL ERROR ANALYSIS")
print("=" * 60)
print("Total validation examples:", len(df))
print("Incorrect predictions:", len(errors))
print(
    "Error rate:",
    f"{len(errors) / len(df) * 100:.2f}%"
)

print("\nErrors by actual sentiment:")
print(errors["sentiment"].value_counts().to_string())

print("\nMost frequent confusion patterns:")
print(errors["error_type"].value_counts().to_string())

print("\nSample misclassified examples:")
for _, row in errors.head(20).iterrows():
    print("-" * 60)
    print("Text:", row["text"])
    print("Actual:", row["sentiment"])
    print("Predicted:", row["predicted_sentiment"])

errors.to_csv(
    RESULTS_DIR / "combined_validation_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    "\nSaved detailed errors to "
    "results/combined_validation_errors.csv"
)
