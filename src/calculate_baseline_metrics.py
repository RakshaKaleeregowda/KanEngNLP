import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

file_path = "results/combined_validation_predictions.csv"

df = pd.read_csv(file_path)

y_true = df["sentiment"]
y_pred = df["predicted_sentiment"]

print("Validation samples:", len(df))
print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    labels=["negative", "positive", "mixed"],
    digits=4,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_true,
    y_pred,
    labels=["negative", "positive", "mixed"]
))
# Save exact classification report to CSV
import os

report = classification_report(
    y_true,
    y_pred,
    labels=["negative", "positive", "mixed"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

os.makedirs("results", exist_ok=True)

report_path = "results/combined_baseline_classification_report.csv"
report_df.to_csv(report_path, index=True)

print(f"\nClassification report saved to: {report_path}")