from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": DATA_DIR / "train_clean.csv",
    "dev": DATA_DIR / "dev_clean.csv",
    "test": DATA_DIR / "test_clean.csv",
}

datasets = {}

for split, path in FILES.items():
    df = pd.read_csv(path, encoding="utf-8-sig")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    datasets[split] = df


pairs = [
    ("train", "dev"),
    ("train", "test"),
    ("dev", "test"),
]

reports = []

for first, second in pairs:
    left = datasets[first]
    right = datasets[second]

    left_groups = left.groupby("text")["sentiment"].agg(set)
    right_groups = right.groupby("text")["sentiment"].agg(set)

    overlap = set(left_groups.index) & set(right_groups.index)

    for text in sorted(overlap):
        reports.append({
            "first_split": first,
            "second_split": second,
            "text": text,
            "first_labels": ", ".join(sorted(left_groups[text])),
            "second_labels": ", ".join(sorted(right_groups[text])),
            "labels_agree": left_groups[text] == right_groups[text],
        })


report_df = pd.DataFrame(reports)

output_path = RESULTS_DIR / "duplicate_audit_report.csv"

report_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("=" * 60)
print("DUPLICATE AUDIT REPORT")
print("=" * 60)

print("Total overlapping records:", len(report_df))

if not report_df.empty:
    conflicts = report_df[~report_df["labels_agree"]]

    print("Records with conflicting labels:", len(conflicts))

    print("\nConflicting examples:")

    if conflicts.empty:
        print("No conflicting examples found.")
    else:
        print(
            conflicts[
                [
                    "first_split",
                    "second_split",
                    "text",
                    "first_labels",
                    "second_labels",
                ]
            ].to_string(index=False)
        )

print("\nFull report saved to:", output_path)
print("Original datasets were not modified.")