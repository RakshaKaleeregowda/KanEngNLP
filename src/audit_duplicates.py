from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

FILES = {
    "train": DATA_DIR / "train_clean.csv",
    "dev": DATA_DIR / "dev_clean.csv",
    "test": DATA_DIR / "test_clean.csv",
}


def load_dataset(name, path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    # Remove empty text for the overlap audit.
    df = df[df["text"] != ""].copy()

    return df


datasets = {
    name: load_dataset(name, path)
    for name, path in FILES.items()
}

pairs = [
    ("train", "dev"),
    ("train", "test"),
    ("dev", "test"),
]

print("=" * 70)
print("DUPLICATE AND LABEL-CONSISTENCY AUDIT")
print("=" * 70)

for first, second in pairs:
    left = datasets[first]
    right = datasets[second]

    left_texts = set(left["text"])
    right_texts = set(right["text"])

    overlap = left_texts & right_texts

    print(f"\n{first.upper()} vs {second.upper()}")
    print("Overlapping sentences:", len(overlap))

    if not overlap:
        continue

    left_labels = (
        left[left["text"].isin(overlap)]
        .groupby("text")["sentiment"]
        .agg(set)
        .to_dict()
    )

    right_labels = (
        right[right["text"].isin(overlap)]
        .groupby("text")["sentiment"]
        .agg(set)
        .to_dict()
    )

    conflicts = []

    for text in sorted(overlap):
        labels_left = left_labels[text]
        labels_right = right_labels[text]

        if labels_left != labels_right:
            conflicts.append(
                (text, labels_left, labels_right)
            )

    print("Overlapping texts with different label sets:", len(conflicts))

    print("\nExamples:")
    for text in sorted(overlap)[:10]:
        print("Text:", text[:160])
        print("First split labels:", left_labels[text])
        print("Second split labels:", right_labels[text])
        print()

print("\nAudit completed. No files were modified.")