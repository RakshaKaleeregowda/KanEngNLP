from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "deduplicated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": DATA_DIR / "train_clean.csv",
    "dev": DATA_DIR / "dev_clean.csv",
    "test": DATA_DIR / "test_clean.csv",
}

# Priority determines which split retains a consistent duplicate.
PRIORITY = ["train", "dev", "test"]


def load_dataset(name, path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"text", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{name} must contain columns: {required}"
        )

    df = df[["text", "sentiment"]].copy()

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["sentiment"] = (
        df["sentiment"].fillna("").astype(str).str.strip().str.lower()
    )

    # Remove rows with missing or empty fields.
    df = df[
        (df["text"] != "") &
        (df["sentiment"] != "")
    ].copy()

    return df


datasets = {
    name: load_dataset(name, path)
    for name, path in FILES.items()
}

print("=" * 65)
print("CREATING DEDUPLICATED EXPERIMENTAL DATASETS")
print("=" * 65)

# Gather every label assigned to each exact sentence.
all_rows = []

for split, df in datasets.items():
    for _, row in df.iterrows():
        all_rows.append({
            "text": row["text"],
            "sentiment": row["sentiment"],
            "split": split,
        })

all_rows_df = pd.DataFrame(all_rows)

labels_by_text = (
    all_rows_df.groupby("text")["sentiment"]
    .agg(lambda labels: set(labels))
    .to_dict()
)

# Exclude sentences with conflicting labels across the source splits.
conflicting_texts = {
    text
    for text, labels in labels_by_text.items()
    if len(labels) > 1
}

print("\nConflicting sentences:", len(conflicting_texts))

for text in sorted(conflicting_texts):
    print("\nConflicting text:", text[:160])
    print("Labels:", labels_by_text[text])

# Retain each consistent sentence in its highest-priority split.
seen_texts = set()
cleaned = {}

for split in PRIORITY:
    df = datasets[split].copy()

    before = len(df)

    # Exclude any sentence whose label conflicts across source splits.
    df = df[~df["text"].isin(conflicting_texts)].copy()

    # Keep only sentences not already retained by a higher-priority split.
    df = df[~df["text"].isin(seen_texts)].copy()

    # If a sentence appears repeatedly within a split, keep one occurrence.
    df = df.drop_duplicates(subset="text", keep="first").copy()

    seen_texts.update(df["text"].tolist())

    cleaned[split] = df.reset_index(drop=True)

    print(f"\n{split.upper()}")
    print("Original rows:", before)
    print("Retained rows:", len(df))
    print("Removed rows:", before - len(df))
    print("Class distribution:")
    print(df["sentiment"].value_counts().to_string())

# Save to a separate directory; source files remain unchanged.
for split, df in cleaned.items():
    output_path = OUTPUT_DIR / f"{split}_deduplicated.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved {split}: {output_path}")

print("\n" + "=" * 65)
print("DEDUPLICATION COMPLETE")
print("Original processed datasets were not modified.")
print("=" * 65)