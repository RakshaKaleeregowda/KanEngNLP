from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "deduplicated"

files = {
    "train": DATA_DIR / "train_deduplicated.csv",
    "dev": DATA_DIR / "dev_deduplicated.csv",
    "test": DATA_DIR / "test_deduplicated.csv",
}

datasets = {}

for name, path in files.items():
    df = pd.read_csv(path, encoding="utf-8-sig")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    datasets[name] = df

    print(f"\n{name.upper()}")
    print("Rows:", len(df))
    print("Duplicate texts within split:",
          df["text"].duplicated().sum())
    print("Labels:", df["sentiment"].value_counts().to_dict())

print("\nCROSS-SPLIT OVERLAP CHECK")

pairs = [
    ("train", "dev"),
    ("train", "test"),
    ("dev", "test"),
]

all_clear = True

for first, second in pairs:
    texts_a = set(datasets[first]["text"])
    texts_b = set(datasets[second]["text"])
    overlap = texts_a & texts_b

    print(f"{first.upper()} vs {second.upper()}: {len(overlap)}")

    if overlap:
        all_clear = False
        print("Examples:", list(overlap)[:5])

print("\nFINAL RESULT:")
if all_clear:
    print("PASS: No exact text overlaps remain.")
else:
    print("CHECK REQUIRED: Overlapping texts still exist.")