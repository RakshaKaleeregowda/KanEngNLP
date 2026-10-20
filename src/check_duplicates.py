from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

FILES = {
    "train": DATA_DIR / "train_clean.csv",
    "dev": DATA_DIR / "dev_clean.csv",
    "test": DATA_DIR / "test_clean.csv",
}


def load_texts(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df["text"].fillna("").astype(str)


datasets = {
    name: load_texts(path)
    for name, path in FILES.items()
}

print("=" * 60)
print("CROSS-DATASET DUPLICATE CHECK")
print("=" * 60)

for name, texts in datasets.items():
    print(f"{name.upper()} examples: {len(texts)}")
    print(f"{name.upper()} unique sentences: {texts.nunique()}")

pairs = [
    ("train", "dev"),
    ("train", "test"),
    ("dev", "test"),
]

for first, second in pairs:
    first_texts = set(datasets[first])
    second_texts = set(datasets[second])

    overlap = first_texts & second_texts

    print(f"\n{first.upper()} vs {second.upper()}")
    print("Overlapping sentences:", len(overlap))

    if overlap:
        print("Examples:")
        for sentence in list(overlap)[:5]:
            print("-", sentence[:150])

print("\nDuplicate check completed.")