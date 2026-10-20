from pathlib import Path
from collections import Counter
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

FILES = {
    "train": DATA_DIR / "train_clean.csv",
    "dev": DATA_DIR / "dev_clean.csv",
    "test": DATA_DIR / "test_clean.csv",
}


def contains_kannada_script(text):
    return any("\u0C80" <= char <= "\u0CFF" for char in text)


def contains_latin_script(text):
    return any(
        ("A" <= char <= "Z") or ("a" <= char <= "z")
        for char in text
    )


for split, path in FILES.items():

    print("\n" + "=" * 55)
    print(f"DATASET: {split.upper()}")
    print("=" * 55)

    df = pd.read_csv(path, encoding="utf-8-sig")

    # Check text and sentiment columns
    print("Total sentences:", len(df))
    print("Missing text:", int(df["text"].isna().sum()))
    print("Missing sentiment:", int(df["sentiment"].isna().sum()))

    # Analyse scripts
    kannada = df["text"].apply(contains_kannada_script)
    latin = df["text"].apply(contains_latin_script)

    print("\nWriting-system statistics:")
    print("Contains Kannada characters:", int(kannada.sum()))
    print("Contains Latin characters:", int(latin.sum()))
    print("Contains both scripts:", int((kannada & latin).sum()))
    print("Contains neither:", int((~kannada & ~latin).sum()))

    # Sentiment distribution
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts().to_string())

    # Show representative sentences
    print("\nSample sentences:")

    for label in ["positive", "negative", "mixed"]:
        subset = df[df["sentiment"] == label]

        print(f"\n{label.upper()}:")

        for sentence in subset["text"].head(3):
            print("-", str(sentence)[:150])