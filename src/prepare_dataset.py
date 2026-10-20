from pathlib import Path
from collections import Counter
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": RAW_DIR / "train.csv",
    "dev": RAW_DIR / "dev.csv",
    "test": RAW_DIR / "test.csv",
}

LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "mixed_feelings": "mixed",
}

# Accepted source labels, including labels we will exclude.
ALL_LABELS = {
    "positive",
    "negative",
    "mixed_feelings",
    "not-kannada",
    "unknown_state",
}


def read_dataset(file_path):
    records = []
    malformed = []

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.reader(file, delimiter=";")

        for line_number, row in enumerate(reader, start=1):
            # The source file has a trailing empty field.
            if row and row[-1].strip() == "":
                row = row[:-1]

            if len(row) < 2:
                malformed.append(line_number)
                continue

            # The final field is the label.
            label = row[-1].strip().lower()

            if label not in ALL_LABELS:
                malformed.append(line_number)
                continue

            # Everything before the label is treated as text.
            text = ";".join(row[:-1]).strip()

            if not text:
                continue

            records.append((text, label))

    return records, malformed


def prepare_dataset(name, file_path):
    print(f"\n{'=' * 55}")
    print(f"Processing {name.upper()}")
    print("=" * 55)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    records, malformed = read_dataset(file_path)

    original_counts = Counter(label for _, label in records)

    cleaned = []
    excluded = Counter()

    for text, label in records:
        if label not in LABEL_MAP:
            excluded[label] += 1
            continue

        cleaned.append({
            "text": text,
            "sentiment": LABEL_MAP[label],
        })

    # Remove exact duplicate text within this split.
    # If a sentence has conflicting labels, report it
    # instead of silently choosing one.
    text_labels = {}

    for row in cleaned:
        text = row["text"]
        label = row["sentiment"]

        if text not in text_labels:
            text_labels[text] = set()

        text_labels[text].add(label)

    conflicts = {
        text: labels
        for text, labels in text_labels.items()
        if len(labels) > 1
    }

    conflict_texts = set(conflicts)

    final_rows = []
    seen = set()

    for row in cleaned:
        text = row["text"]
        label = row["sentiment"]

        if text in conflict_texts:
            continue

        key = (text, label)

        if key not in seen:
            final_rows.append(row)
            seen.add(key)

    output_path = PROCESSED_DIR / f"{name}_clean.csv"

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["text", "sentiment"]
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print("Original valid-label counts:", dict(original_counts))
    print("Excluded label counts:", dict(excluded))
    print("Malformed rows:", len(malformed))
    print("Conflicting-label texts:", len(conflict_texts))
    print("Final examples:", len(final_rows))
    print(
        "Final sentiment counts:",
        dict(Counter(row["sentiment"] for row in final_rows))
    )
    print("Saved to:", output_path)


for name, file_path in FILES.items():
    prepare_dataset(name, file_path)

print("\nDataset preparation finished.")