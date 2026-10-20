from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

FILES = {
    "TRAIN": DATA_DIR / "train.csv",
    "DEV": DATA_DIR / "dev.csv",
    "TEST": DATA_DIR / "test.csv",
}

VALID_LABELS = {
    "Positive",
    "Negative",
    "Mixed_feelings",
    "not-Kannada",
    "unknown_state",
}


def inspect_file(name, path):
    print(f"\n{'=' * 60}")
    print(f"DATASET: {name}")
    print("=" * 60)

    if not path.exists():
        print(f"File not found: {path}")
        return

    # Read raw lines without changing the file.
    lines = path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    print("Raw line count:", len(lines))

    label_counts = Counter()
    malformed = []
    examples = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        # The final semicolon separates the label from
        # an optional empty trailing field.
        parts = line.rsplit(";", 2)

        if len(parts) < 2:
            malformed.append((line_number, line))
            continue

        # The label should be the field immediately
        # before the optional trailing empty field.
        label = parts[-2].strip()

        if label not in VALID_LABELS:
            malformed.append((line_number, line))
            continue

        sentence = ";".join(parts[:-2]).strip()

        label_counts[label] += 1

        if len(examples) < 5:
            examples.append((sentence, label))

    print("\nLabel counts:")
    for label, count in label_counts.most_common():
        print(f"{label}: {count}")

    print("\nFirst 5 parsed examples:")
    for sentence, label in examples:
        print(f"TEXT: {sentence[:150]}")
        print(f"LABEL: {label}\n")

    print("Unparsed or malformed rows:", len(malformed))

    if malformed:
        print("\nFirst 5 problematic rows:")
        for line_number, line in malformed[:5]:
            print(f"Line {line_number}: {line[:200]}")


for name, path in FILES.items():
    inspect_file(name, path)

print("\nInspection complete.")