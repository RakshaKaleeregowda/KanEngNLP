
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "deduplicated"
RESULTS_DIR = ROOT / "results"

train_df = pd.read_csv(DATA_DIR / "train_deduplicated.csv")
dev_df = pd.read_csv(DATA_DIR / "dev_deduplicated.csv")

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["sentiment"]

X_dev = dev_df["text"].fillna("").astype(str)
y_dev = dev_df["sentiment"]


models = {
    "Word TF-IDF": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]),

    "Character TF-IDF": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2,
                max_features=200000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]),
}


results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)
    macro_f1 = f1_score(
        y_dev,
        predictions,
        average="macro",
        zero_division=0,
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "macro_f1",
    ascending=False,
)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    RESULTS_DIR / "word_char_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved comparison to results/word_char_comparison.csv")
