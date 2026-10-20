# CodeMix AI — Kannada-English Code-Mixed Text Analysis

**A Streamlit NLP demo for sentiment analysis and supporting language utilities for Kannada-English text.**

[**Launch the live app**](https://kanengnlp-4if9rjnq3c33wa8ybhqrc.streamlit.app) · [View source code](https://github.com/rakshajeevan8431-source/KanEngNLP)

## Overview
CodeMix AI explores sentiment classification for text that mixes Kannada and English, including Romanized Kannada and Kannada script. The application combines a saved TF-IDF/Logistic Regression classifier with supplementary rule-based text cues and optional language utilities. It is an educational prototype; not every feature has been independently evaluated.

## Features
- **Sentiment classification:** Saved combined word- and character-level TF-IDF model, plus phrase-cue overrides for selected expressions.
- **Emotion fingerprint:** Keyword-based estimate, not a trained emotion classifier.
- **Topic helper:** Predefined keyword groups for entertainment, food, technology, education, weather, and sports.
- **Context and explanation cues:** Rule-based phrase matches, not model-generated token attributions.
- **Script profile:** Detects Kannada and Latin scripts and selected possible Romanized-Kannada cues; no language percentages.
- **Semantic similarity:** Multilingual MiniLM sentence embeddings and cosine similarity.
- **Voice input:** SpeechRecognition with Google's online speech recognition configured for Kannada (`kn-IN`).
- **Translation:** GoogleTranslator followed by MyMemory fallback; service availability and quality can vary.
- **Transliteration:** Aksharamukha for Kannada and standardized Roman (IAST) conversion. Arbitrary colloquial Kanglish is not guaranteed to transliterate correctly.

## Live demo
[Open CodeMix AI on Streamlit Community Cloud](https://kanengnlp-4if9rjnq3c33wa8ybhqrc.streamlit.app)

The first load may take time because the semantic component loads a pretrained multilingual sentence-embedding model. Internet access is needed for model downloads and online voice/translation features.

## Technology
Python, Streamlit, pandas, NumPy, scikit-learn, joblib, TF-IDF, Logistic Regression, Sentence Transformers, SpeechRecognition, deep-translator, Aksharamukha.

## Repository structure
```text
KanEngNLP/
├── data/raw/                  # Original source splits
├── data/processed/            # Cleaned and deduplicated splits
├── models/                    # Saved scikit-learn pipelines
├── notebooks/                 # Exploratory notebooks
├── results/                   # Metrics, predictions and error analysis
├── src/app.py                 # Streamlit UI
├── src/text_analyzer.py       # Model loading and sentiment logic
├── src/semantic_analysis.py   # Embeddings and cosine similarity
├── src/contextual_meaning.py  # Rule-based explanation
├── src/topic_detector.py      # Keyword topic helper
├── src/prepare_dataset.py     # Label mapping and cleaning
├── src/create_clean_splits.py # Cross-split deduplication
├── src/train_baseline.py      # Word TF-IDF baseline
├── src/train_char_baseline.py # Character TF-IDF baseline
├── src/train_word_char_combined.py
├── requirements.txt
├── LICENSE
└── README.md
```
Additional analysis and validation scripts are also under `src/`.

## Run locally
Prerequisites: Python 3.10–3.12 recommended, pip, and internet access for pretrained model downloads and online services.

1. Download and extract the repository ZIP, or clone:
```bash
git clone https://github.com/rakshajeevan8431-source/KanEngNLP.git
cd KanEngNLP
```
2. Create a virtual environment.

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
4. Run from the repository root:
```bash
streamlit run src/app.py
```
The saved sentiment model must be present at `models/combined_word_char_tfidf.joblib`.

## Recorded evaluation results
These values are transcribed from repository result CSVs; they were not independently recomputed for this README.

| Experiment | Split | Samples | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Combined word + character TF-IDF + Logistic Regression | Validation | 551 | 0.7241 | 0.6312 |
| Combined word + character TF-IDF + Logistic Regression | Held-out test | 554 | 0.6913 | 0.5943 |
| Weighted XLM-RoBERTa Experiment 3 | Held-out test | 554 | 0.7094 | 0.5925 |

The recorded test comparison gives mixed-class F1 of 0.3453 for the combined TF-IDF model and 0.3053 for the weighted XLM-RoBERTa experiment. XLM-RoBERTa has higher accuracy, while combined TF-IDF has marginally higher macro-F1. Do not infer a categorical winner from these metrics alone.

## Limitations
- Performance may vary with spelling, script, code-mixing, negation, sarcasm, and domain.
- `TextAnalyzer` applies phrase overrides before falling back to the saved classifier.
- Emotion, topic, script-cue, civility, and contextual-explanation features are heuristic aids, not independently validated classifiers.
- Semantic similarity is not a sentiment score or probability.
- Translation and voice recognition rely on external services.
- The embedding model is initialized during analyzer startup and may require a download even if similarity is not used.
- Package versions are not pinned; exact environment reproduction is not guaranteed.

## Dataset and licensing
The preprocessing script maps `positive`, `negative`, and `mixed_feelings` to `positive`, `negative`, and `mixed`; it excludes `not-kannada` and `unknown_state`. Before redistributing raw or processed data, verify the original dataset source, licence, attribution requirements, and redistribution permissions. The code licence does not automatically cover third-party datasets or pretrained models.

## Responsible use
For educational and research use. Do not use heuristic outputs for high-stakes decisions or treat them as definitive judgments of a person's intent or feelings.

## License
See [LICENSE](LICENSE) for the repository code licence. Dataset and pretrained-model terms may differ.
