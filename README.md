# Kannada-English Code-Mixed Text Analysis

**Sentiment Analysis, Semantic Similarity, and Language Utilities for Kannada-English Code-Mixed Text**

A Python-based NLP project that explores sentiment analysis of Kannada-English code-mixed text, including text written in Romanized Kannada and Kannada script. The project combines machine-learning models with lightweight text-analysis utilities and a Streamlit interface.

## Overview

People frequently mix Kannada and English in everyday digital communication. This creates challenges for traditional sentiment-analysis systems because of spelling variations, mixed scripts, informal expressions, and context-dependent meanings.

This project investigates machine-learning approaches for classifying sentiment in Kannada-English code-mixed text and provides an interactive interface for exploring model predictions and related text-analysis features.

## Features

- **Sentiment analysis:** Predicts sentiment using a saved TF-IDF and Logistic Regression model.
- **Word and character features:** Supports a combined word-level and character-level TF-IDF model, designed to capture lexical information and spelling variations.
- **Emotion cues:** Uses keyword-based indicators to provide an additional view of the text.
- **Topic detection:** Identifies possible topics using keyword-based rules.
- **Prediction explanations:** Highlights relevant lexical cues that may help explain a prediction.
- **Script-related analysis:** Examines selected language and script cues in the input.
- **Translation and transliteration utilities:** Provides language conversion features, subject to the capabilities and availability of the underlying libraries and services.
- **Semantic similarity:** Includes a component for comparing the semantic similarity of text.
- **Voice input:** Supports speech input through a speech-recognition service.
- **Interactive interface:** Provides a Streamlit application for experimenting with text inputs.

*Note:* These features do not all use the same methodology. Sentiment classification uses a trained machine-learning model, while several supplementary analysis features use keyword-based heuristics. Their outputs should not be interpreted as equally validated predictions.

## Technology Stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- TF-IDF feature extraction
- Logistic Regression
- joblib
- Sentence Transformers for semantic similarity
- Translation, transliteration, and speech-recognition libraries

## Project Structure

```text
KannadaEnglishNLP/
├── data/
│   ├── raw/                 # Source dataset splits
│   └── processed/           # Cleaned and deduplicated datasets
├── models/
│   ├── sentiment_tfidf_logreg.joblib
│   ├── char_tfidf_logreg.joblib
│   └── combined_word_char_tfidf.joblib
├── notebooks/               # Exploratory analysis and experiments
├── results/                 # Evaluation metrics, predictions and reports
├── src/
│   ├── app.py               # Streamlit application
│   ├── text_analyzer.py     # Sentiment-analysis logic
│   ├── semantic_analysis.py # Semantic similarity
│   ├── contextual_meaning.py
│   ├── topic_detector.py
│   ├── translation.py
│   ├── train_baseline.py
│   ├── train_char_baseline.py
│   └── train_word_char_combined.py
├── requirements.txt
├── .gitignore
└── README.md
```

Additional scripts and experiment outputs are available in the corresponding directories.

## Getting Started

### Prerequisites

- Python 3.10–3.12 recommended
- pip
- An internet connection for features that rely on online services or download pretrained models

### 1. Download the repository

Download the repository as a ZIP file from GitHub and extract it, or clone it using Git:

```bash
git clone https://github.com/rakshajeevan8431-source/KanEngNLP.git
cd KanEngNLP
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some optional features may require additional system configuration or access to external services.

### 4. Run the application

From the project root directory:

```bash
streamlit run src/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

### 5. Troubleshooting

- **Missing Python module:** Confirm that the virtual environment is activated and install the dependencies from `requirements.txt`.
- **Model file not found:** Run the command from the project root and ensure that the `models/` directory was included in the download.
- **Translation or speech recognition fails:** Check the internet connection and the availability or usage limits of the external service.
- **Semantic similarity takes time to start:** The component may need to download a pretrained model on its first use.
- **Installation errors:** Confirm the Python version and consult the relevant library documentation.

## Model Evaluation

The project includes baseline experiments using word-level TF-IDF, character-level TF-IDF, and combined word-and-character features.

The following results were recorded for the combined TF-IDF and Logistic Regression model:

| Evaluation | Accuracy | Macro F1-score |
|---|---:|---:|
| Validation | 72.41% | 0.6312 |
| Held-out test | 69.13% | 0.5943 |

These figures describe the recorded experiment and are not a guarantee of performance on new data.

The repository's `results/` directory contains additional experiment metrics, classification reports, predictions, and error-analysis outputs. Refer to those files when comparing individual experiments.

## Limitations

- Code-mixed text can contain informal spellings, transliteration differences, sarcasm, and ambiguous expressions.
- Model performance may vary across Kannada script, Romanized Kannada, English, and mixed-script inputs.
- Keyword-based emotion, topic, and explanation features are heuristic aids and are not independently validated classifiers.
- Translation and speech recognition may be inaccurate for colloquial or code-mixed language.
- Semantic similarity depends on its underlying pretrained model and is different from sentiment classification.
- Reported evaluation metrics depend on the dataset, preprocessing, data splits, and evaluation procedure.

## Reproducibility

The saved model files allow supported predictions without retraining the classifiers. The repository also includes datasets and experiment outputs to support further inspection.

Reproducing every experiment may require additional dependencies, pretrained model downloads, and running the relevant scripts in the correct order. The saved evaluation outputs should not be treated as proof that every experiment can be reproduced from a fresh installation without further configuration.

## Responsible Use and Dataset Licensing

This project is intended for educational and research purposes. Predictions should not be treated as definitive assessments of a person's feelings, intent, or behaviour.

Before redistributing the repository's dataset files, verify the original dataset's licence, attribution requirements, and redistribution permissions. Follow the applicable terms for all datasets and pretrained models used.

## Future Improvements

- Improve sentiment classification for Romanized Kannada and Kannada-script inputs.
- Evaluate performance across separate language and script subsets.
- Strengthen contextual handling of negation, sarcasm, and spelling variations.
- Validate supplementary emotion and topic-analysis features against annotated data.
- Improve reproducibility through documented experiment configurations and automated tests.

## License

The repository includes a `LICENSE` file. Refer to it for the project's licensing terms. Dataset and pretrained-model licences may be different from the repository licence.
