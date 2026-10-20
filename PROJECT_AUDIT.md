# PROJECT_AUDIT.md — CodeMix AI

**Repository:** `rakshajeevan8431-source/KanEngNLP` · **Branch:** `main`  
**Scope:** source-based audit of repository files and recorded result CSVs. No fresh install or independent metric recomputation is claimed.

## Executive summary
CodeMix AI is a Streamlit prototype for Kannada-English code-mixed sentiment analysis. Its primary model is a saved scikit-learn pipeline combining word and character TF-IDF with Logistic Regression. Supporting features include phrase-based sentiment overrides, keyword topic/emotion/civility cues, rule-based explanations, script-cue detection, sentence-embedding similarity, online speech recognition, translation, and transliteration.

## Important files
- `src/app.py`: Streamlit UI and UI-level heuristics; text, voice, translation/transliteration, and similarity interfaces.
- `src/text_analyzer.py`: loads `models/combined_word_char_tfidf.joblib`, applies phrase-level sentiment overrides, and delegates helper functions.
- `src/semantic_analysis.py`: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, normalized embeddings, cosine similarity.
- `src/contextual_meaning.py`: rule-based phrase matches and explanations.
- `src/topic_detector.py`: keyword-counting topic helper.
- `src/prepare_dataset.py`: semicolon-delimited input parsing, label mapping, malformed-row handling, within-split duplicate/conflict handling.
- `src/create_clean_splits.py`: exact-text deduplication across train/dev/test, priority train > dev > test, exclusion of cross-split conflicting-label text.
- `src/train_baseline.py`: word TF-IDF + Logistic Regression.
- `src/train_char_baseline.py`: character TF-IDF + Logistic Regression.
- `src/train_word_char_combined.py`: combined word/character TF-IDF + Logistic Regression.
- `requirements.txt`: package names listed without version pins.

## Dataset processing
The scripts retain `positive`, `negative`, and `mixed_feelings`, mapping the last to `mixed`. They exclude `not-kannada` and `unknown_state`. The raw data is semicolon-delimited; the final field is treated as the label and preceding fields are rejoined as text. Malformed rows and exact duplicates are handled by the cleaning script. The cross-split script prioritizes train, then dev, then test, and removes text whose labels conflict across splits.

This is exact-string deduplication; it does not prove that near-duplicates, spelling variants, or author-level correlations are absent.

**Not verified from inspected files:** authoritative dataset title, publisher, URL, licence, collection method, annotation guidelines, or permission to redistribute. Obtain these from the original dataset source before publication.

## Model configurations verified in code
- **Word baseline:** word n-grams (1,2), `min_df=2`, `max_features=50000`, `sublinear_tf=True`; Logistic Regression `max_iter=2000`, `class_weight='balanced'`, `random_state=42`.
- **Character baseline:** character n-grams (2,5), `min_df=2`, `max_features=200000`, `sublinear_tf=True`; same Logistic Regression settings.
- **Combined:** `FeatureUnion` of word n-grams (1,2), `min_df=2`, sublinear TF (no explicit max feature cap) and character n-grams (2,5), `min_df=2`, max 200,000, sublinear TF; same Logistic Regression settings.

## Deployed inference caveat
`TextAnalyzer.predict_sentiment` applies selected phrase rules before using the saved classifier: selected positive cues return positive, negative cues return negative, neutral cues return mixed; conflicting positive/negative cues defer to the classifier. Consequently, live predictions can differ from raw pipeline predictions. Verify the evaluation code path before claiming recorded metrics measure the complete deployed wrapper.

## Recorded results from repository CSVs
| Experiment | Split | N | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Combined word + character TF-IDF | Validation | 551 | 0.7241379 | 0.6312073 |
| Combined word + character TF-IDF | Held-out test | 554 | 0.6913357 | 0.5942740 |
| Weighted XLM-RoBERTa Experiment 3 | Held-out test | 554 | 0.7093863 | 0.5925433 |

Held-out test class-wise F1: combined TF-IDF negative 0.6541, positive 0.7834, mixed 0.3453; weighted XLM-RoBERTa negative 0.6603, positive 0.8120, mixed 0.3053. Transformer accuracy is higher; combined TF-IDF macro-F1 is marginally higher. The mixed class is weak for both.

The combined validation CSV records 4,358 training rows and 551 validation rows. The test comparison records 554 samples. These counts were not regenerated. Other comparison CSVs include earlier experiments; confirm split and protocol comparability before combining them.

## Supporting features
- Emotion: fixed keyword sets and counts, not a trained emotion model.
- Topic: fixed keyword matches.
- Context: phrase rules.
- Script profile: Unicode checks and possible Romanized-Kannada cues; no language percentages.
- Civility/toxicity: limited keyword matching, not a validated toxicity classifier.
- Similarity: multilingual MiniLM embeddings and cosine similarity; not sentiment or probability.
- Voice: SpeechRecognition with online Google recognition configured as `kn-IN`.
- Translation: GoogleTranslator, then MyMemory fallback.
- Transliteration: Aksharamukha Kannada ↔ IAST; arbitrary colloquial Kanglish is not guaranteed to be valid IAST.

## Risks and checks not performed
1. Dependencies are unpinned.
2. `TextAnalyzer` instantiates the sentence embedding model at startup; model download may be needed even if similarity is not used.
3. Online voice and translation services may fail or rate-limit.
4. joblib artifacts should be loaded only from trusted sources and compatible library versions.
5. No fresh-environment installation, fresh-clone launch, training rerun, metric recomputation, external licence verification, or formal security/robustness testing was performed for this audit.

## Before final academic submission
1. Verify and cite dataset source/licence.
2. Freeze tested dependency versions.
3. Run from a clean environment and test every tab.
4. Document all split counts, labels, exclusions, and deduplication.
5. Confirm whether metrics evaluate raw estimator or deployed wrapper.
6. Report class-wise precision, recall, F1, macro-F1, accuracy, and confusion matrices.
7. Keep held-out test data out of iterative tuning.
8. Test Kannada script, Romanized Kannada, English, mixed text, negation, sarcasm, and spelling variants.
9. Consider lazy-loading the sentence embedding model.
10. Distinguish learned models from heuristic components in diagrams and claims.
