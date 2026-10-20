
from pathlib import Path
import re
import joblib

try:
    from .topic_detector import detect_topic
    from .contextual_meaning import explain_context
    from .semantic_analysis import SemanticAnalyzer
except ImportError:
    from topic_detector import detect_topic
    from contextual_meaning import explain_context
    from semantic_analysis import SemanticAnalyzer


ROOT = Path(__file__).resolve().parent.parent
SENTIMENT_MODEL_PATH = ROOT / "models" / "combined_word_char_tfidf.joblib"


class TextAnalyzer:
    def __init__(self):
        if not SENTIMENT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Sentiment model not found: {SENTIMENT_MODEL_PATH}"
            )

        self.sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
        self.semantic_analyzer = SemanticAnalyzer()

    def predict_sentiment(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please provide a non-empty sentence.")

        normalized = text.casefold()

        positive_roman = [
            "super", "sakkath", "sakath", "sakkat", "sakkattu",
            "bombat", "bombaat", "bombatt", "mast", "masth",
            "barjari", "bharjari", "adduri", "adhoori",
            "chindi", "bindas", "jhakaas", "jakkas",
            "awesome", "excellent", "amazing", "fantastic",
            "superb", "brilliant", "vera level", "semma",
            "psycho", "psych", "full khushi", "chennagide",
            "chennagittu", "tumba chennagide", "chanagide",
            "chennagide guru", "ishtavaytu",
        ]

        positive_kannada = [
            "ಸಕ್ಕತ್", "ಸಖತ್", "ಬೊಂಬಾಟ್", "ಬರ್ಜರಿ", "ಭರ್ಜರಿ",
            "ಅದ್ದೂರಿ", "ಮಸ್ತ್", "ಚೆನ್ನಾಗಿದೆ", "ಚೆನ್ನಾಗಿತ್ತು",
            "ಅದ್ಭುತ", "ಅತ್ಯುತ್ತಮ", "ಚಿಂದಿ", "ಖುಷಿ",
            "ಸೂಪರ್", "ಜಕ್ಕಾಸ್", "ಇಷ್ಟವಾಯಿತು",
            "ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
        ]

        negative_roman = [
            "dabba", "galiju", "galeeju", "galeej", "galeech",
            "thu", "thoo", "chi", "chee", "chhee",
            "asayya", "asahya", "ashahya", "holasu",
            "ketta", "kettadu", "chennagilla", "chanagilla",
            "worst", "waste", "useless", "terrible", "awful",
            "horrible", "pathetic", "bekaar", "bekar",
            "bore", "boring", "disgusting", "poor quality",
            "mosam", "sucks", "bad", "hate", "waste fellow",
        ]

        negative_kannada = [
            "ಡಬ್ಬಾ", "ಗಲೀಜು", "ಗಲೀಜ", "ಹೊಲಸು",
            "ಅಸಹ್ಯ", "ಅಸಯ್ಯ", "ಥೂ", "ಛೀ",
            "ಕೆಟ್ಟ", "ಚೆನ್ನಾಗಿಲ್ಲ", "ಕಳಪೆ", "ಬೋರ್",
            "ಹಾಳಾಗಿದೆ", "ಮೋಸ", "ಕಿರಿಕಿರಿ",
            "ನಿಷ್ಪ್ರಯೋಜಕ", "ಅಸಹ್ಯವಾಗಿದೆ",
        ]

        neutral_roman = [
            "mamuli", "mamool", "okay", "ok", "ok ok",
            "okay okay", "eno onthra", "eno ondhara",
            "eno onthara", "yeno anthara", "average",
            "parvagilla", "paravagilla", "sumaaru",
            "not bad", "so so", "medium", "normal",
            "decent", "alright", "manageable",
        ]

        neutral_kannada = [
            "ಮಾಮೂಲಿ", "ಏನೋ ಒಂಥರ", "ಏನೋ ಒಂದರ",
            "ಪರವಾಗಿಲ್ಲ", "ಸಾಧಾರಣ", "ಸರಾಸರಿ",
            "ಓಕೆ", "ಸರಿ", "ಅಡ್ಡಿಯಿಲ್ಲ",
        ]

        def contains_roman(cues):
            for cue in cues:
                pattern = r"(?<![a-z])" + re.escape(cue) + r"(?![a-z])"
                if re.search(pattern, normalized):
                    return True
            return False

        def contains_kannada(cues):
            return any(cue in normalized for cue in cues)

        has_positive = (
            contains_roman(positive_roman)
            or contains_kannada(positive_kannada)
        )
        has_negative = (
            contains_roman(negative_roman)
            or contains_kannada(negative_kannada)
        )
        has_neutral = (
            contains_roman(neutral_roman)
            or contains_kannada(neutral_kannada)
        )

        # Conflicting cues: defer to the trained classifier.
        if has_positive and has_negative:
            return self.sentiment_model.predict([text])[0]

        if has_negative:
            return "negative"

        if has_positive:
            return "positive"

        if has_neutral:
            return "mixed"

        return self.sentiment_model.predict([text])[0]

    def calculate_similarity(self, text1, text2):
        return self.semantic_analyzer.calculate_similarity(text1, text2)

    def predict_topic(self, text):
        return detect_topic(text)

    def explain_context(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please provide a non-empty sentence.")

        sentiment = self.predict_sentiment(text)
        topic = self.predict_topic(text)

        return explain_context(
            text=text,
            sentiment=sentiment,
            topic=topic,
        )
