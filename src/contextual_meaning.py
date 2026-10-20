
import re


POSITIVE_PHRASES = {
    "chennagide": "something is good or nice",
    "ishta": "liking or preference",
    "super": "strong approval or praise",
    "great": "strong approval or praise",
    "love": "a strong positive feeling",
    "happy": "happiness or satisfaction",
}

NEGATIVE_PHRASES = {
    "chennagilla": "something is not good",
    "dislike": "disapproval or dislike",
    "hate": "a strong negative feeling",
    "bad": "a negative opinion",
}


def explain_context(text, sentiment, topic=None):
    """Generate a cautious, rule-based contextual explanation."""

    normalized = re.sub(r"\s+", " ", str(text).lower()).strip()
    sentiment = str(sentiment).lower().strip()

    positive_matches = [
        meaning
        for phrase, meaning in POSITIVE_PHRASES.items()
        if re.search(r"\b" + re.escape(phrase) + r"\b", normalized)
    ]

    negative_matches = [
        meaning
        for phrase, meaning in NEGATIVE_PHRASES.items()
        if re.search(r"\b" + re.escape(phrase) + r"\b", normalized)
    ]

    detected_cues = sorted(set(positive_matches + negative_matches))

    if positive_matches and negative_matches:
        explanation = (
            "The sentence contains both positive and negative language "
            "cues. This may indicate contrasting opinions or mixed "
            "emotions. The exact interpretation depends on context."
        )

    elif sentiment == "positive":
        if positive_matches:
            cues = "; ".join(positive_matches)
            explanation = (
                "The sentiment model classifies the sentence as positive. "
                f"The detected expressions suggest {cues}. Overall, "
                "the speaker appears to express approval, enjoyment, "
                "liking, or another favourable reaction. This is a "
                "rule-based explanation, not a complete translation."
            )
        else:
            explanation = (
                "The sentiment model classifies the sentence as positive. "
                "The sentence likely expresses a favourable opinion, "
                "positive emotion, or satisfaction. The current phrase "
                "rules could not determine the more specific meaning."
            )

    elif sentiment == "negative":
        if negative_matches:
            cues = "; ".join(negative_matches)
            explanation = (
                "The sentiment model classifies the sentence as negative. "
                f"The detected expressions suggest {cues}. Overall, "
                "the speaker appears to express disapproval, dislike, "
                "or dissatisfaction. This is a rule-based explanation, "
                "not a complete translation."
            )
        else:
            explanation = (
                "The sentiment model classifies the sentence as negative. "
                "The sentence likely expresses an unfavourable opinion "
                "or dissatisfaction, but the current rules could not "
                "determine the specific meaning."
            )

    elif sentiment == "mixed":
        explanation = (
            "The sentiment model classifies the sentence as mixed. "
            "This suggests that it may express contrasting opinions "
            "or emotions. The current rules cannot reliably determine "
            "the exact relationship between them."
        )

    else:
        explanation = (
            "A contextual explanation could not be generated because "
            "the sentiment label was not recognized."
        )

    return {
        "contextual_meaning": explanation,
        "detected_cues": detected_cues,
        "topic": topic or "unknown",
        "method": "rule-based explanation",
    }
