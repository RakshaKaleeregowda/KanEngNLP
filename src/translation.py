
import re


# Known words and phrases. These rules cover only selected expressions.
PHRASE_TRANSLATIONS = {
    "chennagilla": "is not good",
    "chennagide": "is good",
    "ishta patte": "liked",
    "ishta": "like",
    "thumba": "very",
    "tumba": "very",
    "naanu": "I",
    "nanu": "I",
    "nange": "I",
    "neenu": "you",
    "avanu": "he",
    "avalu": "she",
    "super": "excellent",
    "great": "great",
    "love": "love",
    "hate": "hate",
    "movie": "movie",
    "song": "song",
}


def translate_to_english(text):
    """
    Translate selected Romanized Kannada-English patterns.

    This is a rule-based prototype, not a general-purpose translator.
    """

    normalized = re.sub(r"\s+", " ", str(text).lower()).strip()

    if not normalized:
        return {
            "translation": "",
            "status": "empty_input",
            "method": "rule_based",
        }

    # Match complete sentence patterns first to preserve English grammar.
    sentence_patterns = [
        (
            r"(?:ee|e) movie (?:thumba|tumba) chennagide",
            "This movie is very good.",
        ),
        (
            r"(?:ee|e) movie chennagide",
            "This movie is good.",
        ),
        (
            r"(?:ee|e) movie (?:thumba|tumba) chennagilla",
            "This movie is very bad.",
        ),
        (
            r"(?:ee|e) movie chennagilla",
            "This movie is not good.",
        ),
        (
            r"(?:naanu|nanu) ee movie (?:thumba|tumba )?ishta patte",
            "I liked this movie very much.",
        ),
        (
            r"(?:naanu|nanu) ee movie ishta patte",
            "I liked this movie.",
        ),
    ]

    for pattern, translation in sentence_patterns:
        if re.fullmatch(pattern, normalized):
            return {
                "translation": translation,
                "status": "translated",
                "method": "rule_based_sentence_pattern",
            }

    # Otherwise, translate only if at least one known phrase is present.
    translated = normalized
    matched = False

    for phrase in sorted(PHRASE_TRANSLATIONS, key=len, reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b"

        translated, count = re.subn(
            pattern,
            PHRASE_TRANSLATIONS[phrase],
            translated,
        )

        if count:
            matched = True

    if not matched:
        return {
            "translation": None,
            "status": "no_matching_rules",
            "method": "rule_based",
        }

    translated = re.sub(r"\s+", " ", translated).strip()
    translated = translated[0].upper() + translated[1:]

    return {
        "translation": translated,
        "status": "partial_rule_based_translation",
        "method": "rule_based",
    }
