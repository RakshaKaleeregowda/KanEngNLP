
import re

TOPIC_KEYWORDS = {
    "movies_entertainment": [
        "movie", "film", "actor", "actress", "song", "cinema",
        "ಸಿನಿಮಾ", "ಹಾಡು"
    ],
    "food_drinks": [
        "food", "coffee", "tea", "lunch", "dinner", "hungry",
        "ಊಟ", "ಕಾಫಿ", "ತಿಂಡಿ"
    ],
    "technology": [
        "computer", "mobile", "phone", "software", "app",
        "internet", "technology", "ಕಂಪ್ಯೂಟರ್", "ಮೊಬೈಲ್"
    ],
    "education": [
        "college", "school", "exam", "study", "teacher",
        "student", "ಪರೀಕ್ಷೆ", "ಕಾಲೇಜು"
    ],
    "weather": [
        "weather", "rain", "sunny", "hot", "cold",
        "ಮಳೆ", "ಬಿಸಿಲು", "ಹವಾಮಾನ"
    ],
    "sports": [
        "cricket", "football", "match", "player", "ನಮ್ಮ ತಂಡ"
    ],
}

def detect_topic(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a non-empty sentence.")

    normalized = text.lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
            if re.search(pattern, normalized):
                score += 1
        scores[topic] = score

    best_topic = max(scores, key=scores.get)

    if scores[best_topic] == 0:
        return "other/unknown"

    return best_topic
