
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SemanticAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def get_embedding(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def calculate_similarity(self, text1, text2):
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)

        score = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]

        return float(score)



if __name__ == "__main__":
    analyzer = SemanticAnalyzer()

    test_pairs = [
        (
            "Naanu ee movie tumba ishta patte.",
            "I really liked this movie."
        ),
        (
            "Naanu ee movie tumba ishta patte.",
            "ನನಗೆ ಈ ಸಿನಿಮಾ ತುಂಬಾ ಇಷ್ಟವಾಯಿತು."
        ),
        (
            "Naanu ee movie tumba ishta patte.",
            "I hate this movie."
        ),
        (
            "Naanu ee movie tumba ishta patte.",
            "The weather is very hot today."
        )
    ]

    for sentence1, sentence2 in test_pairs:
        score = analyzer.calculate_similarity(sentence1, sentence2)

        print("\nSentence 1:", sentence1)
        print("Sentence 2:", sentence2)
        print(f"Similarity: {score:.4f}")
