from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Similarity:
    def __init__(self):
        pass
    def cosine(sefl, feature_vector1, feature_vector2):
        similarity_score = cosine_similarity(feature_vector1.reshape(1, -1), feature_vector2.reshape(1, -1))
        return similarity_score[0][0]

if __name__ == "__main__":
    # Example usage
    train_feature = np.array([0.1, 0.2, 0.3, 0.4]) 
    test_feature = np.array([0.15, 0.25, 0.35, 0.45])
    similarity = Similarity()
    score = similarity.cosine(train_feature, test_feature)
    print(f"Similarity Score: {score}")
