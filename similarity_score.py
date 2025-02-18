from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model import MLPClassifier
import torch
import torch.nn.functional as F

class Similarity:
    def __init__(self):
        pass
    def cosine(self, feature_matrix1, feature_matrix2):
        # similarity_matrix = cosine_similarity(feature_matrix1, feature_matrix2)
        # Normalize the matrices
        feature_matrix1 = F.normalize(feature_matrix1, p=2, dim=-1)
        feature_matrix2 = F.normalize(feature_matrix2, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity_matrix = torch.matmul(feature_matrix1, feature_matrix2.T)
        return similarity_matrix
    
    def el2norm(self, A, B, lambda_factor=1.0):
        """Compute EL2Norm similarity between vectors A and B."""
        # Need to Look for the way of implementing EL2Norm
        euclidean_distance = np.linalg.norm(A - B)
        return np.exp(-lambda_factor * euclidean_distance)
    
    def topk_similar(self, K, feature_matrix1, feature_matrix2):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = self.cosine(feature_matrix1, feature_matrix2)
            sim_weight, sim_indices = sim_matrix.topk(k=K, dim=-1)
            return sim_weight, sim_indices
            # [B, K]
            # sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            # sim_weight = (sim_weight / temperature).exp()
 
            # # counts for each class
            # one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # # [B*K, C]
            # one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # # weighted score ---> [B, C]
            # pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)
 
            # pred_labels = pred_scores.argsort(dim=-1, descending=True)

    def get_majority_label(self, sim_score, sim_indices, feature_labels, batch_size):
         
        sim_labels = torch.gather(feature_labels.expand(batch_size, -1), dim=-1, index=sim_indices)

        # Count occurrences of 0s and 1s along each row
        count_zeros = (sim_labels == 0).sum(dim=1)
        count_ones = (sim_labels == 1).sum(dim=1)

        # Determine the majority value and its count
        majority_value = (count_ones > count_zeros).int()  # 1 if ones are majority, else 0
        majority_count = torch.max(count_zeros, count_ones)

        # Create a (rows, 2) matrix: [majority_value, majority_count]
        result_matrix = torch.stack((majority_value, majority_count), dim=1)
        return result_matrix
        
         

if __name__ == "__main__":
    # Example usage
    # train_feature = np.array([0.1, 0.2, 0.3, 0.4]) 
    # test_feature = np.array([0.15, 0.25, 0.35, 0.45])
    # similarity = Similarity()
    # score = similarity.cosine(train_feature, test_feature)
    # print(f"Similarity Score: {score}")


    # Train and Test Resprentation Similiarity Score Calculation
    mlp_dims = [10, 32, 16, 3]  # Example: input=10, two hidden layers (32, 16), output=3
    model = MLPClassifier(mlp_dims)

    # Generate sample input matrices (10x10 tensors)
    train_input = torch.rand(10, 10)  # 10 samples, 10 features each
    test_input = torch.rand(10, 10)  # 10 samples, 10 features each

    # Extract features from the penultimate layer
    train_features = model.encode(train_input)#.detach().numpy()
    test_features = model.encode(test_input)#.detach().numpy()

    similarity = Similarity()

    topk_sim_weight, topk_sim_indices = similarity.topk_similar(5, test_features, train_features)

    print("Top K similar Weights: ", topk_sim_weight)
    print("Top K similar Indices: ", topk_sim_indices)

    feature_labels = torch.randint(0, 2, (10, 10))  # Generates 0 or 1 randomly

    pseudo_labels = similarity.get_majority_label(topk_sim_weight, topk_sim_indices, feature_labels, batch_size=10)

    print("Pesudo Labels: ", pseudo_labels)
    

    # cosine_score = similarity.cosine(train_features, test_features)
    # el2norm_score = similarity.el2norm(train_features, test_features, lambda_factor=0.5)

    # print(f"Cosine-similarity Score: {cosine_score}")
    # print(f"EL2Norm Score: {el2norm_score}")
