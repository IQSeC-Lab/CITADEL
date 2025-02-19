from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F
import os
from joblib import Parallel, delayed
import pandas as pd
from scipy.spatial.distance import cdist

# local imports
import data
import utils
from models import MLPClassifier, SimpleEncClassifier

class Similarity:
    def __init__(self):
        pass
    def cosine(self, feature_matrix1, feature_matrix2):
         # Convert NumPy arrays to PyTorch tensors if necessary
        if isinstance(feature_matrix1, np.ndarray):
            feature_matrix1 = torch.tensor(feature_matrix1, dtype=torch.float32)
        if isinstance(feature_matrix2, np.ndarray):
            feature_matrix2 = torch.tensor(feature_matrix2, dtype=torch.float32)

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
    
    def euclidean_dist(self, A, B, lambda_factor=1.0):
        """Compute EL2Norm similarity between vectors A and B."""
        # Compute pairwise Euclidean distances
        euclidean_distances = cdist(A, B, metric='euclidean')
        return torch.tensor(np.exp(-lambda_factor * euclidean_distances))
    
    
    def topk_similar(self, K, feature_matrix1, feature_matrix2):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
            # sim_matrix = self.cosine(feature_matrix1, feature_matrix2) # Cosine Similarity
            sim_matrix = self.euclidean_dist(feature_matrix1, feature_matrix2)
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
         
        # Convert NumPy arrays to PyTorch tensors if necessary
        if isinstance(feature_labels, np.ndarray):
            feature_labels = torch.tensor(feature_labels, dtype=torch.float32)
        if isinstance(sim_indices, np.ndarray):
            sim_indices = torch.tensor(sim_indices, dtype=torch.long)

        # sim_labels = torch.gather(feature_labels.expand(feature_labels.shape[0], -1), dim=-1, index=sim_indices)
        # Ensure feature_labels is a 1D tensor
        print("Feature Labels Shape: ", feature_labels.shape)
        if feature_labels.dim() == 1:
            feature_labels = feature_labels.unsqueeze(0)

        # Expand feature_labels to match the dimensions of sim_indices
        expanded_feature_labels = feature_labels.expand(sim_indices.size(0), -1)
        print("Feature Labels Shape after unsqueeze: ", feature_labels.shape)
        print("Sim Indices Shape: ", sim_indices.shape)
        print("Expanded Feature Labels Shape: ", expanded_feature_labels.shape)

        # Gather the labels based on sim_indices
        sim_labels = torch.gather(expanded_feature_labels, dim=1, index=sim_indices)

        print("Sim Labels Shape: ", sim_labels.shape)

        # Initialize an empty tensor to store (index, label) pairs
        index_label_pairs = [[None for _ in range(sim_indices.size(1))] for _ in range(sim_indices.size(0))]

        # Iterate over sim_indices and fetch the corresponding labels from feature_labels
        for i in range(sim_indices.size(0)):
            for j in range(sim_indices.size(1)):
                index = sim_indices[i, j].item()
                label = feature_labels[0, index].item()
                index_label_pairs[i][j] = str((int(index), int(label)))


        # Count occurrences of 0s and 1s along each row
        count_zeros = (sim_labels == 0).sum(dim=1)
        count_ones = (sim_labels == 1).sum(dim=1)

        # Determine the majority value and its count
        majority_value = (count_ones > count_zeros).int()  # 1 if ones are majority, else 0
        majority_count = torch.max(count_zeros, count_ones)

        # Create a (rows, 2) matrix: [majority_value, majority_count]
        result_matrix = torch.stack((majority_value, majority_count), dim=1)
        return result_matrix.numpy(), np.array(index_label_pairs)
        


def load_data_and_model(args):

    X_train, y_train, y_train_family = data.load_range_dataset_w_benign(args.data, args.train_start, args.train_end)
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_train_family shape: {y_train_family.shape}')

    X_test, _, _ = data.load_range_dataset_w_benign(args.data, args.test_start, args.test_end)
    print(f'y_test shape: {X_test.shape}')
    # print(f'y_test shape: {y_test.shape}')
    # print(f'y_test_family shape: {y_test_family.shape}')

    # ENC_MODEL_PATH = os.path.join(MODEL_DIR, f'{encoder_name}_lr{args.learning_rate}_{args.optimizer}_.pth')
    ENC_MODEL_PATH = "/home/ihossain/ISMAIL/SSL-malware/models/gen_apigraph_drebin/simple_enc_classifier_lr0.003_sgd_.pth"
    # only based on malicious training samples
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    # convert y_train to y_train_binary
    y_train_binary = np.array([1 if item != 0 else 0 for item in y_train])
    BIN_NUM_CLASSES = 2


    cls_gpu = True
    if args.encoder == 'simple-enc-mlp':
        # Enc + MLP model 
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = SimpleEncClassifier(enc_dims, mlp_dims)
        encoder_name = 'simple_enc_classifier'
    else:
        raise Exception(f'The encoder {args.encoder} is not supported yet.')

    return encoder, X_train, y_train_binary, X_test, ENC_MODEL_PATH

def pesudo_labeling(args):
    print("Pseudo Labeling for Acutal Datasets started")
    encoder, X_train, y_train_binary, X_test, ENC_MODEL_PATH  = load_data_and_model(args)

    print("Model loading started...")
    state_dict = torch.load(ENC_MODEL_PATH)
    encoder.load_state_dict(state_dict['model'])
    print("Model loading ended!")

    # prepare X_feat and X_feat_tensor if they are embeddings
    if args.cls_feat == 'encoded':
        X_train_tensor = torch.from_numpy(X_train).float()
        X_test_tensor = torch.from_numpy(X_test).float()
        if torch.cuda.is_available():
            # Getting Train Features
            X_train_tensor = X_train_tensor.cuda()
            X_feat_tensor = encoder.cuda().encode(X_train_tensor)
            X_train_feat = X_feat_tensor.cpu().detach().numpy()

            # Getting Test Features
            X_test_tensor = X_test_tensor.cuda()
            X_test_feat = encoder.encode(X_test_tensor).cpu().detach().numpy()
        else:
            X_train_feat = encoder.encode(X_train_tensor).numpy()
    else:
        # args.cls_feat == 'input'
        X_train_feat = X_train
    
    print("X_train_feat Shape: ", X_train_feat.shape)
    print("X_test_feat Shape: ", X_test_feat.shape)

    print("X_train_feat: ", X_train_feat[:1, :10])
    print("X_test_feat: ", X_test_feat[:1, :10])

    # Train and Test Resprentation Similiarity Score Calculation
    similarity = Similarity()
    # Here, we are calculating the top K=5 similar indices
    topk_sim_weight, topk_sim_indices = similarity.topk_similar(5, X_test_feat, X_train_feat)

    # print("Top K similar Weights: ", topk_sim_weight)
    print("Top K similar Indices: ", topk_sim_indices)

    pseudo_labels, topk_info = similarity.get_majority_label(topk_sim_weight, topk_sim_indices, \
                                                   feature_labels=y_train_binary, batch_size=args.bsize)

    print("Pesudo Labels shape: ", pseudo_labels.shape)
    print("Topk_info shape: ", topk_info.shape)

    print("Pseudo Labeling for Acutal Datasets ended")

    # Concatenate the arrays along the second dimension (columns)
    concatenated_array = np.concatenate((pseudo_labels, topk_info), axis=1)

    # Convert the concatenated array to a DataFrame
    column_names = ['Pseudo Label', 'Majority', 'Train Point 1', 'Train Point 2', 'Train Point 3', 'Train Point 4','Train Point 5']

    df = pd.DataFrame(concatenated_array, columns=column_names)

    # Save the DataFrame to a CSV file
    df.to_csv('/home/ihossain/ISMAIL/SSL-malware/pseudo_labels/pseudo_labels_output.csv', index=True, header=True)

    print("Data saved to output.csv")

if __name__ == "__main__":
    """
    Step (0): Init log path and parse args.
    """
    args = utils.parse_args()

    """
    # Pseudo Labeling for Acutal Datasets
    """
    
    pesudo_labeling(args)

    # Example usage
    # train_feature = np.array([0.1, 0.2, 0.3, 0.4]) 
    # test_feature = np.array([0.15, 0.25, 0.35, 0.45])
    # similarity = Similarity()
    # score = similarity.cosine(train_feature, test_feature)
    # print(f"Similarity Score: {score}")


    # Train and Test Resprentation Similiarity Score Calculation
    # mlp_dims = [10, 32, 16, 3]  # Example: input=10, two hidden layers (32, 16), output=3
    # model = MLPClassifier(mlp_dims)

    # # Generate sample input matrices (10x10 tensors)
    # train_input = torch.rand(10, 10)  # 10 samples, 10 features each
    # test_input = torch.rand(10, 10)  # 10 samples, 10 features each

    # # Extract features from the penultimate layer
    # train_features = model.encode(train_input)#.detach().numpy()
    # test_features = model.encode(test_input)#.detach().numpy()

    # similarity = Similarity()

    # topk_sim_weight, topk_sim_indices = similarity.topk_similar(5, test_features, train_features)

    # print("Top K similar Weights: ", topk_sim_weight)
    # print("Top K similar Indices: ", topk_sim_indices)

    # feature_labels = torch.randint(0, 2, (10, 10))  # Generates 0 or 1 randomly

    # pseudo_labels = similarity.get_majority_label(topk_sim_weight, topk_sim_indices, feature_labels, batch_size=10)

    # print("Pesudo Labels: ", pseudo_labels)

    # cosine_score = similarity.cosine(train_features, test_features)
    # el2norm_score = similarity.el2norm(train_features, test_features, lambda_factor=0.5)

    # print(f"Cosine-similarity Score: {cosine_score}")
    # print(f"EL2Norm Score: {el2norm_score}")
