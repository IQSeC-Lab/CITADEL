from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F
import os
from joblib import Parallel, delayed
import pandas as pd
from scipy.spatial.distance import cdist
import logging
import argparse

# local imports
import data
import utils
from models import MLPClassifier, SimpleEncClassifier

# Conformal Prediction Imports

class Similarity:
    def __init__(self):
        pass

    def cosine(self, feature_matrix1, feature_matrix2):
        # Convert NumPy arrays to PyTorch tensors if necessary
        # if isinstance(feature_matrix1, np.ndarray):
        #     feature_matrix1 = torch.tensor(feature_matrix1, dtype=torch.float32)
        # if isinstance(feature_matrix2, np.ndarray):
        #     feature_matrix2 = torch.tensor(feature_matrix2, dtype=torch.float32)

        # # Normalize the matrices
        # feature_matrix1 = F.normalize(feature_matrix1, p=2, dim=-1)
        # feature_matrix2 = F.normalize(feature_matrix2, p=2, dim=-1)
        
        # # Compute cosine similarity
        # similarity_matrix = torch.matmul(feature_matrix1, feature_matrix2.T)
        A = feature_matrix1
        B = feature_matrix2
        dot_product = torch.mm(A, B.T)
        # Compute L2 norm (magnitude) for each row
        A_norm = torch.norm(A, p=2, dim=1, keepdim=True)  # Shape: (3,1)
        B_norm = torch.norm(B, p=2, dim=1, keepdim=True)  # Shape: (5,1)
    
        # Compute cosine similarity matrix (broadcasting norms)
        similarity_matrix = dot_product / (A_norm * B_norm.T)
    
        # print("Cosine Similarity Matrix:\n", cosine_similarity)
        print("Shape:", cosine_similarity.shape)
        return similarity_matrix
    
    def el2norm(self, A, B, lambda_factor=1.0):
        """Compute EL2Norm similarity between vectors A and B."""
        # Need to Look for the way of implementing EL2Norm
        euclidean_distance = np.linalg.norm(A - B)
        return np.exp(-lambda_factor * euclidean_distance)
    
    def euclidean_dist(self, A, B, lambda_factor=1.0):
        """Compute Euclidean similarity between vectors A and B."""
        # Compute pairwise Euclidean distances
        # euclidean_distances = cdist(A, B, metric='euclidean')
        # return torch.tensor(np.exp(-lambda_factor * euclidean_distances))

        # A = torch.randn(3, 10)  # 3 vectors of 10 dimensions
        # B = torch.randn(5, 10)  # 5 vectors of 10 dimensions
        
        # Compute squared norms of A and B
        A_sq = (A ** 2).sum(dim=1, keepdim=True)  # Shape: (3, 1)
        B_sq = (B ** 2).sum(dim=1, keepdim=True)  # Shape: (5, 1)
        
        # Compute pairwise squared Euclidean distance using matrix operations
        pairwise_sq_dist = A_sq - 2 * torch.mm(A, B.T) + B_sq.T  # Shape: (3,5)
        
        # Ensure distances are non-negative due to numerical precision errors
        pairwise_sq_dist = torch.clamp(pairwise_sq_dist, min=0.0)
        
        # Take square root to get final Euclidean distances
        # euclidean_distance = torch.sqrt(pairwise_sq_dist)
        
        # print("Optimized Euclidean Distance Matrix:\n", euclidean_distance)
        print("Shape:", pairwise_sq_dist.shape)
        return pairwise_sq_dist
    
    def topk_similar(self, K, sm_fn, feature_matrix1, feature_matrix2):
        # compute cos similarity between each feature vector
        if sm_fn == 'cosine':
            sim_matrix = self.cosine(feature_matrix1, feature_matrix2)
        else:
            sim_matrix = self.euclidean_dist(feature_matrix1, feature_matrix2)

        sim_weight, sim_indices = sim_matrix.topk(k=K, dim=-1)

        # Calculate the maximum and average values of the top K similarities
        max_sim = sim_weight.max(dim=-1).values
        avg_sim = sim_weight.mean(dim=-1)

        return sim_weight, sim_indices, max_sim, avg_sim

    def get_majority_label(self, sim_score, sim_indices, feature_labels, feature_family=None):
        # Convert NumPy arrays to PyTorch tensors if necessary
        if isinstance(feature_labels, np.ndarray):
            feature_labels = torch.tensor(feature_labels, dtype=torch.float32)
        # if isinstance(feature_family, np.ndarray):
        #     feature_family = torch.tensor([hash(fam) for fam in feature_family], dtype=torch.float32)
        if isinstance(sim_indices, np.ndarray):
            sim_indices = torch.tensor(sim_indices, dtype=torch.long)

        # Ensure feature_labels is a 1D tensor
        logging.info("Feature Labels Shape: %s", feature_labels.shape)
        if feature_labels.dim() == 1:
            feature_labels = feature_labels.unsqueeze(0)

        # Ensure feature_family is a 1D tensor
        # if isinstance(feature_family, np.ndarray):
        #     if feature_family.ndim == 1:
        #         feature_family = torch.tensor(feature_family, dtype=torch.float32).unsqueeze(0)
        # elif feature_family.dim() == 1:
        #     feature_family = feature_family.unsqueeze(0)

        # Expand feature_labels to match the dimensions of sim_indices
        expanded_feature_labels = feature_labels.expand(sim_indices.size(0), -1)
        # Expand feature_family to match the dimensions of sim_indices
        # expanded_feature_family = feature_family.expand(sim_indices.size(0), -1)
        logging.info("Feature Labels Shape after unsqueeze: %s", feature_labels.shape)
        logging.info("Sim Indices Shape: %s", sim_indices.shape)
        logging.info("Expanded Feature Labels Shape: %s", expanded_feature_labels.shape)
        # logging.info("Expanded Feature Family Shape: %s", expanded_feature_family.shape)

        # Gather the labels based on sim_indices
        sim_labels = torch.gather(expanded_feature_labels, dim=1, index=sim_indices)

        # # Gather the Family based on sim_indices
        # sim_families = torch.gather(expanded_feature_labels, dim=1, index=sim_indices)
        logging.info("Sim Labels Shape: %s", sim_labels.shape)
        # logging.info("Sim Families: %s", feature_family)

        # Initialize an empty tensor to store (index, label) pairs
        index_label = [[None for _ in range(sim_indices.size(1))] for _ in range(sim_indices.size(0))]

        # Iterate over sim_indices and fetch the corresponding labels from feature_labels
        # tuple: (index, similarity_score, label, family)
        for i in range(sim_indices.size(0)):
            for j in range(sim_indices.size(1)):
                index = sim_indices[i, j].item()
                label = feature_labels[0, index].item()
                # family = feature_family[index]
                index_label[i][j] = str((int(index), float(sim_score[i,j]), int(label)))

        # Count occurrences of 0s and 1s along each row
        count_zeros = (sim_labels == 0).sum(dim=1)
        count_ones = (sim_labels == 1).sum(dim=1)

        # Determine the majority value and its count
        majority_value = (count_ones > count_zeros).int()  # 1 if ones are majority, else 0
        majority_count = torch.max(count_zeros, count_ones)

        # Create a (rows, 2) matrix: [majority_value, majority_count]
        result_matrix = torch.stack((majority_value, majority_count), dim=1)
        return result_matrix.numpy(), np.array(index_label)

def load_data_and_model(args):
    X_train, y_train, y_train_family = data.load_range_dataset_w_benign(args.data, args.train_start, args.train_end)
    logging.info('X_train shape: %s', X_train.shape)
    logging.info('y_train shape: %s', y_train.shape)
    logging.info('y_train_family shape: %s', y_train_family.shape)

    ben_len = X_train.shape[0] - y_train_family.shape[0]
    y_ben_family = np.full(ben_len, 'benign')
    all_train_family = np.concatenate((y_train_family, y_ben_family), axis=0)

    X_test, y_test, _ = data.load_range_dataset_w_benign(args.data, args.test_start, args.test_end)
    logging.info('X_test shape: %s', X_test.shape)

    ENC_MODEL_PATH = args.pretrined_model
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    y_train_binary = np.array([1 if item != 0 else 0 for item in y_train])
    y_test_binary = np.array([1 if item != 0 else 0 for item in y_test])
    BIN_NUM_CLASSES = 2

    if args.encoder == 'simple-enc-mlp':
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES, args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = SimpleEncClassifier(enc_dims, mlp_dims)
        encoder_name = 'simple_enc_classifier'
    else:
        raise Exception(f'The encoder {args.encoder} is not supported yet.')

    return encoder, X_train, y_train_binary, all_train_family, X_test, y_test_binary, ENC_MODEL_PATH

def get_train_test_reprentation(args):

    logging.info("Pseudo Labeling for Acutal Datasets started")
    encoder, X_train, y_train_binary, all_train_family, \
        X_test, y_test_binary, ENC_MODEL_PATH = load_data_and_model(args)

    logging.info("Model loading started...")
    state_dict = torch.load(ENC_MODEL_PATH, weights_only=False)
    encoder.load_state_dict(state_dict['model'])
    logging.info("Model loading ended!")

    if args.cls_feat == 'encoded':
        X_train_tensor = torch.from_numpy(X_train).float()
        X_test_tensor = torch.from_numpy(X_test).float()
        if torch.cuda.is_available():
            X_train_tensor = X_train_tensor.cuda()
            X_feat_tensor = encoder.cuda().encode(X_train_tensor)
            X_train_feat = X_feat_tensor.cpu().detach().numpy()

            X_test_tensor = X_test_tensor.cuda()
            X_test_feat = encoder.encode(X_test_tensor).cpu().detach().numpy()
        else:
            X_train_feat = encoder.encode(X_train_tensor).numpy()
    else:
        X_train_feat = X_train

    logging.info("X_train_feat Shape: %s", X_train_feat.shape)
    logging.info("X_test_feat Shape: %s", X_test_feat.shape)

    logging.info("X_train_feat: %s", X_train_feat[:1, :10])
    logging.info("X_test_feat: %s", X_test_feat[:1, :10])

    # y_test_labels = y_test_binary.reshape(-1, 1)

    return X_train_feat, y_train_binary, all_train_family, X_test_feat, y_test_binary

def single_k_sm_fn(args, X_train_feat, y_train_binary, X_test_feat, y_test_binary, sm_fn, k, tot_columns):

    topk_sim_weight, topk_sim_indices, max_sim, avg_sim = Similarity().topk_similar(k, sm_fn, X_test_feat, X_train_feat)

    logging.info("Top K similar Indices: %s", topk_sim_indices)

    pseudo_labels, topk_info = Similarity().get_majority_label(topk_sim_weight, topk_sim_indices, \
                                                               feature_labels=y_train_binary)

    logging.info("Pesudo Labels shape: %s", pseudo_labels.shape)
    logging.info("Topk_info shape: %s", topk_info.shape)

    logging.info("Pseudo Labeling for Acutal Datasets ended")

    y_test_binary = y_test_binary.reshape(-1, 1)
    logging.info("Actual Labels shape: %s", y_test_binary.shape)

    max_sim = max_sim.numpy().reshape(-1, 1)
    avg_sim = avg_sim.numpy().reshape(-1, 1)

    concatenated_array = np.concatenate((pseudo_labels, y_test_binary, max_sim, avg_sim, topk_info), axis=1)
    logging.info("Concatenated Array Shape: %s", concatenated_array.shape)

    df = pd.DataFrame(concatenated_array, columns=tot_columns)
    return df

def get_high_similar_samples(X_train_feat, y_train_binary, all_train_family,\
                              X_test_feat, sm_fn='euclidean', K=5, num_samples=50):
    
    topk_sim_weight, topk_sim_indices, max_sim, _ = Similarity().topk_similar(K, sm_fn, X_test_feat, X_train_feat)

    logging.info("Top K similar Indices: %s", topk_sim_indices)

    pseudo_labels, topk_info = Similarity().get_majority_label(topk_sim_weight, topk_sim_indices, \
                                                               feature_labels=y_train_binary, \
                                                                feature_family=all_train_family)

    logging.info("Pesudo Labels shape: %s", pseudo_labels.shape)
    logging.info("Topk_info shape: %s", topk_info.shape)
    logging.info("Pseudo Labeling for Acutal Datasets ended")
    # logging.info("Actual Labels shape: %s", y_test_binary.shape)

    max_sim = max_sim.numpy().reshape(-1, 1)
    # avg_sim = avg_sim.numpy().reshape(-1, 1)

    concatenated_array = np.concatenate((pseudo_labels, max_sim, topk_info), axis=1)
    logging.info("Concatenated Array Shape: %s", concatenated_array.shape)

    df = pd.DataFrame(concatenated_array)
    # create new index column to maintain the original indices of the samples
    df.insert(0, '', df.index)

    # Sort the DataFrame based on the 5th column (index 4) in descending order
    # Sorting desending order based on the Highest Similarity Score
    # Column 2 contains the Max Similarity values
    df_sorted = df.sort_values(by=df.columns[2], ascending=False)
    # Add a new column for original indices before the first column

    # Take the top num_samples rows while keeping the original indices
    df_top_samples = df_sorted.head(num_samples)

    logging.info("Top Samples Shape: %s", df_top_samples.shape)

    # Print the last K columns of the first row
    logging.info("First row data: %s", df_top_samples.iloc[0, (K+1):].tolist())

    # samples = [df_top_samples.iloc[i, (K+1):].tolist() for i in range(df_top_samples.shape[0])]
 
    # Extract the family based on the maximum similarity score among the tuples
    # families = []
    # for row in samples:
    #     max_sim_score = -1
    #     family = None
    #     logging.info("Row: %s", row)
    #     for tup in row:
    #         _, sim_score, _, fam = eval(tup)
    #         # logging.info("Sim Score: %s, Family: %s", sim_score, fam)
    #         if sim_score > max_sim_score:
    #             max_sim_score = sim_score
    #             family = fam
    #     families.append(family)

    #return indices (df_top_samples[:, 0]) of the top samples 
    # and their corresponding pseudo labels (df_top_samples[:, 1])
    indices = df_top_samples.iloc[:, 0].values
    pseudo_labels = df_top_samples.iloc[:, 1].astype(int).values
    # families = np.array(families)
    return indices, pseudo_labels

def pesudo_labeling(args, is_single_k_sm_fn=True, K=5, sm_fn='euclidean', ckpt_index=None, \
                    X_train_feat=None, y_train_binary=None, X_test_feat=None, y_test_binary=None):

    column_names = ['Pseudo Label', 'Majority', 'Actual Label', 'Max_value', 'Avg_value']

    if is_single_k_sm_fn:
        temp_cols = []
        for c in range(1, K+1):
            temp_cols.append(f'Train Point {c}')
        tot_columns = column_names + temp_cols
        logging.info("Total logging.infoColumns: %s", tot_columns)

        return single_k_sm_fn(args, X_train_feat, y_train_binary, \
                        X_test_feat, y_test_binary, sm_fn, K, tot_columns)
    else:   
        for k in range(1, args.k_closest, 2):
            temp_cols = []
            for c in range(1, k+1):
                temp_cols.append(f'Train Point {c}')
            tot_columns = column_names + temp_cols
            logging.info("Total logging.infoColumns: %s", tot_columns)
            
            for sm_fn in ['cosine', 'euclidean']:
                
                df = single_k_sm_fn(args, X_train_feat, y_train_binary, \
                                    X_test_feat, y_test_binary, sm_fn, k, tot_columns)
                # args.pseudo_output_path = f'{output_parent_path}{args.learning_rate}_{args.optimizer}_{ckpt_index}_{sm_fn}_{k}_{args.test_start}.csv'
                # df.to_csv(args.pseudo_output_path, index=True, header=True)

        logging.info("Data saved to output.csv")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to the data')
    # parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--bsize', type=int, required=True, help='Batch size')
    parser.add_argument('--mdate', type=str, required=True, help='Model date')
    parser.add_argument('--train_start', type=str, required=True, help='Training start date')
    parser.add_argument('--train_end', type=str, required=True, help='Training end date')
    parser.add_argument('--test_start', type=str, required=True, help='Test start date')
    parser.add_argument('--test_end', type=str, required=True, help='Test end date')
    parser.add_argument('--encoder', type=str, required=True, help='Encoder type')
    parser.add_argument('--enc_hidden', type=str, required=True, help='Encoder hidden layers')
    parser.add_argument('--mlp_hidden', type=str, required=True, help='Encoder hidden layers')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer type')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--result_epochs', type=int, required=True, help='Number of epochs after to show results')
    parser.add_argument('--k_closest', type=int, required=True, help='Number of Closest Points')
    parser.add_argument('--loss_func', type=str, required=True, help='Loss function')
    parser.add_argument('--cls_feat', type=str, required=True, help='Loss function')
    parser.add_argument('--log_path', type=str, required=True, help='Path to the log file')
    parser.add_argument('--single_checkpoint', required=True, help='Use single checkpoints')
    args = parser.parse_args()

    log_file_path = args.log_path
    logging.basicConfig(filename=log_file_path, filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    project_path = '/home/ihossain/ISMAIL/SSL-malware'
    model_parent_path = f'{project_path}/models/gen_apigraph_drebin/simple_enc_classifier_lr'
    output_parent_path = f'{project_path}/pseudo_labels/csv/pseudo_labels_output'
    
    if args.single_checkpoint:
        logging.info("----------------[Single Check Point]---------------------", )
        args.pretrined_model = f'{model_parent_path}{args.learning_rate}_{args.optimizer}_Final.pth'
        X_train_feat, y_train_binary, all_train_family, X_test_feat, _ = get_train_test_reprentation(args)
        
        indices, pseudo_labels, families = get_high_similar_samples(X_train_feat, y_train_binary, all_train_family,\
                              X_test_feat, sm_fn='euclidean', K=5, num_samples=50)
        
        logging.info("-------[Test indices: %s], [Pseudo Label: %s], [Family: %s]",\
                      indices, pseudo_labels, families)
        # pesudo_labeling(args=args, is_single_k_sm_fn=True, ckpt_index=args.epochs, X_train_feat=X_train_feat, \
        #                     y_train_binary=y_train_binary, X_test_feat=X_test_feat, y_test_binary=y_test_binary)
    else:
        for i in range(0, args.epochs+1, args.result_epochs):
            logging.info("----------------[Epoch: %s]---------------------", i)
            args.pretrined_model = f'{model_parent_path}{args.learning_rate}_{args.optimizer}_{i}.pth'
            X_train_feat, y_train_binary, X_test_feat, y_test_binary = get_train_test_reprentation(args)

            pesudo_labeling(args=args, is_single_k_sm_fn=False, ckpt_index=i, X_train_feat=X_train_feat, \
                            y_train_binary=y_train_binary, X_test_feat=X_test_feat, y_test_binary=y_test_binary)