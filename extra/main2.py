from distutils import core
import os

import datetime as dt
import logging
import numpy as np
import time
import torch
import xgboost as xgb

from collections import Counter, defaultdict
from dateutil.relativedelta import relativedelta
from pprint import pformat
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
import torch.nn.functional as F

# local imports
import data
from joblib import dump
import utils
from models import SimpleEncClassifier, MLPClassifier
from train import train_encoder, train_classifier
from similarity_score import get_high_similar_samples

def main():
    

    """
    Step (1): Prepare the training dataset. Load the feature vectors and labels.
    """
    data_dir = "/home/mhaque3/myDir/data/gen_apigraph_drebin"
    DATA = data_dir.split('/')[-1]
    TRAIN_START="2012-01"
    TRAIN_END="2012-12"
    TEST_START="2013-01"
    TEST_END="2018-12"
    logging.info(f'Dataset directory name {DATA}')

    logging.info(f'For API_GRAPH dataset start with the month 2012-01to2012-12_selected.')
    
    X_train, y_train, y_train_family = data.load_range_dataset_w_benign(data_dir, TRAIN_START, TRAIN_END)
    # all_train_family has 'benign'
    ben_len = X_train.shape[0] - y_train_family.shape[0]
    y_ben_family = np.full(ben_len, 'benign')
    all_train_family = np.concatenate((y_train_family, y_ben_family), axis=0)
    # print(f"Number of y_train_family = {y_train_family}")
        
    train_families = set(all_train_family)
    # print(f"Number of y_train_family = {y_train_family}")
    # print(f"y_train = {y_train}")
    # print(f"unique y_train = {np.unique(y_train)}")
    # print(f'All train family = {all_train_family}')

    labeled_indices, unlabeled_indices = utils.get_labeled_unlabeled_indices(X_train, y_train, percentage_labeled=50)
    # print(f"Number of labeled_indices = {len(labeled_indices)}")
    # print(f"Number of unlabeled_indices = {len(unlabeled_indices)}")
    # print(f"labeled_indices = {labeled_indices}")
    # print(f"unlabeled_indices = {unlabeled_indices}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).cuda()
    y_train_binary_cat_tensor = torch.tensor([1 if item != 0 else 0 for item in y_train], dtype=torch.long).cuda()
    X_unlabeled_tensor = X_train_tensor[unlabeled_indices]
    X_unlabeled_tensor = torch.tensor(X_unlabeled_tensor, dtype=torch.float32).cuda()

    indicies, y_pseudo = get_high_similar_samples(X_train_tensor, y_train_binary_cat_tensor, y_train_tensor, X_unlabeled_tensor, sm_fn='cosine', num_samples=100)
    print(f"Calculating # of correct pseudo_label for unlabeled training data:")
    print(f"Number of indicies = {len(indicies)}")
    print(f"Number of correct label = {torch.sum(y_pseudo == y_train_tensor[indicies].cpu())}")

    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    start_date = datetime.strptime("2013-01", '%Y-%m')
    end_date = datetime.strptime("2018-12", '%Y-%m')
    # print(f"start_date = {start_date}")
    # print(f"end_date = {end_date}")

    MODEL_PATH = "/home/mhaque3/myDir/active-learning/models/2012-01to2012-12/simple_enc_classifier_1159-512-384-256-128_hi-dist-xent_xent100.0_mselambda1_caelambda0.1_sgd_step_lr0.003_decay0.95_half_b1024_e250_mdate20230501.pth"
    # load model
    enc_dims = utils.get_model_dims('Encoder', X_train.shape[1],
                            "512-384-256-128", len(np.unique(y_train)))
    mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], "100-100", 2)
    encoder = SimpleEncClassifier(enc_dims, mlp_dims)
    state_dict = torch.load(MODEL_PATH)
    encoder.load_state_dict(state_dict['model'])
    
    months = []
    current_date = start_date

    while current_date <= end_date:
        c_date = current_date.strftime("%Y-%m")
        X_test, y_test, y_test_family = data.load_range_dataset_w_benign(data_dir, c_date, c_date)
        print(f"X_test shape = {X_test.shape}")
        X_train_feat = encoder.cuda().encode(X_train_tensor)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
        X_test_feat = encoder.cuda().encode(X_test_tensor)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        y_test_binary_cat_tensor = torch.tensor([1 if item != 0 else 0 for item in y_test], dtype=torch.long).cuda()
        indicies, y_pseudo = get_high_similar_samples(X_train_tensor.cpu(), y_train_binary_cat_tensor.cpu(), y_train_tensor.cpu(), X_test_tensor.cpu(), sm_fn='cosine', num_samples=X_test_tensor.shape[0])
        print(f"Month = {c_date}")
        print(f"Number of pseudo labeled samples = {len(indicies)}")
        
        # Move tensors to CPU before converting to NumPy
        y_pseudo_cpu = y_pseudo
        # """
        y_test_tensor_cpu = y_test_tensor.cpu().numpy()
        y_test_binary_cat_tensor_cpu = y_test_binary_cat_tensor.cpu().numpy()

        print(f"Number of correct label = {np.sum(y_pseudo_cpu == y_test_binary_cat_tensor_cpu[indicies])}")
        print(f"Number of malware in pseudo = {np.sum(y_pseudo_cpu != 0)}")
        print(f"Number of malware in test = {np.sum(y_test_binary_cat_tensor_cpu != 0)}")
        print(f"Number of benign in pseudo = {np.sum(y_pseudo_cpu == 0)}")
        print(f"Number of benign in test = {np.sum(y_test_binary_cat_tensor_cpu == 0)}")
        print(f"Number of correct malware samples = {np.sum((y_pseudo_cpu == y_test_binary_cat_tensor_cpu[indicies]) & (y_test_binary_cat_tensor_cpu[indicies] == 1))}")
        print(f"Number of correct benign samples = {np.sum((y_pseudo_cpu == y_test_binary_cat_tensor_cpu[indicies]) & (y_test_binary_cat_tensor_cpu[indicies] == 0))}")
        
        print(f"Accuracy = {np.sum(y_pseudo_cpu == y_test_binary_cat_tensor_cpu[indicies]) / X_test_tensor.shape[0]}")
        print()
        # """
        current_date += relativedelta(months=1)

   
    
    """

    # the index mapping for the first training set
    new_y_mapping = {}
    for _, label in enumerate(np.unique(y_train)):
        new_y_mapping[label] = label
    
    """
    # Step (2): Variable names and file names.
    """
    # some commonly used variables.
    if args.train_start != args.train_end:
        train_dataset_name = f'{args.train_start}to{args.train_end}'
    else:
        train_dataset_name = f'{args.train_start}'

    SAVED_MODEL_FOLDER = 'models/'
    # only based on malicious training samples
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    logging.info(f'Number of features: {NUM_FEATURES}; Number of classes: {NUM_CLASSES}')

    # convert y_train to y_train_binary
    y_train_binary = np.array([1 if item != 0 else 0 for item in y_train])
    BIN_NUM_CLASSES = 2
    """

if __name__ == "__main__":
    main()