import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

def data_flip_and_save(X_train, y_train, X_test, y_test, path, month):

    X_unlabeled_w, X_unlabeled_s, min_hammming, max_hamming = utils.bit_flipping(X_train, y_train, X_test, 0.05, 0.2, 64)
        
    labeled_ds = TensorDataset(X_test, y_test)
    unlabeled_ds_w = TensorDataset(X_unlabeled_w)
    unlabeled_ds_s = TensorDataset(X_unlabeled_s)

    print("Max Hamming Dist: ", max_hamming)
    print("Min Hamming Dist: ", min_hammming)

    # Example for labeled_ds
    torch.save({
        'data': labeled_ds.tensors[0],     # or use labeled_ds.data if it's custom
        'targets': labeled_ds.tensors[1]   # adjust as needed for your dataset
    }, f"{path}{month}_test_labeled_ds.pt")

    torch.save({
        'data': unlabeled_ds_w.tensors[0]
    }, f"{path}{month}_test_unlabeled_ds_weak_5p.pt")

    torch.save({
        'data': unlabeled_ds_s.tensors[0]
    }, f"{path}{month}_test_unlabeled_ds_strong_20p.pt")

# === Main Execution ===
if __name__ == "__main__":
    # print(f"Running {strategy}...")
    # Load data
    path = "/home/mhaque3/myDir/data/gen_apigraph_drebin"
    file_path = f"{path}2012-01to2012-12_selected.npz"
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    y = np.array([0 if label == 0 else 1 for label in y])

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_labeled, y_labeled, X_unlabeled, _ = utils.split_labeled_unlabeled(X, y, labeled_ratio=0.4)

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    # X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()

    # input_dim = X_2012_labeled.shape[1]
    # num_classes = len(torch.unique(y_2012_labeled))

    # data_flip_and_save(X_2012_labeled, y_2012_labeled, X_2012_unlabeled, path, None)

    test_sets_by_year = {}
    for year in tqdm(range(2013, 2019), desc="Processing years"):
        for month in tqdm(range(1, 13), desc="Processing months"):
            try:
                data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                X_raw = data["X_train"]
                y_true = (data["y_train"] > 0).astype(int)
                # X_scaled = scaler.transform(X_raw)
                # X_l, y_l, X_u, _ = utils.split_labeled_unlabeled(X_raw, y_true, labeled_ratio=0.4)

                X_tensor = torch.tensor(X_raw, dtype=torch.float32).cuda()
                y_tensor = torch.tensor(y_true, dtype=torch.long).cuda()
                # X_u_tensor = torch.tensor(X_u, dtype=torch.float32).cuda()

                new_path = f"{path}{year}/"
                os.makedirs(new_path, exist_ok=True)
                data_flip_and_save(X_2012_labeled, y_2012_labeled, X_tensor, y_tensor, new_path, month)

                # X_tensor = torch.tensor(X_raw, dtype=torch.float32).cuda()
                # y_tensor = torch.tensor(y_true, dtype=torch.long).cuda()
                # test_sets_by_year[f"{year}_{month}"] = (X_tensor, y_tensor)
            except FileNotFoundError:
                continue


# nohup python data.py > /home/ihossain/ISMAIL/SSL-malware/baseline_experiments/flexmatch/output.log 2>&1 &
