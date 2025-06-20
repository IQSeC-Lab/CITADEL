import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

import plot_gen
import eval
import utils
from utils import Classifier
import logging

strategy = "flexmatch_al_bit_flip_20p_w_50p_s"

def train_flexmatch_drift_eval(model, optimizer, X_labeled, y_labeled, X_unlabeled_w, X_unlabeled_s, \
                               test_sets_by_year, num_classes=2, threshold=0.95, lambda_u=1.0, \
                                  epochs=250, retrain_epochs=250, batch_size=64, patience=100):
    
    # X_unlabeled_w, X_unlabeled_s = utils.bit_flipping(X_labeled, X_unlabeled, 0.2, 0.5)
    
    labeled_ds = TensorDataset(X_labeled, y_labeled)
    unlabeled_ds_w = TensorDataset(X_unlabeled_w)
    unlabeled_ds_s = TensorDataset(X_unlabeled_s)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    unlabeled_loader_w = DataLoader(unlabeled_ds_w, batch_size=batch_size, shuffle=True)
    unlabeled_loader_s = DataLoader(unlabeled_ds_s, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # Number of unlabeled data
    N = len(unlabeled_ds_w)
    learning_status = [-1] * N

    # mapping function of beta
    mapping = utils.mapping_func("convex")

    best_loss = float('inf')
    best_state_dict = None
    early_stop_counter = 0

    model.train()
    for epoch in tqdm(range(epochs), desc="Training FlexMatch"):
        
        cls_thresholds = torch.zeros(num_classes).cuda()

        total_loss = 0
        labeled_iter = iter(labeled_loader)
        unlabeled_iter_w = iter(unlabeled_loader_w)
        unlabeled_iter_s = iter(unlabeled_loader_s)

        for _ in range(len(labeled_loader)):
            try:
                x_l, y_l = next(labeled_iter)
                (x_u_w,) = next(unlabeled_iter_w)
                (x_u_s,) = next(unlabeled_iter_s)
            except StopIteration:
                break

            x_l, y_l = x_l.cuda(), y_l.cuda()
            x_u_w, x_u_s = x_u_w.cuda(), x_u_s.cuda()
            u_i = torch.arange(x_u_w.size(0)).to(x_u_w.device)

            # Apply random bit flip: weak (1 bit), strong (3 bits)
            # x_u_w = utils.random_bit_flip(x_u, n_bits=1)
            # x_u_s = utils.random_bit_flip(x_u, n_bits=4)

            # x_u_w, x_u_s = utils.bit_flipping(X_labeled, x_u, 0.2, 0.5)
            # x_u_w, x_u_s = x_u_w.cuda(), x_u_s.cuda()

            xw_pred = model(x_l)
            # loss_x = criterion(logits_x, y_l)

            # supervised loss
            ls = criterion(xw_pred, y_l.cuda()).mean()
            total_loss = ls

            # compute a learning status
            counter = Counter(learning_status)

            # normalize the status
            num_unused = counter[-1]
            if num_unused != N:
                max_counter = max([counter[c] for c in range(num_classes)])
                if max_counter < num_unused:
                    # normalize with eq.11
                    sum_counter = sum([counter[c] for c in range(num_classes)])
                    denominator = max(max_counter, N - sum_counter)
                else:
                    denominator = max_counter
                # threshold per class
                for c in range(num_classes):
                    beta = counter[c] / denominator
                    cls_thresholds[c] = mapping(beta) * threshold

            # with torch.no_grad():
            #     pseudo_logits = F.softmax(model(x_u_w), dim=1)
            #     pseudo_labels = torch.argmax(pseudo_logits, dim=1)
            #     max_probs, _ = torch.max(pseudo_logits, dim=1)
            #     mask = max_probs.ge(threshold).float()
                        # update the pseudo label
            with torch.no_grad():
                uw_prob = F.softmax(model(x_u_w), dim=1)
                max_prob, hard_label = torch.max(uw_prob, dim=1)
                over_threshold = max_prob > threshold
                if over_threshold.any():
                    u_i = u_i.cuda() #to(device)
                    sample_index = u_i[over_threshold].tolist()
                    pseudo_labels = hard_label[over_threshold].tolist()
                    for i, l in zip(sample_index, pseudo_labels):
                        learning_status[i] = l

            us_pred = model(x_u_s)
            # loss_u = (F.cross_entropy(logits_u, pseudo_labels, reduction='none') * mask).mean()
            # loss = loss_x + lambda_u * loss_u
             # unsupervised loss

            batch_threshold = torch.index_select(cls_thresholds, 0, hard_label)
            indicator = max_prob > batch_threshold

            lu = (criterion(us_pred, hard_label) * indicator).mean()
            total_loss += lu * lambda_u #unsupervised_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # total_loss += loss.item()

        # EARLY STOPPING LOGIC
        if total_loss < best_loss:
            best_loss = total_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1} due to no improvement in loss.")
                break

        logging.info(f"##Epoch {epoch+1}/{epochs}: loss={total_loss:.4f}, best_loss={best_loss:.4f}")
        scheduler.step()
    # === Evaluate on each year's test set ===
    # eval.model_evaluate(model, test_sets_by_year, strategy)

    # Restore best model after initial training
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    # Active learning loop: month by month
    metrics_list = []

    sorted_test_keys = test_sets_by_year.keys() #sorted(test_sets_by_year.keys())
    # logging.info(f"Keys: {sorted_test_keys}")

    for test_idx, year in tqdm(enumerate(sorted_test_keys), total=len(sorted_test_keys), desc="Active Learning Loop"):
        X_test, y_test, X_w, X_s = test_sets_by_year[year]

        # === Evaluate on this test set BEFORE adding to unlabeled set ===
        metrics = eval.evaluate_model_active(model, X_test, y_test, num_classes=num_classes)
        metrics['year'] = year
        metrics_list.append(metrics)
        print(f"Year {year}: " + ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]))

        # === Add this test set to the UNLABELED set for next round (active learning) ===
        X_unlabeled_w = torch.cat([X_unlabeled_w, X_w.to(X_unlabeled_w.device)], dim=0)
        X_unlabeled_s = torch.cat([X_unlabeled_s, X_s.to(X_unlabeled_s.device)], dim=0)
        
        unlabeled_ds_w = TensorDataset(X_unlabeled_w)
        unlabeled_ds_s = TensorDataset(X_unlabeled_s)

        unlabeled_loader_w = DataLoader(unlabeled_ds_w, batch_size=batch_size, shuffle=True)
        unlabeled_loader_s = DataLoader(unlabeled_ds_s, batch_size=batch_size, shuffle=True)
        
        # Create a new optimizer and scheduler for each retraining phase
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        best_loss = float('inf')
        best_state_dict = None

        # Number of unlabeled data
        N = len(unlabeled_loader_w)
        learning_status = [-1] * N
        early_stop_counter = 0

        model.train()
        for epoch in tqdm(range(retrain_epochs), desc="Retraining..."):
            total_loss = 0
            labeled_iter = iter(labeled_loader)

            unlabeled_iter_w = iter(unlabeled_loader_w)
            unlabeled_iter_s = iter(unlabeled_loader_s)

            for _ in range(len(labeled_loader)):
                try:
                    x_l, y_l = next(labeled_iter)
                    (x_u_w,) = next(unlabeled_iter_w)
                    (x_u_s,) = next(unlabeled_iter_s)
                except StopIteration:
                    break

                x_l, y_l = x_l.cuda(), y_l.cuda()
                x_u_w, x_u_s = x_u_w.cuda(), x_u_s.cuda()
                u_i = torch.arange(x_u_w.size(0)).to(x_u_w.device)

                # Apply random bit flip: weak (1 bit), strong (3 bits)
                # x_u_w = utils.random_bit_flip(x_u, n_bits=1)
                # x_u_s = utils.random_bit_flip(x_u, n_bits=4)

                xw_pred = model(x_l)
                # loss_x = criterion(logits_x, y_l)

                # supervised loss
                ls = criterion(xw_pred, y_l.cuda()).mean()
                total_loss = ls

                # compute a learning status
                counter = Counter(learning_status)

                # normalize the status
                num_unused = counter[-1]
                if num_unused != N:
                    max_counter = max([counter[c] for c in range(num_classes)])
                    if max_counter < num_unused:
                        # normalize with eq.11
                        sum_counter = sum([counter[c] for c in range(num_classes)])
                        denominator = max(max_counter, N - sum_counter)
                    else:
                        denominator = max_counter
                    # threshold per class
                    for c in range(num_classes):
                        beta = counter[c] / denominator
                        cls_thresholds[c] = mapping(beta) * threshold

                with torch.no_grad():
                    uw_prob = F.softmax(model(x_u_w), dim=1)
                    max_prob, hard_label = torch.max(uw_prob, dim=1)
                    over_threshold = max_prob > threshold
                    if over_threshold.any():
                        u_i = u_i.cuda() #to(device)
                        sample_index = u_i[over_threshold].tolist()
                        pseudo_labels = hard_label[over_threshold].tolist()
                        for i, l in zip(sample_index, pseudo_labels):
                            learning_status[i] = l
                
                us_pred = model(x_u_s)
                batch_threshold = torch.index_select(cls_thresholds, 0, hard_label)
                indicator = max_prob > batch_threshold

                lu = (criterion(us_pred, hard_label) * indicator).mean()
                total_loss += lu * lambda_u #unsupervised_weight

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # EARLY STOPPING LOGIC (retraining)
            if total_loss < best_loss:
                best_loss = total_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logging.info(f"Early stopping retraining at epoch {epoch+1} due to no improvement.")
                    break

            logging.info(f"##Re-train epoch {epoch+1}/{retrain_epochs}: loss={total_loss:.4f}, best_loss={best_loss:.4f}")
            scheduler.step()
        
        # Restore best model after retraining
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Save results to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/flexmatch/results/{strategy}_active.csv", index=False)

    print(f"Mean F1 Scores: {metrics_df['f1'].mean():.4f}")
    print(f"Mean False Negative Rates: {metrics_df['fnr'].mean():.4f}")
    print(f"Mean False Positive Rates: {metrics_df['fpr'].mean():.4f}")
    plot_gen.plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], \
                         save_path=f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/flexmatch/results/f1_fnr_{strategy}_active.png")

# === Main Execution ===
if __name__ == "__main__":
    print(f"Running {strategy}...")
    # Load data
    path = "/home/ihossain/ISMAIL/Datasets/data/gen_apigraph_drebin/"
    # file_path = f"{path}2012-01to2012-12_selected.npz"
    # data = np.load(file_path, allow_pickle=True)
    # X, y = data['X_train'], data['y_train']
    # y = np.array([0 if label == 0 else 1 for label in y])

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_labeled, y_labeled, X_unlabeled, _ = utils.split_labeled_unlabeled(X, y, labeled_ratio=0.4)

    # X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    # y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    # X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()

    # Recreate datasets from saved files
    labeled_dict = torch.load(f"{path}train_labeled_ds.pt")
    unlabeled_w_dict = torch.load(f"{path}train_unlabeled_ds_weak_5p.pt")
    unlabeled_s_dict = torch.load(f"{path}train_unlabeled_ds_strong_20p.pt")

    # labeled_ds = TensorDataset(labeled_dict['data'].float(), labeled_dict['targets'].long())
    # unlabeled_ds_w = TensorDataset(unlabeled_w_dict['data'].float())
    # unlabeled_ds_s = TensorDataset(unlabeled_s_dict['data'].float())

    X_2012_labeled = labeled_dict['data'].float()
    y_2012_labeled = labeled_dict['targets'].long()
    X_2012_unlabeled_w = unlabeled_w_dict['data'].float()
    X_2012_unlabeled_s = unlabeled_s_dict['data'].float()


    # Recreate loaders
    # batch_size = 64  # or same as before
    # labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    # unlabeled_loader_w = DataLoader(unlabeled_ds_w, batch_size=batch_size, shuffle=True)
    # unlabeled_loader_s = DataLoader(unlabeled_ds_s, batch_size=batch_size, shuffle=True)

    input_dim = X_2012_labeled.shape[1]  # Access the shape of the first tensor
    num_classes = 2 #len(torch.unique(y_2012_labeled))

    test_sets_by_year = {}
    for year in tqdm(range(2013, 2019), desc="Processing years"):
        for month in range(1, 13):
            try:
                # data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                
                u_dict = torch.load(f"{path}{year}/{month}_test_labeled_ds.pt")
                u_w_dict = torch.load(f"{path}{year}/{month}_test_unlabeled_ds_weak_5p.pt")
                u_s_dict = torch.load(f"{path}{year}/{month}_test_unlabeled_ds_strong_20p.pt")

                # u_ds = TensorDataset(u_dict['data'], u_dict['targets'])
                # u_ds_w = TensorDataset(u_w_dict['data'])
                # u_ds_s = TensorDataset(u_s_dict['data'])

                test_sets_by_year[f"{year}_{month}"] = (u_dict['data'].float(), u_dict['targets'].long(), u_w_dict['data'].float(), u_s_dict['data'].float())
            except FileNotFoundError:
                continue

    model = Classifier(input_dim=input_dim, num_classes=num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    sorted_test_keys = sorted(test_sets_by_year.keys())
    # print(f"Keys: {sorted_test_keys}")

    train_flexmatch_drift_eval(
        model,
        optimizer,
        X_2012_labeled,
        y_2012_labeled,
        X_2012_unlabeled_w,
        X_2012_unlabeled_s,
        test_sets_by_year,
        num_classes=num_classes
    )
    # print(f"Mean F1 Scores: {sum(f1_scores.values())/len(f1_scores)}")
    # print(f"Mean False Negative Rates: {sum(fnrs.values())/len(fnrs)}")
    # plot_f1_fnr(f1_scores, fnrs)


# nohup python active_learning.py > /home/ihossain/ISMAIL/SSL-malware/baseline_experiments/flexmatch/output.log 2>&1 &