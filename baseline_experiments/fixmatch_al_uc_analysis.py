import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import os
import math
import argparse
import random
import pandas as pd

from sklearn.manifold import TSNE

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# Global font settings
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.frameon": False
})

# define strategy as global variable
strategy = ""


# === Classifier Definition ===
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 384), nn.ReLU(),
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, num_classes)
            #nn.Linear(100, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(self.encoder(x))
    def encode(self, x):
        """Encode input features to a lower-dimensional representation."""
        return self.encoder(x)


class ClassifierWB(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 384), nn.BatchNorm1d(384), nn.ReLU(),
            nn.Linear(384, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def encode(self, x):
        """Encode input features to a lower-dimensional representation."""
        return self.encoder(x)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)



def split_labeled_unlabeled(X, y, labeled_ratio=0.1, stratify=True, random_state=42):
    n_samples = len(X)
    n_labeled = int(n_samples * labeled_ratio)
    if stratify:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=n_labeled, stratify=y, random_state=random_state
        )
    else:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=n_labeled, random_state=random_state
        )
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled

def random_bit_flip(x, n_bits=1):
    x_aug = x.clone()
    batch_size, num_features = x.shape
    for i in range(batch_size):
        flip_indices = torch.randperm(num_features)[:n_bits]
        x_aug[i, flip_indices] = 1 - x_aug[i, flip_indices]
    return x_aug

def random_bit_flip_bernoulli(x, p=None, n_bits=None):
    """
    Randomly flip each bit in the input tensor with probability p using Bernoulli distribution.
    If n_bits is given, p is set so that on average n_bits are flipped per sample.
    Args:
        x: Tensor of shape (batch_size, num_features)
        p: Probability of flipping each bit (float, between 0 and 1)
        n_bits: If given, overrides p so that p = n_bits / num_features
    Returns:
        Augmented tensor with bits flipped
    """
    x_aug = x.clone()
    batch_size, num_features = x.shape
    device = x.device
    
    if p is not None:
        p = float(p)
    else:
        p = 0.01  
    flip_mask = torch.bernoulli(torch.full_like(x_aug, p, device=device))
    x_aug = torch.abs(x_aug - flip_mask)
    return x_aug

def random_feature_mask(x, n_mask=1):
    x_aug = x.clone()
    batch_size, num_features = x.shape
    for i in range(batch_size):
        mask_indices = torch.randperm(num_features)[:n_mask]
        x_aug[i, mask_indices] = 0
    return x_aug

def random_bit_flip_and_mask(x, n_bits=1, n_mask=1):
    x_aug = random_bit_flip(x, n_bits=n_bits)
    x_aug = random_feature_mask(x_aug, n_mask=n_mask)
    return x_aug



def evaluate_model(model, X_test, y_test, year_month, num_classes=2):
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.cuda(), y_test.cuda()
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
        preds = logits.argmax(dim=1)
        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()
        if probs.shape[1] == 2:
            y_score = probs[:, 1].cpu().numpy()
        else:
            y_score = probs.cpu().numpy()  # for multi-class

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fnr = fpr = float('nan')

        # ROC-AUC and PR-AUC (binary or multiclass)
        try:
            if probs.shape[1] == 2:
                roc_auc = roc_auc_score(y_true, y_score)
                pr_auc = average_precision_score(y_true, y_score)
            else:
                roc_auc = roc_auc_score(y_true, probs.cpu().numpy(), multi_class='ovr')
                pr_auc = average_precision_score(y_true, probs.cpu().numpy(), average='weighted')
        except Exception:
            roc_auc = pr_auc = float('nan')

        metrics = {
            'year': year_month,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'fnr': fnr,
            'fpr': fpr,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        return metrics


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])



# ---------- Confidence Filtering ----------
def get_low_confidence_indices(model, X_test, threshold=0.7, batch_size=512):
    """
    Returns indices of test samples with confidence below a given threshold.
    Confidence is computed as max softmax probability.
    """
    model.eval()
    confidences = []

    with torch.no_grad():
        for i in range(0, X_test.size(0), batch_size):
            batch = X_test[i:i+batch_size]
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            max_conf = probs.max(dim=1).values
            confidences.append(max_conf)

    confidences = torch.cat(confidences)
    low_conf_mask = confidences < threshold
    low_conf_indices = torch.nonzero(low_conf_mask, as_tuple=True)[0]
    # also return the high confidence indices
    high_conf_indices = torch.nonzero(~low_conf_mask, as_tuple=True)[0]

    return low_conf_indices, high_conf_indices, confidences


# ---------- Lp-Norm Uncertainty Sampling ----------
def get_uncertain_samples(model, X_label, X_test, p=2, top_k=200, batch_size=64):
    """
    Computes Lp-distance-based uncertainty using encoded representations, with batching for memory efficiency.
    """
    device = X_label.device
    X_test = X_test.to(device)

    # Encode feature representations
    X_labeled_enc = model.encode(X_label)
    X_test_enc = model.encode(X_test)

    min_distances = []

    with torch.no_grad():
        N = X_test_enc.size(0)
        for i in range(0, N, batch_size):
            batch = X_test_enc[i:i+batch_size]
            distances = torch.cdist(batch, X_labeled_enc, p=p)
            batch_min_distances, _ = distances.min(dim=1)
            min_distances.append(batch_min_distances.cpu())
        min_distances = torch.cat(min_distances)

        top_k = min(top_k, min_distances.size(0))
        top_values, top_indices = torch.topk(min_distances, top_k, largest=True)

    return top_indices, top_values


def select_boundary_samples(model, X_unlabeled, y_unlabeled, top_k=200, batch_size=512):
    """
    Selects top-K most uncertain samples from X_unlabeled based on softmax margin.
    
    Args:
        model: Trained classification model with output logits.
        X_unlabeled (Tensor): Unlabeled data of shape (N, D).
        top_k (int): Number of most uncertain samples to select.
        batch_size (int): Batch size for efficient inference.
    
    Returns:
        indices (Tensor): Indices of top-K uncertain samples in X_unlabeled.
        margins (Tensor): Margin values for selected samples.
    """
    model.eval()
    all_margins = []

    with torch.no_grad():
        for i in range(0, X_unlabeled.size(0), batch_size):
            xb = X_unlabeled[i:i+batch_size].to(next(model.parameters()).device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            top2 = torch.topk(probs, 2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).cpu()
            all_margins.append(margin)

    all_margins = torch.cat(all_margins)
    k = min(top_k, len(all_margins))
    topk_indices = torch.argsort(all_margins)[:k]  # lowest margins = highest uncertainty
    # we need to see how many boundary samples are misclassified
    # Get predicted labels for the selected boundary samples
    selected_samples = X_unlabeled[topk_indices].to(next(model.parameters()).device)

    top1000_indices = torch.argsort(all_margins)
    selected_samples_1000 = X_unlabeled[top1000_indices].to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(selected_samples)
        logits_1000 = model(selected_samples_1000)
        preds = logits.argmax(dim=1).cpu()
        preds_1000 = logits_1000.argmax(dim=1).cpu()
    # True labels for these samples (if available)
    if y_unlabeled is not None:
        true_labels = y_unlabeled[topk_indices].cpu()
        true_labels_1000 = y_unlabeled[top1000_indices].cpu()
        num_misclassified = (preds != true_labels).sum().item()
        num_misclassified_1000 = (preds_1000 != true_labels_1000).sum().item()
        print(f"Boundary samples misclassified: {num_misclassified} out of {len(topk_indices)}")
        print(f"Boundary samples 1000 misclassified: {num_misclassified_1000} out of {len(top1000_indices)}")

    return topk_indices, all_margins[topk_indices]



# ---------- Active Learning Integration ----------
def active_learning_step(args, cached_tsne, model, year_month, X_labeled, y_labeled, X_test, y_test, top_k=200):
    """
    Active learning step:
    1. Selects top-k most uncertain samples from the entire test set (X_test).
    2. Moves those to labeled set.
    3. Keeps the rest in the unlabeled set.
    """
    device = X_labeled.device

    # # Call per X_test month
    analyze_and_plot_boundary_samples(
        model, X_test, y_test,
        cached_tsne=cached_tsne,
        strategy=year_month,  # or any tag
        boundary_k=400,
        save_dir=args.save_path
    )

    # Step 1: Apply uncertainty sampling over entire test set
    # print(f"Selecting {top_k} most uncertain samples from {X_test.size(0)} total test samples...")
    if args.unc_samp == 'lp-norm':
        # Use Lp-norm based uncertainty sampling
        uncertain_indices, _ = get_uncertain_samples(
            model, X_labeled, X_test, p=args.lp, top_k=top_k, batch_size=256
        )
    elif args.unc_samp == 'boundary':
        # Use boundary selection method
        uncertain_indices, _ = select_boundary_samples(
            model, X_test, y_test, top_k=top_k, batch_size=256
        )
    else:
        raise ValueError(f"Unknown uncertainty sampling method: {args.unc_samp}")
    print(f"Selected {len(uncertain_indices)} uncertain samples from {X_test.size(0)} total test samples.")

    # Step 2: Create a mask to separate selected and remaining samples
    remaining_mask = torch.ones(X_test.size(0), dtype=torch.bool, device=device)
    remaining_mask[uncertain_indices] = False

    # Step 3: Update labeled and unlabeled sets
    X_new_labeled = X_test[uncertain_indices]
    y_new_labeled = y_test[uncertain_indices]
    X_unlabeled = X_test[remaining_mask]
    y_unlabeled = y_test[remaining_mask]  # Optional, for evaluation

    X_labeled = torch.cat([X_labeled, X_new_labeled], dim=0)
    y_labeled = torch.cat([y_labeled, y_new_labeled], dim=0)

    print(f"[✓] Added {len(X_new_labeled)} new labeled samples.")
    print(f"[✓] Remaining unlabeled pool size: {len(X_unlabeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled



# Uncertainty sampling with decision boundary focus (median ± MAD)
def select_boundary_uncertain_samples(model, X_label, y_label, X_test, p=2, band_width=0.5, max_samples=200):
    """
    Select samples near the decision boundary (median ± band_width * MAD) for Lp-norm-based pseudo-labeling.
    """
    X_labeled_enc = model.encode(X_label)
    X_test_enc = model.encode(X_test)

    distances = torch.cdist(X_test_enc, X_labeled_enc, p=p)
    min_distances, nearest_indices = torch.min(distances, dim=1)

    # Compute median and MAD
    dists = min_distances.cpu().numpy()
    median = np.median(dists)
    mad = np.median(np.abs(dists - median))

    lower = median - band_width * mad
    upper = median + band_width * mad

    # Select samples near decision boundary
    mask = (dists >= lower) & (dists <= upper)
    boundary_indices = np.where(mask)[0]

    if len(boundary_indices) == 0:
        return [], torch.empty(0), torch.empty(0, dtype=torch.long)

    # Select top-k samples closest to the median within the boundary band
    sorted_indices = boundary_indices[np.argsort(np.abs(dists[boundary_indices] - median))]
    selected_indices = sorted_indices[:min(max_samples, len(sorted_indices))]

    selected_samples = X_test[selected_indices]
    neighbor_indices = nearest_indices[selected_indices]
    pseudo_labels = y_label[neighbor_indices]

    return selected_indices, selected_samples, pseudo_labels



def analyze_misclassifications(model, X_labeled, y_labeled, X_test, y_test, save_path="misclassified_analysis.csv"):
    """
    Analyze misclassified samples and save information to a CSV.

    Includes:
    - predicted label and true label
    - softmax confidence
    - misclassification flag
    - distance to nearest labeled sample
    - distance to each class centroid (malware, benign)
    - margin to centroid
    """

    model.eval()

    with torch.no_grad():
        # 1. Encode data
        X_labeled_enc = model.encode(X_labeled)
        X_test_enc = model.encode(X_test)

        # 2. Get model predictions
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        conf, y_pred = probs.max(dim=1)
        misclassified = (y_pred != y_test)

        # 3. Compute distances to class centroids
        mal_centroid = X_labeled_enc[y_labeled == 1].mean(dim=0)
        ben_centroid = X_labeled_enc[y_labeled == 0].mean(dim=0)
        d_to_mal = torch.norm(X_test_enc - mal_centroid, dim=1)
        d_to_ben = torch.norm(X_test_enc - ben_centroid, dim=1)
        centroid_margin = torch.abs(d_to_mal - d_to_ben)

        # 4. Compute distance to nearest labeled sample which is malware or benign
        X_lab_mal = X_labeled_enc[y_labeled == 1]
        X_lab_ben = X_labeled_enc[y_labeled == 0]

        d_nearest_mal = torch.cdist(X_test_enc, X_lab_mal).min(dim=1).values
        d_nearest_ben = torch.cdist(X_test_enc, X_lab_ben).min(dim=1).values

        # Compute pairwise distances from test to labeled samples
        pairwise_distances = torch.cdist(X_test_enc, X_labeled_enc, p=2)

        # Find nearest labeled distance and its index
        d_nearest_labeled, nearest_indices = pairwise_distances.min(dim=1)

        # Get the label of the nearest labeled sample for each test sample
        nearest_labels = y_labeled[nearest_indices]  # shape: (num_test,)



    # 5. Build DataFrame
    df = pd.DataFrame({
        'true_label': y_test.cpu().numpy(),
        'pred_label': y_pred.cpu().numpy(),
        'confidence': conf.cpu().numpy(),
        'misclassified': misclassified.cpu().numpy().astype(int),
        'd_to_malware_centroid': d_to_mal.cpu().numpy(),
        'd_to_benign_centroid': d_to_ben.cpu().numpy(),
        'centroid_margin': centroid_margin.cpu().numpy(),
        'd_to_nearest_labeled': d_nearest_labeled.cpu().numpy(),
        'nearest_labeled_class': nearest_labels.cpu().numpy(),  # 0 = benign, 1 = malware
    })


    df.to_csv(save_path, index=False)
    print(f"[✓] Misclassification analysis saved to: {save_path}")
    return df



def hamming_distance_chunked(a, b, chunk_size=1000):
    """
    Compute Hamming distances between binary tensors a (N, D) and b (M, D) in chunks.
    Returns (N, M) distance matrix on CPU.
    """
    a = a.int().cpu()
    b = b.int().cpu()
    N = a.size(0)
    M = b.size(0)

    dist_matrix = torch.empty((N, M), dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        a_chunk = a[start:end]  # (chunk_size, D)
        dists = (a_chunk.unsqueeze(1) != b.unsqueeze(0)).sum(dim=2).float()
        dist_matrix[start:end] = dists

    return dist_matrix

def analyze_misclassifications_hamming(model, X_labeled, y_labeled, X_test, y_test, save_path="misclassified_analysis.csv"):
    """
    Analyze misclassified samples and save information to a CSV.

    Includes:
    - predicted label and true label
    - softmax confidence
    - misclassification flag
    - Hamming distance to nearest labeled sample
    - Hamming distance to each class centroid (malware, benign)
    - margin to centroid
    """
    model.eval()

    with torch.no_grad():
        # 1. Get predictions
        logits = model(X_test)
        probs = F.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
        conf, y_pred = probs.max(dim=1) if probs.ndim > 1 else (probs, (probs >= 0.5).long())
        misclassified = (y_pred != y_test)

        # 2. Compute Hamming distances to class centroids
        mal_centroid = (X_labeled[y_labeled == 1].float().mean(dim=0) > 0.5).int()
        ben_centroid = (X_labeled[y_labeled == 0].float().mean(dim=0) > 0.5).int()

        d_to_mal = (X_test.cpu() != mal_centroid.cpu()).sum(dim=1).float()
        d_to_ben = (X_test.cpu() != ben_centroid.cpu()).sum(dim=1).float()
        centroid_margin = torch.abs(d_to_mal - d_to_ben)

        # 3. Nearest labeled sample (all classes)
        d_nearest_all = hamming_distance_chunked(X_test, X_labeled)
        d_nearest_labeled, nearest_indices = d_nearest_all.min(dim=1)
        nearest_labels = y_labeled[nearest_indices]

        # 4. Nearest to malware and benign samples
        X_lab_mal = X_labeled[y_labeled == 1]
        X_lab_ben = X_labeled[y_labeled == 0]

        d_nearest_mal = hamming_distance_chunked(X_test, X_lab_mal).min(dim=1).values
        d_nearest_ben = hamming_distance_chunked(X_test, X_lab_ben).min(dim=1).values

    # 5. Build DataFrame
    df = pd.DataFrame({
        'true_label': y_test.cpu().numpy(),
        'pred_label': y_pred.cpu().numpy(),
        'confidence': conf.cpu().numpy(),
        'misclassified': misclassified.cpu().numpy().astype(int),
        'd_to_malware_centroid': d_to_mal.numpy(),
        'd_to_benign_centroid': d_to_ben.numpy(),
        'centroid_margin': centroid_margin.numpy(),
        'd_to_nearest_labeled': d_nearest_labeled.numpy(),
        'nearest_labeled_class': nearest_labels.cpu().numpy(),
        'd_to_nearest_malware': d_nearest_mal.numpy(),
        'd_to_nearest_benign': d_nearest_ben.numpy()
    })

    df.to_csv(save_path, index=False)
    print(f"[✓] Misclassification analysis saved to: {save_path}")
    return df




def plot_malware_tsne_with_boundary(
    model,
    X_labeled, y_labeled,
    X_unlabeled, y_unlabeled,
    X_test, y_test,
    strategy,
    boundary_k=200,
    cached_tsne=None  # Optional: precomputed t-SNE projection dict
):
    """
    Plots t-SNE visualization of labeled, unlabeled, and test data with decision boundary samples highlighted.

    Parameters:
        model: Trained model with an encode() method.
        X_labeled, y_labeled: Tensor - labeled data and labels
        X_unlabeled, y_unlabeled: Tensor - unlabeled data and (true or pseudo) labels
        X_test, y_test: Tensor - test data and true labels
        strategy: str - name suffix for plot
        boundary_k: int - number of boundary samples to highlight
        cached_tsne: dict or None - Optional precomputed t-SNE dict with keys: 'X_lab_2d', 'X_unlab_2d', 'y_lab', 'y_unlab'
    """
    model.eval()

    if cached_tsne is None:
        with torch.no_grad():
            X_labeled_enc = model.encode(X_labeled).cpu()
            X_unlabeled_enc = model.encode(X_unlabeled).cpu()

        # Stack for t-SNE
        X_ref_all = torch.cat([X_labeled_enc, X_unlabeled_enc], dim=0)
        X_ref_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_ref_all)

        n_lab = X_labeled.shape[0]
        X_lab_2d = X_ref_2d[:n_lab]
        X_unlab_2d = X_ref_2d[n_lab:]
        y_lab = y_labeled.cpu().numpy()
        y_unlab = y_unlabeled.cpu().numpy()
    else:
        X_lab_2d = cached_tsne['X_lab_2d']
        y_lab = cached_tsne['y_lab']
        X_unlab_2d = cached_tsne['X_unlab_2d']
        y_unlab = cached_tsne['y_unlab']

    # Encode and project only X_test
    with torch.no_grad():
        X_test_enc = model.encode(X_test).cpu()
        logits_test = model(X_test)
        probs_test = F.softmax(logits_test, dim=1)
        top2 = torch.topk(probs_test, 2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]
        boundary_indices = torch.argsort(margins)[:boundary_k]

    X_test_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_test_enc)
    y_tst = y_test.cpu().numpy()
    X_boundary_2d = X_test_2d[boundary_indices.cpu()]

    # ---------- Plot ----------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))

    # Labeled
    plt.scatter(X_lab_2d[y_lab == 0, 0], X_lab_2d[y_lab == 0, 1], c='green', label='Labeled Benign', alpha=0.6, marker='o')
    plt.scatter(X_lab_2d[y_lab == 1, 0], X_lab_2d[y_lab == 1, 1], c='red', label='Labeled Malware', alpha=0.6, marker='o')

    # Unlabeled
    plt.scatter(X_unlab_2d[y_unlab == 0, 0], X_unlab_2d[y_unlab == 0, 1], c='lightgreen', label='Unlabeled Benign', alpha=0.4, marker='s')
    plt.scatter(X_unlab_2d[y_unlab == 1, 0], X_unlab_2d[y_unlab == 1, 1], c='salmon', label='Unlabeled Malware', alpha=0.4, marker='s')

    # Test
    plt.scatter(X_test_2d[y_tst == 0, 0], X_test_2d[y_tst == 0, 1], c='blue', label='Test Benign', alpha=0.6, marker='^')
    plt.scatter(X_test_2d[y_tst == 1, 0], X_test_2d[y_tst == 1, 1], c='purple', label='Test Malware', alpha=0.6, marker='^')

    # Boundary samples
    plt.scatter(X_boundary_2d[:, 0], X_boundary_2d[:, 1], facecolors='none', edgecolors='black',
                linewidths=1.5, s=100, label=f'Boundary Samples (top {boundary_k})')

    plt.title("t-SNE of Malware Dataset with Decision Boundary Samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.save_path}/{strategy}_tsne_boundary.png", dpi=300)
    plt.show()


def plot_malware_tsne_with_misclassified(
    model,
    X_labeled, y_labeled,
    X_unlabeled, y_unlabeled,
    X_test, y_test,
    strategy,
    boundary_k=200,
    cached_tsne=None
):
    """
    Plots t-SNE visualization of labeled, misclassified unlabeled, and misclassified test data with decision boundary samples highlighted.
    """
    import matplotlib.pyplot as plt

    def to_numpy(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    model.eval()

    if cached_tsne is None:
        with torch.no_grad():
            X_labeled_enc = model.encode(X_labeled).cpu()
            X_unlabeled_enc = model.encode(X_unlabeled).cpu()

        # Stack for t-SNE
        X_ref_all = torch.cat([X_labeled_enc, X_unlabeled_enc], dim=0)
        X_ref_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_ref_all)

        n_lab = X_labeled.shape[0]
        X_lab_2d = X_ref_2d[:n_lab]
        X_unlab_2d = X_ref_2d[n_lab:]
        y_lab = to_numpy(y_labeled)
        y_unlab = to_numpy(y_unlabeled)
    else:
        X_lab_2d = cached_tsne['X_lab_2d']
        y_lab = cached_tsne['y_lab']
        X_unlab_2d = cached_tsne['X_unlab_2d']
        y_unlab = cached_tsne['y_unlab']

    # Get test encodings + predictions
    with torch.no_grad():
        X_test_enc = model.encode(X_test).cpu()
        logits_test = model(X_test)
        probs_test = F.softmax(logits_test, dim=1)
        preds_test = probs_test.argmax(dim=1)
        y_tst = to_numpy(y_test)
        preds_test_np = to_numpy(preds_test)

    X_test_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_test_enc)

    # Misclassified test indices
    mis_test_mask = preds_test_np != y_tst
    X_test_2d_mis = X_test_2d[mis_test_mask]
    y_tst_mis = y_tst[mis_test_mask]

    # Decision boundary: low confidence margin
    top2 = torch.topk(probs_test, 2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    boundary_indices = torch.argsort(margins)[:boundary_k]
    X_boundary_2d = X_test_2d[boundary_indices.cpu()]

    # Predict unlabeled
    with torch.no_grad():
        logits_unlab = model(X_unlabeled)
        preds_unlab = logits_unlab.argmax(dim=1)
    preds_unlab_np = to_numpy(preds_unlab)
    y_unlab_np = to_numpy(y_unlabeled)
    mis_unlab_mask = preds_unlab_np != y_unlab_np
    X_unlab_2d_mis = X_unlab_2d[mis_unlab_mask]
    y_unlab_mis = y_unlab_np[mis_unlab_mask]

    # ---------- Plot ----------
    plt.figure(figsize=(10, 8))

    # Labeled (All)
    plt.scatter(X_lab_2d[y_lab == 0, 0], X_lab_2d[y_lab == 0, 1], c='green', label='Labeled Benign', alpha=0.6, marker='o')
    plt.scatter(X_lab_2d[y_lab == 1, 0], X_lab_2d[y_lab == 1, 1], c='red', label='Labeled Malware', alpha=0.6, marker='o')

    # Unlabeled (Misclassified Only)
    plt.scatter(X_unlab_2d_mis[y_unlab_mis == 0, 0], X_unlab_2d_mis[y_unlab_mis == 0, 1],
                c='lightgreen', label='Misclassified Unlabeled Benign', alpha=0.4, marker='s')
    plt.scatter(X_unlab_2d_mis[y_unlab_mis == 1, 0], X_unlab_2d_mis[y_unlab_mis == 1, 1],
                c='salmon', label='Misclassified Unlabeled Malware', alpha=0.4, marker='s')

    # Test (Misclassified Only)
    plt.scatter(X_test_2d_mis[y_tst_mis == 0, 0], X_test_2d_mis[y_tst_mis == 0, 1],
                c='blue', label='Misclassified Test Benign', alpha=0.6, marker='^')
    plt.scatter(X_test_2d_mis[y_tst_mis == 1, 0], X_test_2d_mis[y_tst_mis == 1, 1],
                c='purple', label='Misclassified Test Malware', alpha=0.6, marker='^')

    # Boundary samples
    plt.scatter(X_boundary_2d[:, 0], X_boundary_2d[:, 1], facecolors='none', edgecolors='black',
                linewidths=1.5, s=100, label=f'Boundary Samples (top {boundary_k})')

    plt.title("t-SNE of Malware Dataset (Only Misclassified Test & Unlabeled)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"analysis2/{strategy}_tsne_boundary.png", dpi=300)
    plt.show()


import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

def analyze_and_plot_boundary_samples(
    model, X_test, y_test, cached_tsne,
    strategy, boundary_k=400, save_dir="analysis_boundary"
):
    """
    Analyze and plot t-SNE of X_test samples, including:
    - Correctly and misclassified malware/benign
    - Highlight boundary samples
    - Save details of boundary samples to CSV
    """

    os.makedirs(save_dir, exist_ok=True)

    def to_numpy(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    model.eval()
    device = X_test.device

    with torch.no_grad():
        # Encode and predict
        X_test_enc = model.encode(X_test).cpu()
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        probs_np = probs.cpu().numpy()
        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()

    # t-SNE of test
    X_test_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_test_enc)

    # Confidence margin
    top2 = torch.topk(probs, 2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    _, boundary_indices = torch.topk(margins, boundary_k, largest=False)
    X_boundary = X_test[boundary_indices]
    y_boundary_true = y_test[boundary_indices]
    y_boundary_pred = preds[boundary_indices]
    X_boundary_2d = X_test_2d[boundary_indices.cpu()]
    margins_boundary = margins[boundary_indices].cpu().numpy()

    # Get distances to centroids from labeled set (cached)
    X_labeled_2d = cached_tsne["X_lab_2d"]
    y_labeled_np = cached_tsne["y_lab"]

    with torch.no_grad():
        X_lab_enc = model.encode(cached_tsne["X_labeled_tensor"].to(device)).cpu()
        mal_centroid = X_lab_enc[y_labeled_np == 1].mean(dim=0)
        ben_centroid = X_lab_enc[y_labeled_np == 0].mean(dim=0)

        X_boundary_enc = model.encode(X_boundary)

        d_to_mal = torch.norm(X_boundary_enc - mal_centroid.to(device), dim=1)
        d_to_ben = torch.norm(X_boundary_enc - ben_centroid.to(device), dim=1)
        margin_to_centroid = torch.abs(d_to_mal - d_to_ben).cpu().numpy()

        pairwise = torch.cdist(X_boundary_enc, X_lab_enc.to(device), p=2)
        d_nearest, nearest_idx = pairwise.min(dim=1)
        nearest_labels = y_labeled_np[nearest_idx.cpu()]
        # --- Compute distance to nearest boundary sample (excluding self) ---

        boundary_encodings = X_boundary_enc.cpu()
        boundary_dists = torch.cdist(boundary_encodings, boundary_encodings, p=2)

        # Mask diagonal (self) by setting to large number
        diag_idx = torch.arange(boundary_dists.size(0))
        boundary_dists[diag_idx, diag_idx] = float('inf')

        # Find nearest neighbor per row
        d_nearest_boundary, nearest_boundary_idx = boundary_dists.min(dim=1)

        # Get true labels of nearest boundary neighbors
        nearest_boundary_true_label = y_boundary_true[nearest_boundary_idx].cpu().numpy()


    # Plotting
    plt.figure(figsize=(12, 8))
    correct_mask = y_pred == y_true
    mal_mask = y_true == 1
    ben_mask = y_true == 0
    mis_mask = ~correct_mask

    plt.scatter(X_test_2d[correct_mask & ben_mask, 0], X_test_2d[correct_mask & ben_mask, 1],
                c="green", label="Benign Samples", alpha=0.5)
    plt.scatter(X_test_2d[correct_mask & mal_mask, 0], X_test_2d[correct_mask & mal_mask, 1],
                c="red", label="Malware Samples", alpha=0.5)
    plt.scatter(X_test_2d[mis_mask & ben_mask, 0], X_test_2d[mis_mask & ben_mask, 1],
                c="blue", label="Misclassified Benign", alpha=0.6)
    plt.scatter(X_test_2d[mis_mask & mal_mask, 0], X_test_2d[mis_mask & mal_mask, 1],
                c="purple", label="Misclassified Malware", alpha=0.6)

    # for i, true in enumerate(y_boundary_true.cpu().numpy()):
    #     # if true == 1:
    #     #     color = "black" if y_boundary_pred[i] == 1 else "red"
    #     # else:
    #     #     color = "gray" if y_boundary_pred[i] == 0 else "orange"
    #     color='black'
    #     plt.scatter(X_boundary_2d[i, 0], X_boundary_2d[i, 1], edgecolors=color, facecolors='none',
    #                 s=120, linewidths=2, marker='o', label=None if i else "Boundary Drifted Samples")

    # Only plot misclassified boundary samples
    boundary_pred = y_boundary_pred.cpu().numpy()
    boundary_true = y_boundary_true.cpu().numpy()

    misclassified_boundary_mask = boundary_pred != boundary_true
    X_mis_boundary_2d = X_boundary_2d[misclassified_boundary_mask]

    for i in range(X_mis_boundary_2d.shape[0]):
        plt.scatter(
            X_mis_boundary_2d[i, 0], X_mis_boundary_2d[i, 1],
            edgecolors='black', facecolors='none',
            s=120, linewidths=2, marker='o',
            label="Misclassified Boundary Samples" if i == 0 else None
        )

    # plt.scatter(X_labeled_2d[y_labeled_np == 0, 0], X_labeled_2d[y_labeled_np == 0, 1],
    #             c='green', label='Labeled Benign', alpha=0.2)
    # plt.scatter(X_labeled_2d[y_labeled_np == 1, 0], X_labeled_2d[y_labeled_np == 1, 1],
    #             c='red', label='Labeled Malware', alpha=0.2)

    plt.legend()
    # plt.grid(True)
    # plt.title("t-SNE of X_test with Misclassifications and Boundary Samples")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{strategy}_boundary_plot.png", dpi=300)
    plt.show()

    # Save boundary details
    df = pd.DataFrame({
        "index": boundary_indices.cpu().numpy(),
        "true_label": y_boundary_true.cpu().numpy(),
        "pred_label": y_boundary_pred.cpu().numpy(),
        "margin": margins_boundary,
        "d_to_mal_centroid": d_to_mal.cpu().numpy(),
        "d_to_ben_centroid": d_to_ben.cpu().numpy(),
        "centroid_margin": margin_to_centroid,
        "d_to_nearest_labeled": d_nearest.cpu().numpy(),
        "nearest_labeled_class": nearest_labels,
        "d_to_nearest_boundary": d_nearest_boundary.numpy(),
        "nearest_boundary_true_label": nearest_boundary_true_label
    })

    csv_path = f"{save_dir}/{strategy}_boundary_details.csv"
    df.to_csv(csv_path, index=False)
    print(f"[✓] Boundary analysis saved to: {csv_path}")



# --- Ensure compatibility with PyTorch tensors ---
def ensure_tensor_on_cpu(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x.cpu()




def append_to_strategy(s):
    global strategy
    strategy += s

# === Main Training Function for FixMatch with AL and taking best loss model weight===
def active_learning_fixmatch(
    bit_flip, model, optimizer, X_labeled, y_labeled, X_unlabeled,
    args, num_classes=2, threshold=0.95, lambda_u=1.0, epochs=200, retrain_epochs=70, batch_size=512,
    al_batch_size=512 
):
    labeled_ds = TensorDataset(X_labeled, y_labeled)
    unlabeled_ds = TensorDataset(X_unlabeled)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler


    labeled_loader = DataLoader(labeled_ds, sampler=train_sampler(labeled_ds), batch_size=batch_size, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_ds, sampler=train_sampler(unlabeled_ds), batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, epochs)
    
    best_loss = float('inf')
    best_state_dict = None

    mu = 1  # Number of unlabeled augmentations per sample (FixMatch default is 1)
    interleave_size = 2 * mu + 1  # labeled, unlabeled_weak, unlabeled_strong

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for _ in range(len(labeled_loader)):
            try:
                x_l, y_l = next(labeled_iter)
                (x_u,) = next(unlabeled_iter)
            except StopIteration:
                break

            x_l, y_l = x_l.cuda(), y_l.cuda()
            x_u = x_u.cuda()

            # Weak and strong augmentations for unlabeled data, now with seed
            if args.aug == "random_bit_flip":
                x_u_w = random_bit_flip(x_u, n_bits=1)
                x_u_s = random_bit_flip(x_u, n_bits=bit_flip)
            elif args.aug == "random_bit_flip_bernoulli":
                x_u_w = random_bit_flip_bernoulli(x_u, p=0.01, n_bits=None)
                x_u_s = random_bit_flip_bernoulli(x_u, p=0.05, n_bits=None)
            elif args.aug == "random_feature_mask":
                x_u_w = random_feature_mask(x_u, n_mask=1)
                x_u_s = random_feature_mask(x_u, n_mask=bit_flip)
            elif args.aug == "random_bit_flip_and_mask":
                x_u_w = random_bit_flip_and_mask(x_u, n_bits=1, n_mask=1)
                x_u_s = random_bit_flip_and_mask(x_u, n_bits=bit_flip, n_mask=bit_flip)
            else:
                raise ValueError(f"Unknown augmentation function: {args.aug}")

            # Interleave all inputs for batchnorm consistency
            inputs = torch.cat([x_l, x_u_w, x_u_s], dim=0)

            logits = model(inputs)

            batch_size = x_l.shape[0]
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            # Labeled loss
            loss_x = criterion(logits_x, y_l)

            # Unlabeled loss (FixMatch pseudo-labeling)
            with torch.no_grad():
                pseudo_logits = F.softmax(logits_u_w / args.T, dim=1)
                pseudo_labels = torch.argmax(pseudo_logits, dim=1)
                max_probs, _ = torch.max(pseudo_logits, dim=1)
                mask = max_probs.ge(threshold).float()

            loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()
            loss = loss_x + lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
        scheduler.step()

    # Restore best model after initial training
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)


    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # === Encode labeled and unlabeled data for t-SNE visualization and save===
    with torch.no_grad():
        X_labeled_enc = model.encode(X_labeled).cpu()
        # X_unlabeled_enc = model.encode(X_unlabeled).cpu()

    # X_ref_all = torch.cat([X_labeled_enc, ], dim=0)
    tsne_ref_all = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_labeled_enc)

    # Split back for plotting
    # n_lab = len(X_labeled)
    X_lab_2d = tsne_ref_all

    # Cache for later reuse
    cached_tsne = {
        "X_lab_2d": tsne_ref_all,
        "y_lab": y_labeled.cpu().numpy(),
        "X_labeled_tensor": X_labeled.cpu()
    }

    # Active learning loop
    metrics_list = []
    model.eval()
    
    for year in range(2013, 2019):
        for month in range(1, 13):
            try:
                with torch.no_grad():
                    data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                    X_raw = data["X_train"]
                    y_true = (data["y_train"] > 0).astype(int)
                    X_test = torch.tensor(X_raw, dtype=torch.float32).cuda()
                    y_test = torch.tensor(y_true, dtype=torch.long).cuda()

                    logits = model(X_test)
                    probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
                    preds = logits.argmax(dim=1)
                    y_true = y_test.cpu().numpy()
                    y_pred = preds.cpu().numpy()
                    if probs.shape[1] == 2:
                        y_score = probs[:, 1].cpu().numpy()
                    else:
                        y_score = probs.cpu().numpy()  # for multi-class

                    # Evaluate metrics
                    year_month = f"{year}-{month:02d}"
                    print(f"Evaluating {year_month}...")
                    metrics = evaluate_model(model, X_test, y_test, year_month, num_classes=num_classes)
                    acc = metrics['accuracy']
                    prec = metrics['precision']
                    rec = metrics['recall']
                    f1 = metrics['f1']
                    fnr = metrics['fnr']
                    fpr = metrics['fpr']
                    roc_auc = metrics['roc_auc']
                    pr_auc = metrics['pr_auc']
                    metrics_list.append(metrics)

                    print(f"Year {year_month}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

            except FileNotFoundError:
                continue
            # --- Active learning: select most uncertain samples from X_unlabeled ---
            # X_unlabeled = X_test.clone()
            
            # total number of misclassified samples
            num_misclassified = (y_pred != y_true).sum().item()
            print(f"Total misclassified samples in {year}-{month:02d}: {num_misclassified} out of {len(y_test)} samples")
            
            # Select samples based on uncertainty sampling
            X_labeled, y_labeled, X_test, y_test = active_learning_step(
                args=args,
                cached_tsne=cached_tsne,
                model=model,
                year_month=f"{year}-{month:02d}",
                X_labeled=X_labeled,
                y_labeled=y_labeled,
                X_test=X_test,
                y_test=y_test,
                top_k=args.budget,  # Use budget as top_k
                # confidence_threshold=threshold
            )

            
            
            X_unlabeled = torch.cat([X_unlabeled, X_test], dim=0)
            # X_unlabeled = X_test.clone()
            # Remove selected samples from the unlabeled set
            unlabeled_ds = TensorDataset(X_unlabeled)
            labeled_ds = TensorDataset(X_labeled, y_labeled)


            labeled_loader = DataLoader(labeled_ds, sampler=train_sampler(labeled_ds), batch_size=al_batch_size, drop_last=True)
            unlabeled_loader = DataLoader(unlabeled_ds, sampler=train_sampler(unlabeled_ds), batch_size=al_batch_size, drop_last=True)
            criterion = nn.CrossEntropyLoss(reduction='mean')


            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr,
                                momentum=0.9, nesterov=args.nesterov)
            
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, epochs)
            
            best_loss = float('inf')
            best_state_dict = None


            for epoch in range(retrain_epochs):
                model.train()
                total_loss = 0
                labeled_iter = iter(labeled_loader)
                unlabeled_iter = iter(unlabeled_loader)
                for _ in range(len(labeled_loader)):
                    try:
                        x_l, y_l = next(labeled_iter)
                        (x_u,) = next(unlabeled_iter)
                    except StopIteration:
                        break
                    x_l, y_l = x_l.cuda(), y_l.cuda()
                    x_u = x_u.cuda()
                    if args.aug == "random_bit_flip":
                        x_u_w = random_bit_flip(x_u, n_bits=1)
                        x_u_s = random_bit_flip(x_u, n_bits=bit_flip)
                    elif args.aug == "random_bit_flip_bernoulli":
                        x_u_w = random_bit_flip_bernoulli(x_u, p=0.01)
                        x_u_s = random_bit_flip_bernoulli(x_u, p=0.05)
                    elif args.aug == "random_bit_flip_and_mask":
                        x_u_w = random_bit_flip_and_mask(x_u, n_bits=1, n_mask=1)
                        x_u_s = random_bit_flip_and_mask(x_u, n_bits=bit_flip, n_mask=bit_flip)
                    else:
                        raise ValueError(f"Unknown augmentation function: {args.aug}")
                    inputs = torch.cat([x_l, x_u_w, x_u_s], dim=0)
                    logits = model(inputs)
                    batch_size = x_l.shape[0]
                    logits_x = logits[:batch_size]
                    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                    loss_x = criterion(logits_x, y_l)
                    with torch.no_grad():
                        # Pseudo-labels via FixMatch (confidence-based)
                        pseudo_logits = F.softmax(logits_u_w, dim=1)
                        pseudo_labels = torch.argmax(pseudo_logits, dim=1)
                        max_probs, _ = torch.max(pseudo_logits, dim=1)
                        mask = max_probs.ge(threshold).float()

                        # Low confidence filtering
                        pseudo_logits_low = F.softmax(logits_x, dim=1)
                        max_probs_low, _ = torch.max(pseudo_logits_low, dim=1)
                        low_confidence_mask = max_probs_low < threshold
                        X_test_low_conf = x_u[low_confidence_mask]

                    # Standard FixMatch loss on high-confidence pseudo-labels
                    loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()

                    

                    # Final total loss
                    loss = loss_x + lambda_u * loss_u

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                print(f"[{year}] Retrain Epoch {epoch+1}: loss={total_loss:.4f}")

    # Save results to CSV
    metrics_df = pd.DataFrame(metrics_list)
    save_metrics = os.path.join(args.save_path, f"{strategy}_active.csv")
    metrics_df.to_csv(save_metrics, index=False)
    print(f"Mean F1 Scores: {metrics_df['f1'].mean():.4f}")
    print(f"Mean False Negative Rates: {metrics_df['fnr'].mean()}")
    print(f"Mean False Positive Rates: {metrics_df['fpr'].mean()}")
    save_plots = os.path.join(args.save_path, f"{strategy}_active_plots.png")
    # plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], save_path=save_plots)
    return metrics_df



# === Main Execution ===
if __name__ == "__main__":
    # here we take the number of bit flip as argument
    # using argparse for taking arguments for number of bit flip

    parser = argparse.ArgumentParser(description="Run FixMatch with Bit Flip Augmentation")
    parser.add_argument("--bit_flip", type=int, default=11, help="Number of bits to flip per sample")
    parser.add_argument("--labeled_ratio", type=float, default=0.4, help="Ratio of labeled data")
    parser.add_argument("--aug", type=str, default="random_bit_flip", help="Augmentation function to use")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--lp', default=2.0, type=float, help='Lp norm for uncertainty sampling (e.g., 1.0, 2.0, 2.5)')
    parser.add_argument('--budget', default=200, type=int, help='Budget for active learning (number of samples to select)')
    parser.add_argument('--epochs', default=250, type=int, help='Number of training epochs')
    parser.add_argument('--retrain_epochs', default=50, type=int, help='Number of retraining epochs after initial training')
    parser.add_argument('--save_path', type=str, default='results/al_uc/', help='Path to save results')
    # parse arugments for uncertainty sampling option 1. lp-norm, 2. boundary selection 3. priority 4. hybrid
    parser.add_argument('--al', action='store_true', help='Enable Active Learning (default: False)')
    parser.add_argument('--unc_samp', type=str, default='lp-norm', choices=['lp-norm', 'boundary', 'priority', 'hybrid'], help='Uncertainty sampling method to use')
    parser.add_argument('--lambda_supcon', default=0.5, type=float, help='Coefficient of supervised contrastive loss.')
    parser.add_argument("--strategy", type=str, default="_", help="any strategy (keywork) to use")
    
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load data
    path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
    file_path = f"{path}2012-01to2012-12_selected.npz"
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    y = np.array([0 if label == 0 else 1 for label in y])
    
    n_bit_flip = args.bit_flip
    labeled_ratio = args.labeled_ratio

    strategy = f"fixmatch_w_al_uc_" + args.aug + "_" + str(n_bit_flip) + "_lbr_" + str(labeled_ratio) +  "_seed_" + str(args.seed)
    append_to_strategy(f"_{args.budget}")
    append_to_strategy(f"_lp_{args.lp}")
    append_to_strategy(f"_unc_samp_{args.unc_samp}")

    print(f"Running {strategy}...")
    print(f"Using {n_bit_flip} bits to flip per sample.")

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_ratio=0.4)

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()


    input_dim = X_2012_labeled.shape[1]
    num_classes = len(torch.unique(y_2012_labeled))



    # model = Classifier(input_dim=input_dim, num_classes=num_classes).cuda()
    model = ClassifierWB(input_dim=input_dim, num_classes=num_classes).cuda()
    # append_to_strategy(f"_batch_norm_")
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    
    # calculate the time it takes to run the function
    import time
    start_time = time.time()
    active_learning_fixmatch(
        n_bit_flip,
        model,
        optimizer,
        X_2012_labeled,
        y_2012_labeled,
        X_2012_unlabeled,
        args,
        num_classes=num_classes
    )
    end_time = time.time()
    print(f"Time taken to run the function: {end_time - start_time:.2f} seconds")





# def train_fixmatch_drift_eval(
#     bit_flip, model, optimizer, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
#     args, num_classes=2, threshold=0.95, lambda_u=1.0, epochs=20, batch_size=64
# ):
#     labeled_ds = TensorDataset(X_labeled, y_labeled)
#     unlabeled_ds = TensorDataset(X_unlabeled)

#     train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler


#     labeled_loader = DataLoader(labeled_ds, sampler=train_sampler(labeled_ds), batch_size=batch_size, drop_last=True)
#     unlabeled_loader = DataLoader(unlabeled_ds, sampler=train_sampler(unlabeled_ds), batch_size=batch_size, drop_last=True)
#     criterion = nn.CrossEntropyLoss(reduction='mean')

    
#     # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#     scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, epochs)
    
#     best_loss = float('inf')
#     best_state_dict = None

#     mu = 1  # Number of unlabeled augmentations per sample (FixMatch default is 1)
#     interleave_size = 2 * mu + 1  # labeled, unlabeled_weak, unlabeled_strong

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         labeled_iter = iter(labeled_loader)
#         unlabeled_iter = iter(unlabeled_loader)

#         for _ in range(len(labeled_loader)):
#             try:
#                 x_l, y_l = next(labeled_iter)
#                 (x_u,) = next(unlabeled_iter)
#             except StopIteration:
#                 break

#             x_l, y_l = x_l.cuda(), y_l.cuda()
#             x_u = x_u.cuda()
#             p = 0.0
#             # Weak and strong augmentations for unlabeled data, now with seed
#             if args.aug == "random_bit_flip":
#                 x_u_w = random_bit_flip(x_u, n_bits=1)
#                 x_u_s = random_bit_flip(x_u, n_bits=bit_flip)
#             elif args.aug == "random_bit_flip_bernoulli":
#                 p = 0.05
#                 x_u_w = random_bit_flip_bernoulli(x_u, p=0.01, n_bits=None)
#                 x_u_s = random_bit_flip_bernoulli(x_u, p=p, n_bits=None)
#             else:
#                 raise ValueError(f"Unknown augmentation function: {args.aug}")

#             # Interleave all inputs for batchnorm consistency
#             inputs = torch.cat([x_l, x_u_w, x_u_s], dim=0)
#             inputs = interleave(inputs, interleave_size)

#             logits = model(inputs)
#             logits = de_interleave(logits, interleave_size)

#             batch_size = x_l.shape[0]
#             logits_x = logits[:batch_size]
#             logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
#             del logits

#             # previously, we had:
#             # inputs = torch.cat([x_l, x_u_w, x_u_s], dim=0)

#             # logits = model(inputs)

#             # batch_size = x_l.shape[0]
#             # logits_x = logits[:batch_size]
#             # logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

#             # Labeled loss
#             loss_x = criterion(logits_x, y_l)

#             # Unlabeled loss (FixMatch pseudo-labeling)
#             with torch.no_grad():
#                 pseudo_logits = F.softmax(logits_u_w / args.T, dim=1)
#                 pseudo_labels = torch.argmax(pseudo_logits, dim=1)
#                 max_probs, _ = torch.max(pseudo_logits, dim=1)
#                 mask = max_probs.ge(threshold).float()

#             loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()
#             loss = loss_x + lambda_u * loss_u

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         if total_loss < best_loss:
#             best_loss = total_loss
#             best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
#         print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
#         scheduler.step()

#     # Restore best model after initial training
#     if best_state_dict is not None:
#         model.load_state_dict(best_state_dict)

#     os.environ["OPENBLAS_NUM_THREADS"] = "1"

#     # === Encode labeled and unlabeled data for t-SNE visualization and save===
#     with torch.no_grad():
#         X_labeled_enc = model.encode(X_labeled).cpu()
#         X_unlabeled_enc = model.encode(X_unlabeled).cpu()

#     X_ref_all = torch.cat([X_labeled_enc, X_unlabeled_enc], dim=0)
#     tsne_ref_all = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_ref_all)

#     # Split back for plotting
#     n_lab = len(X_labeled)
#     X_lab_2d = tsne_ref_all[:n_lab]
#     X_unlab_2d = tsne_ref_all[n_lab:]

#     # Cache for later reuse
#     cached_tsne = {
#         "X_lab_2d": X_lab_2d,
#         "y_lab": ensure_tensor_on_cpu(y_labeled),
#         "X_unlab_2d": X_unlab_2d,
#         "y_unlab": ensure_tensor_on_cpu(y_unlabeled),
#     }

    

#     # === Evaluate on each year's test set ===
#     path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
#     metrics_list = []
#     model.eval()
#     with torch.no_grad():
#         for year in range(2013, 2019):
#             for month in range(1, 13):
#                 try:
#                     file_path = f"{path}{year}-{month:02d}_selected.npz"
#                     data = np.load(file_path)
#                     X_raw = data["X_train"]
#                     y_true = (data["y_train"] > 0).astype(int)

#                     X_test = torch.tensor(X_raw, dtype=torch.float32).cuda()
#                     y_test = torch.tensor(y_true, dtype=torch.long).cuda()

#                     # Make sure y_* values are tensors for plotting
#                     y_lab = ensure_tensor_on_cpu(y_labeled)
#                     y_unlab = ensure_tensor_on_cpu(y_unlabeled)
#                     y_tst = ensure_tensor_on_cpu(y_test)

#                     # === Plot with boundary highlights ===
#                     plot_malware_tsne_with_misclassified(
#                         model,
#                         X_labeled, y_lab,
#                         X_unlabeled, y_unlab,
#                         X_test, y_tst,
#                         strategy=strategy + f"_{year}_{month}",
#                         boundary_k=200,
#                         cached_tsne=cached_tsne
#                     )


#                     # === Misclassification analysis ===
#                     save_path = f"analysis2/analysis_misclassified_{year}_{month}.csv"
#                     df_analysis = analyze_misclassifications(
#                         model,
#                         X_labeled, y_labeled,
#                         X_test, y_test,
#                         save_path=save_path
#                     )
