import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.cluster import KMeans

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



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

class MalwareTripletDataset(Dataset):
    def __init__(self, X, y, num_triplets=5000):
        self.X = X
        self.y = y
        self.num_triplets = num_triplets
        self.triplets = []
        self._create_triplets()

    def _create_triplets(self):
        for _ in range(self.num_triplets):
            idx_anchor = random.randint(0, len(self.y) - 1)
            pos_indices = np.where(self.y == self.y[idx_anchor])[0]
            neg_indices = np.where(self.y != self.y[idx_anchor])[0]
            if len(pos_indices) > 1 and len(neg_indices) > 0:
                idx_pos = random.choice(pos_indices)
                idx_neg = random.choice(neg_indices)
                self.triplets.append((self.X[idx_anchor], self.X[idx_pos], self.X[idx_neg]))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative)



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

    # top1000_indices = torch.argsort(all_margins)
    # selected_samples_1000 = X_unlabeled[top1000_indices].to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(selected_samples)
        # logits_1000 = model(selected_samples_1000)
        preds = logits.argmax(dim=1).cpu()
        # preds_1000 = logits_1000.argmax(dim=1).cpu()
    # True labels for these samples (if available)
    if y_unlabeled is not None:
        true_labels = y_unlabeled[topk_indices].cpu()
        # true_labels_1000 = y_unlabeled[top1000_indices].cpu()
        num_misclassified = (preds != true_labels).sum().item()
        # num_misclassified_1000 = (preds_1000 != true_labels_1000).sum().item()
        # print(f"Boundary samples misclassified: {num_misclassified} out of {len(topk_indices)}")
        # print(f"Boundary samples 1000 misclassified: {num_misclassified_1000} out of {len(top1000_indices)}")

    return topk_indices, all_margins[topk_indices]



def select_boundary_samples_(model, X_unlabeled, y_unlabeled, year_month="unknown", top_k=200, batch_size=512):
    """
    Selects top-K most uncertain samples from X_unlabeled based on softmax margin and logs detailed statistics.

    Returns:
        topk_indices (Tensor): Indices of selected boundary samples (lowest margins).
        topk_margins (Tensor): Margin values of selected samples.
    """
    model.eval()
    device = next(model.parameters()).device
    all_margins = []
    all_preds = []

    with torch.no_grad():
        for i in range(0, X_unlabeled.size(0), batch_size):
            xb = X_unlabeled[i:i+batch_size].to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            top2 = torch.topk(probs, 2, dim=1).values
            margins = (top2[:, 0] - top2[:, 1]).cpu()
            preds = probs.argmax(dim=1).cpu()
            all_margins.append(margins)
            all_preds.append(preds)

    all_margins = torch.cat(all_margins)
    all_preds = torch.cat(all_preds)

    k = min(top_k, len(all_margins))
    topk_indices = torch.argsort(all_margins)[:k]
    topk_margins = all_margins[topk_indices]
    topk_preds = all_preds[topk_indices]

    if y_unlabeled is not None:
        y_all = y_unlabeled.cpu()
        y_topk = y_all[topk_indices]
        mis_topk = (y_topk != topk_preds)

        # Misclassified stats
        total_misclassified = (y_all != all_preds).sum().item()
        mis_topk_count = mis_topk.sum().item()
        mis_topk_benign = ((y_topk == 0) & mis_topk).sum().item()
        mis_topk_malware = ((y_topk == 1) & mis_topk).sum().item()

        # For non-boundary stats
        all_indices = torch.arange(len(X_unlabeled))
        non_boundary_mask = torch.ones(len(X_unlabeled), dtype=torch.bool)
        non_boundary_mask[topk_indices] = False
        y_non_boundary = y_all[non_boundary_mask]
        preds_non_boundary = all_preds[non_boundary_mask]
        mis_non_boundary = (y_non_boundary != preds_non_boundary)
        mis_non_boundary_count = mis_non_boundary.sum().item()
        mis_non_boundary_benign = ((y_non_boundary == 0) & mis_non_boundary).sum().item()
        mis_non_boundary_malware = ((y_non_boundary == 1) & mis_non_boundary).sum().item()

        stats = {
            "year_month": year_month,
            "total_samples": len(X_unlabeled),
            "top_k": k,
            "total_misclassified": total_misclassified,
            "boundary_misclassified": mis_topk_count,
            "boundary_mis_benign": mis_topk_benign,
            "boundary_mis_malware": mis_topk_malware,
            "non_boundary_misclassified": mis_non_boundary_count,
            "non_boundary_mis_benign": mis_non_boundary_benign,
            "non_boundary_mis_malware": mis_non_boundary_malware,
        }

        df_stats = pd.DataFrame([stats])
        save_path = f"analysis_boundary/boundary_stats_{year_month}.csv"
        df_stats.to_csv(save_path, index=False)
        print(f"[✓] Saved boundary statistics to {save_path}")

        print(f"Total misclassified samples: {total_misclassified}")
        print(f"Boundary samples misclassified: {mis_topk_count} / {k}")
        print(f"    Malware: {mis_topk_malware}, Benign: {mis_topk_benign}")
        print(f"Non-boundary misclassified: {mis_non_boundary_count}")
        print(f"    Malware: {mis_non_boundary_malware}, Benign: {mis_non_boundary_benign}")

    return topk_indices, topk_margins



from sklearn.cluster import KMeans

def select_diverse_boundary_samples(model, X_test, budget=400, top_k=1000):
    """
    Select diverse samples near the decision boundary using KMeans clustering.

    Args:
        model: Trained model with `.eval()` and softmax output.
        X_test (Tensor): Test samples (N, D)
        top_k (int): Number of most uncertain samples to consider
        budget (int): Final number of diverse samples to return (<= top_k)

    Returns:
        selected_indices (Tensor): indices of selected samples (on device)
    """
    model.eval()
    device = X_test.device

    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        top2 = torch.topk(probs, 2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]  # smaller = more uncertain

    # Step 1: Get top-K most uncertain samples (smallest margins)
    topk_margins, topk_indices = torch.topk(margins, top_k, largest=False)
    X_boundary = X_test[topk_indices]

    # Step 2: Encode boundary features for diversity clustering
    with torch.no_grad():
        if hasattr(model, 'encode'):
            X_encoded = model.encode(X_boundary)
        else:
            X_encoded = X_boundary  # fallback to raw input if no encoder

    # Step 3: Run KMeans to select diverse samples
    X_encoded_np = X_encoded.cpu().numpy()
    n_clusters = min(budget, X_encoded_np.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_encoded_np)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    selected_indices = []
    selected_margins = []
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        if not np.any(cluster_mask):
            continue  # Skip empty clusters

        cluster_points = X_encoded_np[cluster_mask]
        cluster_idxs = topk_indices[cluster_mask]
        cluster_margins = topk_margins[cluster_mask]

        center = centers[cluster_id]
        dists = np.linalg.norm(cluster_points - center, axis=1)
        best_local_idx = np.argmin(dists)

        selected_indices.append(cluster_idxs[best_local_idx].item())
        selected_margins.append(cluster_margins[best_local_idx].item())


    return selected_indices, selected_margins




def select_diverse_boundary_samples_(model, X_test, y_test, top_k=1000, budget=400):
    """
    Select diverse samples near the decision boundary using KMeans clustering.

    Args:
        model: Trained model with `.eval()` and softmax output.
        X_test (Tensor): Test samples (N, D)
        top_k (int): Number of most uncertain samples to consider
        budget (int): Final number of diverse samples to return (<= top_k)

    Returns:
        selected_indices (Tensor): indices of selected samples (on device)
        selected_margins (Tensor): margins of selected samples (on device)
    """
    model.eval()
    device = X_test.device

    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        top2 = torch.topk(probs, 2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]  # smaller = more uncertain

    # Step 1: Get top-K most uncertain samples (smallest margins)
    topk_margins, topk_indices = torch.topk(margins, top_k, largest=False)
    X_boundary = X_test[topk_indices]

    # Step 2: Encode boundary features for diversity clustering
    with torch.no_grad():
        if hasattr(model, 'encode'):
            X_encoded = model.encode(X_boundary)
        else:
            X_encoded = X_boundary

    X_encoded_np = X_encoded.cpu().numpy()
    n_clusters = min(budget, X_encoded_np.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_encoded_np)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    selected_indices = []
    selected_margins = []
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_points = X_encoded_np[cluster_mask]
        cluster_idxs = topk_indices[cluster_mask]
        cluster_margins = topk_margins[cluster_mask]

        center = centers[cluster_id]
        dists = np.linalg.norm(cluster_points - center, axis=1)
        best_local_idx = np.argmin(dists)
        selected_indices.append(cluster_idxs[best_local_idx].item())
        selected_margins.append(cluster_margins[best_local_idx].item())

    # return torch.tensor(selected_indices, device=device), torch.tensor(selected_margins, device=device)
    return selected_indices, selected_margins

def prioritized_uncertainty_selection(
    model,
    X_labeled, y_labeled,
    X_test, y_test=None,
    budget=400,
    lp_norm=2,
    confidence_threshold=0.95,
    w_margin=1.0,
    w_lp=1.0,
    w_conf=1.0,
    batch_size=512
):
    """
    Combines boundary, Lp-norm, and low-confidence metrics with weights,
    and selects top `budget` samples with highest uncertainty.
    
    Returns:
        selected_indices (Tensor): Indices of selected samples (len = budget)
    """
    device = X_test.device
    model.eval()

    # ---------- Softmax Margin & Confidence ----------
    all_margins = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for i in range(0, X_test.size(0), batch_size):
            xb = X_test[i:i+batch_size].to(device)
            probs = F.softmax(model(xb), dim=1)
            preds = probs.argmax(dim=1).cpu()
            top2 = torch.topk(probs, 2, dim=1).values
            margins = (top2[:, 0] - top2[:, 1]).cpu()
            all_margins.append(margins)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    all_margins = torch.cat(all_margins)
    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    max_probs = all_probs.max(dim=1).values

    margin_score = 1 - (all_margins - all_margins.min()) / (all_margins.max() - all_margins.min() + 1e-8)
    conf_score = (1 - max_probs - (1 - max_probs).min()) / ((1 - max_probs).max() - (1 - max_probs).min() + 1e-8)

    # ---------- Lp-Distance Score ----------
    with torch.no_grad():
        enc_lab = model.encode(X_labeled.to(device))
        enc_test = model.encode(X_test.to(device))
        dists = torch.cdist(enc_test, enc_lab, p=lp_norm)
        min_dists = dists.min(dim=1).values.cpu()

    lp_score = (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min() + 1e-8)

    # ---------- Final Combined Score ----------
    final_score = w_margin * margin_score + w_lp * lp_score + w_conf * conf_score

    # ---------- Select top `budget` ----------
    k = min(budget, len(final_score))  # Ensure k is within valid range
    selected_indices = torch.topk(final_score, k).indices

    print(f"[✓] Prioritized Uncertainty Selection (budget={budget})")
    print(f" - margin_score: {w_margin}, lp_score: {w_lp}, conf_score: {w_conf}")
    print(f" - Selected {len(selected_indices)} samples from total {X_test.shape[0]}")
    # how many samples misclassified from the selected_indices
    # Ensure selected_indices is a flat LongTensor
    selected_indices = torch.tensor(selected_indices, dtype=torch.long).view(-1).cpu()

    # Ensure predictions and ground truths are on CPU and same dtype
    all_preds = all_preds.cpu().long()
    y_test = y_test.cpu()

    # Compute misclassified count
    misclassified_selected = (all_preds[selected_indices] != y_test[selected_indices]).sum().item()

    print(f" - Total number of misclassified samples: {(all_preds != y_test).sum().item()}")
    print(f" - Total number of misclassified samples from the selected: {misclassified_selected}")


    return selected_indices




def hybrid_active_sample_selection(
    model, X_labeled, y_labeled, X_test,
    budget=400,
    lp_norm=2,
    confidence_threshold=0.95,
    batch_size=512
):
    device = X_test.device
    model.eval()

    with torch.no_grad():
        # ---------- Compute margins and softmax confidences ----------
        all_margins = []
        all_probs = []
        for i in range(0, X_test.size(0), batch_size):
            xb = X_test[i:i+batch_size].to(device)
            probs = F.softmax(model(xb), dim=1)
            top2 = torch.topk(probs, 2, dim=1).values
            margins = (top2[:, 0] - top2[:, 1]).cpu()
            all_margins.append(margins)
            all_probs.append(probs.cpu())

        all_margins = torch.cat(all_margins)  # shape: (N,)
        all_probs = torch.cat(all_probs)
        max_probs = all_probs.max(dim=1).values  # shape: (N,)

        # ---------- Get top-K boundary and lp samples ----------
        margin_order = torch.argsort(all_margins)  # ascending = most uncertain
        topk_bdy = margin_order[:budget]  # tensor on CPU

        # ---------- Lp-distance uncertainty ----------
        enc_lab = model.encode(X_labeled.to(device))
        enc_test = model.encode(X_test.to(device))
        min_dists = []

        for i in range(0, enc_test.size(0), batch_size):
            test_batch = enc_test[i:i+batch_size]
            dist_batch = torch.cdist(test_batch, enc_lab, p=lp_norm)
            min_dist_batch = dist_batch.min(dim=1).values
            min_dists.append(min_dist_batch.cpu())

        min_dists = torch.cat(min_dists)
        dist_order = torch.argsort(min_dists, descending=True)
        topk_lp = dist_order[:budget]

        # ---------- Low-confidence ----------
        low_conf_mask = max_probs < confidence_threshold
        low_conf_indices = torch.nonzero(low_conf_mask, as_tuple=True)[0]
        topk_conf = low_conf_indices[:budget]

        # ---------- Intersection ----------
        intersection = set(topk_bdy.tolist()) & set(topk_lp.tolist()) & set(topk_conf.tolist())
        intersection = torch.tensor(list(intersection), dtype=torch.long)
        base_size = len(intersection)

        if base_size >= budget:
            return intersection[:budget].to(device)

        # ---------- Remaining budget ----------
        num_remaining = budget - base_size
        num_bdy = int(0.4 * budget)
        num_lp = int(0.4 * budget)
        num_conf = budget - base_size - num_bdy - num_lp

        # ---------- Remainder excluding base ----------
        remaining_bdy = [i for i in topk_bdy.tolist() if i not in intersection]
        remaining_lp = [i for i in topk_lp.tolist() if i not in intersection]
        remaining_conf = [i for i in topk_conf.tolist() if i not in intersection]

        # ---------- Select additional samples ----------
        margin_rest = all_margins[remaining_bdy] if remaining_bdy else torch.tensor([])
        dist_rest = min_dists[remaining_lp] if remaining_lp else torch.tensor([])
        conf_rest = max_probs[remaining_conf] if remaining_conf else torch.tensor([])

        add_bdy = []
        if margin_rest.numel() > 0 and num_bdy > 0:
            _, idx_sorted_bdy = torch.topk(-margin_rest, k=min(num_bdy, margin_rest.numel()))
            add_bdy = [remaining_bdy[i.item()] for i in idx_sorted_bdy]

        add_lp = []
        if dist_rest.numel() > 0 and num_lp > 0:
            _, idx_sorted_lp = torch.topk(dist_rest, k=min(num_lp, dist_rest.numel()))
            add_lp = [remaining_lp[i.item()] for i in idx_sorted_lp]

        add_conf = []
        if conf_rest.numel() > 0 and num_conf > 0:
            _, idx_conf_sorted = torch.topk(-conf_rest, k=min(num_conf, conf_rest.numel()))
            add_conf = [remaining_conf[i.item()] for i in idx_conf_sorted]

        # ---------- Merge all ----------
        final_indices = torch.unique(torch.tensor(
            list(intersection.tolist()) + add_bdy + add_lp + add_conf,
            dtype=torch.long
        ))

        if final_indices.numel() > budget:
            final_indices = final_indices[:budget]

    return final_indices





# ---------- Active Learning Integration ----------
def active_learning_step(args, year_month, model, X_labeled, y_labeled, X_test, y_test, top_k=200):
    """
    Active learning step:
    1. Selects top-k most uncertain samples from the entire test set (X_test).
    2. Moves those to labeled set.
    3. Keeps the rest in the unlabeled set.
    """
    device = X_labeled.device

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

        # uncertain_indices, _ = select_diverse_boundary_samples(
        #     model, X_test, budget=top_k
        # )

        # uncertain_indices, margins = select_boundary_samples_(
        #     model, X_test, y_test, year_month=year_month, top_k=top_k
        # )
    elif args.unc_samp == 'priority':
        uncertain_indices = prioritized_uncertainty_selection(
            model,
            X_labeled, y_labeled,
            X_test, y_test,
            budget=top_k,
            lp_norm=2,
            confidence_threshold=0.95,
            w_margin=1.0,
            w_lp=1.0,
            w_conf=1.0
        )
    elif args.unc_samp == 'hybrid':
        uncertain_indices = hybrid_active_sample_selection(
            model, X_labeled, y_labeled, X_test,
            budget=top_k,
            lp_norm=2,
            confidence_threshold=0.95,
            batch_size=512
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


# ---------- Active learning step considering low-confidence samples separated----------
# def active_learning_step(args, model, X_labeled, y_labeled, X_test, y_test, top_k=200, confidence_threshold=0.7):
#     """
#     Performs one step of active learning:
#     1. Filters low-confidence test samples.
#     2. Selects top-k most uncertain samples from them.
#     3. Moves those to labeled set.
#     4. Puts the rest (including high-confidence) into the unlabeled set.
#     """
#     device = X_labeled.device

#     # Step 0: Get low-confidence sample indices
#     low_conf_indices, high_conf_indices, confidences = get_low_confidence_indices(model, X_test, threshold=0.98)
#     print(f"Found {len(low_conf_indices)} low-confidence samples out of {X_test.size(0)} total samples.")
#     print(f"Found {len(high_conf_indices)} high-confidence samples out of {X_test.size(0)} total samples.")
#     if low_conf_indices.numel() == 0:
#         print("No low-confidence samples found.")
#         return X_labeled, y_labeled, X_test, y_test

#     # Step 1: Filter low-confidence test set
#     X_test_lowconf = X_test[low_conf_indices]
#     y_test_lowconf = y_test[low_conf_indices]

#     X_test_highconf = X_test[high_conf_indices]
#     y_test_highconf = y_test[high_conf_indices]

#     # how many high confident samples are misclassified?
#     with torch.no_grad():
#         logits = model(X_test_lowconf.to(device))
#         logits_highconf = model(X_test_highconf.to(device))
#         preds = logits.argmax(dim=1).cpu()
#         preds_highconf = logits_highconf.argmax(dim=1).cpu()
#         true_labels = y_test_lowconf.cpu()
#         true_labels_highconf = y_test_highconf.cpu()
#         num_misclassified = (preds != true_labels).sum().item()
#         num_misclassified_highconf = (preds_highconf != true_labels_highconf).sum().item()
#         print(f"Low confidence samples misclassified: {num_misclassified} out of {len(y_test_lowconf)}")
#         print(f"High confidence samples misclassified: {num_misclassified_highconf} out of {len(y_test_highconf)}")

#     # Step 2: Get most uncertain from low-confidence subset
#     # NOTE: either use uncertainty sampling or boundary selection
#     # uncertain_indices_in_lowconf, _ = get_uncertain_samples(
#     #     model, X_labeled, X_test_lowconf, p=args.lp, top_k=top_k
#     # )
    
#     # low confident boundary sample
#     # uncertain_indices_in_lowconf, _ = select_boundary_samples(
#     #     model, X_test_lowconf, y_test_highconf, top_k=top_k, batch_size=512
#     # )
#     # print(f"Selected {len(uncertain_indices_in_lowconf)} uncertain samples from low-confidence subset.")

#     # high confidence boundary sample
#     uncertain_indices_in_highconf, _ = select_boundary_samples(
#         model, X_test_highconf, y_test_highconf, top_k=top_k, batch_size=512
#     )
#     print(f"Selected {len(uncertain_indices_in_highconf)} uncertain samples from low-confidence subset.")

#     # Step 3: Map selected uncertain indices back to X_test
#     final_selected_indices = high_conf_indices[uncertain_indices_in_highconf]

#     # Step 4: Create mask to keep remaining samples in unlabeled pool
#     remaining_mask = torch.ones(X_test.size(0), dtype=torch.bool, device=device)
#     remaining_mask[final_selected_indices] = False

#     # Step 5: Update labeled and unlabeled sets
#     X_new_labeled = X_test[final_selected_indices]
#     y_new_labeled = y_test[final_selected_indices]

#     X_unlabeled = X_test[remaining_mask]
#     y_unlabeled = y_test[remaining_mask]  # Optional, for evaluation

#     X_labeled = torch.cat([X_labeled, X_new_labeled], dim=0)
#     y_labeled = torch.cat([y_labeled, y_new_labeled], dim=0)

#     return X_labeled, y_labeled, X_unlabeled, y_unlabeled




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

def append_to_strategy(s):
    global strategy
    strategy += s

def validation_function(args, model, val_st_year, val_st_month, val_end_year, val_end_month, batch_size=512):
    """
    Validation function to evaluate the model on a validation set.
    """
    val_metrics_list = []
    model.eval()
    
    for year in range(val_st_year, val_end_year + 1):
        for month in range(val_st_month, val_end_month+1):
            try:
                with torch.no_grad():
                    data = np.load(f"{path}{val_st_year}-{month:02d}_selected.npz")
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
                    val_metrics_list.append(metrics)

                    print(f"Year {year_month}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

            except FileNotFoundError:
                continue


# === Main Training Function for FixMatch with AL and taking best loss model weight===
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, DistributedSampler
from sklearn.metrics import pairwise_distances

def create_triplets(embeddings, labels):
    """
    From a batch of embeddings and labels, create triplet (anchor, positive, negative) tensors.
    """
    anchors, positives, negatives = [], [], []
    labels = labels.cpu().numpy()
    for i in range(len(labels)):
        anchor = embeddings[i]
        label = labels[i]

        pos_indices = [j for j in range(len(labels)) if labels[j] == label and j != i]
        neg_indices = [j for j in range(len(labels)) if labels[j] != label]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        pos = embeddings[np.random.choice(pos_indices)]
        neg = embeddings[np.random.choice(neg_indices)]

        anchors.append(anchor)
        positives.append(pos)
        negatives.append(neg)

    if len(anchors) == 0:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(labels.shape[0]).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss



def active_learning_fixmatch(
    bit_flip, model, optimizer, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
    args, num_classes=2, threshold=0.95, lambda_u=1.0, epochs=200, retrain_epochs=70, batch_size=512,
    al_batch_size=512, margin=1.0
):
    labeled_ds = TensorDataset(X_labeled, y_labeled)
    unlabeled_ds = TensorDataset(X_unlabeled, y_unlabeled)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_loader = DataLoader(labeled_ds, sampler=train_sampler(labeled_ds), batch_size=batch_size, drop_last=True)
    # labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)

    unlabeled_loader = DataLoader(unlabeled_ds, sampler=train_sampler(unlabeled_ds), batch_size=batch_size, drop_last=True)

    criterion = nn.CrossEntropyLoss(reduction='mean')

    supcon_loss_fn = SupConLoss(temperature=0.07)
    lambda_supcon = 0.5  # You can tune this


    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, epochs)
    
    best_loss = float('inf')
    best_state_dict = None

    mu = 1  # FixMatch default
    interleave_size = 2 * mu + 1
    lambda_triplet = 1
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for _ in range(len(labeled_loader)):
            try:
                x_l, y_l = next(labeled_iter)
                x_u, y_u = next(unlabeled_iter)
            except StopIteration:
                break

            x_l, y_l= x_l.cuda(), y_l.cuda()
            x_u, y_u = x_u.cuda(), y_u.cuda()

            # Weak and strong augmentations
            if args.aug == "random_bit_flip":
                x_l = random_bit_flip(x_l, n_bits=5)
                x_u_w = random_bit_flip(x_u, n_bits=5)
                x_u_s = random_bit_flip(x_u, n_bits=bit_flip)
            elif args.aug == "random_bit_flip_bernoulli":
                x_l = random_bit_flip_bernoulli(x_l, p=0.01)
                x_u_w = random_bit_flip_bernoulli(x_u, p=0.01)
                x_u_s = random_bit_flip_bernoulli(x_u, p=0.05)
            elif args.aug == "random_feature_mask":
                x_l = random_feature_mask(x_l, n_mask=5)
                x_u_w = random_feature_mask(x_u, n_mask=5)
                x_u_s = random_feature_mask(x_u, n_mask=bit_flip)
            elif args.aug == "random_bit_flip_and_mask":
                x_l = random_bit_flip_and_mask(x_l, n_bits=2, n_mask=2)
                x_u_w = random_bit_flip_and_mask(x_u, n_bits=2, n_mask=2)
                x_u_s = random_bit_flip_and_mask(x_u, n_bits=bit_flip, n_mask=bit_flip)
            else:
                raise ValueError(f"Unknown augmentation function: {args.aug}")
            # x_l = torch.cat([x_l1, x_l2, x_l3], dim=0)
            inputs = torch.cat([x_l, x_u_w, x_u_s], dim=0)
            logits = model(inputs)
            batch_size = x_l.shape[0]
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            loss_x = criterion(logits_x, y_l)

            # ====== Unsupervised Loss ======
            with torch.no_grad():
                pseudo_logits = F.softmax(logits_u_w / args.T, dim=1)
                pseudo_labels = torch.argmax(pseudo_logits, dim=1)
                max_probs, _ = torch.max(pseudo_logits, dim=1)
                mask = max_probs.ge(threshold).float()

            loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()

            # X_all = torch.cat([x_l, x_u], dim=0)
            # y_all = torch.cat([y_l, y_u], dim=0)
            # selected_indices, _ = select_boundary_samples(model, X_all, y_all, top_k=100)
            # X_boundary = X_all[selected_indices]
            # y_boundary = y_all[selected_indices]
            

            # Final total loss
            loss = loss_x + lambda_u * loss_u
            
            if args.supcon == True:
                # Contrastive loss on labeled encodings
                features = F.normalize(model.encode(x_l), dim=1)
                loss_supcon = supcon_loss_fn(features, y_l)
                loss += lambda_supcon * loss_supcon

            # ====== Total Loss ======
            # loss = loss_x + lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if total_loss < best_loss:
            best_loss = total_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
        scheduler.step()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)


    # Evaluate on validation set
    
    # validation_function(args, model, args.start_year+1, 1, args.start_year, 6, batch_size=al_batch_size)


    # selecting boundary samples
    # X_all = torch.cat([X_labeled, X_unlabeled], dim=0)
    # y_all = torch.cat([y_labeled, y_unlabeled], dim=0)
    # selected_indices, _ = select_boundary_samples(model, X_all, y_all, top_k=30000)
    # X_labeled = X_all[selected_indices]
    # y_labeled = y_all[selected_indices]


    # Active learning loop
    metrics_list = []
    model.eval()
    args.start_year, args.start_month = 2013, 7
    args.end_year, args.end_month = 2018, 12
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            if (year == args.start_year and month < args.start_month) or (year == args.end_year and month > args.end_month):
                continue
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

            if args.al == True:    
                # Select samples based on uncertainty sampling
                X_labeled, y_labeled, X_test, y_test = active_learning_step(
                    args=args,
                    year_month=year_month,
                    model=model,
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
                # labeled_ds = TensorDataset(X_labeled, y_labeled)
                labeled_ds = TensorDataset(X_labeled, y_labeled)


                labeled_loader = DataLoader(labeled_ds, sampler=train_sampler(labeled_ds), batch_size=al_batch_size, drop_last=True)
                unlabeled_loader = DataLoader(unlabeled_ds, sampler=train_sampler(unlabeled_ds), batch_size=al_batch_size, drop_last=True)
                criterion = nn.CrossEntropyLoss(reduction='mean')

                supcon_loss_fn = SupConLoss(temperature=0.07)
                lambda_supcon = 0.5  # You can tune this




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
                            x_l = random_bit_flip(x_l, n_bits=5)
                            x_u_w = random_bit_flip(x_u, n_bits=5)
                            x_u_s = random_bit_flip(x_u, n_bits=bit_flip)
                        elif args.aug == "random_bit_flip_bernoulli":
                            x_l = random_bit_flip_bernoulli(x_l, p=0.01)
                            x_u_w = random_bit_flip_bernoulli(x_u, p=0.01)
                            x_u_s = random_bit_flip_bernoulli(x_u, p=0.05)
                        elif args.aug == "random_feature_mask":
                            x_l = random_feature_mask(x_l, n_mask=5)
                            x_u_w = random_feature_mask(x_u, n_mask=5)
                            x_u_s = random_feature_mask(x_u, n_mask=bit_flip)
                        elif args.aug == "random_bit_flip_and_mask":
                            x_l = random_bit_flip_and_mask(x_l, n_bits=2, n_mask=2)
                            x_u_w = random_bit_flip_and_mask(x_u, n_bits=2, n_mask=2)
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


                        # Standard FixMatch loss on high-confidence pseudo-labels
                        loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()


                        # Final total loss
                        loss = loss_x + lambda_u * loss_u

                        if args.supcon == True:
                            # Contrastive loss on labeled encodings
                            features = F.normalize(model.encode(x_l), dim=1)
                            loss_supcon = supcon_loss_fn(features, y_l)
                            loss += lambda_supcon * loss_supcon


                        # Final total loss
                        # loss = loss_x + lambda_u * loss_u


                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    scheduler.step()
                    print(f"Epoch {epoch+1}: loss={total_loss:.4f}")

    # Save results to CSV
    metrics_df = pd.DataFrame(metrics_list)
    save_metrics = os.path.join(args.save_path, f"{strategy}_active.csv")
    metrics_df.to_csv(save_metrics, index=False)
    print(f"Mean F1 Scores: {metrics_df['f1'].mean():.4f}")
    print(f"Mean False Negative Rates: {metrics_df['fnr'].mean()}")
    print(f"Mean False Positive Rates: {metrics_df['fpr'].mean()}")
    save_plots = os.path.join(args.save_path, f"{strategy}_active_plots.png")
    plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], save_path=save_plots)
    return metrics_df


# === Plotting Function ===
import matplotlib.pyplot as plt

def plot_f1_fnr(years, f1s, fnrs, save_path="f1_fnr_fixmatch_baseline_with_al.png"):
    # Convert to list if Series
    years = list(years)
    f1s = list(f1s)
    fnrs = list(fnrs)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Year")
    # ax1.set_ylabel("F1 Score", color="blue")
    ax1.plot(years, f1s, color="blue", label="F1 Score")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, 1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    # ax2.set_ylabel("False Negative Rate (FNR)", color="red")
    ax2.plot(years, fnrs, color="red", label="FNR")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)

    # Set only year (4-digit) on x-axis, sampled to reduce overlap
    xtick_positions = []
    xtick_labels = []
    seen_years = set()
    for idx, ym in enumerate(years):
        year = ym.split("-")[0]
        if year not in seen_years:
            xtick_positions.append(idx)
            xtick_labels.append(year)
            seen_years.add(year)

    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_labels, rotation=0)

    # Add legend above plot to avoid overlapping x-label
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.show()

    # plt.savefig("f1_fnr_plot.png")

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
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--lp', default=2.0, type=float, help='Lp norm for uncertainty sampling (e.g., 1.0, 2.0, 2.5)')
    parser.add_argument('--budget', default=200, type=int, help='Budget for active learning (number of samples to select)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--retrain_epochs', default=70, type=int, help='Number of retraining epochs after initial training')
    parser.add_argument('--save_path', type=str, default='results/al_uc/', help='Path to save results')
    # parse arugments for uncertainty sampling option 1. lp-norm, 2. boundary selection 3. hybrid
    parser.add_argument('--unc_samp', type=str, default='lp-norm', choices=['lp-norm', 'boundary', 'priority', 'hybrid'], help='Uncertainty sampling method to use')
    parser.add_argument('--lambda_supcon', default=0.5, type=float, help='Coefficient of supervised contrastive loss.')
    parser.add_argument("--strategy", type=str, default="_", help="any strategy (keywork) to use")
    parser.add_argument('--al', action='store_true', help='Enable Active Learning (default: False)')
    # Time window arguments
    parser.add_argument('--start_year', type=int, default=2013, help='Start year for testing (e.g., 2013)')
    parser.add_argument('--start_month', type=int, default=7, help='Start month for testing (e.g., 7 for July)')
    parser.add_argument('--end_year', type=int, default=2018, help='End year for testing (e.g., 2018)')
    parser.add_argument('--end_month', type=int, default=12, help='End month for testing (e.g., 12 for December)')
    parser.add_argument('--supcon', action='store_true', help='Enable Supervised Contrastive loss (default: False)')
    
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
    append_to_strategy(f"_{args.strategy}")
    append_to_strategy(f"_{args.budget}")
    append_to_strategy(f"_lp_{args.lp}")
    append_to_strategy(f"_unc_samp_{args.unc_samp}")
    

    print(f"Running {strategy}...")
    print(f"Using {n_bit_flip} bits to flip per sample.")

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_labeled, y_labeled, X_unlabeled, y_unlabeled = split_labeled_unlabeled(X, y, labeled_ratio=args.labeled_ratio, random_state=args.seed)

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()
    y_2012_unlabeled = torch.tensor(y_unlabeled, dtype=torch.long).cuda()


    input_dim = X_2012_labeled.shape[1]
    num_classes = len(torch.unique(y_2012_labeled))



    model = Classifier(input_dim=input_dim, num_classes=num_classes).cuda()
    # model = ClassifierWB(input_dim=input_dim, num_classes=num_classes).cuda()
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
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
        y_2012_unlabeled,
        args,
        num_classes=num_classes
    )
    end_time = time.time()
    print(f"Time taken to run the function: {end_time - start_time:.2f} seconds")
