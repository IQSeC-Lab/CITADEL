import math
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import random
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)



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