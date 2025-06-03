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
import pandas as pd
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


strategy = "fixmatch_bit_flip_wo_al_bit_flip_"

# === Classifier Definition ===
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# === Data Split Function ===
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

# === FixMatch + Drift-Aware Evaluation ===
def random_bit_flip(x, n_bits=1):
    """
    Randomly flip n_bits in each sample of the batch.
    Args:
        x: Tensor of shape (batch_size, num_features)
        n_bits: Number of bits (features) to flip per sample
    Returns:
        Augmented tensor with bits flipped
    """
    x_aug = x.clone()
    batch_size, num_features = x.shape
    for i in range(batch_size):
        flip_indices = torch.randperm(num_features)[:n_bits]
        x_aug[i, flip_indices] = 1 - x_aug[i, flip_indices]
    return x_aug

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def train_fixmatch_drift_eval(
    bit_flip, model, optimizer, X_labeled, y_labeled, X_unlabeled, test_sets_by_year,
    num_classes=2, threshold=0.85, lambda_u=1.0, epochs=200, batch_size=64
):
    labeled_ds = TensorDataset(X_labeled, y_labeled)
    unlabeled_ds = TensorDataset(X_unlabeled)
    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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

            # Weak and strong augmentations for unlabeled data
            x_u_w = random_bit_flip(x_u, n_bits=1)
            x_u_s = random_bit_flip(x_u, n_bits=bit_flip)

            # Interleave all inputs for batchnorm consistency
            inputs = torch.cat([x_l, x_u_w, x_u_s], dim=0)
            inputs = interleave(inputs, interleave_size)

            logits = model(inputs)
            logits = de_interleave(logits, interleave_size)

            batch_size = x_l.shape[0]
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            # Labeled loss
            loss_x = criterion(logits_x, y_l)

            # Unlabeled loss (FixMatch pseudo-labeling)
            with torch.no_grad():
                pseudo_logits = F.softmax(logits_u_w, dim=1)
                pseudo_labels = torch.argmax(pseudo_logits, dim=1)
                max_probs, _ = torch.max(pseudo_logits, dim=1)
                mask = max_probs.ge(threshold).float()

            loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()
            loss = loss_x + lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
        scheduler.step()
    # === Evaluate on each year's test set ===

    metrics_list = []
    model.eval()
    with torch.no_grad():
        for year, (X_test, y_test) in test_sets_by_year.items():
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

            metrics_list.append({
                'year': year,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'fnr': fnr,
                'fpr': fpr,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            })

            print(f"Year {year}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    # Save results to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f"results/{strategy+str(bit_flip)}.csv", index=False)

    print(f"Mean F1 Scores: {metrics_df['f1'].mean():.4f}")
    print(f"Mean False Negative Rates: {metrics_df['fnr'].mean()}")
    print(f"Mean False Positive Rates: {metrics_df['fpr'].mean()}")
    plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], save_path=f"results/{strategy+str(bit_flip)}_f1_fnr_plot.png")

    # Optionally, update your plotting function to use metrics_df if you want to plot other metrics.

# === Plotting Function ===
import matplotlib.pyplot as plt

def plot_f1_fnr(years, f1s, fnrs, save_path="f1_fnr_fixmatch_baseline_with_al.png"):
    # Convert to list if Series
    years = list(years)
    f1s = list(f1s)
    fnrs = list(fnrs)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("F1 Score", color="blue")
    ax1.plot(years, f1s, color="blue", label="F1 Score")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, 1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel("False Negative Rate (FNR)", color="red")
    ax2.plot(years, fnrs, color="red", label="FNR")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)

    # Set only year (4-digit) on x-axis, sampled to reduce overlap
    xtick_positions = []
    xtick_labels = []
    seen_years = set()
    for idx, ym in enumerate(years):
        year = ym.split("_")[0]
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
    


    print(f"Running {strategy}...")
    # Load data
    path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
    file_path = f"{path}2012-01to2012-12_selected.npz"
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    y = np.array([0 if label == 0 else 1 for label in y])
    
    import sys
    if len(sys.argv) > 1:
        n_bit_flip = int(sys.argv[1])
    else:
        n_bit_flip = X.shape[1] // 100  # Default to 1% of features
    print(f"Using {n_bit_flip} bits to flip per sample.")

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_ratio=0.4)

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()


    input_dim = X_2012_labeled.shape[1]
    num_classes = len(torch.unique(y_2012_labeled))

    test_sets_by_year = {}
    for year in range(2013, 2019):
        for month in range(1, 13):
            try:
                data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                X_raw = data["X_train"]
                y_true = (data["y_train"] > 0).astype(int)
                # X_scaled = scaler.transform(X_raw)
                X_tensor = torch.tensor(X_raw, dtype=torch.float32).cuda()
                y_tensor = torch.tensor(y_true, dtype=torch.long).cuda()
                test_sets_by_year[f"{year}_{month}"] = (X_tensor, y_tensor)
            except FileNotFoundError:
                continue

    model = Classifier(input_dim=input_dim, num_classes=num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_fixmatch_drift_eval(
        n_bit_flip,
        model,
        optimizer,
        X_2012_labeled,
        y_2012_labeled,
        X_2012_unlabeled,
        test_sets_by_year,
        num_classes=num_classes
    )
    # print(f"Mean F1 Scores: {sum(f1_scores.values())/len(f1_scores)}")
    # print(f"Mean False Negative Rates: {sum(fnrs.values())/len(fnrs)}")
    # plot_f1_fnr(f1_scores, fnrs)

