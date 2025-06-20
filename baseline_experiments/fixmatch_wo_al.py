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

# define strategy as global variable
strategy = ""


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
        )
    def forward(self, x):
        return self.classifier(self.encoder(x))
    def encode(self, x):
        """Encode input features to a lower-dimensional representation."""
        return self.encoder(x)

# --- Utility Functions ---
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def split_labeled_unlabeled(X, y, labeled_ratio=0.1, stratify=True, random_state=42):
    n_samples = len(X)
    n_labeled = int(n_samples * labeled_ratio)
    if stratify:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=n_labeled, stratify=y, random_state=random_state)
    else:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=n_labeled, random_state=random_state)
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled

# --- Augmentation Functions (Unchanged) ---
def random_bit_flip(x, n_bits=1):
    x_aug = x.clone()
    batch_size, num_features = x.shape
    for i in range(batch_size):
        flip_indices = torch.randperm(num_features, device=x.device)[:n_bits]
        x_aug[i, flip_indices] = 1 - x_aug[i, flip_indices]
    return x_aug

def random_bit_flip_bernoulli(x, p=None, n_bits=None):
    x_aug = x.clone()
    batch_size, num_features = x.shape
    device = x.device
    if n_bits is not None: p = float(n_bits * 2) / num_features
    elif p is None: p = 0.01
    flip_mask = torch.bernoulli(torch.full_like(x_aug, p, device=device))
    return torch.abs(x_aug - flip_mask)

def random_feature_mask(x, n_mask=1):
    x_aug = x.clone()
    batch_size, num_features = x.shape
    for i in range(batch_size):
        mask_indices = torch.randperm(num_features, device=x.device)[:n_mask]
        x_aug[i, mask_indices] = 0
    return x_aug

def random_bit_flip_and_mask(x, n_bits=1, n_mask=1):
    x_aug = random_bit_flip(x, n_bits=n_bits)
    return random_feature_mask(x_aug, n_mask=n_mask)


# === CORRECTION 1: Replace with correct, functional interleave/de-interleave logic ===
def interleave(tensors_list):
    """Correctly interleaves a list of tensors for BatchNorm."""
    batch_size = tensors_list[0].shape[0]
    # Get all dimensions after the first (batch) dimension
    shape = list(tensors_list[0].shape[1:])
    # Stack along a new dimension, then reshape to interleave
    return torch.stack(tensors_list, dim=1).reshape(len(tensors_list) * batch_size, *shape)

def de_interleave(interleaved_tensor, num_groups):
    """Correctly de-interleaves a tensor back into groups."""
    batch_size = interleaved_tensor.shape[0] // num_groups
    # Get all dimensions after the first (batch) dimension
    shape = list(interleaved_tensor.shape[1:])
    # Reshape to group, then unbind to create a list of tensors
    return interleaved_tensor.reshape(batch_size, num_groups, *shape).unbind(dim=1)


def train_fixmatch_drift_eval(
    bit_flip, model, optimizer, X_labeled, y_labeled, X_unlabeled,
    args, num_classes=2, threshold=0.95, lambda_u=1.0, epochs=200, batch_size=64
):
    labeled_ds = TensorDataset(X_labeled, y_labeled)
    unlabeled_ds = TensorDataset(X_unlabeled)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(labeled_ds, sampler=train_sampler(labeled_ds), batch_size=batch_size, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_ds, sampler=train_sampler(unlabeled_ds), batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, epochs)

    # === CORRECTION 2: Use the model from the final epoch for more stable results ===
    # Saving based on the lowest training loss can be noisy and lead to selecting an overfit model.
    # Using the model state at the end of training is a more robust practice.
    # (The old best_state_dict logic is removed)

    for epoch in range(epochs):
        model.train()
        total_loss, total_loss_x, total_loss_u = 0, 0, 0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for batch_idx in range(len(labeled_loader)):
            try:
                x_l, y_l = next(labeled_iter)
                (x_u,) = next(unlabeled_iter)
            except StopIteration:
                break

            x_l, y_l, x_u = x_l.cuda(), y_l.cuda(), x_u.cuda()

            if args.aug == "random_bit_flip":
                x_u_w = random_bit_flip(x_u, n_bits=1)
                x_u_s = random_bit_flip(x_u, n_bits=bit_flip)
            elif args.aug == "random_bit_flip_bernoulli":
                x_u_w = random_bit_flip_bernoulli(x_u, p=0.01, n_bits=1)
                x_u_s = random_bit_flip_bernoulli(x_u, p=0.05, n_bits=args.n_bits)
            elif args.aug == "random_feature_mask":
                x_u_w = random_feature_mask(x_u, n_mask=1)
                x_u_s = random_feature_mask(x_u, n_mask=bit_flip)
            elif args.aug == "random_bit_flip_and_mask":
                x_u_w = random_bit_flip_and_mask(x_u, n_bits=1, n_mask=1)
                x_u_s = random_bit_flip_and_mask(x_u, n_bits=bit_flip, n_mask=bit_flip)
            else:
                raise ValueError(f"Unknown augmentation function: {args.aug}")

            # === CORRECTION 1 (continued): Apply interleaving for BatchNorm stability ===
            # The simple `torch.cat` is replaced with the interleave/de-interleave process
            inputs = interleave([x_l, x_u_w, x_u_s])
            logits = model(inputs)
            logits_x, logits_u_w, logits_u_s = de_interleave(logits, 3)

            # Labeled loss
            loss_x = criterion(logits_x, y_l)

            # Unlabeled loss (FixMatch pseudo-labeling)
            with torch.no_grad():
                pseudo_logits = F.softmax(logits_u_w / args.T, dim=1)
                max_probs, pseudo_labels = torch.max(pseudo_logits, dim=1)
                mask = max_probs.ge(threshold).float()

            loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()
            loss = loss_x + lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_u += loss_u.item()

        avg_loss = total_loss / len(labeled_loader)
        avg_loss_x = total_loss_x / len(labeled_loader)
        avg_loss_u = total_loss_u / len(labeled_loader)
        print(f"Epoch {epoch+1}/{epochs}: Avg Loss={avg_loss:.4f} (Labeled: {avg_loss_x:.4f}, Unlabeled: {avg_loss_u:.4f})")
        scheduler.step()

    # (No need to restore best model, we just use the final model state)

    # === Evaluate on each year's test set ===
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for year in range(2013, 2019):
            for month in range(1, 13):
                try:
                    data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                    X_raw = data["X_train"]
                    y_true_np = (data["y_train"] > 0).astype(int)

                    # === CORRECTION 3: Use a DataLoader for evaluation to prevent OOM errors ===
                    test_ds = TensorDataset(torch.tensor(X_raw, dtype=torch.float32),
                                            torch.tensor(y_true_np, dtype=torch.long))
                    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

                    all_preds, all_probs, all_y_true = [], [], []
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.cuda()
                        
                        logits = model(X_batch)
                        # Ensure softmax is always used for multi-class probability calculation
                        probs = torch.softmax(logits, dim=1)
                        preds = logits.argmax(dim=1)
                        
                        all_preds.append(preds.cpu())
                        all_probs.append(probs.cpu())
                        all_y_true.append(y_batch)

                    y_pred = torch.cat(all_preds).numpy()
                    y_score_tensor = torch.cat(all_probs)
                    y_true = torch.cat(all_y_true).numpy()
                    
                    if y_score_tensor.shape[1] == 2:
                        y_score = y_score_tensor[:, 1].numpy()
                    else: # For multi-class
                        y_score = y_score_tensor.numpy()

                    # --- Metric calculations (unchanged, but now using aggregated results) ---
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    cm = confusion_matrix(y_true, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    else: fnr, fpr = float('nan'), float('nan')
                    
                    try:
                        if y_score_tensor.shape[1] == 2:
                            roc_auc = roc_auc_score(y_true, y_score)
                            pr_auc = average_precision_score(y_true, y_score)
                        else:
                            roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
                            pr_auc = average_precision_score(y_true, y_score, average='weighted')
                    except Exception: roc_auc, pr_auc = float('nan'), float('nan')

                    metrics_list.append({'year': f"{year}_{month}", 'accuracy': acc, 'precision': prec,
                                         'recall': rec, 'f1': f1, 'fnr': fnr, 'fpr': fpr, 'roc_auc': roc_auc, 'pr_auc': pr_auc})
                    print(f"Year {year}_{month}: Acc={acc:.4f}, F1={f1:.4f}, FNR={fnr:.4f}")

                except FileNotFoundError:
                    continue

    metrics_df = pd.DataFrame(metrics_list)
    # The global strategy variable should be set in the main block
    csv_path = f"results/baseline_2runs/{strategy}.csv"
    plot_path = f"results/baseline_2runs/{strategy}_f1_fnr_plot.png"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print(f"\nMean F1 Score: {metrics_df['f1'].mean():.4f}")
    print(f"Mean FNR: {metrics_df['fnr'].mean():.4f}")
    print(f"Mean FPR: {metrics_df['fpr'].mean():.4f}")
    
    plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], save_path=plot_path)

# === Plotting Function (Unchanged from your version) ===
def plot_f1_fnr(years, f1s, fnrs, save_path): # Pass save_path as an argument
    years, f1s, fnrs = list(years), list(f1s), list(fnrs)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Year-Month")
    ax1.set_ylabel("F1 Score", color="blue", weight='bold')
    ax1.plot(years, f1s, color="blue", marker='o', linestyle='-', label="F1 Score")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, 1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("False Negative Rate (FNR)", color="red", weight='bold')
    ax2.plot(years, fnrs, color="red", marker='x', linestyle='--', label="FNR")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)
    
    xtick_positions = np.arange(0, len(years), max(1, len(years)//12)) # Sample ticks to prevent overlap
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels([years[i] for i in xtick_positions], rotation=45, ha="right")
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close() # Close plot to free memory

# === Main Execution ===
if __name__ == "__main__":
    import argparse
    import random
    print(f"Process started with PID: {os.getpid()}", flush=True)

    parser = argparse.ArgumentParser(description="Run FixMatch with Bit Flip Augmentation on a 1D-CNN")
    parser.add_argument("--bit_flip", type=int, default=11, help="Number of bits to flip per sample")
    parser.add_argument("--labeled_ratio", type=float, default=0.4, help="Ratio of labeled data")
    parser.add_argument("--aug", type=str, default="random_bit_flip", help="Augmentation function to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    # Use the global strategy variable
    strategy = f"fixmatch_cnn_{args.aug}_{n_bit_flip}_lbr_{labeled_ratio}_seed_{args.seed}"
    print(f"Running strategy: {strategy}...")
    print(f"Hyperparameters: {vars(args)}")

    X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_ratio=labeled_ratio, random_state=args.seed)

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32)
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long)
    X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32)

    input_dim = X_2012_labeled.shape[1]
    num_classes = len(np.unique(y_labeled))

    
    model = Classifier(input_dim=input_dim, num_classes=num_classes).cuda()
    # model = CNNClassifier(input_dim=input_dim, num_classes=num_classes).cuda()

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    train_fixmatch_drift_eval(
        n_bit_flip,
        model,
        optimizer,
        X_2012_labeled,
        y_2012_labeled,
        X_2012_unlabeled,
        args,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size
    )