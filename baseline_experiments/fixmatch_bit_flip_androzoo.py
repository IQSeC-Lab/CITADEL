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
    # if n_bits is not None:
    #     p = float(n_bits * 10) / num_features
    # else:
    #     if p is None:
    #         p = 0.01  # default
    #     else:
    #         p = float(p)
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

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def append_to_strategy(s):
    global strategy
    strategy += s


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
            p = 0.0
            # Weak and strong augmentations for unlabeled data, now with seed
            if args.aug == "random_bit_flip":
                x_u_w = random_bit_flip(x_u, n_bits=1)
                x_u_s = random_bit_flip(x_u, n_bits=bit_flip)
            elif args.aug == "random_bit_flip_bernoulli":
                p = 0.05
                x_u_w = random_bit_flip_bernoulli(x_u, p=0.01, n_bits=None)
                x_u_s = random_bit_flip_bernoulli(x_u, p=p, n_bits=None)
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

    # === Evaluate on each year's test set ===

    metrics_list = []
    model.eval()
    with torch.no_grad():
        for year in range(2020, 2022):
            for month in range(1, 13):
                try:
                    data = np.load(f"/home/mhaque3/myDir/data/gen_androzoo_drebin/{year}-{month:02d}_selected.npz")
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
                        'year': f"{year}_{month}",
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'fnr': fnr,
                        'fpr': fpr,
                        'roc_auc': roc_auc,
                        'pr_auc': pr_auc
                    })

                    print(f"Year {year}_{month}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

                except FileNotFoundError:
                    continue

    # Save results to CSV
    if p != 0: 
        append_to_strategy(f"_berno_p_{p}")
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f"results/baseline_andro/{strategy}.csv", index=False)

    print(f"Mean F1 Scores: {metrics_df['f1'].mean():.4f}")
    print(f"Mean False Negative Rates: {metrics_df['fnr'].mean()}")
    print(f"Mean False Positive Rates: {metrics_df['fpr'].mean()}")
    plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], save_path=f"results/baseline_andro/{strategy}_f1_fnr_plot.png")
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
    path = "/home/mhaque3/myDir/data/gen_androzoo_drebin/2019-01to2019-12_selected.npz"
    data = np.load(path, allow_pickle=True)
    data.files
    X = data['X_train']
    y = data['y_train']
    print(X.shape, y.shape)

    import argparse
    print(f"Process started with PID: {os.getpid()}", flush=True)
    parser = argparse.ArgumentParser(description="Run FixMatch with Bit Flip Augmentation on MLP")
    parser.add_argument("--bit_flip", type=int, default=11, help="Number of bits to flip per sample")
    parser.add_argument("--labeled_ratio", type=float, default=0.4, help="Ratio of labeled data")
    parser.add_argument("--aug", type=str, default="random_bit_flip_bernoulli", help="Augmentation function to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    args = parser.parse_args()

    # === FIX 1: Complete Seeding for Full Reproducibility ===
    import random
    np.random.seed(args.seed)
    random.seed(args.seed) # <-- FIX: Seed the standard random library
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # These are important for reproducibility on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {args.seed}", flush=True)
    # Load data
    # path = "/home/mhaque3/myDir/data/gen_androzoo_drebin/2019-01to2019-12_selected.npz"
    # data = np.load(path, allow_pickle=True)
    # data.files
    # X = data['X_train']
    # y = data['y_train']
    # print(X.shape, y.shape)
    # print(f"Loading initial training data from: {path}")
    # try:
    #     data = np.load(path, allow_pickle=True)
    #     X, y = data['X_train'], data['y_train']
    # except FileNotFoundError:
    #     print(f"ERROR: Initial training file not found at {path}")
    #     print("Please ensure your data is split into monthly files (e.g., 2019-01_selected.npz).")
    #     exit() # Exit if the initial data isn't available

    y = np.array([0 if label == 0 else 1 for label in y])


    n_bit_flip = args.bit_flip
    labeled_ratio = args.labeled_ratio

    strategy = f"fixmatch_androzoo_wo_al_{args.aug}_{n_bit_flip}_lbr_{labeled_ratio}_seed_{args.seed}"
    print(f"Running {strategy}...")
    print(f"Using {n_bit_flip} bits to flip per sample. Labeled ratio: {labeled_ratio}, Seed: {args.seed}")

    # Use the labeled_ratio argument here!
    X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_ratio=labeled_ratio, random_state=args.seed)

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()

    input_dim = X_2012_labeled.shape[1]
    num_classes = len(torch.unique(y_2012_labeled))


    model = Classifier(input_dim=input_dim, num_classes=num_classes).cuda()

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
        num_classes=num_classes
    )
