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
import os
import math
import argparse
import random
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from utils import *
from augmentations import *
from sample_selectors import (
    get_low_confidence_indices, get_uncertain_samples,
    select_boundary_samples, prioritized_uncertainty_selection
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
    

# def interleave(x, size):
#     s = list(x.shape)
#     return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# def de_interleave(x, size):
#     s = list(x.shape)
#     return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])



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



def append_to_strategy(s):
    global strategy
    strategy += s


def active_learning_fixmatch(
    model, optimizer, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
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

    # mu = 1  # FixMatch default
    # interleave_size = 2 * mu + 1
    # lambda_triplet = 1
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
                x_u_s = random_bit_flip(x_u, n_bits=args.bit_flip)
            elif args.aug == "random_bit_flip_bernoulli":
                x_l = random_bit_flip_bernoulli(x_l, p=0.01)
                x_u_w = random_bit_flip_bernoulli(x_u, p=0.01)
                x_u_s = random_bit_flip_bernoulli(x_u, p=0.05)
            elif args.aug == "random_feature_mask":
                x_l = random_feature_mask(x_l, n_mask=5)
                x_u_w = random_feature_mask(x_u, n_mask=5)
                x_u_s = random_feature_mask(x_u, n_mask=args.bit_flip)
            elif args.aug == "random_bit_flip_and_mask":
                x_l = random_bit_flip_and_mask(x_l, n_bits=2, n_mask=2)
                x_u_w = random_bit_flip_and_mask(x_u, n_bits=2, n_mask=2)
                x_u_s = random_bit_flip_and_mask(x_u, n_bits=args.bit_flip, n_mask=args.bit_flip)
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
    # selected_indices, _ = select_boundary_samples(model, X_all, y_all, top_k=10000)
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
                            x_u_s = random_bit_flip(x_u, n_bits=args.bit_flip)
                        elif args.aug == "random_bit_flip_bernoulli":
                            x_l = random_bit_flip_bernoulli(x_l, p=0.01)
                            x_u_w = random_bit_flip_bernoulli(x_u, p=0.01)
                            x_u_s = random_bit_flip_bernoulli(x_u, p=0.05)
                        elif args.aug == "random_feature_mask":
                            x_l = random_feature_mask(x_l, n_mask=5)
                            x_u_w = random_feature_mask(x_u, n_mask=5)
                            x_u_s = random_feature_mask(x_u, n_mask=args.bit_flip)
                        elif args.aug == "random_bit_flip_and_mask":
                            x_l = random_bit_flip_and_mask(x_l, n_bits=2, n_mask=2)
                            x_u_w = random_bit_flip_and_mask(x_u, n_bits=2, n_mask=2)
                            x_u_s = random_bit_flip_and_mask(x_u, n_bits=args.bit_flip, n_mask=args.bit_flip)
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



# === Main Execution ===
if __name__ == "__main__":

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
    parser.add_argument('--budget', default=400, type=int, help='Budget for active learning (number of samples to select)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--retrain_epochs', default=70, type=int, help='Number of retraining epochs after initial training')
    parser.add_argument('--save_path', type=str, default='results/CITADEL/', help='Path to save results')
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
    parser.add_argument('--dataset', type=str.lower, default='apigraph', choices=['apigraph', 'chen-androzoo', 'lamda'], \
        help='Dataset to use: apigraph (2012–2018), chen-androzoo (2019–2021), or lamda (2013–2025).')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to the dataset.')

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
    if args.dataset == 'apigraph':
        path = os.path.join(args.data_dir, '2012-01to2012-12_selected.npz')
        # file_path = f"{args.data_dir}2012-01to2012-12_selected.npz"
        data = np.load(path, allow_pickle=True)
        X, y = data['X_train'], data['y_train']
        y = np.array([0 if label == 0 else 1 for label in y])

    elif args.dataset == 'chen-androzoo':
        path = os.path.join(args.data_dir, '2012-01to2012-12_selected.npz')
        # file_path = f"{path}2019-01to2021-12_selected.npz"
        data = np.load(path, allow_pickle=True)
        X, y = data['X_train'], data['y_train']
        y = np.array([0 if label == 0 else 1 for label in y])
    elif args.dataset == 'lamda':
        path_X = os.path.join(args.data_dir, '2013_X_train.npz')
        X_train = load_npz(path_X)
        X = X_train.toarray()
        path_y = os.path.join(args.data_dir, '2013_meta_train.npz')
        y = np.load(path_y, allow_pickle=True)['y']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    # path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
    # file_path = f"{path}2012-01to2012-12_selected.npz"
    # data = np.load(file_path, allow_pickle=True)
    # X, y = data['X_train'], data['y_train']
    # y = np.array([0 if label == 0 else 1 for label in y])
    
    # n_bit_flip = args.bit_flip
    # labeled_ratio = args.labeled_ratio

    strategy = f"CITADEL_fixmatch_" + args.aug + "_" + "_lbr_" + str(args.labeled_ratio) +  "_seed_" + str(args.seed)
    append_to_strategy(f"_{args.strategy}")
    append_to_strategy(f"_{args.budget}")
    append_to_strategy(f"_lp_{args.lp}")
    append_to_strategy(f"_unc_samp_{args.unc_samp}")
    

    print(f"Running {strategy}...")

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_labeled, y_labeled, X_unlabeled, y_unlabeled = split_labeled_unlabeled(X, y, labeled_ratio=args.labeled_ratio, random_state=args.seed)

    X_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    X_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()
    y_unlabeled = torch.tensor(y_unlabeled, dtype=torch.long).cuda()


    input_dim = X_labeled.shape[1]
    num_classes = len(torch.unique(y_labeled))



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
        model,
        optimizer,
        X_labeled,
        y_labeled,
        X_unlabeled,
        y_unlabeled,
        args,
        num_classes=num_classes
    )
    end_time = time.time()
    print(f"Time taken to run the function: {end_time - start_time:.2f} seconds")
