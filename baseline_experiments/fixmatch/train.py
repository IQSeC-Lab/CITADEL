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
import math
import plot_gen
import eval
import utils
from models import MLPClassifier, Classifier
import logging
import binary_smote
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


SEED = 1  # You can set this to any integer you like

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

strategy = "freematch_bit_flip_wo_mal_feat_w1_s4_r1"

class FreeMatch:
    def __init__(self, model_fn, num_classes, ema_momentum, lambda_unsup, \
                  lambda_fairness, ema_decay=0.8):
        self.model = model_fn
        self.ema = utils.EMA(self.model, ema_momentum)
        self.lambda_u = lambda_unsup
        self.lambda_f = lambda_fairness
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
        self.ema_decay = ema_decay  # <-- Add this line

        # Initialize EMA state variables
        self.tau = 0.3
        self.p_tilde = torch.full((self.num_classes,), 0.5, dtype=torch.float32).cuda()
        self.h_tilde = torch.ones(self.num_classes, dtype=torch.float32).cuda()
        self.h_tilde /= self.h_tilde.sum()  # normalize to sum to 1


    def set_data(self, labeled_loader, unlabeled_loader, eval_loader=None):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        # self.eval_loader = eval_loader

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def get_model(self):
        return self.model

    def warmup(self):
        # Train briefly with labeled data to bootstrap model confidence
        for X, y in self.labeled_loader:
            X, y = X.cuda(), y.cuda()
            logits = self.model(X)
            loss = utils.ce_loss(logits, y)
            if loss.dim() > 0:
                loss = loss.mean()  # Ensure scalar loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def model_train(self):
        self.model.train()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        # === Forward Pass with Gradients ===
        logits_lb = self.model(x_lb)
        Ls = F.cross_entropy(logits_lb, y_lb)

        logits_ulb_w = self.model(x_ulb_w)
        probs_ulb_w = torch.softmax(logits_ulb_w, dim=1)
        max_probs, max_idx = probs_ulb_w.max(dim=1)

        logits_ulb_s = self.model(x_ulb_s)
        mask = max_probs >= (self.p_tilde / self.p_tilde.max() * self.tau)[max_idx]
        Lu = F.cross_entropy(logits_ulb_s[mask], max_idx[mask]) if mask.any() else torch.zeros(1, device=x_lb.device, requires_grad=True).sum()

        # === Fairness Loss (no grad required for EMA stats, only Qb and Lf require autograd) ===
        Qb = torch.softmax(logits_ulb_s, dim=1)
        if mask.any():
            p = Qb[mask].mean(dim=0)
            h = torch.bincount(Qb[mask].argmax(dim=1), minlength=self.num_classes).float().to(x_lb.device)
            h = h / h.sum()
        else:
            p = torch.zeros(self.num_classes, device=x_lb.device)
            h = torch.ones(self.num_classes, device=x_lb.device) / self.num_classes

        def sumnorm(x): return x / (x.sum() + 1e-8)
        Lf = -F.kl_div(sumnorm(self.p_tilde / (self.h_tilde + 1e-8)).log(), sumnorm(p / (h + 1e-8)), reduction='batchmean')

        # === Backward Pass ===
        total_loss = Ls + self.lambda_u * Lu + self.lambda_f * Lf
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # === EMA Updates (no gradient tracking needed here) ===
        with torch.no_grad():
            self.tau = self.ema_decay * self.tau + (1 - self.ema_decay) * max_probs.mean().item()
            self.p_tilde = self.ema_decay * self.p_tilde + (1 - self.ema_decay) * probs_ulb_w.mean(dim=0)
            hist = torch.bincount(max_idx, minlength=self.num_classes).float()
            hist = hist / hist.sum()
            if self.h_tilde is None:
                self.h_tilde = hist
            else:
                self.h_tilde = self.ema_decay * self.h_tilde + (1 - self.ema_decay) * hist

        # self.ema.update()

        return total_loss.item(), Ls.item(), Lu.item(), mask.float().mean().item()
    
    @torch.no_grad()
    def update_scheduler(self):
        self.scheduler.step()

    @torch.no_grad()
    def update_model(self, best_state_dict):
        self.model.load_state_dict(best_state_dict)

    @torch.no_grad()
    def update_thresholds(self, logits_ulb):
        probs = torch.softmax(logits_ulb, dim=1)
        max_probs, max_idx = torch.max(probs, dim=1)
        self.time_p = self.time_p * 0.999 + max_probs.mean() * 0.001 if self.time_p is not None else max_probs.mean()
        self.p_model = self.p_model * 0.999 + probs.mean(dim=0) * 0.001 if self.p_model is not None else probs.mean(dim=0)
        hist = torch.bincount(max_idx, minlength=probs.shape[1]).float()
        hist /= hist.sum()
        self.label_hist = self.label_hist * 0.999 + hist * 0.001 if self.label_hist is not None else hist
    
    @torch.no_grad()    
    def cosine_lr(self, step, total_steps, initial_lr):
        return initial_lr * math.cos(7 * math.pi * step / (16 * total_steps))

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        y_true, y_pred, y_scores = [], [], []
        for _, x, y in self.eval_loader:
            x, y = x.cuda(), y.cuda()
            logits = self.model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_scores.extend(torch.softmax(logits, 1).cpu().numpy())
        acc = accuracy_score(y_true, y_pred)
        return {'accuracy': acc}
    
    def get_model_state(self):
        return self.model.state_dict()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


def train_freematch_drift_eval(model, X_labeled, y_labeled, X_unlabeled, \
                               test_sets_by_year, num_classes=2, threshold=0.95, lambda_u=1.0, \
                                  epochs=200, retrain_epochs=50, batch_size=64):
    labeled_ds = TensorDataset(X_labeled, y_labeled)
    unlabeled_ds = TensorDataset(X_unlabeled)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True)


    freeMatch = FreeMatch(model_fn=model, num_classes=2, ema_momentum=0.80, \
                          lambda_unsup=1.0, lambda_fairness=0.05, ema_decay=0.80)   
    freeMatch.ema.register()  # <-- Add this line
    # logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    total_steps = 2**20 #epochs * len(labeled_loader)
    initial_lr = 0.001
    global_step = 0
    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr) #utils.get_optimizer(net=model, optim_name='AdamW', lr=0.001, momentum=0.9, \
                                    # weight_decay=0, nesterov=True, bn_wd_skip=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=2**20)
    
    ## set AdamW and cosine lr on FreeMatch 
    freeMatch.set_optimizer(optimizer, scheduler)
    freeMatch.set_data(labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader)
    freeMatch.warmup()

    best_loss = float('inf')
    best_state_dict = None

    freeMatch.model_train()

    for epoch in tqdm(range(epochs), desc="Training FreeMatch"):
        total_loss = 0
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        total_loss, sup_loss, unsup_loss = 0, 0 ,0

        for _ in tqdm(range(len(labeled_loader)), desc="FreeMatch Applying..."):
            try:
                x_l, y_l = next(labeled_iter)
                (x_u,) = next(unlabeled_iter)
            except StopIteration:
                break

            x_l, y_l = x_l.cuda(), y_l.cuda()
            x_u = x_u.cuda()
            # u_i = torch.arange(x_u.size(0)).to(x_u.device)

            # Apply random bit flip: weak (1 bit), strong (3 bits)
            x_u_w = utils.random_bit_flip(x_u, n_bits=1, fixed_features=False)
            x_u_s = utils.random_bit_flip(x_u, n_bits=4, fixed_features=False)

            # x_u_w = torch.tensor(binary_smote.weak_augment_batch(x_u, 0.05), dtype=x_u.dtype, device=x_u.device)
            # x_u_s = torch.tensor(binary_smote.strong_augment_batch(x_u, 0.20), dtype=x_u.dtype, device=x_u.device)

            # lr = freeMatch.cosine_lr(global_step, total_steps, initial_lr)
            # for param_group in freeMatch.optimizer.param_groups:
            #     param_group['lr'] = lr

            t_loss, s_loss, u_loss, _ = freeMatch.train_step(x_l, y_l, x_u_w, x_u_s)
            # global_step += 1
            total_loss += t_loss
            sup_loss += s_loss
            unsup_loss += u_loss
        
        # global_step += 1
        freeMatch.update_scheduler()

        avg_loss = total_loss/len(labeled_loader)
        avg_sup_loss = sup_loss/len(labeled_loader)
        avg_unsup_loss = unsup_loss/len(labeled_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = {k: v.cpu().clone() for k, v in freeMatch.get_model_state().items()}
        
        logging.info(f"##Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, sup_loss={avg_sup_loss:.4f}, unsup_loss={avg_unsup_loss:.4f}, global_threshold={freeMatch.tau:.4f}")
    
    # freeMatch.scheduler.step()
    # Restore best model after retraining
    if best_state_dict is not None:
        freeMatch.update_model(best_state_dict)

    # === Evaluate on each year's test set ===
    eval.model_evaluate(freeMatch.get_model(), test_sets_by_year, strategy)
    
# === Main Execution ===
if __name__ == "__main__":
    print(f"Running {strategy}...")
    path = "/home/ihossain/ISMAIL/Datasets/data/gen_apigraph_drebin/"
    # Load data
    X_2012_labeled, y_2012_labeled, X_2012_unlabeled = utils.get_dataset(path)

    # input_dim = X_2012_labeled.shape[1]
    NUM_FEATURES = X_2012_labeled.shape[1]
    mlp_hidden = '100-100'
    num_classes = 2
    mlp_dims = utils.get_model_dims('MLP', NUM_FEATURES, mlp_hidden, num_classes)

    test_sets_by_year = {}
    for year in tqdm(range(2013, 2019), desc="Processing years"):
        for month in range(1, 13):
            try:
                data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                X_raw = data["X_train"]
                y_true = (data["y_train"] > 0).astype(int)
                # Filter out the undesired features from the datasets
                X_raw = utils.create_dataset_without_mal_features(X_raw)
                
                X_tensor = torch.tensor(X_raw, dtype=torch.float32).cuda()
                y_tensor = torch.tensor(y_true, dtype=torch.long).cuda()
                test_sets_by_year[f"{year}_{month}"] = (X_tensor, y_tensor)
            except FileNotFoundError:
                continue

    model = MLPClassifier(mlp_dims=mlp_dims).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_freematch_drift_eval(
        model,
        # optimizer,
        X_2012_labeled,
        y_2012_labeled,
        X_2012_unlabeled,
        test_sets_by_year,
        num_classes=num_classes
    )
    # print(f"Mean F1 Scores: {sum(f1_scores.values())/len(f1_scores)}")
    # print(f"Mean False Negative Rates: {sum(fnrs.values())/len(fnrs)}")
    # plot_f1_fnr(f1_scores, fnrs)


# nohup python train.py > /home/ihossain/ISMAIL/SSL-malware/baseline_experiments/freematch/output.log 2>&1 &