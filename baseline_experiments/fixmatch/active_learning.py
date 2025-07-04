import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
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
from models import MLPClassifier, Classifier, SimpleEncClassifier
import logging
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import argparse
import time
import csv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

strategy = "fixmatch_bit_flip_not_mal_feat_w2_s5_r3_ep200_active"

class FixMatch:
    def __init__(self, model_fn, num_classes, lambda_unsup, threshold=0.95):
        self.model = model_fn
        self.lambda_u = lambda_unsup
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.hiDistanceXent = utils.HiDistanceXentLoss().cuda()
        self.num_classes = num_classes
        self.threshold = threshold

    def set_data(self, labeled_loader, unlabeled_loader, eval_loader=None):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        # self.eval_loader = eval_loader

    def set_optimizer(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def get_model(self):
        return self.model
    
    def get_parameters(self, wdecay):
        no_decay = ['bias', 'bn']
        group_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': wdecay},
        {'params': [p for n, p in self.model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return group_parameters

    # def warmup(self):
    #     # Train briefly with labeled data to bootstrap model confidence
    #     for X, y in self.labeled_loader:
    #         X, y = X.cuda(), y.cuda()
    #         logits = self.model(X)
    #         loss = utils.ce_loss(logits, y)
    #         if loss.dim() > 0:
    #             loss = loss.mean()  # Ensure scalar loss
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    def model_train(self):
        self.model.train()

    def cal_loss(self, X_lb, y_lb, y_bin_lb):
        _, cur_f, y_pred = self.model(X_lb)

        # features: hidden vector of shape [bsz, n_feature_dim].
        features = cur_f

        # features = F.normalize(features, p=2, dim=1)
        # weight_batch = weight_batch / weight_batch.sum()

        # # Our own version of the supervised contrastive learning loss
        # loss, _, _ = self.hiDistanceXent(1.0, \
        #                                 y_pred, y_bin_lb, \
        #                                 features, labels = y_lb, \
        #                                 # margin = args.margin, \
        #                                 weight = None)
        loss, lhc_val, lce_val = utils.combined_loss(y_pred, features, y_bin_lb, y_lb, 1, 1)
        return loss, lhc_val, lce_val

    def train_step(self, x_lb, y_lb, y_bin_lb, x_ulb_w, x_ulb_s):
        # === Forward Pass with Gradients ===
        # inputs = torch.cat([x_lb, x_ulb_w, x_ulb_s], dim=0)
        # logits = model(inputs)
        # batch_size = x_lb.shape[0]
        # logits_x = logits[:batch_size]
        # logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        # loss_x = self.criterion(logits_x, y_lb)

        loss_x, lhc_val, lce_val = self.cal_loss(x_lb, y_lb, y_bin_lb)
        
        with torch.no_grad():
            logits_u_w, logits_u_s = self.model(x_ulb_w)[-1], self.model(x_ulb_s)[-1]
            max_probs, _ = torch.max(logits_u_w, dim=1)
            mask = max_probs.ge(self.threshold).float()
            pseudo_labels = (logits_u_w >= 0.5).float()
            loss_u = (F.binary_cross_entropy(logits_u_s, pseudo_labels.detach()) * mask).mean()
        loss = loss_x + self.lambda_u * loss_u
        
        # === Backward Pass ===
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_x.item(), loss_u.item(), lhc_val, lce_val
    
    @torch.no_grad()
    def update_scheduler(self):
        self.scheduler.step()

    @torch.no_grad()
    def update_model(self, best_state_dict):
        self.model.load_state_dict(best_state_dict)


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
    
    def active_learning_loop(self, unlabeled_data, train_embeddings, train_labels, budget=10, margin=1.0, K=63, p_value=2):
        """
        Selects the top 'budget' most uncertain samples based on pseudo contrastive loss.

        Returns:
            selected_indices (List[int])
            uncertainty_scores (List[float])
        """
        # model.eval()
        # scores = []
        # for i in range(unlabeled_data.size(0)):
        #     x = unlabeled_data[i].unsqueeze(0)  # shape [1, input_dim]
        #     score = eval.pseudo_loss_selector(x, self.model, train_embeddings, train_labels, margin, k)
        #     scores.append((i, score))

        # # Sort samples by uncertainty (descending)
        # scores.sort(key=lambda x: x[1], reverse=True)
        # selected_indices = [i for i, _ in scores[:budget]]
        # selected_scores = [s for _, s in scores[:budget]]
        # return selected_indices, selected_scores
        # Step 1: Normalize
        _, unlabeled_embeddings, preds = self.model(unlabeled_data)

        u_emb = F.normalize(unlabeled_embeddings, dim=1)
        t_emb = F.normalize(train_embeddings, dim=1)

        # Step 2: Predict pseudo labels
        pseudo_labels = (preds.view(-1) >= 0.5).long()

        # Step 3: Compute scores for all at once
        scores = eval.faiss_batch_pseudo_loss_selector(u_emb, pseudo_labels, t_emb, train_labels, margin=1.0, k=K, P=p_value)

        # Step 4: Select top-K
        _, top_indices = torch.topk(scores, k=budget, largest=False, sorted=False)
        return top_indices.cpu().numpy()


def train_fixmatch_drift_eval(args, model, X_labeled, y_labeled, y_labeled_binary, X_unlabeled, \
                               test_sets_by_year, num_classes=2, threshold=0.95, lambda_u=1.0, \
                                  epochs=1, retrain_epochs=1, batch_size=128, al_batch_size=128, weight=None):
    
    # if weight is None:
    #     weight_tensor = torch.ones(X_labeled.shape[0])
    # else:
    #     weight_tensor = torch.from_numpy(weight).float()

    # y_labeled_binary_cat_tensor = torch.from_numpy(utils.to_categorical(y_labeled_binary.cpu().numpy())).float()

    labeled_ds = TensorDataset(X_labeled, y_labeled, y_labeled_binary)
    # labeled_ds = TensorDataset(X_labeled, y_labeled)
    unlabeled_ds = TensorDataset(X_unlabeled)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True)
    # labeled_loader = DataLoader(labeled_ds, sampler=train_sampler(labeled_ds), batch_size=batch_size, drop_last=True)
    # unlabeled_loader = DataLoader(unlabeled_ds, sampler=train_sampler(unlabeled_ds), batch_size=batch_size, drop_last=True)

    fixMatch = FixMatch(model_fn=model, num_classes=2,\
                          lambda_unsup=1.0, threshold=0.95)   

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    grouped_parameters = fixMatch.get_parameters(args.wdecay)

    optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    scheduler = utils.get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs)
    
    ## set AdamW and cosine lr on FreeMatch 
    fixMatch.set_optimizer(optimizer, scheduler)
    fixMatch.set_data(labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader)
    # fixMatch.warmup()

    if args.pretrained_model:
        logging.info(f"Loading pretrained model....")
        fixMatch.load_model(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/checkpoints/{args.pretrained_model_path}")
    else:
        logging.info(f"Training from scratch...")
        best_loss = float('inf')
        best_state_dict = None
        loss_log = []

        fixMatch.model_train()

        for epoch in tqdm(range(args.epochs), desc="Training FreeMatch"):
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)

            total_loss, total_loss_sup, total_loss_unsup, total_lhc_val, total_lce_val = 0, 0, 0, 0, 0
            N = len(labeled_loader)
            for _ in tqdm(range(N), desc="FixMatch Applying..."):
                try:
                    x_l, y_l, y_bin_l = next(labeled_iter)
                    (x_u,) = next(unlabeled_iter)
                except StopIteration:
                    break

                x_l, y_l = x_l.cuda(), y_l.cuda()
                y_bin_l = y_bin_l.cuda()
                x_u = x_u.cuda()

                # Weak and strong augmentations for unlabeled data, now with seed
                if args.aug == "random_bit_flip":
                    x_u_w = utils.random_bit_flip(x_u, n_bits=1)
                    x_u_s = utils.random_bit_flip(x_u, n_bits=args.bit_flip)
                elif args.aug == "random_bit_flip_bernoulli":
                    x_u_w = utils.random_bit_flip_bernoulli(x_u, p=0.01, n_bits=None)
                    x_u_s = utils.random_bit_flip_bernoulli(x_u, p=0.05, n_bits=None)
                elif args.aug == "random_feature_mask":
                    x_u_w = utils.random_feature_mask(x_u, n_mask=1)
                    x_u_s = utils.random_feature_mask(x_u, n_mask=args.bit_flip)
                elif args.aug == "random_bit_flip_and_mask":
                    x_u_w = utils.random_bit_flip_and_mask(x_u, n_bits=1, n_mask=1)
                    x_u_s = utils.random_bit_flip_and_mask(x_u, n_bits=args.bit_flip, n_mask=args.bit_flip)
                else:
                    raise ValueError(f"Unknown augmentation function: {args.aug}")

                loss, loss_s, loss_u, lhc_val, lce_val = fixMatch.train_step(x_l, y_l, y_bin_l, x_u_w, x_u_s)

                total_loss += loss
                total_loss_sup += loss_s
                total_loss_unsup += loss_u
                total_lhc_val += lhc_val
                total_lce_val += lce_val

            fixMatch.update_scheduler()

            avg_loss = total_loss/N
            avg_loss_sup = total_loss_sup/N
            avg_loss_unsup = total_loss_unsup/N
            avg_lhc_val = total_lhc_val/N
            avg_lce_val = total_lce_val/N

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state_dict = {k: v.cpu().clone() for k, v in fixMatch.get_model_state().items()}
            
            logging.info(f"##Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, sup_loss={avg_loss_sup:.4f}, "
                         f"unsup_loss={avg_loss_unsup:.4f}, lhc_val={avg_lhc_val:.4f}, lce_val={avg_lce_val:.4f}, best_loss={best_loss:.4f}")

            # Collect losses for this epoch
            loss_log.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'sup_loss': avg_loss_sup,
                'unsup_loss': avg_loss_unsup,
                'lhc_val': avg_lhc_val,
                'lce_val': avg_lce_val
            })

        # freeMatch.scheduler.step()
        # Restore best model after training
        if best_state_dict is not None:
            fixMatch.update_model(best_state_dict)
            fixMatch.save_model(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/checkpoints/{args.pretrained_model_path}")

        # Save losses to CSV after each epoch
        loss_log_df = pd.DataFrame(loss_log)
        loss_log_df.to_csv(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/{args.strategy}_ssl_loss.csv", index=False)
            
    # === Evaluate on each year's test set ===
    # eval.model_evaluate(fixMatch.get_model(), test_sets_by_year, args.strategy)
    # Active learning loop: month by month
    
    for p_value in [1.0, 1.5, 1.8, 1.9, 2.0, 2.1, 2.2]:
        for budget in [50, 100, 200, 400]:
            start = time.time()
            # Your code block
            args.budget = budget
            args.k_nearest_neighbors = budget + 50
            args.p_value = p_value

            metrics_list = []
            all_month_losses = []

            sorted_test_keys = test_sets_by_year.keys() #sorted(test_sets_by_year.keys())
            # logging.info(f"Keys: {sorted_test_keys}")

            for test_idx, year in tqdm(enumerate(sorted_test_keys), total=len(sorted_test_keys), desc="Active Learning Loop"):
                X_test, y_test_bin, y_test_actual = test_sets_by_year[year]

                # === Evaluate on this test set BEFORE adding to unlabeled set ===
                metrics, mismatch_details = eval.evaluate_model_active(fixMatch.get_model(), X_test, y_test_bin, \
                                                                    y_test_actual, num_classes=num_classes)
                metrics['year'] = year
                metrics_list.append(metrics)
                print(f"Year {year}: " + ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]))

                # Save mismatch details for this year
                # mismatch_df = pd.DataFrame(mismatch_details, columns=['index', 'y_pred', 'y_true', 'actual_label'])
                # mismatch_df.to_csv(
                #     f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/freematch/results/false_pred/mismatch_{strategy}_{year}.csv",
                #     index=False
                # )
                # === Add this test set to the UNLABELED set for next round (active learning) ===
                # X_unlabeled = torch.cat([X_unlabeled, X_test.to(X_unlabeled.device)], dim=0)
                X_unlabeled = X_test.to(X_unlabeled.device)
                unlabeled_ds = TensorDataset(X_unlabeled.to(X_unlabeled.device))
                unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True)
                # unlabeled_loader = DataLoader(unlabeled_ds, sampler=train_sampler(unlabeled_ds), batch_size=al_batch_size, drop_last=True)
                
                # Create a new optimizer and scheduler for each retraining phase
                optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
                scheduler = utils.get_cosine_schedule_with_warmup(optimizer, args.warmup, args.retrain_epochs)

                ## set AdamW and cosine lr on FreeMatch
                fixMatch.set_optimizer(optimizer, scheduler)
                fixMatch.set_data(labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader)
                # freeMatch.warmup()
                
                best_loss = float('inf')
                best_state_dict = None
                loss_log = []

                # Number of unlabeled data
                N = len(labeled_loader)

                fixMatch.model_train()

                for epoch in tqdm(range(args.retrain_epochs), desc="Retraining FixMatch..."):
                    labeled_iter = iter(labeled_loader)

                    unlabeled_iter= iter(unlabeled_loader)

                    total_loss, total_loss_sup, total_loss_unsup, total_lhc_val, total_lce_val = 0, 0, 0, 0, 0
                    for _ in range(N):
                        try:
                            x_l, y_l, y_bin_l = next(labeled_iter)
                            (x_u,) = next(unlabeled_iter)
                        except StopIteration:
                            break

                        x_l, y_l = x_l.cuda(), y_l.cuda()
                        y_bin_l = y_bin_l.cuda()
                        x_u = x_u.cuda()
                        # x_u_w, x_u_s = x_u_w.cuda(), x_u_s.cuda()
                        # u_i = torch.arange(x_u_w.size(0)).to(x_u_w.device)

                        if args.aug == "random_bit_flip":
                            x_u_w = utils.random_bit_flip(x_u, n_bits=1)
                            x_u_s = utils.random_bit_flip(x_u, n_bits=args.bit_flip)
                        elif args.aug == "random_bit_flip_bernoulli":
                            x_u_w = utils.random_bit_flip_bernoulli(x_u, p=0.01)
                            x_u_s = utils.random_bit_flip_bernoulli(x_u, p=0.05)
                        elif args.aug == "random_bit_flip_and_mask":
                            x_u_w = utils.random_bit_flip_and_mask(x_u, n_bits=1, n_mask=1)
                            x_u_s = utils.random_bit_flip_and_mask(x_u, n_bits=args.bit_flip, n_mask=args.bit_flip)
                        else:
                            raise ValueError(f"Unknown augmentation function: {args.aug}")

                        loss, loss_s, loss_u, lhc_val, lce_val = fixMatch.train_step(x_l, y_l, y_bin_l, x_u_w, x_u_s)

                        total_loss += loss
                        total_loss_sup += loss_s
                        total_loss_unsup += loss_u
                        total_lhc_val += lhc_val
                        total_lce_val += lce_val

                    fixMatch.update_scheduler()

                    avg_loss = total_loss/N
                    avg_loss_sup = total_loss_sup/N
                    avg_loss_unsup = total_loss_unsup/N
                    avg_lhc_val = total_lhc_val/N
                    avg_lce_val = total_lce_val/N

                    # Collect losses for this epoch
                    loss_log.append({
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'sup_loss': avg_loss_sup,
                        'unsup_loss': avg_loss_unsup,
                        'lhc_val': avg_lhc_val,
                        'lce_val': avg_lce_val
                    })

                    # EARLY STOPPING LOGIC (retraining)
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_state_dict = {k: v.cpu().clone() for k, v in fixMatch.get_model_state().items()}

                    logging.info(f"##Epoch {epoch+1}/{args.retrain_epochs}: loss={avg_loss:.4f}, sup_loss={avg_loss_sup:.4f}, "
                                f"unsup_loss={avg_loss_unsup:.4f}, lhc_val={avg_lhc_val:.4f}, lce_val={avg_lce_val:.4f}, best_loss={best_loss:.4f}")

                    # scheduler.step()
                
                # Restore best model after retraining
                if best_state_dict is not None:
                    fixMatch.update_model(best_state_dict)
                
                # Save average loss for this month
                # Collect average loss for this month
                avg_month_loss = {
                    'year': year,
                    'avg_loss': np.mean([entry['loss'] for entry in loss_log]),
                    'avg_sup_loss': np.mean([entry['sup_loss'] for entry in loss_log]),
                    'avg_unsup_loss': np.mean([entry['unsup_loss'] for entry in loss_log]),
                    'avg_lhc_val': np.mean([entry['lhc_val'] for entry in loss_log]),
                    'avg_lce_val': np.mean([entry['lce_val'] for entry in loss_log])
                }
                all_month_losses.append(avg_month_loss)
                
                # === Active Learning: Select top uncertain samples from unlabeled set ===
                # Get embeddings for the unlabeled data
                with torch.no_grad():
                    model = fixMatch.get_model()
                    model.eval()
                    _, train_embeddings, _ = model(X_labeled)
                # Assume your model, encoder, and training embeddings are ready
                selected_idxs = fixMatch.active_learning_loop(
                    unlabeled_data=X_unlabeled,
                    train_embeddings=train_embeddings,
                    train_labels=y_labeled,
                    budget=args.budget,  # Select top 10 uncertain
                    margin=1.0,
                    K=args.k_nearest_neighbors,
                    p_value=args.p_value
                )
                print("Selected indices for labeling:", selected_idxs)
                # print("Uncertainty scores:", scores)

                X_labeled = torch.cat([X_labeled, X_test[selected_idxs].to(X_unlabeled.device)], dim=0)

                y_labeled = torch.cat([y_labeled, torch.tensor(y_test_actual[selected_idxs], device=X_unlabeled.device, dtype=y_labeled.dtype)], dim=0)
                y_labeled_binary = torch.cat([y_labeled_binary, torch.tensor(y_test_bin[selected_idxs], device=X_unlabeled.device, dtype=y_labeled_binary.dtype)], dim=0)

                # weight_tensor = torch.ones(X_labeled.shape[0])
                labeled_ds = TensorDataset(X_labeled, y_labeled, y_labeled_binary)
                labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)

            # Save model after all iterations of Active learning
            # fixMatch.save_model(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/checkpoints/{args.strategy}_model_al.pth")

            # Save results to CSV
            metrics_df = pd.DataFrame(metrics_list)
            # metrics_df.to_csv(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/{args.strategy}_al.csv", index=False)

            # Save average losses to CSV
            # avg_loss_df = pd.DataFrame(all_month_losses)
            # avg_loss_df.to_csv(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/{args.strategy}_ssl_loss_al.csv", index=False)
            avg_f1 = metrics_df['f1'].mean()
            avg_fnr = metrics_df['fnr'].mean()
            avg_fpr = metrics_df['fpr'].mean()
            print(f"Mean F1 Scores: {avg_f1:.4f}")
            print(f"Mean False Negative Rates: {avg_fnr:.4f}")
            print(f"Mean False Positive Rates: {avg_fpr:.4f}")

            end = time.time()

            print(f"Program Execution Ended for one combination!!")
            print(f"##Execution time: {end - start:.4f} seconds")

            # Append to CSV file
            with open(f'/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/model_performance_{args.aug}_p_values_budgets.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([args.p_value, args.budget, avg_f1, avg_fnr, avg_fpr, (end - start)])  # appends a single row

            # plot_gen.plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], \
            #                      save_path=f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/f1_fnr_{args.strategy}_al.png")

# === Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FixMatch with Bit Flip Augmentation")
    parser.add_argument("--bit_flip", type=int, default=4, help="Number of bits to flip per sample")
    parser.add_argument("--labeled_ratio", type=float, default=0.4, help="Ratio of labeled data")
    parser.add_argument("--aug", type=str, default="random_bit_flip", help="Augmentation function to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--lambda-u', default=1.0, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--p_value', default=2.0, type=float, help='P value for dist matric')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--strategy', default="", type=str, help='strategy...')
    parser.add_argument('--pretrained_model', default=True, type=bool, help='Whether to use pretrained model or not')
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs for SSL training")
    parser.add_argument("--retrain_epochs", type=int, default=100, help="Number of epochs for SSL retraining")
    parser.add_argument("--budget", type=int, default=150, help="Number of budget for SSL retraining")
    parser.add_argument("--k_nearest_neighbors", type=int, default=200, help="Number of nearest neighbors for SSL retraining")
    parser.add_argument("--pretrained_model_path", type=str, default="", help="Path to the pretrained model if any")

    args = parser.parse_args()
    args.seed = 0
    SEED = args.seed  # You can set this to any integer you like

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_bit_flip = args.bit_flip
    labeled_ratio = args.labeled_ratio

    args.aug = "random_bit_flip"
    args.budget = 400
    args.k_nearest_neighbors = 450
    args.p_value = 1.9

    args.pretrained_model = True
    args.pretrained_model_path = f"fixmatch_hidistloss_w_al_{args.aug}_{str(n_bit_flip)}_lbr_{str(labeled_ratio)}_ep{str(args.epochs)}_model.pth"

    args.strategy = f"fixmatch_hidistloss_w_al_" + args.aug + "_" + str(n_bit_flip) + "_lbr_" + str(labeled_ratio) +  "_seed_" + str(args.seed) + "_ep" + str(args.epochs) + "_retrain_ep" + str(args.retrain_epochs) + "_budget" + str(args.budget) + "_k" + str(args.k_nearest_neighbors) + "_p" + str(args.p_value)
    print(f"Running {strategy}...")
    print(f"Using {n_bit_flip} bits to flip per sample.")

    path = "/home/ihossain/ISMAIL/Datasets/data/gen_apigraph_drebin/"
    # Load data
    X_2012_labeled, y_2012_labeled, y_2012_bin_labeled, X_2012_unlabeled = utils.get_dataset(path)

    test_sets_by_year = {}
    for year in tqdm(range(2013, 2019), desc="Processing years"):
        for month in range(1, 13):
            try:
                data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                X_raw = data["X_train"]
                y_true = (data["y_train"] > 0).astype(int)
                X_tensor = torch.tensor(X_raw, dtype=torch.float32).cuda()
                y_tensor = torch.tensor(y_true, dtype=torch.long).cuda()
                test_sets_by_year[f"{year}_{month}"] = (X_tensor, y_tensor, data["y_train"])
            except FileNotFoundError:
                continue

    NUM_FEATURES = X_2012_labeled.shape[1]
    mlp_hidden = '100-100'
    enc_hidden = '512-384-256-128'
    NUM_CLASSES = 2

    enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES, enc_hidden, NUM_CLASSES)
    mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], mlp_hidden, NUM_CLASSES)
    model = SimpleEncClassifier(enc_dims, mlp_dims).cuda()

    train_fixmatch_drift_eval(
        args,
        model,
        X_2012_labeled,
        y_2012_labeled,
        y_2012_bin_labeled,
        X_2012_unlabeled,
        test_sets_by_year,
        num_classes=NUM_CLASSES
    )

# CUDA_VISIBLE_DEVICES=1 nohup python active_learning.py > /home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/output.log 2>&1 &