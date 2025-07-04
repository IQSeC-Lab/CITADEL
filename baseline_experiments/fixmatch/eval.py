import torch
from sklearn.metrics import f1_score, confusion_matrix
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import plot_gen
import numpy as np
import torch.nn.functional as F
import faiss
from sklearn.metrics import pairwise_distances


def model_evaluate(model, test_sets_by_year, strategy):
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for year, (X_test, y_test, y_actual) in test_sets_by_year.items():
            X_test, y_test = X_test.cuda(), y_test.cuda()
            y_score = model(X_test)[-1]

            pred_label = (y_score >= 0.5).float() #torch.tensor((y_score.item() >= 0.5), dtype=torch.long)

            # # Determine shape-based branching
            # if logits.shape[1] == 1:  # Binary classification (1 logit per sample)
            #     probs = torch.sigmoid(logits).squeeze()
            #     preds = (probs >= 0.5).long()
            #     y_score = probs.detach().cpu().numpy()
            # else:  # Multi-class (or binary with 2 outputs)
            #     probs = torch.softmax(logits, dim=1)
            #     preds = torch.argmax(probs, dim=1)
            #     y_score = probs[:, 1].detach().cpu().numpy() if probs.shape[1] == 2 else probs.detach().cpu().numpy()

            y_true = y_test.cpu().numpy()
            y_pred = pred_label.cpu().numpy()
            y_score_np = y_score.cpu().numpy()
            # print(f"Evaluating year {year} with {strategy} strategy...")
            # print(f"Logits shape: {logits.shape}, Probs shape: {probs.shape}, y_score shape: {y_score.shape}")
            # print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
            # print(f"y_true: {y_true[:10]}, y_pred: {y_pred[:10]}, y_score: {y_score[:10]}")

            # Convert probabilities to hard labels
            # y_pred = (y_score > 0.5).astype(int)  # for sigmoid output
            # # OR
            # y_pred = np.argmax(probs, axis=1)    # for softmax over 2-class logits

            # Metrics
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
            # try:
            #     if probs.shape[1] == 2:
            #         roc_auc = roc_auc_score(y_true, y_score)
            #         pr_auc = average_precision_score(y_true, y_score)
            #     else:
            #         roc_auc = roc_auc_score(y_true, probs.cpu().numpy(), multi_class='ovr')
            #         pr_auc = average_precision_score(y_true, probs.cpu().numpy(), average='weighted')
            # except Exception:
            #     roc_auc = pr_auc = float('nan')
            roc_auc = roc_auc_score(y_true, y_score_np)
            pr_auc = average_precision_score(y_true, y_score_np)

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
    metrics_df.to_csv(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/{strategy}_wo_al.csv", index=False)

    print(f"Mean F1 Scores: {metrics_df['f1'].mean():.4f}")
    print(f"Mean False Negative Rates: {metrics_df['fnr'].mean()}")
    print(f"Mean False Positive Rates: {metrics_df['fpr'].mean()}")
    plot_gen.plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], save_path=f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/fixmatch/results/{strategy}_wo_al_f1_fnr_plot.png")



def evaluate_model_active(model, X_test, y_test, y_actual, num_classes=2):
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.cuda(), y_test.cuda()
        # logits = model(X_test)[-1]
        y_score = model(X_test)[-1]

        pred_label = (y_score >= 0.5).float()
        # probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
        # preds = logits.argmax(dim=1)
        y_true = y_test.cpu().numpy()
        y_pred = pred_label.cpu().numpy()
        y_score_np = y_score.cpu().numpy()

        # if probs.shape[1] == 2:
        #     y_score = probs[:, 1].cpu().numpy()
        # else:
        #     y_score = probs.cpu().numpy()  # for multi-class

        mismatch_indices = np.where(y_true != y_pred)[0]
        mismatch_details = [(idx, y_pred[idx], y_true[idx], int(y_actual[idx])) for idx in mismatch_indices]
        # print("Mismatched sample indices:", mismatch_indices)

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
        # try:
        #     if probs.shape[1] == 2:
        #         roc_auc = roc_auc_score(y_true, y_score)
        #         pr_auc = average_precision_score(y_true, y_score)
        #     else:
        #         roc_auc = roc_auc_score(y_true, probs.cpu().numpy(), multi_class='ovr')
        #         pr_auc = average_precision_score(y_true, probs.cpu().numpy(), average='weighted')
        # except Exception:
        #     roc_auc = pr_auc = float('nan')
        roc_auc = roc_auc_score(y_true, y_score_np)
        pr_auc = average_precision_score(y_true, y_score_np)
        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'fnr': fnr,
            'fpr': fpr,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        return metrics, mismatch_details
    

def pseudo_loss_selector(test_sample, model, train_embeddings, train_labels, margin=1.0, k=63):
    """
    Compute the pseudo contrastive loss for a test sample for active learning selection.

    Args:
        test_sample (torch.Tensor): Shape (1, input_dim)
        model (nn.Module): Trained Hierarchical Contrastive Classifier
        train_embeddings (torch.Tensor): Precomputed normalized embeddings of training set
        train_labels (torch.Tensor): Corresponding binary labels of training set
        margin (float): Margin used in contrastive loss
        k (int): Number of nearest neighbors to consider

    Returns:
        float: Pseudo contrastive loss (uncertainty score)
    """
    model.eval()
    # with torch.no_grad():
    #     _, embedding, pred = model(test_sample)  # [1, embedding_dim]
    #     embedding = F.normalize(embedding, p=2, dim=1)

    #     # Get pseudo label
    #     pseudo_label = torch.tensor((pred.item() >= 0.5), dtype=torch.long)

    #     # Nearest neighbors in training embeddings
    #     dists = torch.cdist(embedding, train_embeddings, p=2).squeeze(0)
    #     _, nn_indices = torch.topk(dists, k=k, largest=False)

    #     neighbors = train_embeddings[nn_indices]
    #     neighbor_labels = train_labels[nn_indices]
    #     device = pseudo_label.device
    #     neighbor_labels = neighbor_labels.to(device)

    #     # Mini-batch construction
    #     batch_embeddings = torch.cat([embedding, neighbors], dim=0)
    #     batch_labels = torch.cat([pseudo_label.unsqueeze(0), neighbor_labels], dim=0)
    #     dists_batch = torch.cdist(batch_embeddings, batch_embeddings, p=2)

    #     # Compute L̂_hc (i) = terms 1 and 3 only
    #     i = 0
    #     Pi = [j for j in range(1, k + 1) if batch_labels[j] == batch_labels[i]]
    #     Ni = [j for j in range(1, k + 1) if batch_labels[j] != batch_labels[i]]

    #     term1 = torch.sum(torch.clamp(dists_batch[i][Pi] - margin, min=0)) / (len(Pi) if Pi else 1)
    #     term3 = torch.sum(torch.clamp(2 * margin - dists_batch[i][Ni], min=0)) / (len(Ni) if Ni else 1)

    #     pseudo_loss = term1 + term3
    #     return pseudo_loss.item()
    with torch.no_grad():
        _, embedding, pred = model(test_sample)  # embedding: [1, D]
        embedding = F.normalize(embedding, p=2, dim=1)
        pseudo_label = torch.tensor((pred.item() >= 0.5), dtype=torch.long, device=test_sample.device)

        # Get k nearest neighbors (distances and indices)
        dists = torch.cdist(embedding, train_embeddings, p=2).squeeze(0)
        nn_dists, nn_indices = torch.topk(dists, k=k, largest=False)

        neighbor_labels = train_labels[nn_indices].to(test_sample.device)

        # Identify Pi and Ni using masks
        same_class = (neighbor_labels == pseudo_label)
        diff_class = ~same_class

        # Calculate contrastive pseudo loss directly
        term1 = torch.sum(torch.clamp(nn_dists[same_class] - margin, min=0)) / (same_class.sum() or 1)
        term3 = torch.sum(torch.clamp(2 * margin - nn_dists[diff_class], min=0)) / (diff_class.sum() or 1)

        return (term1 + term3).item()
    

def batch_pseudo_loss_selector(unlabeled_embeddings, pseudo_labels, train_embeddings, train_labels, margin=1.0, k=63):
    """
    Compute pseudo loss scores for a batch of unlabeled embeddings at once.
    Args:
        unlabeled_embeddings: [N, D]
        pseudo_labels: [N]
        train_embeddings: [T, D]
        train_labels: [T]
    Returns:
        scores: [N]
    """
    N = unlabeled_embeddings.size(0)
    D = unlabeled_embeddings.size(1)
    T = train_embeddings.size(0)

    # [N, T] distances
    dists = torch.cdist(unlabeled_embeddings, train_embeddings, p=2)

    scores = torch.zeros(N, device=unlabeled_embeddings.device)

    for i in range(N):
        d_i = dists[i]
        label_i = pseudo_labels[i]
        topk_dists, topk_indices = torch.topk(d_i, k=k, largest=False)
        neighbor_labels = train_labels[topk_indices]

        same_class = (neighbor_labels == label_i)
        diff_class = ~same_class

        term1 = torch.sum(torch.clamp(topk_dists[same_class] - margin, min=0)) / (same_class.sum() or 1)
        term3 = torch.sum(torch.clamp(2 * margin - topk_dists[diff_class], min=0)) / (diff_class.sum() or 1)

        scores[i] = term1 + term3

    return scores


def get_top_k_neighbors(train_embed, unlabeled_embed, K=63, p_value=2, dim=1, device='cuda'):
    """
    Get the top-k nearest neighbors for each unlabeled embedding from the training embeddings.

    Args:
        unlabeled_embeddings (torch.Tensor): [N, D] normalized embeddings of unlabeled data
        train_embeddings (torch.Tensor): [T, D] normalized embeddings of training data
        k (int): number of nearest neighbors to consider

    Returns:
        torch.Tensor: [N, k] indices of the top-k nearest neighbors in the training set
    """
    # dists = torch.cdist(unlabeled_embeddings, train_embeddings, p=2)  # [N, T]
    # D_k, top_k_indices = torch.topk(dists, k=k, largest=False)  # [N, k]
    # Choose your p-value
    p_value = 1.5  # or 1 for L1, 2 for L2, np.inf for L-infinity

    # if p_value == 2:
    #     # Convert to numpy for FAISS
    #     train_np = train_embed.detach().cpu().numpy().astype(np.float32)
    #     query_np = unlabeled_embed.detach().cpu().numpy().astype(np.float32)
    #     # Build FAISS index and perform search
    #     index = faiss.IndexFlatL2(dim)
    #     index.add(train_np)
    #     D_k, I_k = index.search(query_np, K)  # D_k: [N, k], I_k: [N, k]

    #     # Convert back to torch tensors
    #     D_k = torch.tensor(D_k, device=device)        # distances
    #     I_k = torch.tensor(I_k, device=device)        # indices
    # else:
    train_torch = torch.tensor(train_embed, device='cuda')  # move to GPU
    query_torch = torch.tensor(unlabeled_embed, device='cuda')

    # Compute L2 distance: ||x - y||^2 = sum((x - y)^2)
    # Shape: [N, T] where N = num queries, T = num training
    dists = torch.cdist(query_torch, train_torch, p=p_value)

    # Get top-k
    D_k, I_k = torch.topk(dists, k=K, dim=1, largest=False)
    # # Compute pairwise distances
    # dist_matrix = pairwise_distances(query_np, train_np, metric='minkowski', p=p_value)  # shape: [N, T]

    # # Find indices of top-k nearest neighbors
    # I_k_np = dist_matrix.argsort(axis=1)[:, :K]
    # D_k_np = np.take_along_axis(dist_matrix, I_k_np, axis=1)
    # D_k = torch.tensor(D_k_np, device=device)
    # I_k = torch.tensor(I_k_np, device=device)

    return D_k, I_k

def faiss_batch_pseudo_loss_selector(unlabeled_embeddings, pseudo_labels, train_embeddings, train_labels, margin=1.0, k=63, P=2):
    """
    Use FAISS to compute pseudo contrastive loss for a batch of unlabeled embeddings.

    Args:
        unlabeled_embeddings (torch.Tensor): [N, D] normalized embeddings of unlabeled data
        pseudo_labels (torch.Tensor): [N] binary pseudo labels for the unlabeled data
        train_embeddings (torch.Tensor): [T, D] normalized embeddings of training data
        train_labels (torch.Tensor): [T] binary ground truth labels of training data
        margin (float): margin value for contrastive loss
        k (int): number of nearest neighbors to consider

    Returns:
        torch.Tensor: [N] pseudo contrastive loss scores
    """
    device = unlabeled_embeddings.device
    N, D = unlabeled_embeddings.shape

    D_k, I_k = get_top_k_neighbors(train_embeddings, unlabeled_embeddings, K=k, p_value=P, dim=D, device=device)
    
    train_labels = train_labels.to(device)

    # Get neighbor labels
    neighbor_labels = train_labels[I_k]           # shape: [N, k]
    pseudo_labels_exp = pseudo_labels.unsqueeze(1).expand_as(neighbor_labels)  # [N, k]

    # Masks
    same_class = (neighbor_labels == pseudo_labels_exp)
    diff_class = ~same_class

    # Compute term1: same class (weak positives)
    term1 = torch.sum(torch.clamp(D_k * same_class - margin, min=0), dim=1) / same_class.sum(dim=1).clamp(min=1)

    # Compute term3: different class (negatives)
    term3 = torch.sum(torch.clamp(2 * margin - D_k * diff_class, min=0), dim=1) / diff_class.sum(dim=1).clamp(min=1)

    # Total pseudo loss per sample
    scores = term1 + term3
    return scores

