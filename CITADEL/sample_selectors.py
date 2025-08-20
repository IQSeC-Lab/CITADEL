import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader



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