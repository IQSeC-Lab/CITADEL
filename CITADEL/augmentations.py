import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


# ============================================================
# 3. Label-based gap
#    Always returns CPU tensors
# ============================================================
def compute_label_gap(X_labeled, y_labeled):
    """
    gap[j] = P(feature_j=1 | malware) - P(feature_j=1 | benign)
    """
    X = X_labeled.float().cpu()
    y = y_labeled.long().cpu()

    p_mal = X[y == 1].mean(0)
    p_ben = X[y == 0].mean(0)
    gap = p_mal - p_ben

    return {
        "p_mal": p_mal.cpu(),
        "p_ben": p_ben.cpu(),
        "gap": gap.cpu(),
    }


# ============================================================
# 4. Model gradient info
#    Model runs on device, outputs returned on CPU
# ============================================================
def compute_model_gradient_info(model, X_labeled, device=None):
    model.eval()

    X = X_labeled.float().to(device).clone().detach().requires_grad_(True)
    logits = model(X)

    logits[:, 1].sum().backward()

    grad = X.grad.detach()
    gx = grad * X.detach()

    signed_score = gx.mean(0).detach().cpu()
    usage = gx.abs().mean(0).detach().cpu()
    usage = usage / (usage.max() + 1e-8)

    return {
        "signed_score": signed_score.cpu(),
        "usage": usage.cpu(),
    }



# ============================================================
# 6. Build initial 3 groups
#    Returns ONLY CPU tensors for feature groups
# ============================================================
def build_three_feature_groups(
    X_labeled,
    y_labeled,
    model,
    signal_threshold=0.10,
    direction_threshold=0.01,
    device=None,
):
    """
    Returns exactly 3 non-overlapping groups on CPU:
      combined_label_model_mal_feature
      combined_label_model_ben_feature
      blindspot_feature
    """

    label_info = compute_label_gap(X_labeled, y_labeled)
    grad_info = compute_model_gradient_info(model, X_labeled, device=device)

    gap = label_info["gap"].cpu()
    signed = grad_info["signed_score"].cpu()
    usage = grad_info["usage"].cpu()

    label_mal = gap > signal_threshold
    label_ben = gap < -signal_threshold

    model_mal = signed > direction_threshold
    model_ben = signed < -direction_threshold

    mal_candidate = label_mal | model_mal
    ben_candidate = label_ben | model_ben

    combined_label_model_mal_feature = mal_candidate.cpu()
    combined_label_model_ben_feature = (ben_candidate & (~combined_label_model_mal_feature)).cpu()
    blindspot_feature = (~(combined_label_model_mal_feature | combined_label_model_ben_feature)).cpu()

    

    return {
        "combined_label_model_mal_feature": combined_label_model_mal_feature.bool().cpu(),
        "combined_label_model_ben_feature": combined_label_model_ben_feature.bool().cpu(),
        "blindspot_feature": blindspot_feature.bool().cpu(),
        "gap": gap.cpu(),
        "signed_score": signed.cpu(),
        "usage": usage.cpu(),
        "history": [],
    }


# ============================================================
# 7. Update groups with monthly selected labeled samples
#    All boolean logic happens on CPU
# ============================================================
def update_three_feature_groups(
    current_groups,
    X_month,
    y_month,
    model,
    signal_threshold=0.10,
    direction_threshold=0.01,
    device=None,
):

    old_mal = current_groups["combined_label_model_mal_feature"].clone().bool().cpu()
    old_ben = current_groups["combined_label_model_ben_feature"].clone().bool().cpu()
    old_blind = current_groups["blindspot_feature"].clone().bool().cpu()

    month_label_info = compute_label_gap(X_month, y_month)
    month_grad_info = compute_model_gradient_info(model, X_month, device=device)

    month_gap = month_label_info["gap"].cpu()
    month_signed = month_grad_info["signed_score"].cpu()
    month_usage = month_grad_info["usage"].cpu()

    month_label_mal = month_gap > signal_threshold
    month_label_ben = month_gap < -signal_threshold

    month_model_mal = month_signed > direction_threshold
    month_model_ben = month_signed < -direction_threshold

    month_mal_candidate = month_label_mal | month_model_mal
    month_ben_candidate = month_label_ben | month_model_ben

    promote_to_mal = old_blind & month_mal_candidate
    promote_to_ben = old_blind & month_ben_candidate & (~promote_to_mal)

    new_mal = (old_mal | promote_to_mal).cpu()
    new_ben = (old_ben | promote_to_ben).cpu()
    new_blind = (~(new_mal | new_ben)).cpu()

    

    new_history = list(current_groups.get("history", []))
    new_history.append({
        "n_promoted_to_mal": int(promote_to_mal.sum()),
        "n_promoted_to_ben": int(promote_to_ben.sum()),
        "month_gap": month_gap,
        "month_signed_score": month_signed,
        "month_usage": month_usage,
    })

    return {
        "combined_label_model_mal_feature": new_mal.bool().cpu(),
        "combined_label_model_ben_feature": new_ben.bool().cpu(),
        "blindspot_feature": new_blind.bool().cpu(),
        "gap": month_gap.cpu(),
        "signed_score": month_signed.cpu(),
        "usage": month_usage.cpu(),
        "history": new_history,
    }



# ============================================================
# 7. Internal helper to update groups from evidence
# ============================================================
def _update_groups_from_evidence(
    current_groups,
    gap,
    signed_score,
    signal_threshold=0.10,
    direction_threshold=0.01,
    stage_name="update",
):
    """
    Promote only from blindspot -> malware/benign.
    Existing mal/ben assignments stay fixed.
    Malware gets priority on conflict.
    """
    old_mal = current_groups["combined_label_model_mal_feature"].clone().bool().cpu()
    old_ben = current_groups["combined_label_model_ben_feature"].clone().bool().cpu()
    old_blind = current_groups["blindspot_feature"].clone().bool().cpu()

    gap = gap.cpu()
    signed_score = signed_score.cpu()

    label_mal = gap > signal_threshold
    label_ben = gap < -signal_threshold

    model_mal = signed_score > direction_threshold
    model_ben = signed_score < -direction_threshold

    month_mal_candidate = label_mal | model_mal
    month_ben_candidate = label_ben | model_ben

    promote_to_mal = old_blind & month_mal_candidate
    promote_to_ben = old_blind & month_ben_candidate & (~promote_to_mal)

    new_mal = old_mal | promote_to_mal
    new_ben = old_ben | promote_to_ben
    new_blind = ~(new_mal | new_ben)


    new_history = list(current_groups.get("history", []))
    new_history.append({
        "stage": stage_name,
        "n_promoted_to_mal": int(promote_to_mal.sum()),
        "n_promoted_to_ben": int(promote_to_ben.sum()),
    })

    return {
        "combined_label_model_mal_feature": new_mal.cpu(),
        "combined_label_model_ben_feature": new_ben.cpu(),
        "blindspot_feature": new_blind.cpu(),
        "gap": gap.cpu(),
        "signed_score": signed_score.cpu(),
        "usage": current_groups.get("usage", None),
        "history": new_history,
    }



# ============================================================
# 9. Update groups again after retraining model
# ============================================================
def update_three_feature_groups_after_retrain(
    current_groups,
    X_labeled_all,
    y_labeled_all,
    retrained_model,
    signal_threshold=0.10,
    direction_threshold=0.01,
    device="cuda",
):
    """
    Update again after retraining the model on expanded labeled data.
    Uses the full current labeled pool and the retrained model.
    """
    full_label_info = compute_label_gap(X_labeled_all, y_labeled_all)
    full_grad_info = compute_model_gradient_info(retrained_model, X_labeled_all, device=device)

    updated = _update_groups_from_evidence(
        current_groups=current_groups,
        gap=full_label_info["gap"],
        signed_score=full_grad_info["signed_score"],
        signal_threshold=signal_threshold,
        direction_threshold=direction_threshold,
        stage_name="post_retrain_update",
    )

    updated["usage"] = full_grad_info["usage"].cpu()
    return updated


# ============================================================
# 8. Prepare feature-group masks once on CUDA for fast augmentation
#    Input groups are CPU bool; output masks are CUDA bool
# ============================================================
def prepare_feature_groups_for_cuda(feature_groups, device=None):
    return {
        "mal": feature_groups["combined_label_model_mal_feature"].bool().to(device),
        "ben": feature_groups["combined_label_model_ben_feature"].bool().to(device),
        "blind": feature_groups["blindspot_feature"].bool().to(device),
    }


# ============================================================
# 9. Fast helper: activate missing binary features
#    x and masks MUST already be on same device
# ============================================================
def activate_features_with_probability_fast(x_bool, feature_mask_bool, p):
    """
    x_bool            : [N, d] bool tensor on device
    feature_mask_bool : [d] bool tensor on same device
    p                 : float in [0, 1]

    Only 0 -> 1 activation.
    """
    if x_bool.dtype != torch.bool:
        raise ValueError("x_bool must be torch.bool")
    if feature_mask_bool.dtype != torch.bool:
        raise ValueError("feature_mask_bool must be torch.bool")
    if x_bool.device != feature_mask_bool.device:
        raise ValueError("x_bool and feature_mask_bool must be on the same device")
    if x_bool.dim() != 2:
        raise ValueError("x_bool must have shape [N, d]")

    candidate = (~x_bool) & feature_mask_bool.unsqueeze(0)

    if p <= 0.0:
        return x_bool
    if p >= 1.0:
        return x_bool | candidate

    add_mask = candidate & (torch.rand_like(x_bool, dtype=torch.float32) < p)
    return x_bool | add_mask


# ============================================================
# 10. Malware weak / strong
# ============================================================
def malware_weak_augment_fast(
    x_malware,
    mal_mask,
    blind_mask,
    p_mw=0.05,
    p_blw=0.02,
):
    x = x_malware.bool()
    if p_mw > 0:
        x = activate_features_with_probability_fast(x, mal_mask, p_mw)
    if p_blw > 0:
        x = activate_features_with_probability_fast(x, blind_mask, p_blw)
    return x


def malware_strong_augment_fast(
    x_malware_weak_bool,
    ben_mask,
    blind_mask,
    p_bls=0.08,
    p_bs=0.03,
):
    x = x_malware_weak_bool
    if p_bls > 0:
        x = activate_features_with_probability_fast(x, blind_mask, p_bls)
    if p_bs > 0:
        x = activate_features_with_probability_fast(x, ben_mask, p_bs)
    return x


# ============================================================
# 11. Benign weak / strong
# ============================================================
def benign_weak_augment_fast(
    x_benign,
    ben_mask,
    blind_mask,
    mal_mask=None,
    p_bw=0.05,
    p_blw=0.02,
    p_mw=0.00,
    add_malware_in_weak=False,
):
    x = x_benign.bool()
    if p_bw > 0:
        x = activate_features_with_probability_fast(x, ben_mask, p_bw)
    if p_blw > 0:
        x = activate_features_with_probability_fast(x, blind_mask, p_blw)
    if add_malware_in_weak:
        if mal_mask is None:
            raise ValueError("mal_mask must be provided when add_malware_in_weak=True")
        if p_mw > 0:
            x = activate_features_with_probability_fast(x, mal_mask, p_mw)
    return x


def benign_strong_augment_fast(
    x_benign_weak_bool,
    mal_mask,
    blind_mask,
    p_ms=0.05,
    p_bls=0.08,
):
    x = x_benign_weak_bool
    if p_ms > 0:
        x = activate_features_with_probability_fast(x, mal_mask, p_ms)
    if p_bls > 0:
        x = activate_features_with_probability_fast(x, blind_mask, p_bls)
    return x


# ============================================================
# 12. Per-class pipelines
# ============================================================
def malware_augment_pipeline_fast(
    x_malware,
    feature_groups_cuda,
    p_mw=0.05,
    p_blw=0.02,
    p_bls=0.08,
    p_bs=0.03,
    return_float=True,
):
    mal_mask = feature_groups_cuda["mal"]
    ben_mask = feature_groups_cuda["ben"]
    blind_mask = feature_groups_cuda["blind"]

    x_bool = x_malware.bool()

    x_weak = malware_weak_augment_fast(
        x_bool, mal_mask, blind_mask, p_mw=p_mw, p_blw=p_blw
    )
    x_strong = malware_strong_augment_fast(
        x_weak, ben_mask, blind_mask, p_bls=p_bls, p_bs=p_bs
    )

    if return_float:
        return x_weak.float(), x_strong.float()
    return x_weak, x_strong


def benign_augment_pipeline_fast(
    x_benign,
    feature_groups_cuda,
    p_bw=0.05,
    p_blw=0.02,
    p_mw=0.00,
    add_malware_in_weak=False,
    p_ms=0.05,
    p_bls=0.08,
    return_float=True,
):
    mal_mask = feature_groups_cuda["mal"]
    ben_mask = feature_groups_cuda["ben"]
    blind_mask = feature_groups_cuda["blind"]

    x_bool = x_benign.bool()

    x_weak = benign_weak_augment_fast(
        x_bool,
        ben_mask,
        blind_mask,
        mal_mask=mal_mask,
        p_bw=p_bw,
        p_blw=p_blw,
        p_mw=p_mw,
        add_malware_in_weak=add_malware_in_weak,
    )
    x_strong = benign_strong_augment_fast(
        x_weak,
        mal_mask,
        blind_mask,
        p_ms=p_ms,
        p_bls=p_bls,
    )

    if return_float:
        return x_weak.float(), x_strong.float()
    return x_weak, x_strong


# ============================================================
# 13. Combined batch augmentation using pseudo-labels
# ============================================================
def combined_augment_batch_fast(
    x_batch,
    pseudo_labels,
    feature_groups_cuda,
    # malware params
    p_mw=0.05,
    p_blw_m=0.02,
    p_bls_m=0.08,
    p_bs=0.03,
    # benign params
    p_bw=0.05,
    p_blw_b=0.02,
    p_mw_benign=0.00,
    add_malware_in_benign_weak=False,
    p_ms=0.05,
    p_bls_b=0.08,
    return_float=True,
):
    device = x_batch.device
    x_bool = x_batch.bool()
    pseudo_labels = pseudo_labels.to(device)

    x_weak = x_bool.clone()
    x_strong = x_bool.clone()

    malware_rows = (pseudo_labels == 1)
    benign_rows = (pseudo_labels == 0)

    if malware_rows.any():
        xm = x_bool[malware_rows]
        xm_weak, xm_strong = malware_augment_pipeline_fast(
            xm,
            feature_groups_cuda,
            p_mw=p_mw,
            p_blw=p_blw_m,
            p_bls=p_bls_m,
            p_bs=p_bs,
            return_float=False,
        )
        x_weak[malware_rows] = xm_weak
        x_strong[malware_rows] = xm_strong

    if benign_rows.any():
        xb = x_bool[benign_rows]
        xb_weak, xb_strong = benign_augment_pipeline_fast(
            xb,
            feature_groups_cuda,
            p_bw=p_bw,
            p_blw=p_blw_b,
            p_mw=p_mw_benign,
            add_malware_in_weak=add_malware_in_benign_weak,
            p_ms=p_ms,
            p_bls=p_bls_b,
            return_float=False,
        )
        x_weak[benign_rows] = xb_weak
        x_strong[benign_rows] = xb_strong

    if return_float:
        return x_weak.float(), x_strong.float()
    return x_weak, x_strong


# ============================================================
# 14. Utility
# ============================================================
def count_new_activations(x_original, x_aug):
    x0 = x_original.bool()
    x1 = x_aug.bool()
    return ((~x0) & x1).sum(dim=1)




# ============================================================
# 2. Train classifier
# ============================================================
def train_classifier(X, y, model, optimizer, epochs=20, batch_size=512, lr=1e-3, device=None):
    X = X.float().to(device)
    y = y.long().to(device)


    n = X.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        model.train()

        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X[idx])
            loss = F.cross_entropy(logits, y[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * idx.size(0)

        model.eval()
        with torch.no_grad():
            pred = model(X).argmax(-1)
            acc = (pred == y).float().mean().item()

        if ep % 10 == 0:
            print(f"epoch {ep+1:02d}  loss={total_loss/n:.4f}  acc={acc:.4f}")

    return model


def random_bit_flip(x, n_bits=1):
    x_aug = x.clone()
    batch_size, num_features = x.shape

    # Generate random bit indices to flip for each sample
    flip_indices = torch.randint(0, num_features, (batch_size, n_bits), device=x.device)
    row_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).repeat(1, n_bits)

    # Flip bits
    x_aug[row_indices, flip_indices] = 1 - x_aug[row_indices, flip_indices]
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

    # Random mask indices (may contain duplicates)
    mask_indices = torch.randint(0, num_features, (batch_size, n_mask), device=x.device)
    row_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).repeat(1, n_mask)

    x_aug[row_indices, mask_indices] = 0
    return x_aug


def random_feature_mask_bernoulli(x, p=None, n_mask=None):
    """
    Randomly mask (set to 0) each feature in the input tensor with probability p using Bernoulli distribution.
    If n_mask is given, p is set so that on average n_mask features are masked per sample.

    Args:
        x: Tensor of shape (batch_size, num_features)
        p: Probability of masking each feature (float, between 0 and 1)
        n_mask: If given, overrides p so that p = n_mask / num_features

    Returns:
        Augmented tensor with features masked
    """
    x_aug = x.clone()
    batch_size, num_features = x.shape
    device = x.device

    if n_mask is not None:
        p = n_mask / num_features
    elif p is not None:
        p = float(p)
    else:
        p = 0.01  # default probability

    # Create Bernoulli mask (1 = mask, 0 = keep)
    mask = torch.bernoulli(torch.full_like(x_aug, p, device=device))

    # Apply mask by setting masked features to 0
    x_aug[mask.bool()] = 0

    return x_aug



def random_bit_flip_and_mask(x, n_bits=1, n_mask=1):
    x_aug = random_bit_flip(x, n_bits=n_bits)
    x_aug = random_feature_mask(x_aug, n_mask=n_mask)
    return x_aug
