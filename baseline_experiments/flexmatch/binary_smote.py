import numpy as np
import random
from collections import defaultdict

def binary_smote(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    new_sample = []
    for a, b in zip(x1, x2):
        if a == 1 and b == 1:
            new_sample.append(1)
        elif a == 0 and b == 0:
            new_sample.append(0)
        else:
            new_sample.append(random.choice([0, 1]))
    return np.array(new_sample)

def binary_smote_batch(X_batch, y_batch, n_synthetic=1):
    """
    Args:
        X_batch: binary vectors of shape (B, F)
        y_batch: class labels (pseudo-labels) of shape (B,)
        n_synthetic: number of synthetic samples per class
    Returns:
        X_augmented: original + synthetic samples
        y_augmented: corresponding labels
    """
    class_groups = defaultdict(list)
    for x, y in zip(X_batch, y_batch):
        class_groups[y].append(x)

    synthetic_X = []
    synthetic_y = []

    for label, samples in class_groups.items():
        n = len(samples)
        if n < 2:
            continue
        for _ in range(min(n_synthetic, n * (n - 1) // 2)):
            i, j = random.sample(range(n), 2)
            new_x = binary_smote(samples[i], samples[j])
            synthetic_X.append(new_x)
            synthetic_y.append(label)

    # Combine real + synthetic
    X_out = np.vstack([X_batch, np.array(synthetic_X)])
    y_out = np.concatenate([y_batch, np.array(synthetic_y)])
    return X_out, y_out


def weak_augment_batch(X_batch, drop_prob=0.05):
    """
    Drop a small % of 1s in each sample.
    """
    X_aug = []
    for x in X_batch:
        x = x.clone()
        x_np = x.cpu().numpy()  # Ensure on CPU and NumPy
        one_indices = np.where(x_np == 1)[0]
        drop_count = int(len(one_indices) * drop_prob)
        if drop_count > 0:
            drop_indices = np.random.choice(one_indices, drop_count, replace=False)
            x_np[drop_indices] = 0
        X_aug.append(x_np)
    return np.array(X_aug)

def strong_augment_batch(X_batch, drop_prob=0.3, do_mix=True):
    """
    Apply strong augmentation: heavy dropout + optional OR mixing.
    """
    X_aug = []
    B = X_batch.shape[0]

    for i, x in enumerate(X_batch):
        x = x.clone()
        x_np = x.cpu().numpy()
        # Dropout part
        one_indices = np.where(x_np == 1)[0]
        drop_count = int(len(one_indices) * drop_prob)
        if drop_count > 0:
            drop_indices = np.random.choice(one_indices, drop_count, replace=False)
            x_np[drop_indices] = 0

        # Optional mix with another random sample from the batch
        if do_mix:
            j = np.random.choice([idx for idx in range(B) if idx != i])
            x_mixed = np.logical_or(x_np, X_batch[j].cpu().numpy()).astype(int)
            X_aug.append(x_mixed)
        else:
            X_aug.append(x_np)

    return np.array(X_aug)

