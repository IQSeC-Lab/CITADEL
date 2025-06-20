import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def mapping_func(mode='convex'):
    """Returnt he Mapping func."""
    if mode == 'convex':
        return lambda x: x / (2 - x)
    

def bit_flipping(X_labeled, y_labeled, X_unlabeled, flip_ratio_weak=0.05, flip_ratio_strong=0.20, batch_size=64):
    
    if isinstance(X_unlabeled, torch.Tensor):
        X_unlabeled = X_unlabeled.cpu().numpy()
    if isinstance(X_labeled, torch.Tensor):
        X_labeled = X_labeled.cpu().numpy()
    if isinstance(y_labeled, torch.Tensor):
        y_labeled = y_labeled.cpu().numpy()

    # Ensure binary integer type for bitwise ops
    X_unlabeled = X_unlabeled.astype(np.uint8)
    X_labeled = X_labeled.astype(np.uint8)

    n_unlabeled = X_unlabeled.shape[0]
    X_modified_weak = X_unlabeled.copy()
    X_modified_strong = X_unlabeled.copy()

    # Getting the sample indices which are malwares
    # we will be augmenting all samples in X_unlabeld w.r.t X_labels that means only malware families
    pos_indices = np.where(y_labeled > 0)[0]
    X_labeled = X_labeled[pos_indices]

    min_hamming, max_hamming = 1000009, -1

    for start in tqdm(range(0, n_unlabeled, batch_size), desc="Bit Flipping Batches"):
        end = min(start + batch_size, n_unlabeled)
        X_batch = X_unlabeled[start:end]

        # Compute Hamming distances: shape (batch_size, n_labeled)
        diff = X_batch[:, np.newaxis, :] != X_labeled[np.newaxis, :, :]
        hamming_dist = np.sum(diff, axis=2)
        # Save the hamming distance for this batch to a file

        min_hamming = min(min_hamming, hamming_dist.min())
        max_hamming = max(max_hamming, hamming_dist.max())
        # all_hamming = []

        closest_indices = np.argmin(hamming_dist, axis=1)
        farthest_indices = np.argmax(hamming_dist, axis=1)

        for i, (x_u, close_idx, far_idx) in enumerate(zip(X_batch, closest_indices, farthest_indices)):
            abs_i = start + i  # global index in X_unlabeled

            # Weak flip (closest)
            x_l_close = X_labeled[close_idx]
            mismatch_weak = np.where(x_u != x_l_close)[0]
            n_weak = int(len(mismatch_weak) * flip_ratio_weak)
            if n_weak > 0:
                flip_indices = np.random.choice(mismatch_weak, size=n_weak, replace=False)
                X_modified_weak[abs_i, flip_indices] ^= 1

            # Strong flip (farthest)
            x_l_far = X_labeled[far_idx]
            mismatch_strong = np.where(x_u != x_l_far)[0]
            n_strong = int(len(mismatch_strong) * flip_ratio_strong)
            if n_strong > 0:
                flip_indices = np.random.choice(mismatch_strong, size=n_strong, replace=False)
                X_modified_strong[abs_i, flip_indices] ^= 1

    return torch.tensor(X_modified_weak), torch.tensor(X_modified_strong), min_hamming, max_hamming

