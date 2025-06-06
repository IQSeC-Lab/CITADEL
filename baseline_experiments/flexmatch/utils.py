import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

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
    

def bit_flipping(X_labeled, X_unlabeled, flip_ratio_weak, flip_ratio_strong):
    # Example unlabeled and labeled sets
    # X_unlabeled = np.array([
    #     [1, 0, 1, 0],
    #     [0, 1, 1, 1]
    # ])

    # X_labeled = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 1, 0],
    #     [1, 1, 1, 1]
    # ])

    # Step 1: Compute Hamming distances
    hamming_dist = np.sum(X_unlabeled[:, np.newaxis, :] != X_labeled[np.newaxis, :, :], axis=2)

    # Step 2: Choose the closest labeled sample for each unlabeled sample
    closest_indices = np.argmin(hamming_dist, axis=1)
    farthest_indices = np.argmax(hamming_dist, axis=1)
    # Step 3: Flip 50% of mismatched bits
    X_modified_weak = X_unlabeled.copy()
    X_modified_strong = X_unlabeled.copy()

    for i, closest_idx in enumerate(closest_indices):
        x_u = X_unlabeled[i]
        x_l = X_labeled[closest_idx]
        
        # Find mismatched positions
        mismatches = np.where(x_u != x_l)[0]
        
        # Select 50% of mismatches (rounded down)
        num_to_flip = len(mismatches) * flip_ratio_weak
        if num_to_flip > 0:
            flip_indices = np.random.choice(mismatches, size=num_to_flip, replace=False)
            X_modified_weak[i, flip_indices] = 1 - X_modified_weak[i, flip_indices]  # Flip bits

    for i, farthest_idx in enumerate(farthest_indices):
        x_u = X_unlabeled[i]
        x_l = X_labeled[farthest_idx]
        
        # Find mismatched positions
        mismatches = np.where(x_u != x_l)[0]
        
        # Select 50% of mismatches (rounded down)
        num_to_flip = len(mismatches) * flip_ratio_strong
        if num_to_flip > 0:
            flip_indices = np.random.choice(mismatches, size=num_to_flip, replace=False)
            X_modified_strong[i, flip_indices] = 1 - X_modified_strong[i, flip_indices]  # Flip bits

    # print("Original X_unlabeled:")
    # print(X_unlabeled)
    # print("Modified X_unlabeled (50% flipped where mismatched):")
    # print(X_modified)

    return X_modified_weak, X_modified_strong
