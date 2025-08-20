import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


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