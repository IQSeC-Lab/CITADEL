import logging
import random
import numpy as np

# This structure remains IDENTICAL to the original code.
# We are simply providing new functions with the same names.

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

# ==============================================================================
# 1D VECTOR IMPLEMENTATIONS of FixMatch Augmentation Functions
#
# Each function below is a logical translation of the original PIL function
# to operate on a 1D NumPy binary vector (`vec`).
# ==============================================================================

def AutoContrast(vec, **kwarg):
    """1D Analogue: No direct analogue. Returns vector as is. Equivalent to Identity."""
    return vec.copy()

def Brightness(vec, v, max_v, bias=0):
    """1D Analogue: Flips a percentage of bits from 0->1 or 1->0.
    'Brightness' changes pixel intensity. Here, we change bit intensity.
    """
    p = _float_parameter(v, max_v) + bias
    aug_vec = vec.copy()
    
    # Choose indices to potentially flip
    num_to_modify = int(p * len(vec))
    indices = np.random.choice(len(vec), size=num_to_modify, replace=False)
    
    # Flip the bits at the chosen indices
    aug_vec[indices] = 1 - aug_vec[indices]
    return aug_vec

def Color(vec, v, max_v, bias=0):
    """1D Analogue: Randomly drops active features (1s -> 0s).
    'Color' saturation change is like washing out features.
    """
    p = _float_parameter(v, max_v) + bias
    aug_vec = vec.copy()
    active_indices = np.where(aug_vec == 1)[0]
    
    if len(active_indices) > 0:
        num_to_drop = int(p * len(active_indices))
        indices_to_drop = np.random.choice(active_indices, size=num_to_drop, replace=False)
        aug_vec[indices_to_drop] = 0
    return aug_vec

def Contrast(vec, v, max_v, bias=0):
    """1D Analogue: Randomly adds features (0s -> 1s).
    'Contrast' makes features stand out more.
    """
    p = _float_parameter(v, max_v) + bias
    aug_vec = vec.copy()
    inactive_indices = np.where(aug_vec == 0)[0]
    
    if len(inactive_indices) > 0:
        num_to_add = int(p * len(inactive_indices))
        indices_to_add = np.random.choice(inactive_indices, size=num_to_add, replace=False)
        aug_vec[indices_to_add] = 1
    return aug_vec

def Equalize(vec, **kwarg):
    """1D Analogue: Tries to balance 0s and 1s by flipping from the majority class.
    'Equalize' spreads out the histogram. Here, we balance the feature distribution.
    """
    aug_vec = vec.copy()
    n_ones = np.sum(aug_vec)
    n_zeros = len(aug_vec) - n_ones

    # Flip some of the majority class to the minority class
    if n_ones > n_zeros:
        indices_to_flip = np.where(aug_vec == 1)[0]
        num_to_flip = (n_ones - n_zeros) // 4 # Mild equalization
    else:
        indices_to_flip = np.where(aug_vec == 0)[0]
        num_to_flip = (n_zeros - n_ones) // 4 # Mild equalization

    if len(indices_to_flip) > 0 and num_to_flip > 0:
        flip_candidates = np.random.choice(indices_to_flip, size=min(num_to_flip, len(indices_to_flip)), replace=False)
        aug_vec[flip_candidates] = 1 - aug_vec[flip_candidates]
        
    return aug_vec

def Identity(vec, **kwarg):
    """Returns the original vector, unchanged."""
    return vec.copy()

def Posterize(vec, v, max_v, bias=0):
    """1D Analogue: Not applicable to binary. Returns Identity."""
    return vec.copy()

def Rotate(vec, v, max_v, bias=0):
    """1D Analogue: Circularly shifts the vector elements.
    'Rotate' for an image is a geometric shift. A circular shift is the 1D equivalent.
    """
    amount = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        amount = -amount
    return np.roll(vec, shift=amount)

def Sharpness(vec, v, max_v, bias=0):
    """1D Analogue: No direct analogue. Implemented as a mild bit flip."""
    return Brightness(vec, v, max_v / 2, bias) # Less intense version of brightness

def ShearX(vec, v, max_v, bias=0):
    """1D Analogue: Swaps random pairs of adjacent features."""
    p = _float_parameter(v, max_v) + bias
    aug_vec = vec.copy()
    num_swaps = int(p * len(vec) / 2)
    for _ in range(num_swaps):
        idx = random.randint(0, len(vec) - 2)
        aug_vec[idx], aug_vec[idx+1] = aug_vec[idx+1], aug_vec[idx]
    return aug_vec

def ShearY(vec, v, max_v, bias=0):
    """1D Analogue: Same as ShearX."""
    return ShearX(vec, v, max_v, bias)

def Solarize(vec, v, max_v, bias=0):
    """1D Analogue: Inverts a portion of the vector."""
    p = _float_parameter(v, max_v)
    aug_vec = vec.copy()
    num_to_invert = int(p * len(vec) / 256) # Scale to match original Solarize range
    
    start_idx = random.randint(0, len(vec) - num_to_invert)
    indices_to_invert = range(start_idx, start_idx + num_to_invert)
    aug_vec[indices_to_invert] = 1 - aug_vec[indices_to_invert]
    return aug_vec

def TranslateX(vec, v, max_v, bias=0):
    """1D Analogue: Shifts vector, padding with zeros.
    'Translate' moves the image, leaving empty space. Here, we shift and pad.
    """
    amount = _int_parameter(v * len(vec), max_v) + bias
    if random.random() < 0.5:
        amount = -amount

    aug_vec = np.roll(vec, shift=amount)
    if amount > 0:
        aug_vec[:amount] = 0
    elif amount < 0:
        aug_vec[amount:] = 0
    return aug_vec

def TranslateY(vec, v, max_v, bias=0):
    """1D Analogue: Same as TranslateX for a 1D vector."""
    return TranslateX(vec, v, max_v, bias)
    
def CutoutAbs(vec, v, **kwarg):
    """1D Analogue: Sets a contiguous block of features to 0."""
    v = int(v)
    if v == 0:
        return vec.copy()

    start_idx = np.random.randint(0, len(vec) - v + 1)
    aug_vec = vec.copy()
    aug_vec[start_idx : start_idx + v] = 0
    return aug_vec

# ==============================================================================
# UNCHANGED CODE FROM FIXMATCH
# This section is kept exactly as it is in the original.
# ==============================================================================

def fixmatch_augment_pool():
    # This pool now references our 1D vector functions with the same names
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, 0, 1),
            (Identity, 0, 1),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs

class RandAugmentMC(object):
    def __init__(self, n, m, feature_vec_len):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()
        self.feature_vec_len = feature_vec_len

    def __call__(self, vec):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            # The original code's bias is for images, may not be ideal for vectors
            # but we keep the structure.
            bias_to_use = bias if bias is not None else 0
            v = np.random.randint(1, self.m + 1)
            if random.random() < 0.5:
                # This now calls our new vector-based functions
                vec = op(vec, v=v, max_v=max_v, bias=bias_to_use)
        
        # The final CutoutAbs is a key part of FixMatch's strong augmentation
        cutout_size = int(self.feature_vec_len * 0.5) # Example: Cutout 50% of the vector length
        vec = CutoutAbs(vec, cutout_size)
        return vec

# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':
    FEATURE_VECTOR_LENGTH = 100
    N_OPS = 2
    MAGNITUDE = 9

    # Create a sample malware feature vector (1D binary numpy array)
    original_vector = np.random.choice([0, 1], size=FEATURE_VECTOR_LENGTH, p=[0.7, 0.3])

    # Initialize the augmentation class - it's the SAME class name and structure
    augmenter = RandAugmentMC(n=N_OPS, m=MAGNITUDE, feature_vec_len=FEATURE_VECTOR_LENGTH)

    # Apply the augmentation
    augmented_vector = augmenter(original_vector)

    print("="*60)
    print(f"RandAugmentMC applied to 1D Malware Vector (n={N_OPS}, m={MAGNITUDE})")
    print("Concept: Kept RandAugmentMC class, re-implemented functions for vectors.")
    print("="*60)
    print(f"Original Vector (active features={np.sum(original_vector)}):\n{original_vector}")
    print("-"*60)
    print(f"Augmented Vector (active features={np.sum(augmented_vector)}):\n{augmented_vector}")
    print("-"*60)
    diff = np.sum(original_vector != augmented_vector)
    print(f"Number of bits changed: {diff} ({diff / FEATURE_VECTOR_LENGTH:.2%})")