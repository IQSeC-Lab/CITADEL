import torch
import random

# Parameter normalization base
PARAMETER_MAX = 10

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

# ===== Augmentation Functions =====

def Identity(x, **kwargs):
    return x

def BitFlip(x, v, max_v, **kwargs):
    num_bits = _int_parameter(v, max_v)
    x = x.clone()
    for i in range(x.size(0)):  # apply per sample
        indices = torch.randperm(x.size(1))[:num_bits]
        x[i, indices] = 1 - x[i, indices]
    return x

def FeatureMask(x, v, max_v, **kwargs):
    num_mask = _int_parameter(v, max_v)
    x = x.clone()
    for i in range(x.size(0)):
        indices = torch.randperm(x.size(1))[:num_mask]
        x[i, indices] = 0
    return x

def FeatureDropIn(x, v, max_v, **kwargs):
    num_set = _int_parameter(v, max_v)
    x = x.clone()
    for i in range(x.size(0)):
        indices = torch.randperm(x.size(1))[:num_set]
        x[i, indices] = 1
    return x

# ===== Augmentation Pool =====

def malware_augment_pool():
    return [
        (Identity, None),
        (BitFlip, 10),
        (FeatureMask, 10),
        (FeatureDropIn, 10),
    ]

# ===== RandAugment for 1D Malware Feature Vectors =====

class RandAugment1D:
    def __init__(self, n=2, m=5):
        """
        n: number of transforms to apply
        m: magnitude (strength) of each transform [1,10]
        """
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.pool = malware_augment_pool()

    def __call__(self, x):
        ops = random.choices(self.pool, k=self.n)
        for op, max_v in ops:
            if random.random() < 0.8:  # typical augmentation application probability
                v = random.randint(1, self.m)
                x = op(x, v=v, max_v=max_v)
        return x
