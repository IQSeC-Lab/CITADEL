# RandAugment-style Augmentation for 1D Binary Feature Vectors (Malware)
import torch
import random
import numpy as np

PARAMETER_MAX = 10

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX
)

def Identity(x, **kwargs):
    return x

def Brightness(x, v, max_v, bias=0):
    return x * (_float_parameter(v, max_v) + bias)

def Contrast(x, v, max_v, bias=0):
    factor = _float_parameter(v, max_v) + bias
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * factor + mean

def Cutout(x, v, max_v, bias=0):
    if v == 0:
        return x
    v = _float_parameter(v, max_v) + bias
    num_features = x.size(1)
    mask_len = int(v * num_features)
    return CutoutAbs(x, mask_len)

def CutoutAbs(x, v, **kwargs):
    x = x.clone()
    for i in range(x.size(0)):
        start = random.randint(0, max(0, x.size(1) - v))
        x[i, start:start+v] = 0
    return x

def Invert(x, **kwargs):
    return 1 - x

def Posterize(x, v, max_v, bias=0):
    bits = _int_parameter(v, max_v) + bias
    if bits >= 8:
        return x
    scale = 2 ** bits
    return (x * scale).floor() / scale

def Rotate(x, v, max_v, bias=0):
    shift = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        shift = -shift
    return torch.roll(x, shifts=shift, dims=1)

def Sharpness(x, v, max_v, bias=0):
    weight = _float_parameter(v, max_v) + bias
    smooth = torch.nn.functional.avg_pool1d(x.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    return torch.clamp(x + weight * (x - smooth), 0, 1)

def TranslateX(x, v, max_v, bias=0):
    shift = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        shift = -shift
    return torch.roll(x, shifts=shift, dims=1)

def fixmatch_augment_pool():
    return [
        (Identity, None, 0),
        (Brightness, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Cutout, 0.2, 0),
        (Invert, None, 0),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 0.9, 0.05),
        (TranslateX, 0.3, 0),
    ]

class RandAugmentMC1D:
    def __init__(self, n, m):
        assert n >= 1 and 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, x):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = random.randint(1, self.m)
            if random.random() < 0.5:
                x = op(x, v=v, max_v=max_v, bias=bias)
        x = CutoutAbs(x, int(0.5 * x.size(1)))
        return x

# Example usage:
# x: torch.Tensor of shape [B, D], values in [0, 1] or {0, 1}
# aug = RandAugmentMC1D(n=2, m=5)
# x_aug = aug(x)
