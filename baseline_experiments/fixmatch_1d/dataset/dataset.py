# Dataset support for 1D binary malware feature vectors with FixMatch-style augmentation
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import random
from fixmatch_malware_mlp import RandAugmentMC1D

class MalwareFeatureDataset(Dataset):
    def __init__(self, features, labels=None, transform=None):
        self.features = features  # torch.Tensor [N, D]
        self.labels = labels      # torch.Tensor [N] or None
        self.transform = transform

    def __getitem__(self, index):
        x = self.features[index]
        if self.transform:
            x = self.transform(x.unsqueeze(0)).squeeze(0)
        if self.labels is not None:
            y = self.labels[index]
            return x, y
        else:
            return x,

    def __len__(self):
        return len(self.features)

class TransformFixMatch1D:
    def __init__(self, weak_aug=None, strong_aug=None):
        self.weak = weak_aug if weak_aug else RandAugmentMC1D(n=1, m=2)
        self.strong = strong_aug if strong_aug else RandAugmentMC1D(n=2, m=8)

    def __call__(self, x):
        return self.weak(x), self.strong(x)

def get_malware_dataset(args, features, labels):
    labels = np.array(labels)
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    labeled_idxs, unlabeled_idxs = x_u_split(args, labels.numpy())
    X_l = features[labeled_idxs]
    y_l = labels[labeled_idxs]
    X_u = features[unlabeled_idxs]

    labeled_ds = MalwareFeatureDataset(X_l, y_l)
    unlabeled_ds = MalwareFeatureDataset(X_u, transform=TransformFixMatch1D())
    test_ds = MalwareFeatureDataset(features, labels)  # or use a real test set

    return labeled_ds, unlabeled_ds, test_ds

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labeled_idx = []
    unlabeled_idx = np.arange(len(labels))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, replace=False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx
