import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math
import logging
import sys
from collections import Counter

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def get_dataset(path):
    
    file_path = f"{path}2012-01to2012-12_selected.npz"
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    # y_bin = np.array([0 if label == 0 else 1 for label in y])

    X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_ratio=0.4, random_state=42)

    # Create binary labels for the labeled set
    y_bin_labeled = np.array([0 if label == 0 else 1 for label in y_labeled])

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    y_2012_bin_labeled = torch.tensor(y_bin_labeled, dtype=torch.long).cuda()

    X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()

    return X_2012_labeled, y_2012_labeled, y_2012_bin_labeled, X_2012_unlabeled

def get_model_dims(model_name, input_layer_num, hidden_layer_num, output_layer_num):
    """convert hidden layer arguments to the architecture of a model (list)

    Arguments:
        model_name {str} -- 'MLP' or 'Contrastive AE' or 'Encoder'.
        input_layer_num {int} -- The number of the features.
        hidden_layer_num {str} -- The '-' connected numbers indicating the number of neurons in hidden layers.
        output_layer_num {int} -- The number of the classes.

    Returns:
        [list] -- List represented model architecture.
    """
    try:
        if '-' not in hidden_layer_num:
            if model_name == 'MLP':
                dims = [input_layer_num, int(hidden_layer_num), output_layer_num]
            else:
                dims = [input_layer_num, int(hidden_layer_num)]
        else:
            hidden_layers = [int(dim) for dim in hidden_layer_num.split('-')]
            dims = [input_layer_num]
            for dim in hidden_layers:
                dims.append(dim)
            if model_name == 'MLP':
                dims.append(output_layer_num)
        logging.debug(f'{model_name} dims: {dims}')
    except:
        logging.error(f'get_model_dims {model_name}')
        sys.exit(-1)

    return dims
    
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

# === Data Split Function ===
def split_labeled_unlabeled(X, y, labeled_ratio=0.1, stratify=True, random_state=42):
    # Remove classes with only one sample
    counts = Counter(y)
    mask = np.array([counts[label] > 1 for label in y])
    X = X[mask]
    y = y[mask]

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
    x_aug = x.clone()
    batch_size, num_features = x.shape
    for i in range(batch_size):
        flip_indices = torch.randperm(num_features)[:n_bits]
        x_aug[i, flip_indices] = 1 - x_aug[i, flip_indices]
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
    if n_bits is not None:
        p = float(n_bits * 10) / num_features
    else:
        if p is None:
            p = 0.01  # default
        else:
            p = float(p)
    flip_mask = torch.bernoulli(torch.full_like(x_aug, p, device=device))
    x_aug = torch.abs(x_aug - flip_mask)
    return x_aug

def random_feature_mask(x, n_mask=1):
    x_aug = x.clone()
    batch_size, num_features = x.shape
    for i in range(batch_size):
        mask_indices = torch.randperm(num_features)[:n_mask]
        x_aug[i, mask_indices] = 0
    return x_aug

def random_bit_flip_and_mask(x, n_bits=1, n_mask=1):
    x_aug = random_bit_flip(x, n_bits=n_bits)
    x_aug = random_feature_mask(x_aug, n_mask=n_mask)
    return x_aug

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])



class HiDistanceLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce='mean'):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(HiDistanceLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, features, binary_cat_labels, labels = None, margin = 0.5,
                weight = None, split = None):
        """
        Pair distance loss.

        Args:
            features: hidden vector of shape [bsz, feature_dim]. e.g., (512, 128)
            binary_cat_labels: one-hot binary labels.
            labels: ground truth of shape [bsz].
            margin: margin for dissimilar distance.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore entries for these
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if labels == None:
            raise ValueError('Need to define labels in DistanceLoss')
        
        # Ensure binary_cat_labels is [N, 2] one-hot
        if binary_cat_labels.dim() == 1 or binary_cat_labels.size(-1) == 1:
            binary_cat_labels = F.one_hot(binary_cat_labels.long().view(-1), num_classes=2).float().to(device)
        else:
            binary_cat_labels = binary_cat_labels.float().to(device)

        # features = F.normalize(features, p=2, dim=1)

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # similar masks
        # mask_{i,j}=1 if sample j has the same class as sample i.
        binary_labels = binary_cat_labels[:, 1].view(-1, 1)
        # mask: both malware, or both benign
        binary_mask = torch.eq(binary_labels, binary_labels.T).float().to(device)
        # multi_mask: same malware family, or benign
        multi_mask = torch.eq(labels, labels.T).float().to(device)
        # malware but not the same family. does not have benign.
        other_mal_mask = binary_mask - multi_mask
        # both benign samples
        ben_labels = torch.logical_not(binary_labels).float().to(device)
        same_ben_mask = torch.matmul(ben_labels, ben_labels.T)
        # same malware family mask
        same_mal_fam_mask = multi_mask - same_ben_mask
        
        # logging.debug("=== new batch ===")
        # pseudo loss
        if self.reduce == 'none':
            tmp = other_mal_mask
            other_mal_mask = same_mal_fam_mask
            same_mal_fam_mask = tmp
            # debug
            # split_index = torch.nonzero(split, as_tuple=True)[0]
            # logging.debug(f'split_index, {split_index}')
        # logging.debug(f'binary_labels {binary_labels}')
        # logging.debug(f'binary_mask {binary_mask}')
        # logging.debug(f'labels {labels}')
        # logging.debug(f'multi_mask {multi_mask}')
        # logging.debug(f'other_mal_mask = binary_mask - multi_mask {other_mal_mask}')
        # logging.debug(f'ben_labels {ben_labels}')
        # logging.debug(f'same_ben_mask {same_ben_mask}')
        # logging.debug(f'same_mal_fam_mask = multi_mask - same_ben_mask {same_mal_fam_mask}')
        
        # dissimilar mask. malware vs benign binary labels
        binary_negate_mask = torch.logical_not(binary_mask).float().to(device)
        # multi_negate_mask = torch.logical_not(multi_mask).float().to(device)

        # mask-out self-contrast cases
        diag_mask = torch.logical_not(torch.eye(batch_size)).float().to(device)
        # similar mask
        binary_mask = binary_mask * diag_mask
        multi_mask = multi_mask * diag_mask
        other_mal_mask = other_mal_mask * diag_mask
        same_ben_mask = same_ben_mask * diag_mask
        same_mal_fam_mask = same_mal_fam_mask * diag_mask

        # adjust the masks based on test indices
        if split is not None:
            split_index = torch.nonzero(split, as_tuple=True)[0]
            # instance-level loss, paired with training samples, pseudo loss
            # logging.debug(f'split_index, {split_index}')
            binary_negate_mask[:, split_index] = 0
            # multi_negate_mask[:, split_index] = 0
            binary_mask[:, split_index] = 0
            multi_mask[:, split_index] = 0
            other_mal_mask[:, split_index] = 0
            same_ben_mask[:, split_index] = 0
            same_mal_fam_mask[:, split_index] = 0

        # reference: https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/functional/pairwise/euclidean.py
        # not taking the sqrt for numerical stability
        x = features
        y = features
        x_norm = x.norm(dim=1, keepdim=True)
        y_norm = y.norm(dim=1).T
        distance_matrix = x_norm * x_norm + y_norm * y_norm - 2 * x.mm(y.T)
        # distance_matrix = torch.maximum(torch.tensor(1e-10), distance_matrix)
        # distance_matrix = torch.clamp(distance_matrix, min=1e-10, max=1e3)  # or 1e2
        distance_matrix = torch.clamp(distance_matrix, min=1e-6, max=10)

        # logging.debug(f'distance_matrix {distance_matrix}')
        # #logging.debug(f'torch.isnan(distance_matrix).any() {torch.isnan(distance_matrix).any()}')
        # logging.debug(f'same_ben_mask {same_ben_mask}')
        # logging.debug(f'other_mal_mask {other_mal_mask}')
        # logging.debug(f'same_mal_fam_mask {same_mal_fam_mask}')
        # logging.debug(f'binary_negate_mask {binary_negate_mask}')
        
        # four types of pairs
        # 1. ben, ben. same_ben_mask
        # 2. mal, mal from different families. other_mal_mask
        # 3. mal, mal from same families. same_mal_fam_mask
        # 4. ben, mal. binary_negate_mask

        # default is to compute mean for these values per sample
        if self.sample_reduce == 'mean' or self.sample_reduce == None:
            if weight == None:
                sum_same_ben = (same_ben_mask * distance_matrix).sum(1)
                # sum_same_ben = torch.maximum(
                #                     torch.sum(same_ben_mask * distance_matrix, dim=1) - \
                #                             same_ben_mask.sum(1) * torch.tensor(margin),
                #                     torch.tensor(0))
                sum_other_mal = torch.maximum(
                                    torch.sum(other_mal_mask * distance_matrix, dim=1) - \
                                            other_mal_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix, dim=1)
                # sum_bin_neg = torch.maximum(
                #                     binary_negate_mask.sum(1) * torch.tensor(2 * margin) - \
                #                             torch.sum(binary_negate_mask * distance_matrix,
                #                                     dim=1),
                #                     torch.tensor(0))
                
                sum_bin_neg = F.relu(margin - (binary_negate_mask * distance_matrix).sum(1))
                # logging.debug(f'sum_same_ben {sum_same_ben}, same_ben_mask.sum(1) {same_ben_mask.sum(1)}')
                # logging.debug(f'sum_other_mal {sum_other_mal}, other_mal_mask.sum(1) {other_mal_mask.sum(1)}')
                # logging.debug(f'sum_same_mal_fam {sum_same_mal_fam}, same_mal_fam_mask.sum(1) {same_mal_fam_mask.sum(1)}')
                # logging.debug(f'sum_bin_neg {sum_bin_neg}, binary_negate_mask.sum(1) {binary_negate_mask.sum(1)}')
            # weighted loss
            else:
                
                weight_matrix = torch.matmul(weight.view(-1, 1), weight.view(1, -1)).to(device)
                sum_same_ben = torch.maximum(
                                    torch.sum(same_ben_mask * distance_matrix * weight_matrix, dim=1) - \
                                            same_ben_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_other_mal = torch.maximum(
                                    torch.sum(other_mal_mask * distance_matrix * weight_matrix, dim=1) - \
                                            other_mal_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix * weight_matrix, dim=1)
                weight_prime = torch.div(1.0, weight)
                weight_matrix_prime = torch.matmul(weight_prime.view(-1, 1), weight_prime.view(1, -1)).to(device)
                sum_bin_neg = torch.maximum(
                                    binary_negate_mask.sum(1) * torch.tensor(2 * margin) - \
                                            torch.sum(binary_negate_mask * distance_matrix * weight_matrix_prime,
                                                    dim=1),
                                    torch.tensor(0))
            loss = sum_same_ben / torch.maximum(same_ben_mask.sum(1), torch.tensor(1)) + \
                    sum_other_mal / torch.maximum(other_mal_mask.sum(1), torch.tensor(1)) + \
                    sum_same_mal_fam / torch.maximum(same_mal_fam_mask.sum(1), torch.tensor(1)) + \
                    sum_bin_neg / torch.maximum(binary_negate_mask.sum(1), torch.tensor(1))
        elif self.sample_reduce == 'max':
            max_same_ben = torch.maximum(
                                torch.amax(same_ben_mask * distance_matrix, 1) - \
                                        torch.tensor(margin),
                                torch.tensor(0))
            max_other_mal = torch.maximum(
                                torch.amax(other_mal_mask * distance_matrix, 1) - \
                                        torch.tensor(margin),
                                torch.tensor(0))
            max_same_mal_fam = torch.amax(same_mal_fam_mask * distance_matrix, 1)
            max_bin_neg = torch.maximum(
                                torch.tensor(2 * margin) - \
                                        torch.amin(binary_negate_mask * distance_matrix, 1),
                                torch.tensor(0))
            loss = max_same_ben + max_other_mal + max_same_mal_fam + max_bin_neg
        else:
            raise Exception(f'sample_reduce = {self.sample_reduce} not implemented yet.')

        if self.reduce == 'mean':
            loss = loss.mean()
        
        return loss

class HiDistanceXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(HiDistanceXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, xent_lambda,
            y_bin_pred, y_bin_batch,
            features, labels = None,
            margin = 1.0,
            weight = None,
            split = None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        Dist = HiDistanceLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
        # try not giving any weight to HiDistanceLoss
        supcon_loss = Dist(features, y_bin_batch, labels = labels, margin = margin, weight = None, split = split)
        
        xent_bin_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_bin_pred[:, 1], y_bin_batch.float(), reduction = 'mean')
        
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()

        loss = supcon_loss + xent_lambda * xent_bin_loss
        
        del Dist
        torch.cuda.empty_cache()

        return supcon_loss, supcon_loss, supcon_loss
    


# def hierarchical_contrastive_loss(z, y, y_fam, margin=1.0):
#     batch_size = z.size(0)
#     dists = torch.cdist(z, z, p=2)
#     loss = 0.0

#     for i in range(batch_size):
#         Pi = [j for j in range(batch_size) if y[j] == y[i] and (y[i] == 0 or y_fam[j] != y_fam[i]) and j != i]
#         Pzi = [j for j in range(batch_size) if y[i] == y[j] == 1 and y_fam[j] == y_fam[i] and j != i]
#         Ni = [j for j in range(batch_size) if y[j] != y[i]]

#         term1 = torch.sum(torch.clamp(dists[i][Pi] - margin, min=0)) / (len(Pi) if Pi else 1)
#         term2 = torch.sum(dists[i][Pzi]) / (len(Pzi) if Pzi else 1)
#         term3 = torch.sum(torch.clamp(2 * margin - dists[i][Ni], min=0)) / (len(Ni) if Ni else 1)

#         loss += term1 + term2 + term3

#     return loss / batch_size

def hierarchical_contrastive_loss(z, y, y_fam, margin=1.0):
    """
    Fully vectorized hierarchical contrastive loss without Python loops.
    """
    batch_size = z.size(0)
    dists = torch.cdist(z, z, p=2)  # [B, B]

    y = y.view(-1, 1)  # [B, 1]
    y_fam = y_fam.view(-1, 1)

    same_class = (y == y.T)  # [B, B]
    same_family = (y_fam == y_fam.T)
    same_sample = torch.eye(batch_size, device=z.device).bool()

    # Set masks
    Pz_mask = (y == 1) & same_class & same_family & ~same_sample  # term2
    P_mask = same_class & ((y == 0) | ((y == 1) & ~same_family)) & ~same_sample  # term1
    N_mask = (y != y.T)  # term3

    # Term 1: weak positives (same label but different family or benign)
    term1 = torch.clamp(dists - margin, min=0) * P_mask
    term1_sum = term1.sum(dim=1)
    term1_count = P_mask.sum(dim=1).clamp(min=1)

    # Term 2: strong positives (same family)
    term2 = dists * Pz_mask
    term2_sum = term2.sum(dim=1)
    term2_count = Pz_mask.sum(dim=1).clamp(min=1)

    # Term 3: negatives (different label)
    term3 = torch.clamp(2 * margin - dists, min=0) * N_mask
    term3_sum = term3.sum(dim=1)
    term3_count = N_mask.sum(dim=1).clamp(min=1)

    # Combine
    total_loss = (term1_sum / term1_count +
                  term2_sum / term2_count +
                  term3_sum / term3_count).mean()

    return total_loss


def classification_loss(preds, targets):
    return F.binary_cross_entropy(preds.view(-1), targets.float())

def combined_loss(preds, embeddings, y, y_fam, margin=1.0, lambda_weight=1.0):
    lce = classification_loss(preds, y)
    lhc = hierarchical_contrastive_loss(embeddings, y, y_fam, margin)
    return lhc + lambda_weight * lce, lhc.item(), lce.item()


