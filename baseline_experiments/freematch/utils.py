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

def get_dataset(path):
    
    file_path = f"{path}2012-01to2012-12_selected.npz"
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    y = np.array([0 if label == 0 else 1 for label in y])

    X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_ratio=0.4, random_state=42)

    # Filter out the undesired features from the datasets
    X_labeled, X_unlabeled = create_dataset_without_mal_features(X_labeled, X_unlabeled)

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()

    return X_2012_labeled, y_2012_labeled, X_2012_unlabeled

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
    
class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


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
def random_bit_flip(x, n_bits=1, fixed_set="all"):
    """
    Randomly flip n_bits in each sample of the batch.
    Args:
        x: Tensor of shape (batch_size, num_features)
        n_bits: Number of bits (features) to flip per sample
    Returns:
        Augmented tensor with bits flipped
    """
    # if fixed_set == "notmal": # Malware features are not being taken
    #     x_aug = x.clone()
    #     batch_size, num_features = x.shape
    #     exclude_indices = set(get_mal_feature_indices())
    #     for i in range(batch_size):
    #         # Flip random bits, but exclude indices from get_mal_feature_indices()
    #         available_indices = [i for i in range(num_features) if i not in exclude_indices]
    #         flip_indices = torch.tensor(
    #             np.random.choice(available_indices, size=n_bits, replace=False),
    #             device=x.device,
    #             dtype=torch.long
    #         )
    #         x_aug[i, flip_indices] = 1 - x_aug[i, flip_indices]
    #     return x_aug
    
    # elif fixed_set == "all": # All features are being taken
    x_aug = x.clone()
    batch_size, num_features = x.shape
    for i in range(batch_size):
        flip_indices = torch.randperm(num_features)[:n_bits]
        x_aug[i, flip_indices] = 1 - x_aug[i, flip_indices]
    return x_aug
    
    # elif fixed_set == "mal": # Only malware features are being taken
    #     x_aug = x.clone()
    #     batch_size, num_features = x.shape
    #     feat_indices = get_mal_feature_indices()
    #     for i in range(batch_size):
    #         flip_indices = torch.tensor(np.random.choice(feat_indices, size=n_bits, replace=False))
    #         x_aug[i, flip_indices] = 1 - x_aug[i, flip_indices]
    #     return x_aug

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


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name or 'bias' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)
    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def consistency_loss(logits_s, logits_w, time_p, p_model, name='ce', use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        uw_prob = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(uw_prob, dim=-1)
        p_cutoff = time_p
        p_model_cutoff = p_model / torch.max(p_model, dim=-1)[0]
        threshold = p_cutoff * p_model_cutoff[max_idx]
        # if dataset == 'svhn':
        #     threshold = torch.clamp(threshold, min=0.9, max=0.95)
        mask = max_probs > threshold #max_probs.ge(threshold)
        
        if use_hard_labels:
            loss_fn = nn.CrossEntropyLoss(reduction='none')  # Create the loss function
            # logits_s = logits_s[:, 1]
            masked_loss = loss_fn(logits_s, max_idx) * mask.float()
        # else:
        #     pseudo_label = torch.softmax(logits_w / T, dim=-1)
        #     masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
        return masked_loss.mean(), mask

    else:
        assert Exception('Not Implemented consistency_loss')

def create_dataset_without_mal_features(X_labeled, X_unlabeled=None):
    """
    Returns modified datasets after excluding the specified feature indices.

    Parameters:
        X_labeled (np.ndarray): The labeled data features.
        X_unlabeled (np.ndarray): The unlabeled data features.
        ignore_features (list): List of column indices to remove.

    Returns:
        np.ndarray, np.ndarray: The filtered labeled and unlabeled data.
    """
    ignore_features = get_mal_feature_indices()
    X_labeled_filtered = np.delete(X_labeled, ignore_features, axis=1)

    if X_unlabeled is None:
        return X_labeled_filtered
    
    X_unlabeled_filtered = np.delete(X_unlabeled, ignore_features, axis=1)
    return X_labeled_filtered, X_unlabeled_filtered

def get_mal_feature_indices():
    return [0, 8, 25, 31, 32, 34, 95, 111, 118, 120, 121, 122, 129, 147, 148, 149, 151, 152, 242, 243, 244, 282, 285, 286, 288, 289, 292, 296, 326, 335, 353, 354, 363, 364, 421, 422, 423, 481, 482, 484, 485, 486, 491, 492, 493, 494, 497, 498, 513, 516, 518, 521, 527, 528, 531, 543, 544, 545, 546, 547, 553, 554, 595, 596, 616, 656, 658, 659, 660, 699]
