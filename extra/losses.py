import logging
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, reduce = 'mean'):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(TripletLoss, self).__init__()
        self.reduce = reduce

    def forward(self, features, labels = None, margin = 10.0,
                weight = None, split = None):
        """
        Triplet loss for model.

        Args:
            features: hidden vector of shape [bsz, feature_dim]. e.g., (512, 128)
            labels: ground truth of shape [bsz].
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        batch_size = features.shape[0]

        pass_size = batch_size // 3
        """
        three shares of pass_size
        1) training data sample
        2) positive samples
        3) negative samples
        """
        anchor = features[:pass_size]
        positive = features[pass_size:pass_size*2]
        negative = features[pass_size*2:]
        positive_losses = torch.maximum(torch.tensor(1e-10), torch.linalg.norm(anchor - positive, ord = 2, dim = 1))
        negative_losses = torch.maximum(torch.tensor(0), margin - torch.linalg.norm(anchor - negative, ord = 2, dim = 1))

        if weight is not None:
            anchor_weight = weight[:pass_size]
            positive_weight = weight[pass_size:pass_size*2]
            negative_weight = weight[pass_size*2:]
            positive_losses = positive_losses * anchor_weight * positive_weight
            negative_losses = negative_losses * positive_weight * negative_weight
        
        loss = positive_losses + negative_losses

        if self.reduce == 'mean':
            loss = loss.mean()

        return loss

class TripletMSELoss(nn.Module):
    def __init__(self, reduce = 'mean'):
        super(TripletMSELoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce

    def forward(self, cae_lambda,
            x, x_prime,
            features, labels = None,
            margin = 10.0,
            weight = None,
            split = None):
        """
        Args:
            cae_lambda: scale the CAE loss
            x: input to the Autoencoder
            x_prime: decoded x' from Autoencoder
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data
        Returns:
            A loss scalar.
        """
        Triplet = TripletLoss(reduce = self.reduce)
        supcon_loss = Triplet(features, labels = labels, margin = margin, weight = weight, split = split)

        mse_loss = torch.nn.functional.mse_loss(x, x_prime, reduction = self.reduce)
        
        loss = cae_lambda * supcon_loss + mse_loss
        
        del Triplet
        torch.cuda.empty_cache()

        return loss, supcon_loss, mse_loss

import torch
import torch.nn as nn

class HiDistanceLoss(nn.Module):
    def __init__(self, reduce='mean', sample_reduce='mean'):
        super(HiDistanceLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, features, binary_cat_labels, labels=None, margin=10.0,
                weight=None, split=None):
        if labels is None:
            raise ValueError('Need to define labels in DistanceLoss')

        device = features.device
        features = features.to(device)
        binary_cat_labels = binary_cat_labels.to(device)
        labels = labels.to(device)
        if weight is not None:
            weight = weight.to(device)
        if split is not None:
            split = split.to(device)

        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)

        binary_labels = binary_cat_labels[:, 1].view(-1, 1)
        binary_mask = (binary_labels == binary_labels.T).float()
        multi_mask = (labels == labels.T).float()
        other_mal_mask = binary_mask - multi_mask
        ben_labels = (~binary_labels.bool()).float()
        same_ben_mask = ben_labels @ ben_labels.T
        same_mal_fam_mask = multi_mask - same_ben_mask
        binary_negate_mask = (~(binary_mask.bool())).float()

        diag_mask = (~torch.eye(batch_size, dtype=torch.bool, device=device)).float()
        binary_mask *= diag_mask
        multi_mask *= diag_mask
        other_mal_mask *= diag_mask
        same_ben_mask *= diag_mask
        same_mal_fam_mask *= diag_mask
        binary_negate_mask *= diag_mask

        if split is not None:
            split_index = torch.nonzero(split, as_tuple=True)[0]
            binary_mask[:, split_index] = 0
            multi_mask[:, split_index] = 0
            other_mal_mask[:, split_index] = 0
            same_ben_mask[:, split_index] = 0
            same_mal_fam_mask[:, split_index] = 0
            binary_negate_mask[:, split_index] = 0

        # Compute distance matrix (no sqrt for stability)
        x = features
        x_norm = x.norm(dim=1, keepdim=True)
        distance_matrix = x_norm ** 2 + x_norm.T ** 2 - 2 * x @ x.T
        distance_matrix = torch.clamp(distance_matrix, min=1e-10)

        if self.sample_reduce in ['mean', None]:
            margin = torch.tensor(margin, device=device)
            if weight is None:
                sum_same_ben = torch.clamp(
                    (same_ben_mask * distance_matrix).sum(1) - same_ben_mask.sum(1) * margin,
                    min=0.0
                )
                sum_other_mal = torch.clamp(
                    (other_mal_mask * distance_matrix).sum(1) - other_mal_mask.sum(1) * margin,
                    min=0.0
                )
                sum_same_mal_fam = (same_mal_fam_mask * distance_matrix).sum(1)
                sum_bin_neg = torch.clamp(
                    binary_negate_mask.sum(1) * 2 * margin - (binary_negate_mask * distance_matrix).sum(1),
                    min=0.0
                )
            else:
                weight_matrix = weight.view(-1, 1) @ weight.view(1, -1)
                sum_same_ben = torch.clamp(
                    (same_ben_mask * distance_matrix * weight_matrix).sum(1) - same_ben_mask.sum(1) * margin,
                    min=0.0
                )
                sum_other_mal = torch.clamp(
                    (other_mal_mask * distance_matrix * weight_matrix).sum(1) - other_mal_mask.sum(1) * margin,
                    min=0.0
                )
                sum_same_mal_fam = (same_mal_fam_mask * distance_matrix * weight_matrix).sum(1)
                weight_prime = 1.0 / weight
                weight_matrix_prime = weight_prime.view(-1, 1) @ weight_prime.view(1, -1)
                sum_bin_neg = torch.clamp(
                    binary_negate_mask.sum(1) * 2 * margin - (binary_negate_mask * distance_matrix * weight_matrix_prime).sum(1),
                    min=0.0
                )

            loss = (
                sum_same_ben / torch.clamp(same_ben_mask.sum(1), min=1) +
                sum_other_mal / torch.clamp(other_mal_mask.sum(1), min=1) +
                sum_same_mal_fam / torch.clamp(same_mal_fam_mask.sum(1), min=1) +
                sum_bin_neg / torch.clamp(binary_negate_mask.sum(1), min=1)
            )

        elif self.sample_reduce == 'max':
            loss = (
                torch.clamp(same_ben_mask * distance_matrix, min=0.0).amax(dim=1) -
                margin
            ) + (
                torch.clamp(other_mal_mask * distance_matrix, min=0.0).amax(dim=1) -
                margin
            ) + (
                torch.clamp(same_mal_fam_mask * distance_matrix, min=0.0).amax(dim=1)
            ) + (
                2 * margin - torch.clamp(binary_negate_mask * distance_matrix, min=0.0).amin(dim=1)
            )
        else:
            raise ValueError(f"Unsupported sample_reduce={self.sample_reduce}")

        return loss.mean() if self.reduce == 'mean' else loss


class HiDistanceXentLoss(nn.Module):
    def __init__(self, reduce='mean', sample_reduce='mean'):
        super(HiDistanceXentLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, xent_lambda,
                y_bin_pred, y_bin_batch,
                features, labels=None,
                margin=10.0, weight=None, split=None):
        
        device = features.device
        y_bin_pred = y_bin_pred.to(device)
        y_bin_batch = y_bin_batch.to(device)
        if weight is not None:
            weight = weight.to(device)

        # Compute the HiDistance loss (distance-based contrastive part)
        dist_fn = HiDistanceLoss(reduce=self.reduce, sample_reduce=self.sample_reduce)
        loss_dist = dist_fn(features, y_bin_batch, labels=labels,
                            margin=margin, weight=weight, split=split)

        # device = features.device  # Ensure we use the same device as the features

        # probs = torch.sigmoid(y_bin_pred[:, 1]).to(device)
        # target = y_bin_batch[:, 1].to(device)

        # loss_bce = nn.functional.binary_cross_entropy(probs, target, reduction=self.reduce, weight=weight)


        # Final combined loss
        loss = loss_dist # + xent_lambda * loss_bce
        return loss, loss_dist #, loss_bce


        return loss, supcon_loss, xent_bin_loss
class SSL_Loss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(SSL_Loss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, pseudo_preds=None, pseudo_labels=None, pseudo_weight=0.5, 
            temperature=1.0, uncertainty_weight=1.0):
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
            pseudo_preds: predicted features for pseudo labels
            pseudo_labels: pseudo labels for the features
        Returns:
            A loss scalar.
        """
        # HiDistanceXent = HiDistanceXentLoss().cuda()
        # loss, supcon_loss, xent_loss = HiDistanceXent(xent_lambda, \
        #                                     y_bin_pred, y_bin_batch, \
        #                                     features, labels = labels, \
        #                                     margin = margin, \
        #                                     weight = weight, split=split)
        # pseudo loss
        if pseudo_preds is not None and pseudo_labels is not None:
            pseudo_preds = pseudo_preds.float()
            pseudo_labels = pseudo_labels.float()
            
            bce_loss_fn = nn.BCELoss()  # Create an instance of BCELoss
            pseudo_loss = bce_loss_fn(pseudo_preds[:, 1], pseudo_labels)  ### check this // some used pseudo_preds/temperature
            
            #pseudo_loss = nn.BCELoss(pseudo_preds, pseudo_labels) ### check this // some used pseudo_preds/temperature
            
            # Weight pseudo-labels by prediction confidence
            confidence_weights = self.get_confidence_weights(pseudo_preds)
            weighted_pseudo_loss = pseudo_loss * confidence_weights > 0.9
            
            # Add uncertainty weighting for pseudo-labeled samples
            uncertainty = self.calculate_uncertainty(pseudo_preds)
            normalized_uncertainty = uncertainty / (uncertainty.max() + 1e-10)
            weighted_pseudo_loss = weighted_pseudo_loss * (1 + uncertainty_weight * normalized_uncertainty)
            pseudo_loss = pseudo_weight * weighted_pseudo_loss.mean()
            
        return pseudo_loss
    
    def calculate_uncertainty(self, probs):
        """
        Calculates uncertainty scores based on prediction entropy.
        
        Entropy measures the uncertainty of a prediction, with higher values indicating greater uncertainty.
        
        Args:
            outputs (torch.Tensor): Model outputs (logits) for the input data.
        
        Returns:
            torch.Tensor: Uncertainty values computed as entropy for each sample in the batch.
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy
    
    def get_confidence_weights(self, probs):
        """
        Calculates confidence-based weights for pseudo-labels based on prediction confidence.
        
        Confidence is computed as the highest probability predicted for each sample.
        
        Args:
            outputs (torch.Tensor): Model outputs (logits) for the input data.
        
        Returns:
            torch.Tensor: Confidence weights for each sample based on the predicted probabilities.
        """
        confidence, _ = torch.max(probs, dim=-1)
        return confidence
            
