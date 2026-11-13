"""Pairwise logistic ranking on the very same four CE negatives."""
import torch
import torch.nn.functional as F


def components(logits):
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError('Expected [batch, positive + four sampled negatives].')
    scores = logits.float()
    ce = F.cross_entropy(scores, torch.zeros(len(scores), dtype=torch.long,
                                           device=scores.device), reduction='none')
    pair = F.softplus(scores[:, 1:] - scores[:, :1]).mean(dim=1)
    return ce, pair


def summed_loss(logits, pair_weight, full_batch_size):
    ce, pair = components(logits)
    return (ce.sum() + pair_weight * pair.sum()) / full_batch_size
