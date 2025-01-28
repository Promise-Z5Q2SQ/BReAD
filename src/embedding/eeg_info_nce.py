import torch
import torch.nn.functional as F
from torch import nn

class EEGInfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def info_nce(query, positive_key,
             negative_keys=None, temperature=0.1,
             reduction='mean', negative_mode='unpaired'):
    """
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.  即给每个query对应一组negative_keys
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.   即给每个query都是同一组negative_keys
    """
    if query.dim() != 2:
        raise ValueError(f'query must be 2D tensor, query shape {query.shape}')
    if positive_key.dim() != 2:
        raise ValueError('positive_key must be 2D tensor')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError('negative_keys must be 2D tensor for negative_mode=unpaired')
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError('negative_keys must be 3D tensor for negative_mode=paired')

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)  # (N, 1)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)  # (N, M)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)  # (N, 1, D)
            negative_logits = query @ transpose(negative_keys) # (N, 1, M)
            negative_logits = negative_logits.squeeze(1)  # (N, M)

        logits = torch.cat([positive_logit, negative_logits], dim=1)  # (N, 1+M)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)  # (N,)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)  # (N, N)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def get_negative_samples(batch):
    negative_samples = torch.zeros(batch.shape[0], batch.shape[0]-1, 768)
    for i in range(batch.shape[0]):
        negative_samples[i] = torch.cat((batch[:i], batch[i+1:]))
    return negative_samples
