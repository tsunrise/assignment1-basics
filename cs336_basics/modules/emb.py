import torch
from torch import nn


class Embedding(nn.Module):
    """
    - `token_ids`: (...batch)
    - return: (...batch, embedding_dim)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.x = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), mean = 0, std = 1, a = -3, b = 3))
    
    def forward(self, token_ids: torch.Tensor):
        return self.x[token_ids]

    
