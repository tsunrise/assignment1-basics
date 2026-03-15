import torch
from torch import nn

def initialized_linear_weights(d_in: int, d_out: int, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    return: (d_out, d_in)
    """
    std = 2 / (d_in + d_out)
    return nn.init.trunc_normal_(torch.empty(d_out, d_in, dtype=dtype, device=device), mean=0, std=std, a = -3*std, b = 3 * std)

    