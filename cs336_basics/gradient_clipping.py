from collections.abc import Iterable
import math

import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Clip gradient for each parameter such that
    - If the gradient norm < max_l2_norm, we leave it as is.
    - Otherwise we scale it to be to max_l2_norm / (l2_norm + eps) * old_gradient
    """

    norm_sq = 0.
    for param in parameters:
        if param.grad is None:
            continue
        norm_sq += float(torch.sum(param.grad * param.grad))
    norm = math.sqrt(norm_sq)
    if norm > max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(max_l2_norm / (norm + eps))
            