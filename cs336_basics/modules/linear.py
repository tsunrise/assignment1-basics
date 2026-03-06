import torch
from torch import nn
from cs336_basics.modules.util import initialized_linear_weights

class Linear(nn.Module):
    """
    - x: (...batch, in_features)
    - return: (...batch, out_features)
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.w = nn.Parameter(initialized_linear_weights(in_features, out_features, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (...batch, in_features) , (out_features, in_features) -> (...batch, out_features)
        result = x @ self.w.T
        return result
