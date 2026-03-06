import torch
from torch import nn


class Linear(nn.Module):
    """
    - x: (...batch, in_features)
    - return: (...batch, out_features)
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        std=2/(in_features + out_features)
        self.w = nn.Parameter(nn.init.trunc_normal_(torch.empty(out_features, in_features), mean=0, std=std, a = -3*std, b = 3 * std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (...batch, in_features) , (out_features, in_features) -> (...batch, out_features)
        result = x @ self.w.T
        return result
