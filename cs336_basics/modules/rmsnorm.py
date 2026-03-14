import torch
from torch import nn


class RmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, dtype=None):
        super().__init__()
        # gain for each element: (d_model, )
        self.g = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - x: (...batch, d_model)

        - return: (...batch, d_model)
        """
        # RMS: (...batch, 1)
        original_type = x.dtype
        x = x.to(torch.float32)  # for numeric stability during square and average
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)

        # (...batch, d_model) * d_model -> (...batch, d_model)
        return (x / rms * self.g).to(original_type)


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, dtype=None):
        super().__init__()
        """
        y_hat = a * (x - mean(x))/(std(x) + eps) + b
        """
        self.a = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(d_model, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model)

        return: (..., d_model)
        """
        original_type = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(dim=-1,keepdim=True)  # (...,1)
        var = ((x - mean)**2).mean(dim=-1, keepdim=True) # (...,1)

        return (self.a * (x - mean) / torch.sqrt(var + self.eps) + self.b).to(original_type)
