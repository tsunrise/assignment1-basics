import torch
from torch import nn
from cs336_basics.modules.util import initialized_linear_weights

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        r"""
        SiLU(x) = x\dot\sigmoid(x)
        SwiGLU = W_2(SiLU(W_1x) * W_3x)
        """
        
        if d_ff is None:
            d_ff = round((8/3) * d_model)
            # increase to the next 64 multiples
            if d_ff % 64 != 0:
                d_ff = d_ff + 64 - (d_ff % 64)

        # w1 w3 project d_model to d_ff
        self.w1 = nn.Parameter(initialized_linear_weights(d_model, d_ff, dtype=dtype, device=device)) # (d_ff, d_model)
        self.w3 = nn.Parameter(initialized_linear_weights(d_model, d_ff, dtype=dtype, device=device)) # (d_ff, d_model)
        # w2 project d_ff back to d_model
        self.w2 = nn.Parameter(initialized_linear_weights(d_ff, d_model, dtype=dtype, device=device)) # (d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        - x: (...batch, d_model)
        - return: (...batch, d_model)
        """

        # (...batch, d_model) @ (d_model, dff) -> (...batch, d_ff)
        left_projected = x @ self.w1.T 
        right_projected = x @ self.w3.T

        left_activated = left_projected * torch.sigmoid(left_projected) # (...batch, d_ff)
        combined_projected = left_activated * right_projected # (...batch, d_ff)

        # (...batch, d_ff) @ (d_ff, d_model) -> (...batch, d_model)
        projected_back = combined_projected @ self.w2.T
        return projected_back