import torch
from torch import nn


class RoPE(nn.Module):
    r"""
    Rotary Position Embeddings.
    For the kth pair (x[i][k//2], x[i][k//2 + 1]) of embedding at position i, we rotate the this pair
    by $\frac{i}{\Theta^{(2k-2)/d}}$. We do this for each k and i to get the embedding with position information.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        r"""
        - `theta`: $\Theta$ parameter
        - `d_k`: dimension of query and key vectors
        """
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k")
        # (max_seq_len, d_k // 2)
        # angle[i,k] = angle to rotate at position i, pair dimension k
        angle = torch.outer(
            torch.arange(max_seq_len, device=device),
            1 / torch.pow(theta, (2 * torch.arange(d_k // 2, device=device)) / d_k),
        )
        self.register_buffer("angle", angle, persistent=False)
        self.d_k = d_k

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        r"""
        - `x`: (...batch, seq_len, d_k)
        - `token_positions`: (...batch, seq_len)

        - return: (...batch, seq_len, d_k)
        """
        # angle rotated for each pair
        if token_positions is not None:
            angles = self.get_buffer("angle")[token_positions].to(
                device=x.device, dtype=x.dtype
            )  # (...batch, seq_len, d_k // 2)
        else:
            angles = self.get_buffer("angle")[: x.shape[-2]].to(device=x.device, dtype=x.dtype)  # (seq_len, d_k // 2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        # (...batch, seq_len, d_k // 2) * (...batch, seq_len, d_k // 2) -> same shape
        x_pairs_rotated_0 = cos * x_pairs[..., 0] - sin * x_pairs[..., 1]
        x_pairs_rotated_1 = sin * x_pairs[..., 0] + cos * x_pairs[..., 1]
        x_pairs_rotated = torch.stack(
            (x_pairs_rotated_0, x_pairs_rotated_1), dim=-1
        )  # (...batch, seq_len, d_k // 2, 2)
        x_rotated = x_pairs_rotated.reshape_as(x)
        return x_rotated
