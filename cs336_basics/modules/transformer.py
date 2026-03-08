import torch
from torch import nn
from cs336_basics.modules.attention import MultiHeadSelfAttention
from cs336_basics.modules.swiglu import SwiGLU
from cs336_basics.modules.rope import RoPE
from cs336_basics.modules.rmsnorm import RmsNorm
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None):
        """
        rope.d_k must be d_model // num_heads
        """
        super().__init__()
        
        self.ln1 = RmsNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)

        self.ln2 = RmsNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - `x`: (...batch, tokens, d_model)
        - return: (...batch, tokens, d_model)
        """

        x1 = self.ln1(x)
        x1 = self.attn.forward(x1)
        x = x + x1

        x1 = self.ln2(x)
        x1 = self.ff(x1)
        x = x + x1

        return x
