import torch
from torch import nn
from cs336_basics.modules.attention import MultiHeadSelfAttention
from cs336_basics.modules.swiglu import SwiGLU
from cs336_basics.modules.rope import RoPE
from cs336_basics.modules.rmsnorm import RmsNorm
from cs336_basics.modules.emb import Embedding
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.softmax import softmax
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

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, vocab_size: int, context_length: int, num_layers: int):
        """
        - `d_model`: Dimensionality of transformer block input/output
        - `num_heads`: Number of heads for multi-head self-attention
        - `d_ff`: Dimensionality of position-wise feed-forward hidden layer
        - `rope_theta`: base parameter for rope
        - `vocab_size`: size of vocabulary, needed for determining the dim of token embedding matrix
        - `context_length`: maximum sequence length, useful for determining the position embedding matrix
        - `num_layers`: number of transform blocks to use
        """
        rope = RoPE(rope_theta, d_model // num_heads, context_length)

        self.emb = Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)]
        self.final_ln = RmsNorm(d_model)
        self.final_linear = Linear(d_model, d_model)

    def forward(self, token_ids: torch.Tensor):
        """
        - `token_ids`: (...batch, tokens) int token id
        - return: (...batch, tokens, d_model) probability of word at token
        """
        x = self.emb.forward(token_ids)
        for tf in self.blocks:
            x = tf(x)
        x = self.final_ln(x)
        x = self.final_linear(x)
        x = softmax(x, -1)
        return x


