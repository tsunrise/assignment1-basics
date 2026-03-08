import torch
from torch import nn
from cs336_basics.modules.softmax import softmax
from cs336_basics.modules.util import initialized_linear_weights
from cs336_basics.modules.rope import RoPE
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    - `Q`: (...batch, queries, d_k)
    - `K`: (...batch, keys, d_k)
    - `V`: (...batch, keys, d_v)
    - `mask`: (...batch, queries_len, key_len) boolean
      `mask[i][j] = False` iff token_i should not consider token_j

    return: weighted average of `V` for each token based on attention
            (...batch, queries_len, d_v)
    """
    # score[i,j] := attention weight of token j for token i
    d_k = Q.shape[-1]
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[:-1] == V.shape[:-1]
    raw_score: torch.Tensor = (torch.einsum('...qd,...kd->...qk', Q, K)) / d_k**0.5 # (...batch, queries, keys)
    if mask is not None:
        raw_score = raw_score.masked_fill(~mask, float('-inf'))

    scores = softmax(raw_score, -1)

    # (...batch, queries, keys) @ (...batch, keys, d_v)
    results = torch.einsum('...qk,...kd->...qd',scores,V)
    return results

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        """
        - `rope`: Rotatary Positional Embedding with `d_k = d_model / num_heads`
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model > num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.WQ = nn.Parameter(initialized_linear_weights(d_model, d_model))
        self.WK = nn.Parameter(initialized_linear_weights(d_model, d_model))
        self.WV = nn.Parameter(initialized_linear_weights(d_model, d_model))
        self.WO = nn.Parameter(initialized_linear_weights(d_model, d_model))

        self.rope = rope
        if rope is not None:
            assert rope.d_k == self.d_k

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        `x`: (...batch, tokens, d_model)
        return: (...batch, tokens, d_model)
        """

        assert x.shape[-1] == self.d_model, "invalid model"
        tokens = x.shape[-2] # number of tokens
        Q_flat = x @ self.WQ.T # (...batch, tokens, d_model)
        Q = Q_flat.reshape(*Q_flat.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3) # (...batch, h, tokens, d_k)
        K_flat = x @ self.WK.T
        K = K_flat.reshape(*K_flat.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3) # (...batch, h, tokens, d_k)
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions) # shape unchanged
        V_flat = x @ self.WV.T
        V = V_flat.reshape(*V_flat.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3) # (...batch, h, tokens, d_k)

        # mask[i][j] = True iff j <= i
        mask = torch.tril(torch.ones((tokens, tokens), dtype=torch.bool)) # (tokens, tokens)
        out = scaled_dot_product_attention(Q, K, V, mask) # (...batch, h, tokens, d_k)
        out_flat = out.transpose(-2,-3).flatten(-2, -1) # (...batch, tokens, d_model)
        return out_flat @ self.WO.T # (...batch, tokens, d_model)

