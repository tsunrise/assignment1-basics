import torch
from torch import nn
from cs336_basics.modules.softmax import softmax
from cs336_basics.modules.util import initialized_linear_weights
from cs336_basics.modules.rope import RoPE


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
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
    assert K.shape[:-2] == V.shape[:-2]
    raw_score: torch.Tensor = (torch.einsum("...qd,...kd->...qk", Q, K)) / d_k**0.5  # (...batch, queries, keys)
    if mask is not None:
        raw_score = raw_score.masked_fill(~mask, float("-inf"))

    scores = softmax(raw_score, -1)

    # (...batch, queries, keys) @ (...batch, keys, d_v)
    results = torch.einsum("...qk,...kd->...qd", scores, V)
    return results


def scaled_dot_product_attention_flash(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    q_block_size: int,
    k_block_size: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    - `Q`: (...batch, queries, d)
    - `K`: (...batch, keys, d)
    - `V`: (...batch, keys, d)
    - `mask`: (...batch, queries_len, key_len) boolean
      `mask[i][j] = False` iff token_i should not consider token_j

    return: weighted average of `V` for each token based on attention
            (...batch, queries_len, d)
    """
    d = Q.shape[-1]
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[:-1] == V.shape[:-1]
    assert Q.shape[-2] % q_block_size == 0
    assert K.shape[-2] % k_block_size == 0

    # sum over k: softmax numerator * V
    numerator = torch.zeros_like(Q, device=Q.device, dtype=Q.dtype)  # (...batch, queries, d)
    # running softmax denominator
    denominator = torch.zeros(*Q.shape[:-1], device=Q.device, dtype=Q.dtype)  # (...batch, queries)
    # a running max for this query
    running_max = torch.full(Q.shape[:-1], float("-inf"), device=Q.device, dtype=Q.dtype)  # (...batch, queries)
    for q in range(0, Q.shape[-2], q_block_size):
        for k in range(0, K.shape[-2], k_block_size):
            logits = torch.einsum(
                "...qd,...kd->...qk", Q[..., q : q + q_block_size, :], K[..., k : k + k_block_size, :]
            ) / (d**0.5)  # (...,q_block,k_block)
            if mask is not None:
                logits = logits.masked_fill(~mask[..., q : q + q_block_size, k : k + k_block_size], float("-inf"))
            prev_max = running_max[..., q : q + q_block_size].clone()  # (..., q_block)
            new_max = torch.maximum(prev_max, logits.max(dim=-1).values)
            running_max[..., q : q + q_block_size] = new_max
            denominator[..., q : q + q_block_size] *= torch.exp(prev_max - new_max)  # rescale
            denominator[..., q : q + q_block_size] += torch.sum(torch.exp(logits - new_max.unsqueeze(-1)), dim=-1)
            numerator[..., q : q + q_block_size, :] *= torch.exp(prev_max - new_max).unsqueeze(-1)
            numerator[..., q : q + q_block_size, :] += torch.einsum(
                "...qk,...kd->...qd", torch.exp(logits - new_max.unsqueeze(-1)), V[..., k : k + k_block_size, :]
            )
    return numerator / denominator.unsqueeze(-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        """
        - `rope`: Rotatary Positional Embedding with `d_k = d_model / num_heads`
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model >= num_heads
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
        `token_positions`: (...batch, tokens)
        return:
            - result: (...batch, tokens, d_model)
        """
        return self.forward_returning_kv(x, token_positions)[0]

    def forward_returning_kv(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        `x`: (...batch, tokens, d_model)
        `token_positions`: (...batch, tokens)
        return:
            - result: (...batch, tokens, d_model)
            - K: (..., h, tokens, d_k)
            - V: (..., h, tokens, d_v) # note that d_v = d_k in this architecture
        """

        assert x.shape[-1] == self.d_model, "invalid model"
        tokens = x.shape[-2]  # number of tokens
        Q_flat = x @ self.WQ.T  # (...batch, tokens, d_model)
        Q = Q_flat.reshape(*Q_flat.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)  # (...batch, h, tokens, d_k)
        K_flat = x @ self.WK.T
        K = K_flat.reshape(*K_flat.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)  # (...batch, h, tokens, d_k)
        if self.rope:
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(-2) # (...,h,tokens)
            Q: torch.Tensor = self.rope(Q, token_positions)
            K: torch.Tensor = self.rope(K, token_positions)  # shape unchanged
        V_flat = x @ self.WV.T
        V = V_flat.reshape(*V_flat.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)  # (...batch, h, tokens, d_k)

        # mask[i][j] = True iff j <= i
        mask = torch.tril(torch.ones((tokens, tokens), dtype=torch.bool, device=x.device))  # (tokens, tokens)
        out = scaled_dot_product_attention(Q, K, V, mask)  # (...batch, h, tokens, d_k)
        out_flat = out.transpose(-2, -3).flatten(-2, -1)  # (...batch, tokens, d_model)
        return out_flat @ self.WO.T, K, V

    def forward_one_token(
        self, K_prefix: torch.Tensor, V_prefix: torch.Tensor, x: torch.Tensor, x_position: torch.Tensor | None = None
    ):
        """
        `K`: (..., heads, tokens, d_k)
        `V`: (..., heads, tokens, d_v)
        `x`: (..., d_model) the new token at last
        `x_position`: (..., )

        return:
            - result: (..., d_model)
            - K: (..., h, tokens + 1, d_k)
            - V: (..., h, tokens + 1, d_v) # note that d_v = d_k in this architecture
        """
        num_batch_dimension = len(x.shape) - 1 
        x = x.unsqueeze(-2)  # (...,1,d_model)

        k_new_flat = x @ self.WK.T  # (..., 1, d_model)
        k_new = split_heads(k_new_flat, self.num_heads)  # (...,h,1,d_k)

        q_new_flat = x @ self.WQ.T  # (...,1,d_model)
        q_new = split_heads(q_new_flat, self.num_heads)  # (...,h,1,d_k)

        v_new_flat = x @ self.WV.T  # (...,1,d_model)
        v_new = split_heads(v_new_flat, self.num_heads)  # (...,h,1,d_v)

        if self.rope is not None:
            if x_position is None:
                # if none, new token position is len(prefix)
                x_position = torch.full([1] * num_batch_dimension, K_prefix.shape[-2], device=x.device, dtype=torch.long)
            # add head dimension
            x_position = x_position.unsqueeze(-1) # (...,1)
            # add token dimension
            x_position = x_position.unsqueeze(-1) # (...,1,1)
            k_new = self.rope(k_new, x_position)
            q_new = self.rope(q_new, x_position)

        # we copy full KV for this assignment, but for further optimization we could reduce the cost of copy using a better allocaation
        # strategy
        K = torch.cat((K_prefix, k_new), dim=-2)  # (...,h,tokens+1,d_k)
        V = torch.cat((V_prefix, v_new), dim=-2)  # (...,h,tokens+1,d_v)

        # no mask required here. x has access to itself and all token in prefix
        out = scaled_dot_product_attention(
            q_new, # (...,h,1,d_k)
            K, # (...,h,tokens+1,d_k)
            V, # (...,h,tokens+1,d_v)
        ) # (...,h,1,d_v)

        out_flat = out.transpose(-2,-3).flatten(-2,-1) # (...,1,d_model)
        result = (out_flat @ self.WO.T).flatten(-2,-1) # (...,d_model)
        return result, K, V

def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    `x`: (...,tokens,d)
    return: (...,heads,tokens,d//heads)
    """
    return x.reshape(*x.shape[:-1], num_heads, x.shape[-1] // num_heads).transpose(-2, -3)
