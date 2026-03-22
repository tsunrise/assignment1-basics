import torch
from torch import nn
from cs336_basics.modules.util import initialized_linear_weights
from cs336_basics.modules.rope import RoPE
import math


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
    d_k = Q.shape[-1]
    assert K.shape[-1] == d_k
    assert V.shape[-2] == K.shape[-2]

    logits = Q @ K.transpose(-1, -2) / math.sqrt(d_k)  # (..., queries, keys)
    if mask is not None:
        logits.masked_fill_(~mask, float("-inf"))
    out = torch.softmax(logits, dim=-1) @ V  # (..., queries, d_v)
    return out


def scaled_dot_product_attention_flash(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    q_block_size: int,
    k_block_size: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    - `Q`: (...batch, queries, d_k)
    - `K`: (...batch, keys, d_k)
    - `V`: (...batch, keys, d_v)
    - `mask`: (...batch, queries_len, key_len) boolean
      `mask[i][j] = False` iff token_i should not consider token_j

    return: weighted average of `V` for each token based on attention
            (...batch, queries, d_v)
    """
    d_k = Q.shape[-1]
    assert K.shape[-1] == d_k
    num_queries = Q.shape[-2]
    num_keys = K.shape[-2]
    assert V.shape[-2] == num_keys
    batch_dims = Q.shape[:-2]
    d_v = V.shape[-1]
    assert num_queries % q_block_size == 0
    assert num_keys % k_block_size == 0

    result = torch.zeros(*batch_dims, num_queries, d_v, dtype=Q.dtype)

    for q in range(0, num_queries, q_block_size):
        # numerator part in result, updated incrementally
        # [e^{q_ij - max_r q_ir}]_j . V_i*
        numerator = torch.zeros(*batch_dims, q_block_size, d_v, dtype=Q.dtype)
        # denominator part in result, updated incrementally
        # \sum_{j} e^{q_ij - max_r q_ir}
        denominator = torch.zeros(*batch_dims, q_block_size, dtype=Q.dtype)
        # for tracking max_r q_ir
        running_max = torch.full((*batch_dims, q_block_size), float("-inf"), dtype=Q.dtype)
        for k in range(0, num_keys, k_block_size):
            q_block = Q[..., q : q + q_block_size, :]  # (...,q_block,d_k)
            k_block = K[..., k : k + k_block_size, :]  # (...,k_block,d_k)
            v_block = V[..., k : k + k_block_size, :]  # (...,k_block,d_v)
            logits = q_block @ k_block.transpose(-1, -2) / math.sqrt(d_k)  # (...,q_block,k_block)
            if mask is not None:
                logits.masked_fill_(~mask[..., q : q + q_block_size, k : k + k_block_size], float("-inf"))
            curr_block_max = torch.max(logits, dim=-1).values  # (...,q_block)
            curr_max = torch.maximum(running_max, curr_block_max)
            rescale = torch.exp(running_max - curr_max)  # (..., q_block)
            running_max = curr_max
            numerator *= rescale.unsqueeze(-1)
            curr_softmax_term = torch.exp(logits - curr_max.unsqueeze(-1))  # (...,q_block,k_block)
            numerator += curr_softmax_term @ v_block  # (...,q_block,d_v)
            denominator *= rescale
            denominator += torch.sum(curr_softmax_term, dim=-1)
        result[..., q : q + q_block_size, :] = numerator / denominator.unsqueeze(-1)
    return result


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
        if token_positions is not None:
            assert token_positions.shape[-1] == x.shape[-2]

        tokens = x.shape[-2]
        Qf = x @ self.WQ.T  # (..., tokens, d_k * num_heads)
        Kf = x @ self.WK.T  # (..., tokens, d_k * num_heads)
        Vf = x @ self.WV.T  # (..., tokens, d_v * num_heads)

        Q = Qf.reshape(*Qf.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)  # (..., heads, tokens, d_k)
        K = Kf.reshape(*Kf.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)  # (..., heads, tokens, d_k)
        V = Vf.reshape(*Vf.shape[:-1], self.num_heads, self.d_v).transpose(-2, -3)  # (..., heads, tokens, d_v)

        if self.rope:
            Q = self.rope(Q)
            K = self.rope(K)

        mask = torch.tril(torch.ones(tokens, tokens, dtype=torch.bool))
        out = scaled_dot_product_attention(Q, K, V, mask)  # (..., heads, tokens, d_v)
        outf = out.transpose(-2, -3).flatten(-2, -1)  # (..., tokens, heads * d_v)
        result = outf @ self.WO.T  # (..., tokens, d_model)
        return result, K, V

    def forward_one_token(
        self, K_prefix: torch.Tensor, V_prefix: torch.Tensor, x: torch.Tensor, x_position: torch.Tensor | None = None
    ):
        """
        `K_prefix`: (..., heads, tokens, d_k)
        `V_prefix`: (..., heads, tokens, d_v)
        `x`: (..., d_model) the new token at last
        `x_position`: (..., )

        return:
            - result: (..., d_model)
            - K: (..., h, tokens + 1, d_k)
            - V: (..., h, tokens + 1, d_v) # note that d_v = d_k in this architecture
        """
        num_heads, num_tokens, d_k = K_prefix.shape[-3:]

        kf = x @ self.WK.T  # (...,d_k * num_heads)
        vf = x @ self.WV.T  # (...,d_v * num_heads)
        qf = x @ self.WQ.T  # (...,d_q * num_heads)

        if x_position is None:
            x_position = torch.full((), num_tokens, dtype=torch.long)

        k = kf.reshape(*kf.shape[:-1], num_heads, self.d_k).unsqueeze(-2)  # (...,num_heads,1,d_k)
        v = vf.reshape(*vf.shape[:-1], num_heads, self.d_v).unsqueeze(-2)  # (...,num_heads,1,d_v)
        q = qf.reshape(*qf.shape[:-1], num_heads, self.d_k).unsqueeze(-2)  # (...,num_heads,1,d_k)
        if self.rope:
            k = self.rope(k, x_position)
            q = self.rope(q, x_position)

        K = torch.cat([K_prefix, k], dim=-2)  # (...,num_heads,tokens + 1, d_k)
        V = torch.cat([V_prefix, v], dim=-2)  # same

        out = scaled_dot_product_attention(q, K, V)  # (...,heads,1,d_v)
        out_f = out.transpose(-2, -3).flatten(-2, -1)  # (...,1,heads * d_v)

        result = out_f @ self.WO.T  # (...,1,d_model)
        return result.flatten(-2, -1), K, V


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    `x`: (...,tokens,d)
    return: (...,heads,tokens,d//heads)
    """
    return x.reshape(*x.shape[:-1], num_heads, x.shape[-1] // num_heads).transpose(-2, -3)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, group_size: int, rope: RoPE | None = None):
        """
        - `rope`: Rotatary Positional Embedding with `d_k = d_model / num_heads`
        """
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.group_size = group_size
        assert num_q_heads % group_size == 0
        self.num_kv_heads = num_q_heads // group_size
        assert d_model % num_q_heads == 0
        self.d_k = d_model // self.num_q_heads
        self.d_v = self.d_k

        self.WQ = nn.Parameter(initialized_linear_weights(d_model, self.num_q_heads * self.d_k))
        self.WK = nn.Parameter(initialized_linear_weights(d_model, self.num_kv_heads * self.d_k))
        self.WV = nn.Parameter(initialized_linear_weights(d_model, self.num_kv_heads * self.d_v))
        self.WO = nn.Parameter(initialized_linear_weights(self.num_q_heads * self.d_v, d_model))

        self.rope = rope
        if rope is not None:
            assert rope.d_k == self.d_k

    def forward(self, x: torch.Tensor):
        """
        x: (...,tokens,d_model)
        return: (...,tokens,d_model)
        """
        Qf = x @ self.WQ.T  # (...,tokens,d_k * num_q_heads)
        Kf = x @ self.WK.T  # (...,tokens,d_k * num_kv_heads)
        Vf = x @ self.WV.T  # (...,tokens,d_v * num_kv_heads)

        tokens = x.shape[-2]

        Q = Qf.reshape(*Qf.shape[:-1], self.num_q_heads, self.d_k).transpose(-2, -3)  # (...,num_q_heads,tokens,d_k)
        K = Kf.reshape(*Kf.shape[:-1], self.num_kv_heads, self.d_k).transpose(-2, -3)  # (...,num_kv_heads,tokens,d_k)
        V = Vf.reshape(*Vf.shape[:-1], self.num_kv_heads, self.d_v).transpose(-2, -3)  # (...,num_kv_heads,tokens,d_v)

        Qg = Q.reshape(
            *Q.shape[:-3], self.num_kv_heads,self.group_size, tokens, self.d_k
        )  # (...,groups,kv_heads,tokens,d_k)
        Kg = K.reshape(*K.shape[:-3], self.num_kv_heads, 1, tokens, self.d_k)  # (...,1,kv_heads,tokens,d_k)
        Vg = V.reshape(*V.shape[:-3], self.num_kv_heads, 1, tokens, self.d_v)  # (...,1,kv_heads,tokens,d_v)

        if self.rope:
            Qg = self.rope(Qg)
            Kg = self.rope(Kg)

        mask = torch.tril(torch.ones((tokens, tokens), dtype=torch.bool))

        out_g = scaled_dot_product_attention(Qg, Kg, Vg, mask)  # (...,kv_heads,groups,tokens,d_v)
        out_f = out_g.flatten(-4, -3)  # (...,num_q_heads,tokens,d_v)
        out = out_f.transpose(-2, -3).flatten(-2, -1)  # (...,tokens,num_q_heads * d_v)
        result = out @ self.WO.T  # (...,tokens, d_model)
        return result
