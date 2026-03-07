import torch
from cs336_basics.modules.softmax import softmax
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

