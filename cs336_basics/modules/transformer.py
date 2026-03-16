from typing import cast

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

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        - `x`: (...batch, tokens, d_model)
        - `token_positions`: (...batch, tokens)
        - return: (...batch, tokens, d_model)
        """

        return self.forward_returning_kv(x, token_positions)[0]

    def forward_returning_kv(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - `x`: (...batch, tokens, d_model)
        - `token_positions`: (...batch, tokens)
        - return:
            - output: (...batch, tokens, d_model)
            - K: (...,h,tokens,d_k)
            - V: (...,h,tokens,d_v)
        """

        x1 = self.ln1(x)
        x1, K, V = self.attn.forward_returning_kv(x1, token_positions)
        x = x + x1

        x1 = self.ln2(x)
        x1 = self.ff(x1)
        x = x + x1

        return x, K, V

    def forward_one_token(
        self,
        K_prefix: torch.Tensor,
        V_prefix: torch.Tensor,
        x: torch.Tensor,
        token_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - `K_prefix`: (...,h,tokens,d_k)
        - `V_prefix`: (...,h,tokens,d_v)
        - x: new token to append, (...,d_model)
        - `token_position`: (...,)

        return:
        - output: output for new token (...,d_model)
        - K: (...,h,tokens+1,d_k)
        - V: (...,h,tokens+1,d_v)
        """

        x1 = self.ln1(x)  # (...,d_model)
        x1, K, V = self.attn.forward_one_token(K_prefix, V_prefix, x1, token_position)
        x = x + x1

        x1 = self.ln2(x)
        x1 = self.ff(x1)
        x = x + x1

        return x, K, V


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
    ):
        """
        - `d_model`: Dimensionality of transformer block input/output
        - `num_heads`: Number of heads for multi-head self-attention
        - `d_ff`: Dimensionality of position-wise feed-forward hidden layer
        - `rope_theta`: base parameter for rope
        - `vocab_size`: size of vocabulary, needed for determining the dim of token embedding matrix
        - `context_length`: maximum sequence length, useful for determining the position embedding matrix
        - `num_layers`: number of transform blocks to use
        """
        super().__init__()
        rope = RoPE(rope_theta, d_model // num_heads, context_length)

        self.emb = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers))
        self.final_ln = RmsNorm(d_model)
        self.final_linear = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        - `token_ids`: (..., tokens) int token id
        - `token_positions`: (..., tokens) int
        - return: (..., tokens, vocab_size) logits of next word at token
        """
        x = self.emb(token_ids)
        for tf in self.blocks:
            x = tf(x, token_positions)
        x = self.final_ln(x)
        x = self.final_linear(x)
        return x

    def prefill(
        self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        - `token_ids`: (..., tokens) int token id
        - `token_positions`: (..., tokens) int
        - return:
            - (..., tokens, vocab_size) logits of next word at token
            - `kv_cache`: KV cache for decoding: list of tuple[(...,h,tokens,d_k), (...,h,tokens,d_v)]
        """
        x = self.emb(token_ids)
        kv_cache = []
        for tf in self.blocks:
            tf = cast(TransformerBlock, tf)
            x, K, V = tf.forward_returning_kv(x, token_positions)
            kv_cache.append((K, V))
        x = self.final_ln(x)
        x = self.final_linear(x)
        return x, kv_cache

    def decode_step(
        self, kv_cache: list[tuple[torch.Tensor, torch.Tensor]], token_id: torch.Tensor, token_position: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        - `kv_cache`: KV for each layer for prefix tokens
        - `token_id`: (...,)
        - `token_positions`:(...,)
        - return:
            - (...,vocab_size): logits for next word at token
            - updated kv cache
        """

        x = self.emb(token_id)  # (...,d_model)
        new_kv_cache = []
        assert len(kv_cache) == len(self.blocks)
        for (K_prefix, V_prefix), tf in zip(kv_cache, self.blocks):
            tf = cast(TransformerBlock, tf)
            x, K, V = tf.forward_one_token(K_prefix, V_prefix, x, token_position)
            new_kv_cache.append((K, V))
        x = self.final_ln(x)
        x = self.final_linear(x)
        return x, new_kv_cache
