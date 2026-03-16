import torch
from cs336_basics.modules.rope import RoPE
from cs336_basics.modules.attention import MultiHeadSelfAttention
from cs336_basics.modules.transformer import TransformerLM

def test_forward_one_token():
    rope = RoPE(10000, 2, 100)
    attn = MultiHeadSelfAttention(8, 4, rope)
    torch.manual_seed(2)
    
    batch_size = 32
    prompt_tokens = 4
    response_tokens = 5
    d_model = 8
    x = torch.randn(batch_size, prompt_tokens + response_tokens, d_model)

    x_next = attn(x) # (...,t,d)

    # prefill
    x_inc_next, K, V = attn.forward_returning_kv(x[:, :prompt_tokens, :])
    torch.testing.assert_close(x_inc_next[:, prompt_tokens - 1, :], x_next[:, prompt_tokens - 1])

    # decode
    for i in range(prompt_tokens, prompt_tokens + response_tokens):
        x_next_i, K, V = attn.forward_one_token(K, V, x[:,i,:])
        torch.testing.assert_close(x_next_i, x_next[:,i,:])


def test_transformer_lm_prefill_and_decode_step():
    torch.manual_seed(3)

    batch_size = 16
    prompt_tokens = 4
    response_tokens = 5
    total_tokens = prompt_tokens + response_tokens
    vocab_size = 32

    lm = TransformerLM(
        d_model=8,
        num_heads=4,
        d_ff=16,
        rope_theta=10000,
        vocab_size=vocab_size,
        context_length=32,
        num_layers=2,
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, total_tokens))
    token_positions = torch.arange(total_tokens).unsqueeze(0).expand(batch_size, total_tokens)

    logits_full = lm(token_ids, token_positions)

    logits_prefill, kv_cache = lm.prefill(token_ids[:, :prompt_tokens], token_positions[:, :prompt_tokens])
    torch.testing.assert_close(
        logits_prefill[:, prompt_tokens - 1, :],
        logits_full[:, prompt_tokens - 1, :],
    )

    for i in range(prompt_tokens, total_tokens):
        logits_step, kv_cache = lm.decode_step(kv_cache, token_ids[:, i], token_positions[:, i])
        torch.testing.assert_close(logits_step, logits_full[:, i, :])
