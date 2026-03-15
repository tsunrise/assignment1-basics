import torch
from cs336_basics.modules.rope import RoPE
from cs336_basics.modules.attention import MultiHeadSelfAttention

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
        print(f"x_next[:,i,:]: {x_next[:,i,:].shape}")
        torch.testing.assert_close(x_next_i, x_next[:,i,:])
