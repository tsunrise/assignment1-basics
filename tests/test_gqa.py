import torch

from cs336_basics.modules.attention import (
    GroupedQueryAttention,
    MultiHeadSelfAttention,
    scaled_dot_product_attention,
)


def _gqa_reference_output(attn: GroupedQueryAttention, x: torch.Tensor) -> torch.Tensor:
    tokens = x.shape[-2]

    q_flat = x @ attn.WQ.T
    k_flat = x @ attn.WK.T
    v_flat = x @ attn.WV.T

    q = q_flat.reshape(*q_flat.shape[:-1], attn.num_q_heads, attn.d_k).transpose(-2, -3)
    k = k_flat.reshape(*k_flat.shape[:-1], attn.num_kv_heads, attn.d_k).transpose(-2, -3)
    v = v_flat.reshape(*v_flat.shape[:-1], attn.num_kv_heads, attn.d_v).transpose(-2, -3)

    # Head layout is grouped by KV head, then by query head within the group.
    q = q.reshape(*q.shape[:-3], attn.num_kv_heads, attn.group_size, tokens, attn.d_k)
    k = k.unsqueeze(-3).expand(*k.shape[:-3], attn.num_kv_heads, attn.group_size, tokens, attn.d_k)
    v = v.unsqueeze(-3).expand(*v.shape[:-3], attn.num_kv_heads, attn.group_size, tokens, attn.d_v)

    q = q.flatten(-4, -3)
    k = k.flatten(-4, -3)
    v = v.flatten(-4, -3)

    mask = torch.tril(torch.ones(tokens, tokens, dtype=torch.bool))
    out = scaled_dot_product_attention(q, k, v, mask)
    out = out.transpose(-2, -3).flatten(-2, -1)
    return out @ attn.WO.T


def test_grouped_query_attention_output_shape():
    torch.manual_seed(0)

    attn = GroupedQueryAttention(d_model=16, num_q_heads=4, group_size=2)
    x = torch.randn(3, 5, 16)

    out = attn(x)

    assert out.shape == x.shape


def test_grouped_query_attention_matches_reference():
    torch.manual_seed(1)

    attn = GroupedQueryAttention(d_model=16, num_q_heads=4, group_size=2)
    x = torch.randn(2, 6, 16)

    actual = attn(x)
    expected = _gqa_reference_output(attn, x)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_grouped_query_attention_matches_multihead_when_group_size_one():
    torch.manual_seed(2)

    gqa = GroupedQueryAttention(d_model=16, num_q_heads=4, group_size=1)
    mha = MultiHeadSelfAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 4, 16)

    with torch.no_grad():
        mha.WQ.copy_(gqa.WQ)
        mha.WK.copy_(gqa.WK)
        mha.WV.copy_(gqa.WV)
        mha.WO.copy_(gqa.WO)

    actual = gqa(x)
    expected = mha(x)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
