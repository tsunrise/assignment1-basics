import torch

from cs336_basics.modules.attention import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_flash,
)


def _run_case(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> None:
    expected = scaled_dot_product_attention(q, k, v, mask)
    for q_block_size in (1, 2, q.shape[-2]):
        for k_block_size in (1, 2, k.shape[-2]):
            if q.shape[-2] % q_block_size != 0 or k.shape[-2] % k_block_size != 0:
                continue
            actual = scaled_dot_product_attention_flash(q, k, v, q_block_size, k_block_size, mask)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_flash_attention_matches_vanilla_unmasked() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    _run_case(q, k, v, mask=None)


def test_flash_attention_matches_vanilla_causal_mask() -> None:
    torch.manual_seed(1)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    _run_case(q, k, v, mask)


def test_flash_attention_matches_vanilla_batched_heads() -> None:
    torch.manual_seed(2)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 4, 8)
    v = torch.randn(2, 3, 4, 8)
    mask = torch.tril(torch.ones(1, 1, 4, 4, dtype=torch.bool)).expand(2, 3, 4, 4)
    _run_case(q, k, v, mask)


def test_flash_attention_matches_vanilla_single_full_block() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 4, 8, dtype=torch.float64)
    k = torch.randn(1, 4, 8, dtype=torch.float64)
    v = torch.randn(1, 4, 8, dtype=torch.float64)
    expected = scaled_dot_product_attention(q, k, v, None)
    actual = scaled_dot_product_attention_flash(q, k, v, q_block_size=4, k_block_size=4, mask=None)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_flash_attention_matches_vanilla_streaming_keys_only() -> None:
    torch.manual_seed(4)
    q = torch.randn(1, 4, 8, dtype=torch.float64)
    k = torch.randn(1, 4, 8, dtype=torch.float64)
    v = torch.randn(1, 4, 8, dtype=torch.float64)
    expected = scaled_dot_product_attention(q, k, v, None)
    actual = scaled_dot_product_attention_flash(q, k, v, q_block_size=4, k_block_size=1, mask=None)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
