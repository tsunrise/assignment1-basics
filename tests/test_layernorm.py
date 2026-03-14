import torch

from cs336_basics.modules.rmsnorm import LayerNorm


def _run_case(x: torch.Tensor, eps: float) -> None:
    layernorm = LayerNorm(x.shape[-1], eps=eps, dtype=x.dtype)
    with torch.no_grad():
        layernorm.a.copy_(torch.randn_like(layernorm.a))
        layernorm.b.copy_(torch.randn_like(layernorm.b))

    expected = torch.nn.functional.layer_norm(x, (x.shape[-1],), layernorm.a, layernorm.b, eps=eps)
    actual = layernorm(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_layernorm_matches_pytorch_2d() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float64)
    _run_case(x, eps=1e-5)


def test_layernorm_matches_pytorch_high_rank() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4, 8, dtype=torch.float64)
    _run_case(x, eps=1e-5)


def test_layernorm_handles_constant_input_with_eps() -> None:
    x = torch.full((2, 5), 7.0, dtype=torch.float64)
    layernorm = LayerNorm(x.shape[-1], eps=1e-5, dtype=x.dtype)
    with torch.no_grad():
        layernorm.a.fill_(1.0)
        layernorm.b.fill_(0.0)

    expected = torch.nn.functional.layer_norm(x, (x.shape[-1],), layernorm.a, layernorm.b, eps=1e-5)
    actual = layernorm(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
