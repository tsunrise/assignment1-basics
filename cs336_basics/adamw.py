from collections.abc import Callable
from typing import Any

import torch


class AdamW(torch.optim.Optimizer):
    """
    Memory Usage: Roughly 2 params required for each param.
    For model with (vocab_size, context_length, num_layers, d_models, num_heads)

    M(AdamW) "Memory of AdamW"
    = 2 * (M(Emb) + num_layers * (M(WQ, WK, WV, WO) + M(ff) + 2*M(RMSNorm)) + M(RMSNorm) + M(Final Linear))
    = 2 * (vocab_size * d_models + num_layers * (4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model) + d_model + d_models * vocab_size)

    Memory usage does not depend on batch size. 

    Flops: Roughly O(P), around 10-20flops
    """
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
        if lr < 0:
            raise ValueError("Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                p: torch.Tensor
                if p.grad is None:
                    continue
                grad = p.grad.data

                state: dict[str, Any] = self.state[p]

                m: torch.Tensor = beta1 * state.get("m", torch.zeros_like(p)) + (1 - beta1) * grad
                state["m"] = m
                v: torch.Tensor = beta2 * state.get("v", torch.zeros_like(p)) + (1 - beta2) * grad * grad
                state["v"] = v
                t: float = state.get("t", 0) + 1
                state["t"] = t

                lrt = lr * ((1 - beta2**t) ** 0.5) / (1 - beta1**t)
                p.data -= lrt * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

        return loss
