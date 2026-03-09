import torch

def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    r"""
    - `x`: (...batch, D)
    - return: (...batch) \log(\sum_{i=D} exp(x[i]))
    """
    # \log(\sum_{i=D} exp(x[i])) = \log(exp(max(x))\sum_{i in D} exp(x[i]-max(x)))
    #                            = max(x) + \log(\sum_{i in D} exp(x[i]-max(x)))
    # for numerical stability.
    x_max = torch.max(x, dim=-1).values # (...batch)
    left = x_max
    right = torch.log(torch.sum(torch.exp(x-x_max.unsqueeze(-1)),dim=-1))
    return left + right

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    - `logits`: (...batch, vocabs),
    - `targets`: (...batch) each represents a token id
    - return: scalar
    """
    # loss = mean_{t in batch} -log(p(target[t]))
    #      = mean_{t in batch} -log(exp(logits[t, targets[t]]) / sum_{a\in vocabs}exp(logits[t, targets[a]]))
    #      = mean_{t in batch} -logits[t, targets[t]] + log(sum_{a\in vocabs}exp(logits[t, targets[a]]))

    # for each batch item
    left = -torch.gather(logits, -1, targets.unsqueeze(-1)).reshape_as(targets)
    right = log_sum_exp(logits) # (...batch)
    assert left.shape == right.shape
    return torch.mean(left + right)


