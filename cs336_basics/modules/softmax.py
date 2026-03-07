import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    `x`: (...L, d, ...R)
    return: (...L, d, ...R)
    """

    x_max = torch.max(x, dim=dim, keepdim=True).values # (...L, 1,...R)
    x_hat = x - x_max # equivalent to multiple e(-x_max) for both numerator and denominator
    numerator = torch.exp(x_hat) # (...L,d,...R)
    denominator = torch.sum(numerator, dim = dim, keepdim=True) # (...L,1,...R)
    return numerator / denominator