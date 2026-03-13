import os
from typing import IO, BinaryIO, TypeAlias, Union

import numpy as np
import torch

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a training batch. 
    - `dataset`: ndarray (tokens,)
    - return: 
        - `input_tokens`: (batch_size, context_length)
        - `target_tokens`: (batch_size, context_length)
    """
    assert len(dataset.shape) == 1
    assert dataset.shape[0] >= context_length + 1 # minimum length need for a training batch
    # sample random starting indices
    training_starts = np.random.randint(0, dataset.shape[0] - context_length, batch_size)
    input_tokens = torch.from_numpy(np.array([dataset[training_starts[i]:training_starts[i] + context_length] for i in range(batch_size)], dtype=np.int64)).to(device=device) # (batch_size, context_length)
    target_tokens = torch.from_numpy(np.array([dataset[training_starts[i]+1:training_starts[i] + context_length+1] for i in range(batch_size)], dtype=np.int64)).to(device=device) # (batch_size, context_length)
    return input_tokens, target_tokens

FILE_LIKE: TypeAlias = str | os.PathLike | BinaryIO | IO[bytes]

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: FILE_LIKE):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)

def load_checkpoint(src: FILE_LIKE, model: torch.nn.Module, optimizer: torch.optim.Optimizer, map_location: str | torch.device = "cpu") -> int:
    obj: dict = torch.load(src, map_location=map_location)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]

    

