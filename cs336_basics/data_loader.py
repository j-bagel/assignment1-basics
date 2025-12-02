import numpy as np
import torch
from numpy.typing import NDArray
from typing import Tuple

def get_batch(
    dataset: NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of data from the dataset.
    """
    length = dataset.shape[0]
    if length <= context_length + 1:
        raise ValueError("Dataset too small for this context length")

    ix = np.random.randint(0, length - context_length, size=batch_size)

    x = dataset[ix[:, None] + np.arange(context_length)]
    y = dataset[ix[:, None] + np.arange(1, context_length + 1)]

    return (
        torch.as_tensor(x, dtype=torch.long, device=device),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )
