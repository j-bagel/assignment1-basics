import torch
from torch import Tensor, nn
from einops import rearrange, einsum
import numpy as np


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((self.out_features, self.in_features), device=device, dtype=dtype)
        )
        self.initialize()

    def forward(self, input: Tensor) -> Tensor:
        return einsum(input, self.weight, "... d_in, d_out d_in -> ... d_out")

    def initialize(self):
        sd = (2.0 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sd, a=-3*sd, b=3*sd)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings  # vocab size
        self.embedding_dim = embedding_dim  # d_{model}
        self.weight = nn.Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype)
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        token_ids: (batch_size, sequence_length)
        return shape (batch_size, sequence_length, self.embedding_dim)
        """
        return self.weight[token_ids]

    def initialize(self):
        sd = 1
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sd, a=-3 * sd, b=3 * sd)
