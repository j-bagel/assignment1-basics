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
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.initialize()

    def forward(self, input: Tensor) -> Tensor:
        return einsum(input, self.weight, "... d_in, d_out d_in -> ... d_out")

    def initialize(self):
        sd = np.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sd, a=-3*sd, b=3*sd)


