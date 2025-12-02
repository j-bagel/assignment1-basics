import torch
from torch import Tensor, nn
from cs336_basics.nn_utils import RMSNorm, MultiHeadSelfAttentionRoPE, SwiGLU, Embedding, Linear, softmax


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int | None,
            max_seq_len: int,
            theta: float,
            device=None,
            dtype=None
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionRoPE(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype
        )

        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)  # pointwise fead forward using SwiGLU
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # pre-norm
        mid = x + self.attn(self.ln1(x))
        return mid + self.ffn(self.ln2(mid))


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int | None,
            max_seq_len: int,
            theta: float,
            device=None,
            dtype=None
    ):
        super().__init__()

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    device=device,
                    dtype=dtype
                ) for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))

