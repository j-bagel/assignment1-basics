import torch
from torch import Tensor, nn
from einops import rearrange, einsum
import math


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
        # return einsum(input, self.weight, "... d_in, d_out d_in -> ... d_out")
        return input @ self.weight.T  # faster

    def initialize(self):
        sd = (2.0 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sd, a=-3*sd, b=3*sd)


class LinearQKV(nn.Module):
    def __init__(self, num_heads: int, d_proj: int, d_model: int, device=None, dtype=None):
        super().__init__()

        self.num_heads = num_heads
        self.d_proj= d_proj
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.empty((self.num_heads, self.d_proj, self.d_model), device=device, dtype=dtype)
        )
        self.initialize()

    def forward(self, input: Tensor) -> Tensor:
        return einsum(input, self.weight, "... seq_len d_in, h d_out d_in -> ... h seq_len d_out")

    def initialize(self):
        sd = (2.0 / (self.num_heads * self.d_proj + self.d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sd, a=-3*sd, b=3*sd)


class LinearO(nn.Module):
    def __init__(self, num_heads: int, d_proj: int, d_model: int, device=None, dtype=None):
        super().__init__()

        self.num_heads = num_heads
        self.d_proj = d_proj
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.empty((self.num_heads, self.d_model, self.d_proj), device=device, dtype=dtype)
        )
        self.initialize()

    def forward(self, input: Tensor) -> Tensor:
        return einsum(input, self.weight, "... h seq_len d_in, h d_out d_in -> ... seq_len d_out")

    def initialize(self):
        sd = (2.0 / (self.num_heads * self.d_proj + self.d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sd, a=-3 * sd, b=3 * sd)


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

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))  # "gain"

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # let's say it's of shape (batch_size, sequence_length, d_model)

        # one_over_RMS = 1 / (torch.sqrt(torch.sum(x * x, dim=-1) / self.d_model + self.eps))  # shape (batch_size, sequence_length)
        # result = x * einsum(one_over_RMS, self.gain, "..., d_out -> ... d_out")
        result = x * self.weight * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        # Return the result in the original dtype
        return result.to(in_dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = math.ceil(d_model / 24) * 64
        self.d_ff = d_ff

        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # w1x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        w1x = self.w1(x)
        # w3x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        w3x = self.w3(x)
        silu_w1x = silu(w1x)
        right = silu_w1x * w3x
        # return einsum(right, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
        return self.w2(right)  # faster


def softmax(x: Tensor, dim: int) -> Tensor:
    scaled_x = x - torch.amax(x, dim=dim, keepdim=True)
    exp_x = torch.exp(scaled_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    scaled_inputs = inputs - torch.amax(inputs, dim=-1, keepdim=True)  # " batch_size vocab_size"
    per_token_loss = (
        - scaled_inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        + scaled_inputs.logsumexp(dim=-1, keepdim=False)
    )  # " batch_size"
    return per_token_loss.mean()


def scaled_dot_product_attention(
    Q: Tensor,  # Float[Tensor, " ... queries d_k"]
    K: Tensor,  # Float[Tensor, " ... keys d_k"]
    V: Tensor,  # Float[Tensor, " ... keys d_v"]
    mask: Tensor | None,  # Bool[Tensor, " ... queries keys"] | None = None
) -> Tensor:  # Float[Tensor, " ... queries d_v"]
    d_k = Q.shape[-1]
    pre_sf = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        pre_sf = pre_sf.masked_fill(~mask, float('-inf'))
    sf = softmax(pre_sf, dim=-1)
    return einsum(sf, V, " ... queries keys, ... keys d_v -> ... queries d_v")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even"

        cos, sin = self.build_rope_cache(theta, max_seq_len, d_k, device)

        # Register as buffers — these are NOT parameters
        self.register_buffer("cos_cached", cos, persistent=False)  # shape (max_seq_len, d_k//2)
        self.register_buffer("sin_cached", sin, persistent=False)  # shape (max_seq_len, d_k//2)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k//2)
        x1 = x[..., 0::2]  # (..., seq_len, d_k//2)
        x2 = x[..., 1::2]  # (..., seq_len, d_k//2)
        return torch.stack([cos*x1 - sin*x2, sin*x1 + cos*x2], dim=-1).reshape_as(x)

    @staticmethod
    def build_rope_cache(theta: float, max_seq_len: int, d: int, device: None):
        i_tensor = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        k_tensor = torch.arange(0, d, step=2, device=device, dtype=torch.float32)
        theta_tensor = torch.outer(i_tensor, theta ** (-k_tensor/d))
        return torch.cos(theta_tensor), torch.sin(theta_tensor)


class MultiHeadSelfAttentionEinsum(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.d_model = d_model

        self.device = device

        self.WQ = LinearQKV(self.num_heads, self.d_k, self.d_model, device=device, dtype=dtype)
        self.WK = LinearQKV(self.num_heads, self.d_k, self.d_model, device=device, dtype=dtype)
        self.WV = LinearQKV(self.num_heads, self.d_v, self.d_model, device=device, dtype=dtype)
        self.WO = LinearO(self.num_heads, self.d_v, self.d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., seq_len, d_model)
        Q = self.WQ(x)  # (..., h, seq_len, d_k)
        K = self.WK(x)  # (..., h, seq_len, d_k)
        V = self.WV(x)  # (..., h, seq_len, d_v)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        attn = scaled_dot_product_attention(Q, K, V, mask)  # (..., h, seq_len, d_v)
        return self.WO(attn)  # (..., seq_len, d_model)


class MultiHeadSelfAttentionRoPEEinsum(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.d_model = d_model

        self.device = device

        self.WQ = LinearQKV(self.num_heads, self.d_k, self.d_model, device=device, dtype=dtype)
        self.WK = LinearQKV(self.num_heads, self.d_k, self.d_model, device=device, dtype=dtype)
        self.WV = LinearQKV(self.num_heads, self.d_v, self.d_model, device=device, dtype=dtype)
        self.WO = LinearO(self.num_heads, self.d_v, self.d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., seq_len, d_model)
        token_positions = torch.arange(0, x.shape[-2])
        Q = self.rope(self.WQ(x), token_positions)  # (..., h, seq_len, d_k)
        K = self.rope(self.WK(x), token_positions)  # (..., h, seq_len, d_k)
        V = self.WV(x)  # (..., h, seq_len, d_v)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        attn = scaled_dot_product_attention(Q, K, V, mask)  # (..., h, seq_len, d_v)
        return self.WO(attn)  # (..., seq_len, d_model)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.d_model = d_model

        self.device = device

        self.q_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_v, device=device, dtype=dtype)
        self.o_proj = Linear(in_features=self.num_heads*self.d_v, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., seq_len, d_model)
        Q = self.q_proj(x)  # (..., seq_len, h*d_k)
        K = self.k_proj(x)  # (..., seq_len, h*d_k)
        V = self.v_proj(x)  # (..., seq_len, h*d_v)

        Q = Q.unflatten(-1, (self.num_heads, self.d_k)).transpose(-3, -2)
        K = K.unflatten(-1, (self.num_heads, self.d_k)).transpose(-3, -2)  # (..., h, seq_len, d_k)
        V = V.unflatten(-1, (self.num_heads, self.d_v)).transpose(-3, -2)  # (..., h, seq_len, d_v)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        attn = scaled_dot_product_attention(Q, K, V, mask)  # (..., h, seq_len, d_v)
        attn = attn.transpose(-3, -2).flatten(start_dim=-2, end_dim=-1)
        return self.o_proj(attn)  # (..., seq_len, d_model)


class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.d_model = d_model

        self.device = device

        self.q_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_v, device=device, dtype=dtype)
        self.o_proj = Linear(in_features=self.num_heads*self.d_v, out_features=d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., seq_len, d_model)
        Q = self.q_proj(x)  # (..., seq_len, h*d_k)
        K = self.k_proj(x)  # (..., seq_len, h*d_k)
        V = self.v_proj(x)  # (..., seq_len, h*d_v)

        token_positions = torch.arange(0, x.shape[-2])
        Q = self.rope(Q.unflatten(-1, (self.num_heads, self.d_k)).transpose(-3, -2), token_positions)
        K = self.rope(K.unflatten(-1, (self.num_heads, self.d_k)).transpose(-3, -2), token_positions)  # (..., h, seq_len, d_k)
        V = V.unflatten(-1, (self.num_heads, self.d_v)).transpose(-3, -2)  # (..., h, seq_len, d_v)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        attn = scaled_dot_product_attention(Q, K, V, mask)  # (..., h, seq_len, d_v)
        attn = attn.transpose(-3, -2).flatten(start_dim=-2, end_dim=-1)
        return self.o_proj(attn)  # (..., seq_len, d_model)
