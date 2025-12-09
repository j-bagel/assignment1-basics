import torch
from torch import Tensor, nn
from cs336_basics.nn_utils import RMSNorm, MultiHeadSelfAttentionRoPE, SwiGLU, Embedding, Linear, softmax, top_p_sample


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

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_tokens: int,
        top_p: float = 0.95,
        temperature: float = 1.0
    ) -> Tensor:
        """
        Autoregressive generation with top-p sampling.
        
        Args:
            input_ids: (batch_size, seq_len) tensor of input token indices
            max_tokens: maximum total sequence length (including input)
            top_p: nucleus sampling probability threshold
            temperature: sampling temperature
            
        Returns:
            (batch_size, output_len) tensor where output_len <= max_tokens
        """
        batch_size, seq_len = input_ids.shape
        
        if seq_len >= max_tokens:
            raise ValueError(f"Input length ({seq_len}) must be smaller than max_tokens ({max_tokens})")
        
        # Start with input
        generated = input_ids
        
        for _ in range(max_tokens - seq_len):
            # Get logits for the last position
            logits = self.forward(generated)  # (batch_size, current_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Sample next token
            next_token = top_p_sample(next_token_logits, top_p=top_p, temperature=temperature)  # (batch_size,)
            
            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        
        return generated


class TransformerLMWithTiedWeights(TransformerLM):
    """TransformerLM with weight tying between token embeddings and LM head."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings.conservative_initialize()  # shrink the tied weights
        self.lm_head.weight = self.token_embeddings.weight

