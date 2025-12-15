import torch
import torch.nn as nn

from elworld.model.architectures.memory.masked_mha import MaskedMultiHeadAttention
from elworld.model.architectures.memory.ff_norm import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    """
    Transformer block = Attention + FeedForward with residual connections and layer norm.
    
    Architecture (Pre-LN variant):
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """
    
    def __init__(self, n_emb, num_heads, block_size, dropout=0.1):
        """
        Args:
            n_emb: Embedding dimension (e.g., 64)
            num_heads: Number of attention heads (e.g., 4)
            block_size: Maximum sequence length (e.g., 1537)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Layer normalization
        self.ln1 = LayerNorm(n_emb)
        self.ln2 = LayerNorm(n_emb)
        
        # Multi-head attention
        self.attn = MaskedMultiHeadAttention(
            n_emb=n_emb,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForward(n_emb=n_emb, dropout=dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C]
        
        Returns:
            Output tensor [B, T, C]
        """
        # Pre-LN: normalize before attention
        x = x + self.attn(self.ln1(x))
        
        # Pre-LN: normalize before feedforward
        x = x + self.ffn(self.ln2(x))
        
        return x
