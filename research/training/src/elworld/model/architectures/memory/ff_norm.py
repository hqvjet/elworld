import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, n_emb, dropout=0.1):
        """
        Args:
            n_emb: Embedding dimension (e.g., 64)
            dropout: Dropout rate
        """
        super().__init__()
        # Typically hidden dimension is 4x embedding dimension
        hidden_dim = 4 * n_emb
        
        self.net = nn.Sequential(
            nn.Linear(n_emb, hidden_dim),
            nn.GELU(),  # Gaussian Error Linear Unit (better than ReLU for transformers)
            nn.Linear(hidden_dim, n_emb),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C]
        
        Returns:
            Output tensor [B, T, C]
        """
        return self.net(x)


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional bias.
    Normalizes across the feature dimension.
    """
    
    def __init__(self, n_emb, bias=True):
        """
        Args:
            n_emb: Embedding dimension
            bias: Whether to use bias parameter
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_emb))
        self.bias = nn.Parameter(torch.zeros(n_emb)) if bias else None
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C]
        
        Returns:
            Normalized tensor [B, T, C]
        """
        # Normalize across last dimension (feature dimension)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        
        # Scale and shift
        if self.bias is not None:
            return self.weight * x_norm + self.bias
        else:
            return self.weight * x_norm
