import torch
import torch.nn as nn

from elworld.model.architectures.memory.mingpt import MinGPT


class MemoryModel(nn.Module):
    """
    Memory Model wrapper for MinGPT transformer.
    Predicts next frame's visual tokens given current frame + action.
    
    Architecture:
        Input: visual_tokens [B, T] + actions [B, T, action_dim]
        Output: next_token_predictions [B, T, vocab_size]
    
    Usage similar to VisionModel for consistency.
    """
    
    def __init__(
        self,
        vocab_size=512,      # VQ-VAE codebook size
        block_size=1537,     # Max sequence length
        n_emb=64,           # Embedding dimension
        num_layers=6,       # Number of transformer blocks
        num_heads=4,        # Number of attention heads
        dropout=0.1,        # Dropout rate
        action_dim=22       # Action dimension
    ):
        """
        Args:
            vocab_size: Size of VQ-VAE codebook (default 512)
            block_size: Maximum context window length (default 1537)
            n_emb: Token embedding dimension (default 64)
            num_layers: Number of transformer layers (default 6)
            num_heads: Number of attention heads (default 4)
            dropout: Dropout probability (default 0.1)
            action_dim: Action vector dimension (default 22)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.action_dim = action_dim
        
        # MinGPT transformer
        self.model = MinGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_emb=n_emb,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            action_dim=action_dim
        )
    
    def forward(self, visual_tokens, actions=None, targets=None):
        """
        Forward pass through memory model.
        
        Args:
            visual_tokens: Token indices [B, T] from VQ-VAE encoding_indices
                         For spatial tokens [B, H, W], flatten to [B, H*W]
            actions: Action vectors [B, T, action_dim] for conditioning
            targets: Target tokens [B, T] for supervised learning (next frame tokens)
        
        Returns:
            Dictionary containing:
                - 'logits': [B, T, vocab_size] - Predicted token distributions
                - 'loss': Scalar loss (only if targets provided)
                - 'predictions': [B, T] - Argmax predictions
        """
        # Ensure visual_tokens is 2D [B, T]
        if visual_tokens.dim() == 3:
            # If input is [B, H, W], flatten spatial dimensions
            B, H, W = visual_tokens.shape
            visual_tokens = visual_tokens.view(B, H * W)
        
        # Forward through MinGPT
        if targets is not None:
            logits, loss = self.model(visual_tokens, actions=actions, targets=targets)
            predictions = torch.argmax(logits, dim=-1)
            return {
                'logits': logits,
                'loss': loss,
                'predictions': predictions
            }
        else:
            logits = self.model(visual_tokens, actions=actions)
            predictions = torch.argmax(logits, dim=-1)
            return {
                'logits': logits,
                'predictions': predictions
            }
    
    def generate(self, start_tokens, actions=None, num_steps=768, temperature=1.0, top_k=None):
        """
        Generate next frame tokens autoregressively.
        
        Args:
            start_tokens: Starting tokens [B, T_start]
            actions: Action sequence [B, num_steps, action_dim]
            num_steps: Number of tokens to generate (default 768 = 24x32)
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            Generated tokens [B, T_start + num_steps]
        """
        return self.model.generate(
            idx=start_tokens,
            max_new_tokens=num_steps,
            temperature=temperature,
            top_k=top_k,
            actions=actions
        )
    
    def get_num_params(self):
        """Return number of parameters."""
        return self.model.get_num_params()
