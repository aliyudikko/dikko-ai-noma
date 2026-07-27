import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import ModelConfig
from .rmsnorm import RMSNorm
from .attention import CausalSelfAttention
from .mlp import MLP

class TransformerBlock(nn.Module):
    """
    A single Transformer block with pre-normalization.
    
    Architecture:
    1. RMSNorm
    2. Causal Self-Attention
    3. Residual connection
    4. RMSNorm
    5. MLP (with SwiGLU)
    6. Residual connection
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Pre-normalization for attention
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Causal self-attention
        self.attention = CausalSelfAttention(config)
        
        # Pre-normalization for MLP
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MLP with SwiGLU
        self.mlp = MLP(config)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-norm + attention + residual
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.attention(normed_x, mask, return_attention)
        x = x + attn_output  # Residual connection
        
        # Pre-norm + MLP + residual
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output  # Residual connection
        
        return x, attn_weights