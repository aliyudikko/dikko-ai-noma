"""Module: attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import ModelConfig
from .rope import RotaryPositionalEmbedding

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with Rotary Positional Embeddings.
    
    This implements multi-head attention where each position can only attend
    to previous positions and itself (causal masking).
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Rotary Positional Embeddings
        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask will be created lazily
        self.register_buffer('mask', None, persistent=False)
        
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal attention mask.
        
        The mask is a triangular matrix where mask[i, j] = -inf if j > i (future tokens),
        and 0 otherwise (including i == j, self-attention).
        
        Args:
            seq_len: Sequence length
            device: Device to place the mask on
        
        Returns:
            Mask tensor of shape (1, 1, seq_len, seq_len)
        """
        if self.mask is None or self.mask.size(-1) < seq_len:
            # Create a causal mask: upper triangular matrix of -inf
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            # Add batch and head dimensions
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            self.mask = mask
        return self.mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for causal self-attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            mask: Optional custom mask (batch_size, 1, seq_len, seq_len)
            return_attention: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights) where output has shape
            (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Project to Q, K, V
        # Shape: (batch_size, seq_len, hidden_size)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        # (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply Rotary Positional Embeddings
        q, k = self.rope.apply_rotary(q, k)
        
        # Compute attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        causal_mask = self._create_causal_mask(seq_len, x.device)
        scores = scores + causal_mask
        
        # Apply optional custom mask
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Output projection with residual dropout
        output = self.out_proj(output)
        output = self.resid_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None