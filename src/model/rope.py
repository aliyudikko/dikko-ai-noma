import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    RoPE rotates the query and key vectors in complex space based on their positions.
    This provides relative position information through the dot product of query and key.
    """
    
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute the rotation matrices for all positions
        self._build_rotary_cache()
    
    def _build_rotary_cache(self):
        """Precompute sine and cosine values for each position and dimension pair."""
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        
        # Create frequency bands: theta^{-2i/d} for i in [0, d/2-1]
        # where d = head_dim
        freq_bands = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        
        # Compute angles: positions[:, None] * freq_bands[None, :]
        # Shape: (max_seq_len, head_dim // 2)
        angles = positions[:, None] * freq_bands[None, :]
        
        # Precompute sine and cosine for each position and dimension pair
        self.register_buffer('cos', torch.cos(angles))  # (max_seq_len, head_dim // 2)
        self.register_buffer('sin', torch.sin(angles))  # (max_seq_len, head_dim // 2)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings to query and key tensors.
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            seq_len: Current sequence length
        
        Returns:
            Tuple of (cos_emb, sin_emb) for applying rotation
        """
        # Get the relevant positions
        cos_emb = self.cos[:seq_len, :]  # (seq_len, head_dim // 2)
        sin_emb = self.sin[:seq_len, :]  # (seq_len, head_dim // 2)
        
        # Reshape for broadcasting
        # (seq_len, head_dim // 2) -> (1, 1, seq_len, head_dim // 2)
        cos_emb = cos_emb.view(1, 1, seq_len, -1)
        sin_emb = sin_emb.view(1, 1, seq_len, -1)
        
        return cos_emb, sin_emb
    
    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to the input tensor.
        
        For a vector [x0, x1, x2, x3, ...] in R^d, we pair dimensions:
        [x0, x1] -> rotation, [x2, x3] -> rotation, etc.
        
        The rotation matrix for each pair is:
        [[cos, -sin],
         [sin,  cos]]
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            cos: Cosine values (1, 1, seq_len, head_dim // 2)
            sin: Sine values (1, 1, seq_len, head_dim // 2)
        
        Returns:
            Rotated tensor of same shape as input
        """
        # Split the head_dim into pairs
        # Shape: (batch_size, num_heads, seq_len, head_dim // 2, 2)
        x1 = x[..., 0::2]  # Even indices: x0, x2, x4, ...
        x2 = x[..., 1::2]  # Odd indices: x1, x3, x5, ...
        
        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        # Shape: (batch_size, num_heads, seq_len, head_dim // 2)
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave the rotated pairs back
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rotated.view(*x.shape[:-1], -1)
        
        return rotated
    
    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor (batch_size, num_heads, seq_len, head_dim)
        
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.size(2)
        cos, sin = self.forward(q, seq_len)
        
        # Rotate queries and keys
        q_rotated = self.rotate(q, cos, sin)
        k_rotated = self.rotate(k, cos, sin)
        
        return q_rotated, k_rotated