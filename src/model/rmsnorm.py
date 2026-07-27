import torch
import torch.nn as nn
from typing import Optional

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm normalizes the input by the RMS of the activations, without mean centering.
    This is computationally cheaper than LayerNorm and performs well in transformers.
    
    Formula: output = input / (RMS(input) + eps) * weight
    where RMS(x) = sqrt(mean(x^2))
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size)
        
        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS along the hidden dimension
        # rms = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return x / rms * self.weight