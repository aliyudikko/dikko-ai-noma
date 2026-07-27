import torch
import torch.nn as nn
from .config import ModelConfig

class MLP(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    
    SwiGLU is a variant of GLU (Gated Linear Unit) that uses Swish/SiLU
    as the activation function. It has been shown to perform well in transformers.
    
    Formula: SwiGLU(x) = SiLU(x @ W1) * (x @ W2)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ff_dim = config.ff_dim
        self.dropout = config.dropout
        
        # Gate projection (for activation)
        self.gate_proj = nn.Linear(self.hidden_size, self.ff_dim, bias=False)
        
        # Up projection (for value)
        self.up_proj = nn.Linear(self.hidden_size, self.ff_dim, bias=False)
        
        # Down projection (back to hidden_size)
        self.down_proj = nn.Linear(self.ff_dim, self.hidden_size, bias=False)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        # SwiGLU: SiLU(gate) * up
        # SiLU(x) = x * sigmoid(x)
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        
        # SwiGLU activation
        # F.silu is the same as Swish: x * sigmoid(x)
        hidden = F.silu(gate_out) * up_out
        
        # Down projection
        output = self.down_proj(hidden)
        
        # Dropout
        output = self.dropout_layer(output)
        
        return output

# Import F for silu
import torch.nn.functional as F