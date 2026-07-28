"""Module: config."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the DikkoHausaLM model."""
    
    # Vocabulary and tokenization
    vocab_size: int = 8000
    
    # Model architecture
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    head_dim: int = 64  # hidden_size // num_heads = 64
    ff_dim: int = 682   # Updated to match checkpoint dimension (was 1024)
    
    # Context and position
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    
    # Dropout and regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training
    rms_norm_eps: float = 1e-6
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        assert self.head_dim == self.hidden_size // self.num_heads, \
            f"head_dim ({self.head_dim}) must equal hidden_size // num_heads ({self.hidden_size // self.num_heads})"