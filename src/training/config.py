from dataclasses import dataclass, field
from typing import Optional, List
import os

@dataclass
class TrainingConfig:
    """Configuration for pretraining."""
    
    # Data configuration
    train_data_path: str = "data/pretraining/train.bin"
    val_data_path: str = "data/pretraining/val.bin"
    tokenizer_path: str = "data/tokenizer/hausa_tokenizer.model"
    
    # Model architecture (must match model config)
    vocab_size: int = 8000
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    head_dim: int = 64
    ff_dim: int = 1024
    max_seq_len: int = 256
    
    # Training hyperparameters
    batch_size: int = 32
    sequence_length: int = 256
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # Evaluation and logging
    eval_interval: int = 500
    eval_steps: int = 100
    checkpoint_interval: int = 500
    log_interval: int = 50
    
    # Output directories
    output_dir: str = "checkpoints"
    run_name: str = "dikko_hausa_lm_pretrain"
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Random seed
    seed: int = 42
    
    # Resume training
    resume_from: Optional[str] = None
    
    # Device
    device: Optional[str] = None  # Auto-detect if None
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        # Create run-specific directory
        self.run_dir = os.path.join(self.output_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
    
    def get_checkpoint_path(self, step: Optional[int] = None) -> str:
        """Get checkpoint file path."""
        if step is None:
            return os.path.join(self.run_dir, "latest.pt")
        return os.path.join(self.run_dir, f"step_{step:06d}.pt")
    
    def get_best_checkpoint_path(self) -> str:
        """Get best checkpoint file path."""
        return os.path.join(self.run_dir, "best.pt")