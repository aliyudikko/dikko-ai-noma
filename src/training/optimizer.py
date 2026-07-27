import torch
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from typing import List, Tuple, Optional
from .config import TrainingConfig

def create_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig
) -> Optimizer:
    """
    Create AdamW optimizer with proper weight decay handling.
    
    Weight decay is applied only to weight matrices, not to:
    - Biases
    - Layer normalization parameters (RMSNorm)
    - Embedding tables (optional, but we tie weights anyway)
    """
    
    # Separate parameters into those with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Don't apply weight decay to:
        # - Biases
        # - Layer norm parameters (weight in RMSNorm)
        # - Embedding layers (optional)
        if param.dim() < 2:
            # All biases and 1D parameters (like RMSNorm weights)
            no_decay_params.append(param)
        elif 'rmsnorm' in name.lower() or 'norm' in name.lower():
            # Normalization parameters
            no_decay_params.append(param)
        elif 'embedding' in name.lower():
            # Embedding tables (weight tying with LM head)
            no_decay_params.append(param)
        else:
            # Weight matrices get weight decay
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {
            "params": decay_params,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        }
    ]
    
    # Count parameters in each group
    num_decay_params = sum(p.numel() for p in decay_params)
    num_no_decay_params = sum(p.numel() for p in no_decay_params)
    
    print(f"Parameters with weight decay: {num_decay_params:,}")
    print(f"Parameters without weight decay: {num_no_decay_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    
    return optimizer


def create_lr_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig
):
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    
    Returns a function that computes the learning rate for each step.
    """
    
    def get_lr(step: int) -> float:
        """Compute learning rate for the given step."""
        if step < config.warmup_steps:
            # Linear warmup from 0 to learning_rate
            return config.learning_rate * (step / config.warmup_steps)
        elif step < config.max_steps:
            # Cosine decay from learning_rate to min_learning_rate
            progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            # Cosine annealing: 0.5 * (1 + cos(pi * progress))
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(3.141592653589793 * progress)).item())
            return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine_decay
        else:
            return config.min_learning_rate
    
    return get_lr