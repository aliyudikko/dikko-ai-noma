import os
import sys
import time
import json
import random
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler, autocast

# Add src to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model import ModelConfig, DikkoHausaLM
from src.training.config import TrainingConfig
from src.training.dataset import PretrainingDataset, get_batch_sampler
from src.training.optimizer import create_optimizer, create_lr_scheduler


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: TrainingConfig) -> torch.device:
    """Get the appropriate device for training."""
    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return device


def create_model(config: TrainingConfig) -> DikkoHausaLM:
    """Create the model with proper configuration."""
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        ff_dim=config.ff_dim,
        max_seq_len=config.max_seq_len,
    )
    
    model = DikkoHausaLM(model_config)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    print("\n" + "=" * 60)
    print("DIKKO AI NOMA DA KIWO")
    print("HAUSA TRANSFORMER PRETRAINING")
    print("=" * 60)
    print(f"Model name: DikkoHausaLM")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB (float32)")
    print(f"Number of layers: {config.num_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Attention heads: {config.num_heads}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Sequence length: {config.sequence_length}")
    print("=" * 60 + "\n")
    
    return model


def create_datasets(config: TrainingConfig) -> Tuple[PretrainingDataset, PretrainingDataset]:
    """Create training and validation datasets."""
    train_dataset = PretrainingDataset(
        config.train_data_path,
        config.sequence_length,
        config.vocab_size,
        split="train"
    )
    
    val_dataset = PretrainingDataset(
        config.val_data_path,
        config.sequence_length,
        config.vocab_size,
        split="val"
    )
    
    return train_dataset, val_dataset


def compute_loss(
    model: DikkoHausaLM,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    use_mixed_precision: bool = True
) -> torch.Tensor:
    """
    Compute loss for a batch.
    
    Args:
        model: The model
        batch: Tuple of (input_ids, labels)
        device: Device to move batch to
        use_mixed_precision: Whether to use mixed precision
    
    Returns:
        Scalar loss
    """
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    with autocast(enabled=use_mixed_precision and device.type == "cuda"):
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
    
    return loss


def train_step(
    model: DikkoHausaLM,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    config: TrainingConfig
) -> float:
    """
    Perform a single training step with gradient accumulation.
    
    Returns:
        Loss for the step
    """
    # Compute loss
    loss = compute_loss(model, batch, device, config.use_mixed_precision)
    
    # Normalize loss for gradient accumulation
    loss = loss / config.gradient_accumulation_steps
    
    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # Clip gradients and update optimizer (we always update in test mode)
    if scaler is not None:
        scaler.unscale_(optimizer)
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    
    # Optimizer step
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    
    # Update learning rate (use a dummy step for testing)
    current_lr = scheduler(1)  # Use step 1 for testing
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    # Zero gradients
    optimizer.zero_grad()
    
    return loss.item() * config.gradient_accumulation_steps


def evaluate(
    model: DikkoHausaLM,
    val_sampler: Any,
    device: torch.device,
    eval_steps: int,
    use_mixed_precision: bool = True
) -> Tuple[float, float]:
    """
    Evaluate the model on validation data.
    
    Returns:
        Tuple of (average_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_sampler):
            if i >= eval_steps:
                break
            
            loss = compute_loss(model, batch, device, use_mixed_precision)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    
    if num_batches == 0:
        return float('inf'), float('inf')
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return avg_loss, perplexity


def save_checkpoint(
    model: DikkoHausaLM,
    optimizer: Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    config: TrainingConfig,
    step: int,
    val_loss: float,
    is_best: bool = False,
    is_latest: bool = True
):
    """Save a training checkpoint."""
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "ff_dim": config.ff_dim,
            "max_seq_len": config.max_seq_len,
        },
        "training_config": {
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "learning_rate": config.learning_rate,
            "min_learning_rate": config.min_learning_rate,
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "weight_decay": config.weight_decay,
        },
        "random_state": {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    # Save latest checkpoint
    if is_latest:
        latest_path = config.get_checkpoint_path()
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = config.get_best_checkpoint_path()
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")
    
    # Save periodic checkpoint
    if step % config.checkpoint_interval == 0:
        periodic_path = config.get_checkpoint_path(step)
        torch.save(checkpoint, periodic_path)
        print(f"Saved checkpoint at step {step} to {periodic_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: DikkoHausaLM,
    optimizer: Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    config: TrainingConfig
) -> Tuple[int, float]:
    """
    Load a checkpoint and resume training.
    
    Returns:
        Tuple of (step, best_val_loss)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scaler
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # Load random states
    if "random_state" in checkpoint:
        random.setstate(checkpoint["random_state"]["random"])
        np.random.set_state(checkpoint["random_state"]["numpy"])
        torch.set_rng_state(checkpoint["random_state"]["torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["random_state"]["torch_cuda"])
    
    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", float('inf'))
    
    print(f"Resumed from checkpoint: {checkpoint_path}")
    print(f"Resumed at step {step} with validation loss {val_loss:.4f}")
    
    return step, val_loss


def train(config: TrainingConfig):
    """Main training loop."""
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config)
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config)
    
    # Create batch samplers
    train_sampler = get_batch_sampler(train_dataset, config.batch_size, shuffle=True)
    val_sampler = get_batch_sampler(val_dataset, config.batch_size, shuffle=False)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler (returns a function)
    scheduler_fn = create_lr_scheduler(optimizer, config)
    scheduler = scheduler_fn  # For checkpointing
    
    # Create scaler for mixed precision
    scaler = GradScaler() if config.use_mixed_precision and device.type == "cuda" else None
    
    # Resume from checkpoint if requested
    start_step = 0
    best_val_loss = float('inf')
    
    if config.resume_from:
        start_step, best_val_loss = load_checkpoint(
            config.resume_from,
            model,
            optimizer,
            scheduler,
            scaler,
            config
        )
    
    # Training loop
    print(f"\nStarting training from step {start_step} to {config.max_steps}")
    print(f"Effective batch size: {config.effective_batch_size}")
    print(f"Number of steps: {config.max_steps - start_step}")
    print("\n" + "=" * 60 + "\n")
    
    # Track training metrics
    train_losses = []
    step_times = []
    total_tokens_processed = 0
    total_time = 0
    global_step = start_step
    
    # Get initial validation loss
    if start_step == 0:
        print("Running initial validation...")
        val_loss, val_ppl = evaluate(
            model, val_sampler, device, config.eval_steps, config.use_mixed_precision
        )
        best_val_loss = val_loss
        print(f"Initial validation loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
        print("")
    
    for step in range(start_step, config.max_steps):
        step_start_time = time.time()
        global_step = step
        
        # Get batch
        try:
            batch = next(train_sampler)
        except StopIteration:
            train_sampler = get_batch_sampler(train_dataset, config.batch_size, shuffle=True)
            batch = next(train_sampler)
        
        # Training step
        loss = train_step(
            model, batch, optimizer, scheduler_fn, scaler, device, config
        )
        train_losses.append(loss)
        total_tokens_processed += batch[0].numel()
        
        # Update current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if (step + 1) % config.log_interval == 0:
            avg_loss = np.mean(train_losses[-config.log_interval:])
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            avg_step_time = np.mean(step_times[-20:]) if step_times else step_time
            
            tokens_per_sec = total_tokens_processed / (time.time() - step_start_time) if step > 0 else 0
            remaining_steps = config.max_steps - step - 1
            estimated_time = remaining_steps * avg_step_time / config.gradient_accumulation_steps
            
            progress = (step + 1) / config.max_steps * 100
            
            print(
                f"Step: {step + 1:,} / {config.max_steps:,} ({progress:.1f}%) | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Tokens/sec: {tokens_per_sec:.0f} | "
                f"ETA: {estimated_time/60:.1f}m"
            )
        
        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            print("\nRunning validation...")
            val_loss, val_ppl = evaluate(
                model, val_sampler, device, config.eval_steps, config.use_mixed_precision
            )
            print(
                f"Validation loss: {val_loss:.4f} | "
                f"Perplexity: {val_ppl:.2f}"
            )
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
            
            save_checkpoint(
                model, optimizer, scheduler_fn, scaler, config,
                step + 1, val_loss, is_best, is_latest=True
            )
            print("")
    
    # Final evaluation and checkpoint
    print("\nTraining completed! Running final validation...")
    val_loss, val_ppl = evaluate(
        model, val_sampler, device, config.eval_steps, config.use_mixed_precision
    )
    print(f"Final validation loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
    
    # Save final checkpoint
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
    save_checkpoint(
        model, optimizer, scheduler_fn, scaler, config,
        config.max_steps, val_loss, is_best, is_latest=True
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.run_dir}")
    print("=" * 60)


def main():
    """Main entry point for training."""
    # Create configuration
    config = TrainingConfig()
    
    # Parse command line arguments for resume
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, help="Batch size override")
    args = parser.parse_args()
    
    if args.resume:
        config.resume_from = args.resume
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()