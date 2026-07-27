"""
Overfitting sanity test.

Trains on a tiny subset of data for a few hundred steps to verify
that the model can overfit to the training data.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model import ModelConfig, DikkoHausaLM
from src.training.config import TrainingConfig
from src.training.dataset import PretrainingDataset
from src.training.train import create_model, set_seed, get_device, train


class OverfitConfig(TrainingConfig):
    """Configuration for overfitting test."""
    
    def __init__(self):
        super().__init__()
        # Small configuration for overfitting
        self.batch_size = 4
        self.sequence_length = 64
        self.max_steps = 3
        self.warmup_steps = 10
        self.eval_interval = 50
        self.checkpoint_interval = 1
        self.log_interval = 1
        self.run_name = "dikko_overfit_test"
        self.learning_rate = 1e-3  # Higher LR for faster overfitting
        self.weight_decay = 0.0  # No weight decay for overfitting
        self.gradient_accumulation_steps = 1
        self.use_mixed_precision = False  # Disable for simplicity


def test_overfit():
    """Run overfitting test."""
    print("\n" + "=" * 60)
    print("DIKKO AI NOMA- OVERFITTING TEST")
    print("=" * 60)
    print("\nThis test trains on a tiny dataset to verify the pipeline.")
    print("The model should quickly overfit and reach near-zero loss.\n")
    
    # Create configuration
    config = OverfitConfig()
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(config)
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Load datasets
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
    
    # Use only a tiny subset for overfitting
    small_size = min(50, len(train_dataset))
    train_indices = np.random.choice(len(train_dataset), small_size, replace=False)
    train_subset = Subset(train_dataset, train_indices)
    print(f"Using {len(train_subset)} sequences for training")
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create optimizer
    from src.training.optimizer import create_optimizer, create_lr_scheduler
    optimizer = create_optimizer(model, config)
    scheduler_fn = create_lr_scheduler(optimizer, config)
    
    # Training loop
    print("\nStarting overfitting test...")
    print(f"Training for {config.max_steps} steps\n")
    
    losses = []
    val_losses = []
    
    for step in range(config.max_steps):
        # Get batch
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            train_loader = DataLoader(
                train_subset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            batch = next(iter(train_loader))
        
        # Forward pass
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        
        # Update learning rate
        current_lr = scheduler_fn(step + 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        losses.append(loss.item())
        
        # Logging
        if (step + 1) % config.log_interval == 0:
            avg_loss = np.mean(losses[-config.log_interval:])
            print(f"Step {step + 1:3d}/{config.max_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # Validation
        if (step + 1) % config.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, val_batch in enumerate(val_loader):
                    if i >= 5:  # Only 5 validation batches
                        break
                    val_input_ids, val_labels = val_batch
                    val_input_ids = val_input_ids.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_input_ids, labels=val_labels)
                    val_loss += val_outputs["loss"].item()
            val_loss /= min(5, len(val_loader))
            val_losses.append(val_loss)
            model.train()
            print(f"  Validation loss: {val_loss:.4f}")
    
    # Results
    print("\n" + "=" * 60)
    print("OVERFITTING TEST RESULTS")
    print("=" * 60)
    
    final_loss = losses[-1] if losses else float('inf')
    final_val_loss = val_losses[-1] if val_losses else float('inf')
    min_loss = min(losses) if losses else float('inf')
    
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Minimum training loss: {min_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Determine if test passed
    if min_loss < 0.1:
        print("\n✅ OVERFITTING TEST PASSED!")
        print("The model successfully overfitted to the tiny dataset.")
        print("The training pipeline is working correctly.")
    else:
        print("\n⚠️ OVERFITTING TEST FAILED!")
        print("The model did not reach near-zero loss on the tiny dataset.")
        print("This may indicate issues with the model or training pipeline.")
    
    print("\n" + "=" * 60)
    
    return min_loss < 0.1


if __name__ == "__main__":
    success = test_overfit()
    sys.exit(0 if success else 1)