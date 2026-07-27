#!/usr/bin/env python
"""
Sanity check script for the training pipeline.

Tests:
1. Data loading
2. Model creation
3. Forward pass
4. Backward pass
5. Training step
"""

import sys
import os
import tempfile
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model import ModelConfig, DikkoHausaLM
from src.training.config import TrainingConfig
from src.training.dataset import PretrainingDataset, get_batch_sampler
from src.training.train import create_model, compute_loss, train_step, set_seed
from src.training.optimizer import create_optimizer, create_lr_scheduler


def test_pipeline():
    """Run sanity checks on the training pipeline."""
    print("\n" + "=" * 60)
    print("DIKKO AI NOMA - SANITY CHECK")
    print("=" * 60 + "\n")
    
    # Test 1: Dataset loading
    print("1. Testing dataset loading...")
    try:
        config = TrainingConfig()
        # Override for testing
        config.sequence_length = 64  # Use shorter sequences for faster testing
        config.batch_size = 2
        
        train_dataset = PretrainingDataset(
            config.train_data_path,
            config.sequence_length,
            config.vocab_size,
            split="train"
        )
        print(f"✅ Dataset loaded successfully")
        print(f"   Number of sequences: {len(train_dataset):,}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Test 2: Batch sampling
    print("\n2. Testing batch sampling...")
    try:
        sampler = get_batch_sampler(train_dataset, batch_size=2, shuffle=True)
        batch = next(sampler)
        input_ids, labels = batch
        print(f"✅ Batch sampled successfully")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Label shape: {labels.shape}")
        print(f"   Token range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        
        # Verify token range
        assert input_ids.min() >= 0
        assert input_ids.max() < config.vocab_size
        assert labels.min() >= 0
        assert labels.max() < config.vocab_size
    except Exception as e:
        print(f"❌ Failed to sample batch: {e}")
        return False
    
    # Test 3: Model creation
    print("\n3. Testing model creation...")
    try:
        model = create_model(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False
    
    # Test 4: Forward pass
    print("\n4. Testing forward pass...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        outputs = model(input_ids, labels=labels)
        logits = outputs["logits"]
        loss = outputs["loss"]
        
        print(f"✅ Forward pass successful")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")
        
        # Verify shapes
        assert logits.shape == (2, config.sequence_length, config.vocab_size)
        assert loss is not None
        assert torch.isfinite(loss)
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test 5: Backward pass
    print("\n5. Testing backward pass...")
    try:
        loss.backward()
        print(f"✅ Backward pass successful")
        
        # Check gradients
        has_grads = any(p.grad is not None for p in model.parameters())
        print(f"   Gradients computed: {has_grads}")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    # Test 6: Training step
    print("\n6. Testing training step...")
    try:
        optimizer = create_optimizer(model, config)
        scheduler_fn = create_lr_scheduler(optimizer, config)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Create scaler
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Single training step
        batch = next(get_batch_sampler(train_dataset, batch_size=2, shuffle=True))
        loss = train_step(
            model, batch, optimizer, scheduler_fn, scaler,
            device, config
        )
        print(f"✅ Training step successful")
        print(f"   Loss: {loss:.4f}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe training pipeline is ready for full pretraining.")
    print("To start training:")
    print("  python scripts/train.py")
    print("\nTo resume training:")
    print("  python scripts/train.py --resume checkpoints/run_name/latest.pt")
    print("\nTo run a tiny overfitting test:")
    print("  python scripts/test_overfit.py")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)