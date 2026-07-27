#!/usr/bin/env python
"""
Training script for DikkoHausaLM.

Usage:
    python scripts/train.py
    python scripts/train.py --resume checkpoints/run_name/latest.pt
    python scripts/train.py --batch_size 16
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.train import main

if __name__ == "__main__":
    main()