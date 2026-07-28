# src/api/model_loader.py (COMPLETE FIX)
import os
import torch
import sentencepiece as spm
from typing import Tuple, Optional
import sys
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import ModelConfig from the correct module
from src.model.config import ModelConfig
from src.model.model import DikkoHausaLM

class ModelLoader:
    """Load and manage the Dikko AI Noma model and tokenizer."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = None
        self.checkpoint_path = None
        
    def load(self, checkpoint_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        """Load the model and tokenizer."""
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
        
        # Find checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint()
        self.checkpoint_path = checkpoint_path
        
        # Find tokenizer
        if tokenizer_path is None:
            tokenizer_path = self._find_tokenizer()
        
        # Load tokenizer
        print(f"🔤 Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        # Load checkpoint with custom unpickler
        print(f"📦 Loading checkpoint from: {checkpoint_path}")
        checkpoint = self._load_checkpoint(checkpoint_path)
        
        # Get model config from checkpoint
        if "model_config" in checkpoint:
            self.config = checkpoint["model_config"]
            print(f"✅ Loaded config from checkpoint")
        else:
            print("⚠️ No model_config found in checkpoint, using default config")
            self.config = ModelConfig()
            
        # Create model
        print(f"\n🤖 Creating model with config:")
        print(f"   vocab_size: {self.config.vocab_size}")
        print(f"   hidden_size: {self.config.hidden_size}")
        print(f"   num_layers: {self.config.num_layers}")
        print(f"   num_heads: {self.config.num_heads}")
        print(f"   block_size: {self.config.block_size}")
        
        self.model = DikkoHausaLM(self.config)
        
        # Load weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded model_state_dict from checkpoint")
        else:
            # Try loading directly
            self.model.load_state_dict(checkpoint)
            print("✅ Loaded model weights directly")
            
        self.model.to(self.device)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n✅ Model loaded successfully!")
        print(f"   Parameters: {total_params:,}")
        print(f"   Model size: {total_params * 4 / (1024 * 1024):.2f} MB")
        print(f"   Vocabulary size: {self.config.vocab_size}")
        print(f"   Training step: {checkpoint.get('step', 'N/A')}")
        print(f"   Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        return self.model, self.tokenizer
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint with custom handling for ModelConfig."""
        # First, try the standard way with weights_only=False
        try:
            # Create a custom unpickler that knows about ModelConfig
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # If it's trying to load ModelConfig from __main__, redirect to src.model.config
                    if name == 'ModelConfig' and module in ['__main__', 'src.model.config']:
                        return ModelConfig
                    # If it's trying to load DikkoHausaLM from __main__, redirect to src.model.model
                    if name == 'DikkoHausaLM' and module in ['__main__', 'src.model.model']:
                        return DikkoHausaLM
                    # For everything else, use the default behavior
                    return super().find_class(module, name)
            
            # Read the file
            with open(checkpoint_path, 'rb') as f:
                # Use the custom unpickler
                unpickler = CustomUnpickler(f)
                checkpoint = unpickler.load()
                print("✅ Loaded checkpoint with custom unpickler")
                return checkpoint
                
        except Exception as e:
            print(f"⚠️ Custom unpickler failed: {e}")
            
            # Fallback: Try torch.load with weights_only=False
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                print("✅ Loaded checkpoint with torch.load(weights_only=False)")
                return checkpoint
            except Exception as e2:
                print(f"❌ All loading methods failed: {e2}")
                raise
    
    def _find_checkpoint(self) -> str:
        """Find the best checkpoint in the project."""
        # Your specific checkpoint paths
        possible_paths = [
            "checkpoints/checkpoint_finetune.pt",
            "checkpoints/checkpoint_finetune20k.pt",
            "checkpoints/checkpoint_best.pt",
            "checkpoints/checkpoint_latest.pt",
            "checkpoints/best.pt",
            "checkpoints/latest.pt",
        ]
        
        # Also search recursively for any .pt files
        for root, dirs, files in os.walk("checkpoints"):
            for file in files:
                if file.endswith(".pt"):
                    full_path = os.path.join(root, file)
                    if full_path not in possible_paths:
                        possible_paths.append(full_path)
        
        # Check each path
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found checkpoint: {path}")
                return path
        
        # If no checkpoint found, list available ones
        print("\n❌ No checkpoint found. Available files in checkpoints/:")
        if os.path.exists("checkpoints"):
            for root, dirs, files in os.walk("checkpoints"):
                for file in files:
                    print(f"   - {os.path.join(root, file)}")
        else:
            print("   checkpoints/ directory does not exist")
        
        raise FileNotFoundError(f"No checkpoint found. Please ensure your checkpoint files exist in the checkpoints/ directory.")
    
    def _find_tokenizer(self) -> str:
        """Find the tokenizer model file."""
        possible_paths = [
            "data/tokenizer/hausa_tokenizer.model",
            "src/tokenizer/hausa_tokenizer.model",
            "tokenizer/hausa_tokenizer.model",
        ]
        
        # Also search recursively
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".model") and "tokenizer" in file.lower():
                    possible_paths.append(os.path.join(root, file))
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found tokenizer: {path}")
                return path
        
        raise FileNotFoundError(f"No tokenizer found. Please ensure hausa_tokenizer.model exists.")

# Singleton instance
model_loader = ModelLoader()