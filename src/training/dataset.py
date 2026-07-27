"""Module: dataset."""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import os

class PretrainingDataset(Dataset):
    """
    Memory-efficient dataset for pretraining on token IDs.
    Uses numpy memmap to avoid loading entire dataset into RAM.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int,
        vocab_size: int,
        split: str = "train"
    ):
        """
        Args:
            data_path: Path to the .bin file containing token IDs
            sequence_length: Number of tokens per sequence
            vocab_size: Vocabulary size for validation
            split: "train" or "val" for logging purposes
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.split = split
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Memory-map the binary file
        # The .bin files contain uint16 token IDs
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # Total number of tokens
        self.num_tokens = len(self.data)
        
        # Number of possible sequences
        # Each sequence needs sequence_length + 1 tokens (for labels shift)
        self.num_sequences = max(0, self.num_tokens - sequence_length)
        
        print(f"[{split}] Loaded {self.num_tokens:,} tokens from {data_path}")
        print(f"[{split}] Number of sequences: {self.num_sequences:,}")
        
        if self.num_sequences == 0:
            raise ValueError(
                f"Data file {data_path} has {self.num_tokens} tokens, "
                f"but sequence_length {sequence_length} requires at least "
                f"{sequence_length + 1} tokens"
            )
        
        # Validate token IDs are within vocabulary range
        self._validate_tokens()
    
    def _validate_tokens(self):
        """Validate that all token IDs are within vocabulary range."""
        # Sample a small portion to avoid iterating over entire dataset
        sample_size = min(10000, self.num_tokens)
        sample_indices = np.random.choice(self.num_tokens, sample_size, replace=False)
        sample_tokens = self.data[sample_indices]
        
        if np.max(sample_tokens) >= self.vocab_size:
            max_token = np.max(sample_tokens)
            raise ValueError(
                f"Token ID {max_token} found in {self.split} data, "
                f"but vocab_size is {self.vocab_size}. "
                f"Please check tokenizer and data generation."
            )
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of input_ids and labels.
        
        Args:
            idx: Sequence index
        
        Returns:
            Tuple of (input_ids, labels)
            - input_ids: [sequence_length] token IDs
            - labels: [sequence_length] shifted token IDs
        """
        if idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range for dataset")
        
        # Get token IDs for this sequence
        # Input: tokens[idx : idx + sequence_length]
        # Target: tokens[idx + 1 : idx + sequence_length + 1]
        start_idx = idx
        end_idx = start_idx + self.sequence_length + 1
        
        # Extract tokens
        tokens = self.data[start_idx:end_idx].astype(np.int64)
        
        # Create input and labels
        input_ids = tokens[:-1]  # First sequence_length tokens
        labels = tokens[1:]      # Last sequence_length tokens (shifted by 1)
        
        # Convert to torch tensors
        input_ids = torch.from_numpy(input_ids).long()
        labels = torch.from_numpy(labels).long()
        
        return input_ids, labels


def get_batch_sampler(
    dataset: PretrainingDataset,
    batch_size: int,
    shuffle: bool = True
):
    """
    Create an efficient batch sampler that yields batches of sequences.
    
    This is more memory-efficient than using a DataLoader for large datasets
    because we can use the dataset's __getitem__ directly.
    
    Args:
        dataset: PretrainingDataset instance
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle indices
    
    Returns:
        Generator yielding batches of (input_ids, labels)
    """
    num_sequences = len(dataset)
    indices = np.arange(num_sequences)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_sequences, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Collect batch data
            batch_inputs = []
            batch_labels = []
            
            for idx in batch_indices:
                input_ids, labels = dataset[idx]
                batch_inputs.append(input_ids)
                batch_labels.append(labels)
            
            # Stack into tensors
            batch_inputs = torch.stack(batch_inputs)
            batch_labels = torch.stack(batch_labels)
            
            yield batch_inputs, batch_labels