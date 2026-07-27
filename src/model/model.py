import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .config import ModelConfig
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock

class DikkoHausaLM(nn.Module):
    """
    DikkoHausaLM: A decoder-only Transformer language model for Hausa.
    
    This is a custom implementation of a causal language model trained
    for next-token prediction on Hausa text.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final RMSNorm
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output head (LM head)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )
        
        # Tie weights: token embeddings and output head share weights
        self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model parameters with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization with appropriate scaling
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            labels: Optional target token IDs for loss calculation (batch_size, seq_len)
            mask: Optional attention mask (batch_size, seq_len, seq_len)
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing:
                - logits: Output logits (batch_size, seq_len, vocab_size)
                - loss: Cross-entropy loss (if labels provided)
                - attention_weights: List of attention weights (optional)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        # (batch_size, seq_len, hidden_size)
        hidden_states = self.token_embeddings(input_ids)
        
        # Store attention weights if requested
        attention_weights = [] if return_attention else None
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states, layer_attn = layer(
                hidden_states,
                mask=mask,
                return_attention=return_attention
            )
            if return_attention and layer_attn is not None:
                attention_weights.append(layer_attn)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Output head
        # (batch_size, seq_len, vocab_size)
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            # Input:  [t1, t2, t3, ..., tn]
            # Target: [t2, t3, t4, ..., t_{n+1}]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore padding tokens if -100 is used
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "attention_weights": attention_weights
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding
        
        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get the last max_seq_len tokens (or less if starting)
                seq_len = input_ids.size(1)
                if seq_len > self.config.max_seq_len:
                    # Crop to max_seq_len
                    input_ids = input_ids[:, -self.config.max_seq_len:]
                
                # Forward pass to get logits for the next token
                outputs = self.forward(input_ids)
                logits = outputs["logits"]  # (batch_size, seq_len, vocab_size)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                    min_top_k = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                # Sample or greedy
                if do_sample:
                    # Convert logits to probabilities
                    probs = nn.functional.softmax(next_token_logits, dim=-1)
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy: take the argmax
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append the new token to the sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        self.train()
        return input_ids