# src/api/generator.py
import torch
import torch.nn.functional as F
from typing import Optional, List
import re

class TextGenerator:
    """Generate text using the Dikko AI Noma model."""
    
    SPECIAL_TOKENS = {
        "begin": "<|begin_of_sample|>",
        "instruction": "<|instruction|>",
        "response": "<|response|>",
        "end": "<|end_of_sample|>"
    }
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # Get token IDs for special tokens
        self.begin_id = self.tokenizer.piece_to_id(self.SPECIAL_TOKENS["begin"])
        self.instruction_id = self.tokenizer.piece_to_id(self.SPECIAL_TOKENS["instruction"])
        self.response_id = self.tokenizer.piece_to_id(self.SPECIAL_TOKENS["response"])
        self.end_id = self.tokenizer.piece_to_id(self.SPECIAL_TOKENS["end"])
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate a response from the model."""
        
        # Format prompt with special tokens
        formatted_prompt = f"{self.SPECIAL_TOKENS['begin']}{self.SPECIAL_TOKENS['instruction']}{prompt}{self.SPECIAL_TOKENS['response']}"
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted_prompt, out_type=int)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Track generated tokens
        generated_tokens = []
        response_started = False
        
        for _ in range(max_new_tokens):
            # Crop to block size
            if input_ids.size(1) > self.config.block_size:
                input_ids = input_ids[:, -self.config.block_size:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Apply top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            token_id = next_token.item()
            
            # Check if we should stop
            if token_id == self.end_id:
                break
            
            # If this is the first response token, start tracking
            if token_id == self.response_id:
                response_started = True
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                continue
            
            # Only add tokens after response started
            if response_started:
                generated_tokens.append(token_id)
                
                # Prevent generating another response token
                if token_id == self.response_id:
                    break
                
                # Add to input
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Safety: prevent infinite generation
            if len(generated_tokens) > max_new_tokens:
                break
        
        # Decode
        if generated_tokens:
            response = self.tokenizer.decode(generated_tokens)
        else:
            response = "Ba a samu amsa ba. "  # "No response found" in Hausa
        
        # Clean response
        response = self._clean_response(response)
        
        return response
    
    def _clean_response(self, text: str) -> str:
        """Clean the generated response by removing special tokens."""
        # Remove special tokens
        for token in self.SPECIAL_TOKENS.values():
            text = text.replace(token, "")
        
        # Remove any remaining special patterns
        text = text.replace("<|", "").replace("|>", "")
        
        # Clean whitespace
        text = " ".join(text.split())
        
        return text.strip()