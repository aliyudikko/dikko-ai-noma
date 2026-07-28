from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message in Hausa")
    max_new_tokens: int = Field(100, ge=1, le=512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    do_sample: bool = Field(True, description="Whether to sample or use greedy decoding")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Generated response in Hausa")
    model: str = Field("dikko-ai-noma", description="Model name")

class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model: str

class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""
    model_name: str
    vocabulary_size: int
    num_layers: int
    hidden_size: int
    num_heads: int
    block_size: int
    tokenizer_path: str
    checkpoint_path: str
    device: str
    total_parameters: int