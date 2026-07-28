from fastapi import APIRouter, HTTPException
from typing import Optional
import time

from ..schemas import ChatRequest, ChatResponse
from ..model_loader import model_loader
from ..generator import TextGenerator

router = APIRouter()

# Initialize generator
generator = None

def get_generator():
    """Lazy load the generator."""
    global generator
    if generator is None:
        if model_loader.model is None:
            model_loader.load()
        generator = TextGenerator(
            model_loader.model,
            model_loader.tokenizer,
            model_loader.config
        )
    return generator

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generate a response from the model.
    """
    try:
        # Validate input length
        if len(request.message) > 500:
            raise HTTPException(status_code=400, detail="Message too long (max 500 characters)")
        
        # Get generator
        gen = get_generator()
        
        # Generate response
        start_time = time.time()
        response = gen.generate(
            prompt=request.message,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        generation_time = time.time() - start_time
        
        # Log generation time
        print(f"⏱️ Generation time: {generation_time:.2f}s")
        print(f"📝 Response length: {len(response)} chars")
        
        return ChatResponse(
            response=response,
            model="dikko-ai-noma"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))