"""
DIKKO AI NOMA - FastAPI Backend with RAG Integration
All-in-one API file with Retrieval-Augmented Generation support
"""

import os
import sys
import pickle
import time
import json
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import sentencepiece as spm
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# For RAG support using Sentence Transformers and FAISS (or fallback cosine similarity)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import model
from src.model.config import ModelConfig
from src.model.model import DikkoHausaLM

# ============================================================================
# SCHEMAS (Request/Response Models)
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message in Hausa")
    max_new_tokens: Optional[int] = Field(100, ge=1, le=512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(50, ge=1, le=100, description="Top-k sampling")
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    do_sample: Optional[bool] = Field(True, description="Whether to sample or use greedy decoding")
    use_rag: Optional[bool] = Field(True, description="Whether to augment prompt with RAG context")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID from UI")

    class Config:
        extra = "allow"

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Generated response in Hausa")
    model: str = Field("dikko-ai-noma", description="Model name")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    retrieved_context: Optional[List[str]] = Field(None, description="Retrieved RAG knowledge chunks used")

class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model: str
    rag_enabled: bool

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
    training_step: Optional[int] = None
    rag_status: str

# ============================================================================
# RAG RETRIEVER COMPONENT
# ============================================================================

class RAGRetriever:
    """Handles loading vector index / knowledge base and retrieving relevant snippets."""

    def __init__(self, index_dir: str = "data/rag_index"):
        self.index_dir = index_dir
        self.encoder = None
        self.documents = []
        self.embeddings = None
        self.is_loaded = False

    def load(self):
        if not HAS_RAG_DEPS:
            print("⚠️ sentence-transformers or numpy not installed. RAG functionality disabled.")
            return

        try:
            # 1. Load documents/chunks
            docs_path = os.path.join(self.index_dir, "documents.json")
            embeddings_path = os.path.join(self.index_dir, "embeddings.npy")
            model_name_path = os.path.join(self.index_dir, "model_name.txt")

            if not (os.path.exists(docs_path) and os.path.exists(embeddings_path)):
                print(f"⚠️ RAG index files not found in {self.index_dir}. RAG will be inactive.")
                return

            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            self.embeddings = np.load(embeddings_path)

            # 2. Load embedding model (default to all-MiniLM-L6-v2 or fallback)
            encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
            if os.path.exists(model_name_path):
                with open(model_name_path, "r", encoding="utf-8") as f:
                    encoder_name = f.read().strip()

            print(f"🔍 Loading RAG embedding model: {encoder_name}")
            self.encoder = SentenceTransformer(encoder_name)
            self.is_loaded = True
            print(f"✅ RAG retriever loaded successfully with {len(self.documents)} knowledge chunks.")

        except Exception as e:
            print(f"❌ Failed to initialize RAG retriever: {e}")
            self.is_loaded = False

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        if not self.is_loaded or len(self.documents) == 0:
            return []

        try:
            # Encode query
            query_vector = self.encoder.encode([query], convert_to_numpy=True)[0]

            # Compute cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)
            norms[norms == 0] = 1e-10
            similarities = np.dot(self.embeddings, query_vector) / norms

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = [self.documents[idx] for idx in top_indices if similarities[idx] > 0.1]
            return results
        except Exception as e:
            print(f"⚠️ Error during RAG retrieval: {e}")
            return []

# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """Load and manage the Dikko AI Noma model and tokenizer."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = None
        self.checkpoint_path = None
        self.training_step = None
    
    def load(self, checkpoint_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        """Load the model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
        
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint()
        self.checkpoint_path = checkpoint_path
        
        if tokenizer_path is None:
            tokenizer_path = self._find_tokenizer()
        
        print(f"🔤 Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        print(f"📦 Loading checkpoint from: {checkpoint_path}")
        checkpoint = self._load_checkpoint(checkpoint_path)
        
        if "model_config" in checkpoint:
            self.config = checkpoint["model_config"]
            print(f"✅ Loaded config from checkpoint")
        else:
            print("⚠️ No model_config found, using default config")
            self.config = ModelConfig()
        
        self.training_step = checkpoint.get("step", None)
        
        print(f"\n🤖 Creating model with config:")
        print(f"    vocab_size: {self.config.vocab_size}")
        print(f"    hidden_size: {self.config.hidden_size}")
        print(f"    num_layers: {self.config.num_layers}")
        print(f"    num_heads: {self.config.num_heads}")
        
        self.model = DikkoHausaLM(self.config)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        remapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            new_key = new_key.replace("tok_embeddings.weight", "token_embeddings.weight")
            new_key = new_key.replace("norm.weight", "final_norm.weight")
            new_key = new_key.replace("output.weight", "lm_head.weight")
            new_key = new_key.replace(".attn.", ".attention.")
            remapped_state_dict[new_key] = value
            
        self.model.load_state_dict(remapped_state_dict, strict=False)
        print("✅ Loaded and remapped model_state_dict successfully")
        
        self.model.to(self.device)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n✅ Model loaded successfully!")
        print(f"    Parameters: {total_params:,}")
        print(f"    Model size: {total_params * 4 / (1024 * 1024):.2f} MB")
        print(f"    Training step: {self.training_step}")
        
        return self.model, self.tokenizer
    
    def _load_checkpoint(self, checkpoint_path: str):
        try:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == 'ModelConfig':
                        return ModelConfig
                    if name == 'DikkoHausaLM':
                        return DikkoHausaLM
                    return super().find_class(module, name)
            
            with open(checkpoint_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                checkpoint = unpickler.load()
                print("✅ Loaded with custom unpickler")
                return checkpoint
        except Exception as e:
            print(f"⚠️ Custom unpickler failed: {e}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            print("✅ Loaded with torch.load")
            return checkpoint
        except Exception as e:
            print(f"❌ All loading methods failed: {e}")
            raise
    
    def _find_checkpoint(self) -> str:
        possible_paths = [
            "checkpoints/checkpoint_finetune20k.pt",
        ]
        
        for root, dirs, files in os.walk("checkpoints"):
            for file in files:
                if file.endswith(".pt"):
                    full_path = os.path.join(root, file)
                    if full_path not in possible_paths:
                        possible_paths.append(full_path)
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found checkpoint: {path}")
                return path
        
        raise FileNotFoundError(f"No checkpoint found in checkpoints/")
    
    def _find_tokenizer(self) -> str:
        possible_paths = [
            "src/tokenizer/hausa_tokenizer.model",
        ]
        
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".model") and "tokenizer" in file.lower():
                    possible_paths.append(os.path.join(root, file))
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found tokenizer: {path}")
                return path
        
        raise FileNotFoundError(f"No tokenizer found")

# ============================================================================
# TEXT GENERATOR
# ============================================================================

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
        self.block_size = getattr(config, "block_size", getattr(config, "max_seq_len", 128))
        
        stop_token_ids = self.tokenizer.encode(self.SPECIAL_TOKENS["end"])
        self.stop_id = stop_token_ids[-1] if len(stop_token_ids) > 0 else None
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.35,
        top_k: int = 40,
        top_p: float = 0.6,
        do_sample: bool = True
    ) -> tuple[str, float]:
        start_time = time.time()
        
        formatted_prompt = f"{self.SPECIAL_TOKENS['begin']}{self.SPECIAL_TOKENS['instruction']}{prompt}{self.SPECIAL_TOKENS['response']}"
        
        encoded_prompt = self.tokenizer.encode(formatted_prompt)
        idx = torch.tensor([encoded_prompt], dtype=torch.long, device=self.device)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            with torch.no_grad():
                outputs = self.model(idx_cond)
                logits = outputs["logits"][:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            if self.stop_id is not None and idx_next.item() == self.stop_id:
                idx = torch.cat((idx, idx_next), dim=1)
                break
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        full_output = self.tokenizer.decode(idx[0].tolist())
        
        response = full_output
        if self.SPECIAL_TOKENS["response"] in response:
            response = response.split(self.SPECIAL_TOKENS["response"])[-1]
        
        for token in self.SPECIAL_TOKENS.values():
            response = response.replace(token, "")
        response = response.replace("<|", "").replace("|>", "")
        response = " ".join(response.split()).strip()
        
        if not response:
            response = "Ba a samu amsa ba."
            
        generation_time = time.time() - start_time
        return response, generation_time

# ============================================================================
# FASTAPP INSTANTIATION & LIFESPAN
# ============================================================================

model_loader = ModelLoader()
rag_retriever = RAGRetriever()
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and RAG index on startup."""
    print("\n🚀 Starting Dikko AI Noma API with RAG...")
    try:
        model_loader.load()
        global generator
        generator = TextGenerator(
            model_loader.model,
            model_loader.tokenizer,
            model_loader.config
        )
        # Load RAG retriever vector indices
        rag_retriever.load()
        print("✅ API ready with RAG support!")
    except Exception as e:
        print(f"❌ Error during startup sequence: {e}")
        raise
    yield
    print("👋 Shutting down...")

app = FastAPI(
    title="Dikko AI Noma API",
    description="Hausa Agricultural Language Model with RAG",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dikko AI Noma API (RAG Enabled)",
        "version": "1.1.0",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "model-info": "/api/model-info"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model="dikko-ai-noma",
        rag_enabled=rag_retriever.is_loaded
    )

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model and RAG status information."""
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model_loader.model.parameters())
    return ModelInfoResponse(
        model_name="dikko-ai-noma",
        vocabulary_size=model_loader.config.vocab_size,
        num_layers=model_loader.config.num_layers,
        hidden_size=model_loader.config.hidden_size,
        num_heads=model_loader.config.num_heads,
        block_size=getattr(model_loader.config, "block_size", 128),
        tokenizer_path=model_loader._find_tokenizer(),
        checkpoint_path=model_loader.checkpoint_path,
        device=str(model_loader.device),
        total_parameters=total_params,
        training_step=model_loader.training_step,
        rag_status="Active" if rag_retriever.is_loaded else "Inactive/Not Loaded"
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate a RAG-augmented response from the model."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.message) > 500:
        raise HTTPException(status_code=400, detail="Message too long (max 500 characters)")
    
    try:
        context_chunks = []
        final_prompt = request.message

        # Perform RAG retrieval if enabled and available
        if request.use_rag and rag_retriever.is_loaded:
            context_chunks = rag_retriever.retrieve(request.message, top_k=2)
            if context_chunks:
                context_str = "\n".join(context_chunks)
                # Formulate augmented prompt structure
                final_prompt = f"Bayani: {context_str}\nTambaya: {request.message}"

        response, gen_time = generator.generate(
            prompt=final_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        
        return ChatResponse(
            response=response,
            model="dikko-ai-noma",
            generation_time=round(gen_time, 2),
            retrieved_context=context_chunks if context_chunks else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )