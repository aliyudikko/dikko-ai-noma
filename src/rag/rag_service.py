import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAGService:
    """Lightweight RAG service for Dikko AI Noma using pure Python chunking & ChromaDB."""
    
    def __init__(
        self,
        data_path: str = "data/rag/noma.txt",
        persist_directory: str = "data/rag/chroma_db",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = None
        self.vector_store = None

    def initialize(self):
        """Initialize embeddings and load persistent ChromaDB collection."""
        print("🔍 Initializing Embeddings & ChromaDB...")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"📦 Loading existing ChromaDB from: {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print(f"⚠️ ChromaDB index not found. Building index...")
            self.build_index()
            
        print("✅ RAG Service initialized successfully.")

    def build_index(self):
        """Read noma.txt, chunk manually with Python, and build ChromaDB index."""
        if not os.path.exists(self.data_path):
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, "w", encoding="utf-8") as f:
                f.write("Noma sana'a ce mai muhimmanci a Najeriya. Ana noman masara, shinkafa, da tumatir a lokacin damina.")
            print(f"⚠️ Created placeholder {self.data_path}")

        # Read file natively
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Simple paragraph/sentence-based text splitting in pure Python
        raw_chunks = text.split("\n\n")
        docs = []
        for chunk in raw_chunks:
            cleaned = chunk.strip()
            if cleaned:
                # Further split long paragraphs if needed
                if len(cleaned) > 400:
                    sub_chunks = [cleaned[i:i+350] for i in range(0, len(cleaned), 300)]
                    docs.extend(sub_chunks)
                else:
                    docs.append(cleaned)

        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize and store directly
        self.vector_store = Chroma.from_texts(
            texts=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ Built and persisted ChromaDB index with {len(docs)} chunks.")

    def is_agricultural_query(self, query: str) -> bool:
        """Check if query falls within agricultural scope."""
        query_lower = query.lower()
        out_of_scope_terms = ["programming", "code", "python", "javascript", "react", "software", "football"]
        if any(term in query_lower for term in out_of_scope_terms):
            return False
        return True

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve relevant context chunks with confidence scoring."""
        if not self.is_agricultural_query(query):
            return {
                "rag_used": False,
                "confidence": "LOW",
                "context": "",
                "sources": [],
                "reason": "Out of agricultural scope"
            }

        if not self.vector_store:
            return {"rag_used": False, "confidence": "LOW", "context": "", "sources": []}

        # Similarity search with scores
        docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        
        sources = []
        valid_chunks = []
        
        for doc, score in docs_with_scores:
            relevance = float(score) if score is not None else 0.5
            sources.append({
                "source": "noma.txt",
                "relevance": round(relevance, 2),
                "chunk": doc.page_content
            })
            if relevance >= 0.2:
                valid_chunks.append(doc.page_content)

        if not valid_chunks and docs_with_scores:
            for doc, score in docs_with_scores[:2]:
                valid_chunks.append(doc.page_content)

        confidence = "HIGH" if len(valid_chunks) >= 2 else ("MEDIUM" if len(valid_chunks) == 1 else "LOW")
        combined_context = "\n\n".join(valid_chunks) if valid_chunks else ""

        return {
            "rag_used": bool(combined_context),
            "confidence": confidence,
            "context": combined_context,
            "sources": sources
        }

rag_service = RAGService()