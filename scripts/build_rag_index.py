import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.rag.rag_service import RAGService

def main():
    print("🚀 Starting RAG Vector Store Build Script...")
    rag = RAGService()
    rag.build_index()
    print("🎉 RAG Vector Store successfully built and saved to data/rag/chroma_db/")

if __name__ == "__main__":
    main()