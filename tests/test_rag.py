import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.rag.rag_service import rag_service

def test_rag_retrieval():
    print("🧪 Running RAG Tests...")
    rag_service.initialize()
    
    test_cases = [
        ("Menene noma?", True),
        ("Yaya ake noman masara?", True),
        ("Yaya ake noman tumatir?", True),
        ("Ta yaya ake amfani da taki?", True),
        ("Menene programming?", False),
    ]
    
    for query, expected_scope in test_cases:
        print(f"\nQuery: '{query}' (Expected Scope: {expected_scope})")
        result = rag_service.retrieve(query, top_k=3)
        is_in_scope = rag_service.is_agricultural_query(query)
        
        print(f"  - In Scope: {is_in_scope}")
        print(f"  - RAG Used: {result['rag_used']}")
        print(f"  - Confidence: {result['confidence']}")
        print(f"  - Sources Count: {len(result['sources'])}")
        
        if is_in_scope != expected_scope:
            print(f"  ❌ Failed scope match for: {query}")
        else:
            print(f"  ✅ Passed test case.")

if __name__ == "__main__":
    test_rag_retrieval()