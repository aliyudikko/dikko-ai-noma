#!/usr/bin/env python
"""
Example client for Dikko AI Noma API.
"""

import requests
import json

class DikkoClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def chat(self, message, max_new_tokens=100, temperature=0.7, top_k=50):
        """Send a chat request."""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "message": message,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k
            }
        )
        return response.json()
    
    def health(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/api/health")
        return response.json()
    
    def model_info(self):
        """Get model information."""
        response = requests.get(f"{self.base_url}/api/model-info")
        return response.json()

# Example usage
if __name__ == "__main__":
    client = DikkoClient()
    
    # Check health
    print("Health:", client.health())
    
    # Get model info
    print("\nModel Info:", json.dumps(client.model_info(), indent=2))
    
    # Chat
    print("\nChatting...")
    response = client.chat("Menene amfanin noman masara?")
    print("\nResponse:", response["response"])