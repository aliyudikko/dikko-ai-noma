import requests
import json

BASE_URL = "http://127.0.0.1:8000/api"

def test_health():
    print("Testing /api/health...")
    res = requests.get(f"{BASE_URL}/health")
    print(json.dumps(res.json(), indent=2))
    assert res.status_code == 200

def test_model_info():
    print("\nTesting /api/model-info...")
    res = requests.get(f"{BASE_URL}/model-info")
    print(json.dumps(res.json(), indent=2))
    assert res.status_code == 200

def test_chat():
    print("\nTesting /api/chat...")
    payload = {
        "message": "Menene noma?",
        "max_new_tokens": 80,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload)
    print(json.dumps(res.json(), indent=2, ensure_ascii=False))
    assert res.status_code == 200

if __name__ == "__main__":
    try:
        test_health()
        test_model_info()
        test_chat()
        print("\nAll API tests passed successfully!")
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to FastAPI server. Ensure it is running via uvicorn.")