"""
Quick manual test of the running API.
Make sure the server is already running (uvicorn app.main:app --reload --port 8000)
then run:  python test_api.py
"""
import requests

URL = "http://127.0.0.1:8000/chat"

test_messages = [
    "Hi there!",
    "Where is my order #10234, it hasn't arrived yet?",
    "This is the third time I've been charged twice, this is ridiculous!",
    "How do I reset my password?",
    "What's the weather like today?",
]

for msg in test_messages:
    resp = requests.post(URL, json={"message": msg})
    print("=" * 70)
    print("USER:", msg)
    print(resp.json())
