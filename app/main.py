"""
FastAPI deployment server for the e-commerce RAG support chatbot.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then test with:
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \\
         -d '{"message": "Where is my order?"}'

or open http://127.0.0.1:8000/docs for an interactive Swagger UI.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from app.pipeline import handle_message

app = FastAPI(title="E-commerce Support Chatbot", version="1.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    language: str
    sentiment: str
    intent: str
    reply: str
    escalated: bool
    sources: list


@app.get("/")
def root():
    return {"status": "ok", "message": "Chatbot API is running. POST to /chat."}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = handle_message(req.message)
    return result
