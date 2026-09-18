"""
Shared inference pipeline: loads all 4 trained artifacts once at startup and
exposes a single `handle_message()` function that runs the full 4-stage flow:

    language detection -> sentiment -> intent -> route (direct reply / RAG / escalate)

This module is imported by app/main.py (the FastAPI server). It does not run
by itself.
"""

import os
import joblib
import faiss
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from groq import Groq


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# 1. LOAD ALL ARTIFACTS
# ============================================================

print("Loading language detector...")

language_pipeline = joblib.load(
    os.path.join(
        MODELS_DIR,
        "language_detector.joblib"
    )
)


print("Loading sentiment model...")

SENTIMENT_ID2NAME = {
    0: "negative",
    1: "positive",
    2: "neutral"
}

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(
        MODELS_DIR,
        "sentiment_model"
    )
)

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(
        MODELS_DIR,
        "sentiment_model"
    )
)

sentiment_model.eval()


print("Loading intent classifier...")

intent_pipeline = joblib.load(
    os.path.join(
        MODELS_DIR,
        "intent_classifier.joblib"
    )
)


print("Loading RAG store (embedder + FAISS index + knowledge base)...")

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

rag_index = faiss.read_index(
    os.path.join(
        MODELS_DIR,
        "rag_store",
        "kb.index"
    )
)

kb_df = pd.read_pickle(
    os.path.join(
        MODELS_DIR,
        "rag_store",
        "kb_metadata.pkl"
    )
)


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY"
)

groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)

GROQ_MODEL = "openai/gpt-oss-20b"


print("All models loaded. Ready.")


# ============================================================
# 2. LANGUAGE DETECTION
# ============================================================

def detect_language(text: str) -> str:

    return language_pipeline.predict(
        [text]
    )[0]


# ============================================================
# 3. SENTIMENT DETECTION
# ============================================================

def detect_sentiment(text: str) -> str:

    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():

        logits = sentiment_model(
            **inputs
        ).logits

    pred_id = int(
        torch.argmax(
            logits,
            dim=-1
        )[0]
    )

    return SENTIMENT_ID2NAME[pred_id]


# ============================================================
# 4. INTENT DETECTION
# ============================================================

def detect_intent(text: str) -> str:

    return intent_pipeline.predict(
        [text]
    )[0]


# ============================================================
# 5. RAG RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    k: int = 3
):

    query_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, idxs = rag_index.search(
        query_vec,
        k
    )

    hits = []

    for score, idx in zip(
        scores[0],
        idxs[0]
    ):

        row = kb_df.iloc[idx]

        hits.append(
            {
                "score": float(score),
                "matched_question": row["instruction"],
                "response": row["response"],
            }
        )

    return hits


# ============================================================
# 6. BUILD GROQ PROMPT
# ============================================================

def build_prompt(
    user_message: str,
    retrieved_chunks: list,
    detected_sentiment: str
) -> list:

    context_block = "\n\n".join(
        f"Q: {c['matched_question']}\n"
        f"A: {c['response']}"
        for c in retrieved_chunks
    )

    system_prompt = (
        "You are a helpful and professional customer support "
        "assistant for an online retailer. "

        "Answer the customer's question using ONLY the "
        "information in the retrieved support responses below. "

        "Do not invent information. "

        "If the customer is clearly frustrated, you may "
        "briefly acknowledge their frustration naturally, "
        "but do not use a fixed or repeated apology. "

        "If the retrieved context does not cover the question, "
        "say so honestly and offer to escalate to a human agent. "

        "For simple greetings such as hi, hello, or hey, "
        "respond naturally and briefly. "

        f"The detected sentiment is: {detected_sentiment}."
    )

    user_prompt = (
        f"Context (retrieved past support responses):\n"
        f"{context_block}\n\n"
        f'Customer question: "{user_message}"'
    )

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]


# ============================================================
# 7. GENERATE RAG ANSWER
# ============================================================

def generate_rag_answer(
    user_message: str,
    detected_sentiment: str,
    k: int = 3
) -> dict:

    hits = retrieve(
        user_message,
        k=k
    )

    if not groq_client:

        return {
            "answer": (
                "Groq API key is not configured. "
                "Please set the GROQ_API_KEY environment variable."
            ),
            "sources": hits,
        }

    messages = build_prompt(
        user_message,
        hits,
        detected_sentiment
    )

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=300,
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "sources": hits
    }


# ============================================================
# 8. ROUTING
# ============================================================

SMALL_TALK_REPLIES = {
    "greeting": (
        "Hello! 👋 How can I help you with your "
        "order, account, or billing today?"
    ),
}


RAG_CATEGORIES = {
    "order_status",
    "order_management",
    "billing_and_refunds",
    "account_management"
}


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def handle_message(
    user_message: str
) -> dict:

    # ------------------------------------------
    # Stage 1: Language
    # ------------------------------------------

    language = detect_language(
        user_message
    )

    # ------------------------------------------
    # Stage 2: Sentiment
    # ------------------------------------------

    sentiment = detect_sentiment(
        user_message
    )

    # ------------------------------------------
    # Stage 3: Intent
    # ------------------------------------------

    intent = detect_intent(
        user_message
    )

    # ------------------------------------------
    # Initial result
    # ------------------------------------------

    result = {
        "language": language,
        "sentiment": sentiment,
        "intent": intent,
        "reply": None,
        "escalated": False,
        "sources": [],
    }


    # ========================================================
    # GREETING
    # ========================================================

    if intent == "greeting":

        result["reply"] = (
            SMALL_TALK_REPLIES["greeting"]
        )

        return result


    # ========================================================
    # COMPLAINT
    # ========================================================

    if intent == "complaint":

        result["escalated"] = True

        result["reply"] = (
            "I'm sorry you're having trouble. "
            "I'm escalating this to a member of our "
            "support team who can help you directly."
        )

        return result


    # ========================================================
    # OUT OF SCOPE
    # ========================================================

    if intent == "out_of_scope":

        result["reply"] = (
            "I'm not able to help with that directly here. "
            "You can ask me about an order, refund, delivery, "
            "or your account."
        )

        return result


    # ========================================================
    # RAG
    # ========================================================

    if intent in RAG_CATEGORIES:

        rag_result = generate_rag_answer(
            user_message,
            detected_sentiment=sentiment
        )

        result["reply"] = rag_result["answer"]

        result["sources"] = rag_result["sources"]

        return result


    # ========================================================
    # FALLBACK
    # ========================================================

    result["reply"] = (
        "Let me connect you with a member of our "
        "support team who can help further."
    )

    result["escalated"] = True

    return result