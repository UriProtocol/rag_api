from fastapi import FastAPI
import requests
import chromadb
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
load_dotenv()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Connect to Chroma Cloud ---
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)

# --- Embedding using Ollama ---
def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    response.raise_for_status()  # <-- ADD THIS
    return response.json()["embedding"]


# --- RAG Pipeline ---
def rag(query, collection_name):
    collection = client.get_collection(collection_name)

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    documents = results.get("documents")

    if not documents or not documents[0]:
        return "No encontré información relevante en la base de datos."

    context = "\n\n".join(documents[0])

    prompt = f"""
Responde usando principalmente el contexto proporcionado.
Si no encuentras información suficiente, indícalo claramente.

Contexto:
{context}

Pregunta:
{query}
"""

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "gpt-oss:20b",
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un asistente útil. Responde SIEMPRE en español."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
    )

    response.raise_for_status()
    return response.json()["message"]["content"]


@app.post("/chat")
def chat(payload: dict):
    message = payload.get("message")
    collection_name = payload.get("collection")

    if not message:
        return {"error": "message is required"}

    if not collection_name:
        return {"error": "collection is required"}

    return {
        "response": rag(message, collection_name)
    }