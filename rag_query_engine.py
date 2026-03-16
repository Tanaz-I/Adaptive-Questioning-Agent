"""
RAG Query Engine
================
Accepts a user query, retrieves relevant chunks from ChromaDB,
and uses arcee-ai/Trinity-Nano-P (served via LM Studio) to generate
a grounded answer.

Usage:
    python rag_query_engine.py                        # interactive mode
    python rag_query_engine.py "What is inheritance?" # single query mode
"""

import sys
import json
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# Configuration  — edit if needed
# ─────────────────────────────────────────────

CHROMA_DB_DIR    = "./chroma_db"
COLLECTION_NAME  = "pptx_rag"
EMBED_MODEL      = "all-MiniLM-L6-v2"
TOP_K            = 5                          # number of chunks to retrieve

# LM Studio server settings
LM_STUDIO_URL    = "http://localhost:11434/v1/chat/completions"
LLM_MODEL_NAME = "qwen2.5:1.5b-instruct"
LLM_TEMPERATURE  = 0.2                        # low = more factual
LLM_MAX_TOKENS   = 1024


# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful teaching assistant. 
Answer the user's question using ONLY the context provided below from lecture slides.
Be clear, accurate, and concise. If the context does not contain enough information 
to answer the question, say so honestly — do not make up information.

Always structure your answer as:
1. A direct answer to the question
2. Supporting explanation from the slides (cite slide titles when relevant)
"""


# ─────────────────────────────────────────────
# Step 1 — Retrieve relevant chunks from ChromaDB
# ─────────────────────────────────────────────

def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and fetch the top-k most relevant chunks."""
    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query])[0].tolist()

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":         doc,
            "score":        round(1 - dist, 4),
            "file_name":    meta["file_name"],
            "slide_number": meta["slide_number"],
            "slide_title":  meta["slide_title"],
        })

    return chunks


# ─────────────────────────────────────────────
# Step 2 — Build the RAG context block
# ─────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string for the LLM."""
    lines = ["=== Retrieved Context from Lecture Slides ===\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] File: {chunk['file_name']}  |  "
            f"Slide {chunk['slide_number']}: {chunk['slide_title'] or 'Untitled'}"
            f"  (relevance: {chunk['score']})"
        )
        lines.append(chunk["text"])
        lines.append("")  # blank line between chunks
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Step 3 — Query Trinity-Nano-P via LM Studio
# ─────────────────────────────────────────────

def ask_llm(query: str, context: str) -> str:
    """
    Send the query + context to Trinity-Nano-P through the LM Studio
    OpenAI-compatible /v1/chat/completions endpoint.
    """
    user_message = f"{context}\n\n=== Question ===\n{query}"

    payload = {
        "model":       LLM_MODEL_NAME,
        "temperature": LLM_TEMPERATURE,
        "max_tokens":  LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    }

    try:
        response = requests.post(
            LM_STUDIO_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return (
            "[Error] Could not connect to LM Studio.\n"
            "Make sure LM Studio is running and the local server is started on port 1234."
        )
    except requests.exceptions.Timeout:
        return "[Error] Request timed out. The model may still be loading."
    except Exception as e:
        return f"[Error] Unexpected error: {e}"


# ─────────────────────────────────────────────
# Full RAG pipeline: query → retrieve → generate
# ─────────────────────────────────────────────

def rag_answer(query: str, verbose: bool = False) -> str:
    """End-to-end RAG: retrieve chunks then generate a grounded answer."""

    # 1. Retrieve
    chunks = retrieve_chunks(query, top_k=TOP_K)

    if verbose:
        print(f"\n📚 Retrieved {len(chunks)} chunks:")
        for c in chunks:
            print(f"   • [{c['score']}] Slide {c['slide_number']} — {c['slide_title']} ({c['file_name']})")

    # 2. Build context
    context = build_context(chunks)

    # 3. Generate answer
    answer = ask_llm(query, context)
    return answer


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════╗")
    print("║   RAG Query Engine — Trinity-Nano-P      ║")
    print("║   Type 'exit' or 'quit' to stop          ║")
    print("╚══════════════════════════════════════════╝\n")

    # Single query mode (passed as CLI argument)
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}\n")
        print(rag_answer(query, verbose=True))
        return

    # Interactive mode
    while True:
        try:
            query = input(" Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("\n Thinking...\n")
        answer = rag_answer(query, verbose=True)
        print(f"\n Answer:\n{answer}\n")
        print("─" * 60 + "\n")


if __name__ == "__main__":
    main()