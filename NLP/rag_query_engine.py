"""
RAG Query Engine (Hybrid Scoring Version)

Uses:
- 70% vector similarity
- 30% metadata matching

Usage:
    python rag_query_engine.py
    python rag_query_engine.py "What is inheritance?"
"""

import sys
import json
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Configuration

CHROMA_DB_DIR    = "./chroma_db"
COLLECTION_NAME  = "rag_kb"
EMBED_MODEL      = "all-MiniLM-L6-v2"
TOP_K            = 5

# LM Studio
LM_STUDIO_URL    = "http://localhost:11434/v1/chat/completions"
LLM_MODEL_NAME   = "qwen2.5:1.5b-instruct"
LLM_TEMPERATURE  = 0.2
LLM_MAX_TOKENS   = 1024


# System Prompt

SYSTEM_PROMPT = """You are a helpful teaching assistant. 
Answer the user's question using ONLY the context provided below from lecture slides.
Be clear, accurate, and concise. If the context does not contain enough information 
to answer the question, say so honestly — do not make up information.

Always structure your answer as:
1. A direct answer to the question
2. Supporting explanation from the slides (cite slide titles when relevant)
"""


# Metadata Scoring (30%)

def compute_metadata_score(query: str, metadata: dict) -> float:
    """
    Compute keyword overlap score between query and metadata.
    Returns value between 0 and 1.
    """
    query_terms = set(query.lower().split())

    meta_text = " ".join([
        str(metadata.get("file_name", "")),
        str(metadata.get("slide_title", "")),
    ]).lower()

    meta_terms = set(meta_text.split())

    if not meta_terms or not query_terms:
        return 0.0

    overlap = query_terms.intersection(meta_terms)
    return len(overlap) / len(query_terms)


# Retrieve + Hybrid Re-ranking

def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve chunks using hybrid scoring (vector + metadata)."""

    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query])[0].tolist()

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)

    # Fetch more candidates for better reranking
    initial_k = top_k * 3

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        vector_score = 1 - dist  # similarity

        metadata_score = compute_metadata_score(query, meta)

        final_score = (0.7 * vector_score) + (0.3 * metadata_score)

        chunks.append({
            "text": doc,
            "vector_score": round(vector_score, 4),
            "metadata_score": round(metadata_score, 4),
            "score": round(final_score, 4),
            "file_name": meta["file_name"],
            "slide_number": meta["slide_number"],
            "slide_title": meta["slide_title"],
        })

    # Sort by hybrid score
    chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)

    return chunks[:top_k]


# Build Context

def build_context(chunks: list[dict]) -> str:
    """Format chunks into readable context."""

    lines = ["=== Retrieved Context from Lecture Slides ===\n"]

    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] File: {chunk['file_name']} | "
            f"Slide {chunk['slide_number']}: {chunk['slide_title'] or 'Untitled'} "
            f"(score: {chunk['score']}, vec={chunk['vector_score']}, meta={chunk['metadata_score']})"
        )
        lines.append(chunk["text"])
        lines.append("")

    return "\n".join(lines)


# LLM Call

def ask_llm(query: str, context: str) -> str:
    """Send query + context to LM Studio."""

    user_message = f"{context}\n\n=== Question ===\n{query}"

    payload = {
        "model": LLM_MODEL_NAME,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
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
            "Make sure LM Studio is running."
        )
    except requests.exceptions.Timeout:
        return "[Error] Request timed out."
    except Exception as e:
        return f"[Error] {e}"


# Full Pipeline

def rag_answer(query: str, verbose: bool = False) -> str:

    chunks = retrieve_chunks(query)

    if verbose:
        print(f"\n Retrieved {len(chunks)} chunks (hybrid scoring):")
        for c in chunks:
            print(
                f"   • [{c['score']}] "
                f"(vec={c['vector_score']}, meta={c['metadata_score']}) "
                f"Slide {c['slide_number']} — {c['slide_title']} ({c['file_name']})"
            )

    context = build_context(chunks)

    return ask_llm(query, context)


# Main

def main():
    print("   RAG Query Engine — Hybrid Retrieval   ")
    print("   Type 'exit' or 'quit' to stop         ")
    print("\n")

    # CLI mode
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