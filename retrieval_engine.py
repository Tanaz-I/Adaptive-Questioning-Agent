"""
Retrieval Engine (FINAL CORRECTED VERSION)
=========================================

✔ Fully dynamic (no hardcoding)
✔ Semantic-first retrieval
✔ Soft metadata filtering (optional)
✔ Robust fallback mechanism
✔ RL-ready
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "pptx_rag"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


# ─────────────────────────────────────────────
# Load Model (once)
# ─────────────────────────────────────────────

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)


# ─────────────────────────────────────────────
# Connect DB
# ─────────────────────────────────────────────

def connect_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        return client.get_collection(COLLECTION_NAME)

    except Exception:
        print("\nERROR: ChromaDB collection not found.")
        print("Run:\n")
        print("1) python knowledge_base_construction.py")
        print("2) python enrich_metadata.py\n")
        raise SystemExit(1)


# ─────────────────────────────────────────────
# Retrieve Chunks (FIXED CORE)
# ─────────────────────────────────────────────

def retrieve_chunks(query, difficulty=None, top_k=TOP_K):

    collection = connect_collection()

    # Step 1: Encode query (FULL query, not just topic)
    query_embedding = embed_model.encode([query])[0].tolist()

    # Step 2: Initial semantic retrieval (NO FILTER)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 3,   # fetch more → better filtering
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []

    # Step 3: Build chunks
    for doc, meta, dist in zip(documents, metadatas, distances):

        chunk = {
            "text": doc,
            "topic": meta.get("topic"),
            "subtopic": meta.get("subtopic"),
            "difficulty": meta.get("difficulty"),
            "concept_type": meta.get("concept_type"),
            "keywords": meta.get("keywords"),
            "file_name": meta.get("file_name"),
            "slide_number": meta.get("slide_number"),
            "slide_title": meta.get("slide_title"),
            "similarity_score": round(1 - dist, 4) if dist else 0.0
        }

        chunks.append(chunk)

    # ─────────────────────────────
    # Step 4: Optional Difficulty Filter (SOFT)
    # ─────────────────────────────

    if difficulty:
        filtered = [c for c in chunks if c["difficulty"] == difficulty]

        if filtered:
            chunks = filtered

    # ─────────────────────────────
    # Step 5: Sort by similarity
    # ─────────────────────────────

    chunks = sorted(chunks, key=lambda x: x["similarity_score"], reverse=True)

    return chunks[:top_k]


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\nTesting retrieval engine...\n")

    query = "explain inheritance in object oriented programming"

    chunks = retrieve_chunks(query, difficulty="medium")

    print(f"Retrieved {len(chunks)} chunks\n")

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print(f"Topic       : {c['topic']}")
        print(f"Subtopic    : {c['subtopic']}")
        print(f"Difficulty  : {c['difficulty']}")
        print(f"Similarity  : {c['similarity_score']}")
        print(f"Slide Title : {c['slide_title']}")
        print(f"Text        : {c['text'][:150]}...\n")