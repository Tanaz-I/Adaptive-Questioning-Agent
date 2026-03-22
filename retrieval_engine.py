"""
Retrieval Engine
================

Retrieves relevant chunks from ChromaDB using:
• Topic OR Subtopic filtering
• Difficulty filtering
• Semantic similarity ranking

Author: Member 2
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
# Load Embedding Model (ONLY ONCE)
# ─────────────────────────────────────────────

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)


# ─────────────────────────────────────────────
# Connect to ChromaDB
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
        print("Run these first:\n")
        print("1) python knowledge_base_construction.py")
        print("2) python enrich_metadata.py\n")
        raise SystemExit(1)


# ─────────────────────────────────────────────
# Build Filter (CORE FIX)
# ─────────────────────────────────────────────

def build_filter(topic, difficulty):
    """
    Matches:
    - topic OR subtopic
    - AND difficulty
    """

    return {
        "$and": [
            {
                "$or": [
                    {"topic": topic},
                    {"subtopic": topic}
                ]
            },
            {"difficulty": difficulty}
        ]
    }


# ─────────────────────────────────────────────
# Retrieve Chunks (FINAL VERSION)
# ─────────────────────────────────────────────

def retrieve_chunks(topic, difficulty, top_k=TOP_K):

    collection = connect_collection()

    filter_condition = build_filter(topic, difficulty)

    # 🔥 ALWAYS use semantic search (important improvement)
    query_embedding = embed_model.encode([topic])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filter_condition,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ─────────────────────────────
    # Build Output
    # ─────────────────────────────

    chunks = []

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

            "similarity_score": round(1 - dist, 4) if dist else None
        }

        chunks.append(chunk)

    return chunks


# ─────────────────────────────────────────────
# Test Mode
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\nTesting retrieval engine...\n")

    # TEST 1: Topic-based
    chunks = retrieve_chunks(
        topic="OOP",
        difficulty="easy"
    )

    print(f"Retrieved {len(chunks)} chunks (OOP easy)\n")

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print(f"Topic       : {c['topic']}")
        print(f"Subtopic    : {c['subtopic']}")
        print(f"Difficulty  : {c['difficulty']}")
        print(f"Similarity  : {c['similarity_score']}")
        print(f"Slide Title : {c['slide_title']}")
        print(f"Text        : {c['text'][:150]}...\n")

    print("\n" + "="*50 + "\n")

    # TEST 2: Subtopic-based (IMPORTANT)
    chunks = retrieve_chunks(
        topic="Inheritance",
        difficulty="medium"
    )

    print(f"Retrieved {len(chunks)} chunks (Inheritance medium)\n")

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print(f"Topic       : {c['topic']}")
        print(f"Subtopic    : {c['subtopic']}")
        print(f"Difficulty  : {c['difficulty']}")
        print(f"Similarity  : {c['similarity_score']}")
        print(f"Slide Title : {c['slide_title']}")
        print(f"Text        : {c['text'][:150]}...\n")