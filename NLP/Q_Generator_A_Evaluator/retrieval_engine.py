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
COLLECTION_NAME = "rag_kb"
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

def retrieve_chunks(topic, difficulty, question_type, top_k=TOP_K):

    collection = connect_collection()

    # ─────────────────────────────────────────────
    # 1. Build richer query (QUERY EXPANSION)
    # ─────────────────────────────────────────────
    query_variants = [
        topic,
        f"{topic} explanation",
        f"{topic} example",
        f"{topic} definition",
        f"{topic} concepts",
        f"{topic} applications",
        f"{topic} {difficulty}",
        f"{topic} {question_type}"
    ]

    # ─────────────────────────────────────────────
    # 2. Metadata filter (same as before)
    # ─────────────────────────────────────────────
    filter_condition = build_filter(topic, difficulty)

    all_results = []

    # ─────────────────────────────────────────────
    # 3. Multi-query retrieval
    # ─────────────────────────────────────────────
    for q in query_variants:

        q_embed = embed_model.encode([q])[0].tolist()

        results = collection.query(
            query_embeddings=[q_embed],
            n_results=top_k,
            where=filter_condition,
            include=["documents", "metadatas", "distances"]
        )

        # 🔥 FALLBACK if no results
        if len(results["documents"][0]) == 0:
            print("[DEBUG] No results with filter → fallback")

            results = collection.query(
                query_embeddings=[q_embed],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            all_results.append({
                "text": doc,
                "meta": meta,
                "score": 1 - dist if dist else 0
            })

    # ─────────────────────────────────────────────
    # 4. Deduplication
    # ─────────────────────────────────────────────
    seen = set()
    unique_chunks = []

    for c in all_results:
        key = c["text"][:100]

        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    # ─────────────────────────────────────────────
    # 5. Diversity selection (MMR-style)
    # ─────────────────────────────────────────────
    final_chunks = []
    seen_subtopics = set()
    # ─────────────────────────────────────────────
    # 5.1 Add metadata-based bonus (IMPORTANT)
    # ─────────────────────────────────────────────
    for c in unique_chunks:
        bonus = 0

        concept = c["meta"].get("concept_type", "")

        if concept == "explanation":
            bonus += 0.05
        elif concept == "example":
            bonus += 0.03

        c["score"] += bonus

    # ─────────────────────────────────────────────
    # 5.2 Sort after boosting scores
    # ─────────────────────────────────────────────
    sorted_chunks = sorted(unique_chunks, key=lambda x: x["score"], reverse=True)

    for c in sorted_chunks:

        subtopic = c["meta"].get("subtopic", "") or "unknown"

        if subtopic not in seen_subtopics:
            final_chunks.append(c)
            seen_subtopics.add(subtopic)

        if len(final_chunks) >= top_k:
            break

    # ─────────────────────────────────────────────
    # 6. Format output (same structure)
    # ─────────────────────────────────────────────
    output = []

    for c in final_chunks:
        meta = c["meta"]

        output.append({
            "text": c["text"],
            "topic": meta.get("topic"),
            "subtopic": meta.get("subtopic"),
            "difficulty": meta.get("difficulty"),
            "concept_type": meta.get("concept_type"),
            "keywords": meta.get("keywords"),
            "file_name": meta.get("file_name"),
            "slide_number": meta.get("slide_number"),
            "slide_title": meta.get("slide_title"),
            "similarity_score": round(c["score"], 4),
            "retrieval_source": "multi_query_mmr"
        })

    return output


# ─────────────────────────────────────────────
# Test Mode
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\nTesting retrieval engine...\n")

    # TEST 1: Topic-based
    # chunks = retrieve_chunks(
    #     topic="OOP",
    #     difficulty="easy",
    #     question_type='factual'
    # )

    # print(f"Retrieved {len(chunks)} chunks (OOP easy)\n")

    # for i, c in enumerate(chunks, 1):
    #     print(f"Chunk {i}")
    #     print(f"Topic       : {c['topic']}")
    #     print(f"Subtopic    : {c['subtopic']}")
    #     print(f"Difficulty  : {c['difficulty']}")
    #     print(f"Similarity  : {c['similarity_score']}")
    #     print(f"Slide Title : {c['slide_title']}")
    #     print(f"Text        : {c['text'][:150]}...\n")

    # print("\n" + "="*50 + "\n")

    # TEST 2: Subtopic-based (IMPORTANT)
    chunks = retrieve_chunks(
    topic="Constructors and destructors",
    difficulty="medium",
    question_type="inferential")

    print("\nSubtopics retrieved:")
    print([c["subtopic"] for c in chunks])
    for c in chunks:
        print("\n---")
        print("Concept Type:", c["concept_type"])
        print("Text:", c["text"])

    print(f"Retrieved {len(chunks)} chunks \n")

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print(f"Topic       : {c['topic']}")
        print(f"Subtopic    : {c['subtopic']}")
        print(f"Difficulty  : {c['difficulty']}")
        print(f"Similarity  : {c['similarity_score']}")
        print(f"Slide Title : {c['slide_title']}")
        print(f"Text        : {c['text'][:150]}...\n")