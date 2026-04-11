"""
Retrieval Engine
================

Retrieves relevant chunks from ChromaDB using:
• Topic OR Subtopic filtering
• Difficulty filtering
• Semantic similarity ranking

Requirements : pip install rank_bm25

Author: Member 2
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


_CLIENT = None
_COLLECTION = None

_BM25_INDEX = None
_CORPUS_DOCS = None
_CORPUS_METAS = None

_RERANKER = None

def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        try:
            _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[Retrieval] Cross-encoder reranker loaded.")
        except Exception as e:
            print(f"[Retrieval] Reranker unavailable: {e}")
            _RERANKER = False
    return _RERANKER if _RERANKER else None


def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-scores chunks using a cross-encoder. Falls back silently if unavailable.
    """
    reranker = _get_reranker()
    if not reranker or len(chunks) <= top_k:
        return chunks[:top_k]

    pairs  = [(query, c["text"][:512]) for c in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]

DIFF_ORDER = {
    'easy': 0, 'basic': 0,
    'medium': 1, 'intermediate': 1,
    'hard': 2, 'advanced': 2
}

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
    global _CLIENT, _COLLECTION

    if _COLLECTION is None:
        _CLIENT = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            _COLLECTION = _CLIENT.get_collection(COLLECTION_NAME)
        except Exception:
            print("\nERROR: ChromaDB collection not found.")
            print("Run these first:\n")
            print("1) python knowledge_base_construction.py")
            print("2) python enrich_metadata.py\n")
            raise SystemExit(1)

    return _COLLECTION

def get_bm25_index():
    global _BM25_INDEX, _CORPUS_DOCS, _CORPUS_METAS

    if _BM25_INDEX is None:
        collection = connect_collection()
        data = collection.get(include=["documents", "metadatas"])

        _CORPUS_DOCS = data["documents"]
        _CORPUS_METAS = data["metadatas"]

        tokenized = [doc.lower().split() for doc in _CORPUS_DOCS]
        _BM25_INDEX = BM25Okapi(tokenized)

    return _BM25_INDEX, _CORPUS_DOCS, _CORPUS_METAS


# ─────────────────────────────────────────────
# Build Filter (CORE FIX)
# ─────────────────────────────────────────────

def build_filter(topics: list):
    conditions = []
    for t in topics:
        conditions.append({"topic": t})
        conditions.append({"subtopic": t})
    return {"$or": conditions}


# ─────────────────────────────────────────────
# Retrieve Chunks (FINAL VERSION)
# ─────────────────────────────────────────────

def rrf_score(rank, k=60):
    return 1 / (k + rank)

# def expand_topic_smart_with_prereq_concept_graph(topic, prerequisites, concept_graph, collection):
#     """
#     Smart topic expansion:
#     - Expands ONLY if insufficient chunks exist
#     - Adds at most:
#         • 1 prerequisite
#         • 2 strongly connected graph neighbors
#     """

#     # ---- Step 1: count exact topic chunks ----
#     try:
#         results = collection.get(
#             where={"topic": topic},
#             include=["documents"]
#         )
#         n_exact = len(results.get("documents", []))
#     except Exception:
#         n_exact = 10  # assume sufficient → skip expansion

#     expanded = [topic]

#     # ---- Step 2: expand ONLY if needed ----
#     if n_exact < 5:

#         # ---- add ONE prerequisite (most relevant = last) ----
#         if prerequisites:
#             prereqs = prerequisites.get(topic, [])
#             if prereqs:
#                 expanded.append(prereqs[-1])

#         # ---- add top-2 graph neighbors (most connected) ----
#         if concept_graph:
#             topic_key = topic.lower()

#             if topic_key in concept_graph:
#                 neighbors = list(concept_graph[topic_key])

#                 # sort by connectivity (degree)
#                 neighbors = sorted(
#                     neighbors,
#                     key=lambda n: len(concept_graph.get(n, [])),
#                     reverse=True
#                 )

#                 expanded.extend(neighbors[:2])

#     # ---- Step 3: deduplicate while preserving order ----
#     seen = set()
#     final = []

#     for t in expanded:
#         t_clean = t.strip()
#         if t_clean and t_clean not in seen:
#             seen.add(t_clean)
#             final.append(t_clean)

#     return final

def expand_topic_smart_with_prereq_concept_graph(topic, prerequisites, concept_graph, collection):
    """
    Smart topic expansion:
    - Expands ONLY if insufficient chunks exist for the topic
    - Returns:
        expanded_topics  : list of topic-level strings for ChromaDB filter + BM25
        concept_keywords : list of concept-level strings for query text enrichment only
    """

    # ── Step 1: count exact topic chunks ────────────────────────
    try:
        results = collection.get(
            where={"topic": topic},
            include=["documents"]
        )
        n_exact = len(results.get("documents", []))
    except Exception:
        n_exact = 10  # assume sufficient → skip expansion

    expanded = [topic]

    # ── Step 2: expand topic-level list (for filter) ─────────────
    # Only fires when the topic is genuinely sparse
    if n_exact < 5:

        # Add ONE prerequisite topic (last = most recent/relevant)
        if prerequisites:
            prereqs = prerequisites.get(topic, [])
            if prereqs:
                expanded.append(prereqs[-1])

    # ── Step 3: collect concept keywords (for query text only) ───
    # concept_graph keys are concept-level strings like "pointer arithmetic",
    # NOT topic-level strings like "Pointers to Class Members".
    # So we NEVER add them to expanded (filter), only use them to
    # enrich query text so the embedding search finds more relevant chunks.
    concept_keywords = []
    if concept_graph:
        # Try progressively looser key matches:
        # 1. Full topic lowercased       e.g. "pointers to class members"
        # 2. Each individual word        e.g. "pointers", "class", "members"
        # 3. Each bigram in the topic    e.g. "pointers to", "to class", "class members"
        topic_lower = topic.lower()
        topic_words = topic_lower.split()

        bigrams = [
            f"{topic_words[i]} {topic_words[i+1]}"
            for i in range(len(topic_words) - 1)
        ] if len(topic_words) >= 2 else []

        candidate_keys = [topic_lower] + bigrams + topic_words

        for key in candidate_keys:
            if key in concept_graph:
                neighbors = list(concept_graph[key])
                # Sort by connectivity degree — most central concepts first
                neighbors = sorted(
                    neighbors,
                    key=lambda n: len(concept_graph.get(n, [])),
                    reverse=True
                )
                concept_keywords.extend(neighbors[:2])
                break  # stop at first key that hits — avoid over-expansion

    # ── Step 4: deduplicate expanded_topics, preserving order ────
    seen = set()
    final = []
    for t in expanded:
        t_clean = t.strip()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            final.append(t_clean)

    return final, concept_keywords

def difficulty_match_bonus(chunk_diff, requested_diff):
    req = DIFF_ORDER.get(requested_diff, 1)
    ch  = DIFF_ORDER.get(chunk_diff, 1)

    distance = abs(req - ch)

    # closer → higher bonus
    if distance == 0:
        return 0.08
    elif distance == 1:
        return 0.03
    else:
        return 0.0

def retrieve_chunks(topic, difficulty, question_type, used_chunk_ids = None, prerequisites=None, concept_graph=None, top_k=TOP_K):

    collection = connect_collection()
    bm25, corpus_docs, corpus_metas = get_bm25_index()

    # ─────────────────────────────────────────────
    # 1. Build richer query (QUERY EXPANSION)
    # ─────────────────────────────────────────────
    
    expanded_topics, concept_keywords = expand_topic_smart_with_prereq_concept_graph(topic,
        prerequisites,
        concept_graph,
        collection)

    query_variants = [topic, f"{topic} explanation", f"{topic} example",
                    f"{topic} definition", f"{topic} concepts", f"{topic} applications"]

    # add concept keyword enriched queries (these stay as query text only, not filter)
    for kw in concept_keywords:
        query_variants.append(f"{topic} {kw}")

    # add prerequisite/expanded topic queries
    for t in expanded_topics[1:]:
        query_variants.extend([t, f"{t} explanation", f"{t} example", f"{t} definition"])

    filter_condition = build_filter(expanded_topics)

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

        # FALLBACK if no results
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
    # 4. Deduplication (MANDATORY BEFORE RRF)
    # ─────────────────────────────────────────────

    seen = set()
    unique_chunks = []

    for c in all_results:
        key = c["text"][:100]

        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    # ─────────────────────────────────────────────
    # 4.5 BM25 Retrieval + RRF Fusion
    # ─────────────────────────────────────────────

    # ---- BM25 retrieval ----
    query_tokens = " ".join(expanded_topics + ["concepts", "usage", "working"]).lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k * 3]

    bm25_results = []

    for rank, idx in enumerate(bm25_ranked_indices):
        bm25_results.append({
            "text": corpus_docs[idx],
            "meta": corpus_metas[idx],
            "score": 0,            # will be updated
            "bm25_rank": rank
        })


    # ---- RRF Fusion ----
    rrf_scores = {}

    # embedding ranking
    for rank, c in enumerate(unique_chunks):
        key = c["text"]
        rrf_scores[key] = rrf_score(rank)

    # BM25 ranking
    for rank, c in enumerate(bm25_results):
        key = c["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score(rank)


    # ---- Merge embedding + BM25 chunks ----
    combined_chunks = {}

    # add embedding chunks
    for c in unique_chunks:
        combined_chunks[c["text"]] = c

    # add BM25-only chunks
    for c in bm25_results:
        if c["text"] not in combined_chunks:
            combined_chunks[c["text"]] = {
                "text": c["text"],
                "meta": c["meta"],
                "score": 0
            }

    # ---- Assign fused score ----
    fused_chunks = []

    for text, c in combined_chunks.items():
        c["score"] += rrf_scores.get(text, 0)
        fused_chunks.append(c)


    
    unique_chunks = fused_chunks

    # ─────────────────────────────────────────────
    # 5. Diversity selection (MMR-style)
    # ─────────────────────────────────────────────
    final_chunks = []
    seen_subtopics = set()
    seen_text_keys = set()
    
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

        bonus += difficulty_match_bonus(c["meta"].get("difficulty"), difficulty)

        c["score"] += bonus

    # ─────────────────────────────────────────────
    # 5.2 Sort after boosting scores
    # ─────────────────────────────────────────────
    sorted_chunks = sorted(unique_chunks, key=lambda x: x["score"], reverse=True)
    if used_chunk_ids is None:
        used_chunk_ids = []

    def chunk_key(text):
        return text[:80].strip()

    # ---- filter recent chunks ----
    recent_set = set(used_chunk_ids)

    fresh_chunks = [
        c for c in sorted_chunks
        if chunk_key(c["text"]) not in recent_set
    ]

    # fallback if too restrictive
    if len(fresh_chunks) < 3:
        fresh_chunks = sorted_chunks

    for c in fresh_chunks:

        subtopic = c["meta"].get("subtopic", "") or "unknown"
        if subtopic == "unknown":
            print(c)

        if subtopic and subtopic != "Unknown":
            if subtopic in seen_subtopics:
                continue
            seen_subtopics.add(subtopic)
        else:
            # Fallback: dedup by first 60 chars of text
            text_key = c["text"][:60].strip()
            if text_key in seen_text_keys:
                continue
            seen_text_keys.add(text_key)

        final_chunks.append(c)
        if len(final_chunks) >= top_k:
            break

    rerank_query = f"{topic} {difficulty} {question_type}"
    final_chunks = rerank_chunks(rerank_query, final_chunks, top_k=top_k)
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
            "page_number": meta.get("page_number"),
            "section": meta.get("section"),
            "similarity_score": round(c["score"], 4),
            "retrieval_source": "multi_query_mmr",
            "contains_code": meta.get('contains_code'),
            "contains_example": meta.get('contains_example'),
            "parent_id" : meta.get('parent_id')
        })

    used_keys = [chunk_key(c["text"]) for c in final_chunks]

    return output, used_keys

def get_neighbor_chunks(chunk, window=1):
    _, docs, metas = get_bm25_index()

    # Support both dict formats (with and without "meta" nesting)
    if isinstance(chunk, dict) and "meta" in chunk:
        parent_id = chunk["meta"].get("parent_id", "")
        file_name = chunk["meta"].get("file_name", "")
        page      = chunk["meta"].get("page_number")
    else:
        parent_id = chunk.get("parent_id", "")
        file_name = chunk.get("file_name", "")
        page      = chunk.get("page_number")

    neighbors = []
    for doc, meta in zip(docs, metas):
        # Best: exact sibling from same slide via parent_id
        if parent_id and meta.get("parent_id") == parent_id:
            neighbors.append({"text": doc, "meta": meta})
            continue
        # Fallback: page proximity in same file (when parent_id not yet stored)
        if not parent_id and meta.get("file_name") == file_name:
            p2 = meta.get("page_number")
            if p2 is not None and page is not None and abs(p2 - page) <= window:
                neighbors.append({"text": doc, "meta": meta})

    return neighbors


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
    chunks, _ = retrieve_chunks(
    topic="Pointers to Class Members",
    difficulty="medium",
    question_type="factual")

    print("\nSubtopics retrieved:")
    # print(chunks)
    print([c["subtopic"] for c in chunks])
    for c in chunks:
        print("\n---")
        print("Concept Type:", c["concept_type"])
        print("Text:", c["text"])

    print(f"Retrieved {len(chunks)} chunks \n")

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}")
        print(f"Text        : {c['text']}")
        print(f"Topic       : {c['topic']}")
        print(f"Subtopic    : {c['subtopic']}")
        print(f"Difficulty  : {c['difficulty']}")
        print(f"Similarity  : {c['similarity_score']}")
        print(f"Slide Title : {c['section']}")
        print(f"Text        : {c['text'][:150]}...\n")

    # print("\n========== TEST: PARENT_ID GROUPING ==========\n")

    # target_chunk = chunks[0]

    # print("TARGET CHUNK:")
    # print("Text:", target_chunk["text"][:200])
    # print("Page:", target_chunk.get("page_number"))
    # print("Section:", target_chunk.get("section"))

    # # If meta exists (safe handling)
    # if "meta" in target_chunk:
    #     print("Parent ID:", target_chunk["meta"].get("parent_id"))
    # else:
    #     print("Parent ID:", target_chunk.get("parent_id"))

    # print("\n---- NEIGHBORS ----\n")

    # neighbors = get_neighbor_chunks(chunk=target_chunk)

    # parent_ids = set()
    # pages = set()

    # for i, n in enumerate(neighbors, 1):

    #     meta = n["meta"]

    #     print(f"\nNeighbor {i}")
    #     print("Text:", n["text"][:150])
    #     print("Page:", meta.get("page_number"))
    #     print("Section:", meta.get("section"))
    #     print("Parent ID:", meta.get("parent_id"))

    #     parent_ids.add(meta.get("parent_id"))
    #     pages.add(meta.get("page_number"))

    # print("\n========== SUMMARY ==========")
    # print("Unique parent_ids:", parent_ids)
    # print("Unique pages:", pages)
    # print("Total neighbors:", len(neighbors))