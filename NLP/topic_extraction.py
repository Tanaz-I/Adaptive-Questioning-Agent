"""
Global Two-Pass Topic Extraction
==================================
Phase 1: Extract raw topics per file → consolidate into canonical list
Phase 2: Tag each chunk against the canonical list

Domain-agnostic — works for any subject matter.
"""

import json
import requests
import chromadb
from pathlib import Path
from chromadb.config import Settings
from tqdm import tqdm


CHROMA_DB_DIR   = "./chroma_db"
COLLECTION_NAME = "rag_kb"
OLLAMA_URL      = "http://localhost:11434/v1/chat/completions"
LLM_MODEL_NAME  = "llama3"
CANONICAL_TOPICS_FILE = "./canonical_topics.json"


# ─────────────────────────────────────────────
# LLM call helper
# ─────────────────────────────────────────────

def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model":       LLM_MODEL_NAME,
            "temperature": 0.1,
            "max_tokens":  max_tokens,
            "messages":    [{"role": "user", "content": prompt}],
        },
        timeout=12000,
    )
    data = response.json()

    if "choices" in data:
        raw = data["choices"][0]["message"]["content"]
    elif "message" in data:
        raw = data["message"]["content"]
    elif "response" in data:
        raw = data["response"]
    else:
        raise ValueError(f"Unexpected response format: {data}")

    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


# ─────────────────────────────────────────────
# Phase 1a — Extract raw topics from each file
# ─────────────────────────────────────────────

def extract_raw_topics_from_file(file_text: str, file_name: str) -> list[str]:
    """Ask LLM to list topics covered in one file."""
    prompt = f"""You are an expert educator.

Read the following learning material and list ALL unique topics covered in it.
- Use concise, general topic names
- Avoid overly specific names
- Do NOT include subtopics as separate entries if they belong under a broader topic

Material from: {file_name}
---
{file_text[:10000]}
---

Return ONLY a JSON array of topic name strings. No markdown. No explanation.
No unnecessary statements. Give only the JSON.
Example: ["Topic A", "Topic B", "Topic C"]
"""
    raw    = call_llm(prompt, max_tokens=512)
    topics = json.loads(raw)
    return [t.strip() for t in topics if t.strip()]


# ─────────────────────────────────────────────
# Phase 1b — Consolidate all raw topics into
#             a clean canonical list
# ─────────────────────────────────────────────

def consolidate_topics(all_raw_topics: list[str]) -> list[str]:
    """
    Feed all raw topics (from all files) to LLM.
    LLM merges duplicates, resolves overlaps, and returns
    a clean canonical list.
    """
    prompt = f"""You are an expert educator.

Below is a messy list of topics extracted from multiple learning documents.
Many entries are duplicates or near-duplicates with slightly different names.

Your task:
1. Merge duplicates and near-duplicates into a single canonical name
2. Remove overly specific topics that are subtopics of a broader topic already in the list
3. Keep the list as concise as possible while covering all important concepts
4. Use clear, general topic names that would work as category labels

Raw topic list:
{json.dumps(sorted(set(all_raw_topics)), indent=2)}

Return ONLY a JSON array of canonical topic name strings.
No markdown. No explanation.
"""
    raw    = call_llm(prompt, max_tokens=1024)
    topics = json.loads(raw)
    return sorted([t.strip() for t in topics if t.strip()])


# ─────────────────────────────────────────────
# Phase 2 — Tag each chunk against canonical list
# ─────────────────────────────────────────────

def tag_chunk_against_canonical(
    chunk_text:       str,
    canonical_topics: list[str],
) -> str:
    """Pick the single best matching topic from the canonical list."""
    prompt = f"""You are an expert educator.

Given this fixed list of topics:
{json.dumps(canonical_topics)}

Which single topic from the list best describes the following content?
Return ONLY the topic name exactly as it appears in the list. No explanation.

Content:
{chunk_text[:8000]}
"""
    raw = call_llm(prompt, max_tokens=50)

    # Validate — if LLM returns something not in list, find closest match
    raw = raw.strip().strip('"')
    if raw in canonical_topics:
        return raw

    # Fallback: find closest by substring match
    for t in canonical_topics:
        if t.lower() in raw.lower() or raw.lower() in t.lower():
            return t

    return canonical_topics[0]   # last resort


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def run_global_topic_extraction():
    client     = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)
    all_data   = collection.get(include=["documents", "metadatas"])
    ids        = all_data["ids"]
    documents  = all_data["documents"]
    metadatas  = all_data["metadatas"]

    # ── Phase 1a: extract raw topics per file ────────────────────────────
    print("\n[Phase 1a] Extracting raw topics from each file ...")

    # Group chunks by file
    file_chunks = {}
    for doc, meta in zip(documents, metadatas):
        fname = meta.get("file_name", "unknown")
        file_chunks.setdefault(fname, []).append(doc)

    all_raw_topics = []
    for fname, chunks in tqdm(file_chunks.items(), desc="Files"):
        # Concatenate first N chunks as a file summary
        file_text  = "\n\n".join(chunks[:20])
        raw_topics = extract_raw_topics_from_file(file_text, fname)
        print(f"  {fname}: {raw_topics}")
        all_raw_topics.extend(raw_topics)

    print(f"\n  Total raw topics across all files: {len(all_raw_topics)}")
    print(f"  Unique raw topics: {len(set(all_raw_topics))}")

    # ── Phase 1b: consolidate into canonical list ────────────────────────
    print("\n[Phase 1b] Consolidating into canonical topic list ...")
    canonical_topics = consolidate_topics(all_raw_topics)

    print(f"\n  Canonical topics ({len(canonical_topics)}):")
    for t in canonical_topics:
        print(f"    • {t}")

    # Save canonical topics
    Path(CANONICAL_TOPICS_FILE).write_text(
        json.dumps(canonical_topics, indent=2), encoding="utf-8"
    )
    print(f"\n  Saved → {CANONICAL_TOPICS_FILE}")

    # ── Phase 2: re-tag all chunks ───────────────────────────────────────
    print("\n[Phase 2] Re-tagging all chunks against canonical topics ...")

    batch_ids  = []
    batch_meta = []
    BATCH      = 50

    for doc_id, doc, meta in tqdm(
        zip(ids, documents, metadatas), total=len(ids), desc="Chunks"
    ):
        canonical_topic   = tag_chunk_against_canonical(doc, canonical_topics)
        meta["topic"]     = canonical_topic
        batch_ids.append(doc_id)
        batch_meta.append(meta)

        if len(batch_ids) >= BATCH:
            collection.update(ids=batch_ids, metadatas=batch_meta)
            batch_ids.clear()
            batch_meta.clear()

    if batch_ids:
        collection.update(ids=batch_ids, metadatas=batch_meta)

    print("\n✓ All chunks re-tagged with canonical topics.")
    print(f"✓ Canonical topics saved to {CANONICAL_TOPICS_FILE}\n")

    return canonical_topics


if __name__ == "__main__":
    run_global_topic_extraction()
