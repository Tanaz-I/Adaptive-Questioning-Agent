"""
Chunk Metadata Enrichment
=========================
Reads all existing chunks from ChromaDB, sends each to qwen2.5:1.5b-instruct
via Ollama to auto-generate rich metadata tags, then updates ChromaDB.

Tags generated per chunk:
    - topic         : broad subject  (e.g. "OOP")
    - subtopic      : specific idea  (e.g. "Single Inheritance")
    - concept_type  : role of chunk  (definition / example / explanation / summary / other)
    - difficulty    : estimated level (easy / medium / hard)
    - keywords      : comma-separated key terms

Prerequisites:
    1. pptx_rag_pipeline.py must have been run (ChromaDB populated)
    2. Ollama must be running:  ollama run qwen2.5:1.5b-instruct
    3. pip install chromadb requests tqdm

Usage:
    python enrich_metadata.py
"""

import json
import time
import requests
import chromadb
from chromadb.config import Settings
from tqdm import tqdm


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CHROMA_DB_DIR    = "./chroma_db"
COLLECTION_NAME  = "pptx_rag"

OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:1.5b-instruct"

BATCH_SIZE       = 10      # save progress to ChromaDB every N chunks
RETRY_LIMIT      = 3       # retries per chunk on failure
RETRY_DELAY      = 2       # seconds between retries


# ─────────────────────────────────────────────
# Tagging prompt
# ─────────────────────────────────────────────

def build_prompt(chunk_text: str) -> str:
    return f"""You are an educational content tagger for C++ and Object-Oriented Programming lecture slides.

Analyze the following lecture slide chunk and return ONLY a JSON object with these fields:
- "topic"        : The broad subject area (e.g. "OOP", "C++ Basics", "Constructors", "Operator Overloading", "Inheritance", "Polymorphism", "Abstraction", "Encapsulation", "Templates", "STL")
- "subtopic"     : A specific concept within the topic (e.g. "Single Inheritance", "Copy Constructor", "Virtual Functions")
- "concept_type" : One of: "definition", "example", "explanation", "summary", "other"
- "difficulty"   : One of: "easy", "medium", "hard"
- "keywords"     : A comma-separated string of 3-5 key terms from the chunk

Return ONLY valid JSON. No explanation, no markdown, no extra text.

Chunk:
\"\"\"
{chunk_text[:1000]}
\"\"\"

JSON:"""


# ─────────────────────────────────────────────
# Call Ollama
# ─────────────────────────────────────────────

DEFAULT_TAGS = {
    "topic":        "Unknown",
    "subtopic":     "Unknown",
    "concept_type": "other",
    "difficulty":   "medium",
    "keywords":     "",
}

def tag_chunk(chunk_text: str) -> dict:
    """Send chunk to Ollama and return parsed metadata tags."""
    prompt = build_prompt(chunk_text)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,   # low = consistent structured output
                        "num_predict": 200,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

            # Extract JSON even if model adds surrounding text
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")

            tags = json.loads(raw[start:end])

            # Validate and fill missing fields
            for key, default in DEFAULT_TAGS.items():
                if key not in tags or not isinstance(tags[key], str):
                    tags[key] = default

            # Normalize concept_type and difficulty to allowed values
            if tags["concept_type"] not in ("definition", "example", "explanation", "summary", "other"):
                tags["concept_type"] = "other"
            if tags["difficulty"] not in ("easy", "medium", "hard"):
                tags["difficulty"] = "medium"

            return tags

        except requests.exceptions.ConnectionError:
            print("\n[Error] Cannot connect to Ollama. Is it running?")
            print("        Run: ollama run qwen2.5:1.5b-instruct")
            raise SystemExit(1)

        except Exception as e:
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  [WARN] Tagging failed after {RETRY_LIMIT} attempts: {e}")
                return DEFAULT_TAGS.copy()

    return DEFAULT_TAGS.copy()


# ─────────────────────────────────────────────
# Main enrichment pipeline
# ─────────────────────────────────────────────

def enrich_metadata():
    # ── Connect to ChromaDB ──────────────────────────────────────────────
    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)

    total = collection.count()
    print(f"\n{'='*55}")
    print(f"  Chunk Metadata Enrichment")
    print(f"{'='*55}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Total chunks: {total}")
    print(f"  Model      : {OLLAMA_MODEL}")
    print(f"{'='*55}\n")

    # ── Fetch all chunks ─────────────────────────────────────────────────
    all_data = collection.get(include=["documents", "metadatas"])
    ids       = all_data["ids"]
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    # ── Track progress — skip already-enriched chunks ────────────────────
    to_process = [
        (i, ids[i], documents[i], metadatas[i])
        for i in range(len(ids))
        if "topic" not in metadatas[i]      # skip if already tagged
    ]

    skipped = total - len(to_process)
    if skipped:
        print(f"  ⏭  Skipping {skipped} already-enriched chunks.\n")

    if not to_process:
        print("✓ All chunks already enriched. Nothing to do.")
        return

    print(f"  Tagging {len(to_process)} chunks...\n")

    # ── Tag each chunk and batch-update ChromaDB ──────────────────────────
    batch_ids  = []
    batch_meta = []
    enriched   = 0
    failed     = 0

    for i, chunk_id, text, existing_meta in tqdm(to_process, desc="Enriching"):
        tags = tag_chunk(text)

        # Merge new tags into existing metadata
        updated_meta = {**existing_meta, **tags}

        batch_ids.append(chunk_id)
        batch_meta.append(updated_meta)

        if tags == DEFAULT_TAGS:
            failed += 1
        else:
            enriched += 1

        # Save batch to ChromaDB
        if len(batch_ids) >= BATCH_SIZE:
            collection.update(ids=batch_ids, metadatas=batch_meta)
            batch_ids.clear()
            batch_meta.clear()

    # Save remaining
    if batch_ids:
        collection.update(ids=batch_ids, metadatas=batch_meta)

    print(f"\n{'='*55}")
    print(f"  ✓ Enrichment complete!")
    print(f"  Successfully tagged : {enriched}")
    print(f"  Used defaults       : {failed}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# Preview enriched chunks
# ─────────────────────────────────────────────

def preview(n: int = 5):
    """Print a sample of enriched chunks to verify quality."""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"], limit=n)

    print(f"\n── Sample of {n} enriched chunks ──\n")
    for doc, meta in zip(data["documents"], data["metadatas"]):
        print(f"  File      : {meta.get('file_name')}  Slide {meta.get('slide_number')}")
        print(f"  Topic     : {meta.get('topic')}  →  {meta.get('subtopic')}")
        print(f"  Type      : {meta.get('concept_type')}  |  Difficulty: {meta.get('difficulty')}")
        print(f"  Keywords  : {meta.get('keywords')}")
        print(f"  Text      : {doc[:120]}...")
        print()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    enrich_metadata()
    preview(n=5)