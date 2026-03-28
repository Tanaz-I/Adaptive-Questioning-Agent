"""
Chunk Metadata Enrichment

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


# Configuration

CHROMA_DB_DIR    = "./chroma_db"
COLLECTION_NAME  = "rag_kb"

OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:1.5b-instruct"

BATCH_SIZE       = 10      # save progress to ChromaDB every N chunks
RETRY_LIMIT      = 3       # retries per chunk on failure
RETRY_DELAY      = 2       # seconds between retries


# Tagging prompt

def build_prompt(chunk_text: str, course: str, level: str, slide_title: str = "") -> str:
    title_hint = f"\nThe slide title is: \"{slide_title}\"." if slide_title else ""
    return f"""You are an expert educator analyzing lecture slides for a course on "{course}" ({level} level).{title_hint}

Analyze the following lecture slide chunk and return ONLY a JSON object with these fields:
- "topic"        : The lecture or chapter this chunk belongs to.
                   - If a slide title is provided, use it directly as the topic (cleaned to Title Case)
                   - If no title, infer the topic from the content at lecture/chapter granularity
                   - Must NOT be the course name itself ("{course}")
                   - Must NOT be a single concept or function — should represent a full lecture unit
                   - All chunks from the same slide must share the same topic string
- "subtopic"     : The specific concept discussed in this chunk within the topic
                   - Should be a single focused idea (e.g. a specific method, rule, theorem, or technique)
                   - More specific than topic, but not a line of code or trivial detail
- "concept_type" : One of: "definition", "example", "explanation", "summary", "other"
- "difficulty"   : One of: "easy", "medium", "hard"
- "keywords"     : A comma-separated string of 3-5 key terms from the chunk

Return ONLY valid JSON. No explanation, no markdown, no extra text.
No unnecessary statements. Give only the JSON.

Chunk:
\"\"\"
{chunk_text[:1000]}
\"\"\"

JSON:"""
# Call Ollama

DEFAULT_TAGS = {
    "topic":        "Unknown",
    "subtopic":     "Unknown",
    "concept_type": "other",
    "difficulty":   "medium",
    "keywords":     "",
}

def tag_chunk(chunk_text: str, course: str = "", level: str = "", slide_title: str = "") -> dict:
    """Send chunk to Ollama and return parsed metadata tags."""
    prompt = build_prompt(chunk_text, course, level, slide_title)

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

def detect_course(collection) -> str:
    """Sample chunks from the collection and infer the course/domain."""
    sample = collection.get(include=["documents"], limit=20)
    sample_text = "\n---\n".join(sample["documents"][:20])

    prompt = f"""You are an expert educator. Read the following lecture slide excerpts and identify:
1. The course or subject being taught (e.g. "Data Structures and Algorithms", "C++ Programming", "Machine Learning", "Discrete Mathematics")
2. The academic level (e.g. "undergraduate", "postgraduate", "professional")

Return ONLY a JSON object with two fields:
- "course": the name of the course (concise, 2-6 words)
- "level": the academic level

Return ONLY valid JSON. No explanation, no markdown.
No unnecessary statements. Give only the JSON.

Slide excerpts:
\"\"\"
{sample_text[:3000]}
\"\"\"

JSON:"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 100},
        },
        timeout=60,
    )
    raw = response.json().get("response", "").strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    print(raw)
    result = json.loads(raw[start:end])
    course = result.get("course", "the subject")
    level  = result.get("level", "undergraduate")
    print(f"\n  Detected course : {course}")
    print(f"  Detected level  : {level}")
    return course, level

def normalize_topics(collection, course: str, level: str) -> dict:
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    raw_topics = sorted(set(
        m["topic"].lower()
        for m in all_meta
        if m.get("topic") not in (None, "", "Unknown")
    ))

    # Deterministically remove course name
    course_lower = course.lower()
    filtered_topics = [t for t in raw_topics if t != course_lower]
    removed = set(raw_topics) - set(filtered_topics)
    if removed:
        print(f"  Removed course-name topics: {removed}")

    # --- Step 1: Ask model to GROUP similar topics (easier task) ---
    prompt_cluster = f"""You are an expert in "{course}" ({level} level).

Here is a list of lecture topics:
{json.dumps(filtered_topics, indent=2)}

Group the topics that refer to the same concept into clusters.
Each cluster should have one representative name in Title Case (2-4 words).

Return ONLY a JSON array of objects, each with:
- "canonical": the representative name for the group (Title Case)
- "members": list of raw topic strings from the input that belong to this group

Every raw topic must appear in exactly one group.
Return ONLY valid JSON. No explanation, no markdown.
No unnecessary statements. Give only the JSON.

JSON:"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt_cluster,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 600},
        },
        timeout=60,
    )
    raw = response.json().get("response", "").strip()
    start = raw.find("[")
    end   = raw.rfind("]") + 1

    mapping = {}  # raw_topic -> canonical

    if start != -1 and end > 0:
        clusters = json.loads(raw[start:end])
        for cluster in clusters:
            canonical = cluster.get("canonical", "").strip()
            members   = cluster.get("members", [])
            if not canonical:
                continue
            for member in members:
                mapping[member.lower()] = canonical
    else:
        # Fallback: just title-case each topic as its own canonical
        print("  [WARN] Clustering failed, falling back to title-case normalization")
        mapping = {t: t.title() for t in filtered_topics}

    # Ensure every filtered topic has a mapping (catch any the model missed)
    for t in filtered_topics:
        if t not in mapping:
            mapping[t] = t.title()

    # Map removed course-name topics to None (drop them)
    for t in removed:
        mapping[t] = None

    print(f"\n  Topic normalization map:\n{json.dumps(mapping, indent=2)}")
    return mapping


def apply_topic_normalization(collection, mapping: dict):
    all_data  = collection.get(include=["metadatas"])
    ids       = all_data["ids"]
    metadatas = all_data["metadatas"]

    batch_ids, batch_meta = [], []
    for chunk_id, meta in zip(ids, metadatas):
        raw_topic = meta.get("topic", "").lower()
        canonical = mapping.get(raw_topic)
        if canonical is None:
            # Drop: remap to "Unknown" so it's excluded downstream
            meta["topic"] = "Unknown"
            batch_ids.append(chunk_id)
            batch_meta.append(meta)
        elif canonical != meta.get("topic"):
            meta["topic"] = canonical
            batch_ids.append(chunk_id)
            batch_meta.append(meta)

    if batch_ids:
        collection.update(ids=batch_ids, metadatas=batch_meta)
        print(f"  Updated {len(batch_ids)} chunks with canonical topics.")

def enforce_per_slide_topic_consistency(collection):
    """
    For each (file_name, page_number) group, 
    assign the majority-voted topic to all chunks in that slide.
    """
    from collections import Counter, defaultdict

    all_data  = collection.get(include=["metadatas"])
    ids       = all_data["ids"]
    metadatas = all_data["metadatas"]

    # Group chunk indices by slide
    slide_groups = defaultdict(list)   # (file, page) -> list of (idx, meta)
    for idx, meta in enumerate(metadatas):
        key = (meta.get("file_name", ""), meta.get("page_number", 0))
        slide_groups[key].append((idx, meta))

    batch_ids, batch_meta = [], []
    for (fname, page), group in slide_groups.items():
        topics = [m.get("topic", "") for _, m in group if m.get("topic", "") not in ("", "Unknown")]
        if not topics:
            continue
        majority_topic = Counter(topics).most_common(1)[0][0]

        for idx, meta in group:
            if meta.get("topic") != majority_topic:
                meta["topic"] = majority_topic
                batch_ids.append(ids[idx])
                batch_meta.append(meta)

    if batch_ids:
        collection.update(ids=batch_ids, metadatas=batch_meta)
        print(f"  Enforced consistent topic across {len(batch_ids)} chunks.")

# Main enrichment pipeline


def enrich_metadata():
    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)
    course, level = detect_course(collection)

    total = collection.count()
    print(f"\n{'='*55}")
    print(f"  Chunk Metadata Enrichment")
    print(f"{'='*55}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Total chunks: {total}")
    print(f"  Model      : {OLLAMA_MODEL}")
    print(f"{'='*55}\n")

    # Fetch all chunks
    all_data = collection.get(include=["documents", "metadatas"])
    ids       = all_data["ids"]
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    # Track progress — skip already-enriched chunks
    to_process = [
        (i, ids[i], documents[i], metadatas[i])
        for i in range(len(ids))
        if "topic" not in metadatas[i]      # skip if already tagged
    ]

    skipped = total - len(to_process)
    if skipped:
        print(f"    Skipping {skipped} already-enriched chunks.\n")

    if not to_process:
        print(" All chunks already enriched. Nothing to do.")
        return

    print(f"  Tagging {len(to_process)} chunks...\n")

    # Tag each chunk and batch-update ChromaDB
    batch_ids  = []
    batch_meta = []
    enriched   = 0
    failed     = 0

    for i, chunk_id, text, existing_meta in tqdm(to_process, desc="Enriching"):
        slide_title = existing_meta.get("section", "")
        tags = tag_chunk(text, course = course, level = level, slide_title=slide_title)

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
    print(f"  Enrichment complete!")
    print(f"  Successfully tagged : {enriched}")
    print(f"  Used defaults       : {failed}")
    print(f"{'='*55}\n")

    print("\n  Normalizing and deduplicating topics...")
    mapping = normalize_topics(collection, course, level)
    apply_topic_normalization(collection, mapping)

    print("\n  Enforcing per-slide topic consistency...")
    enforce_per_slide_topic_consistency(collection)


# Preview enriched chunks

def preview(n: int = 5):
    """Print a sample of enriched chunks to verify quality."""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"], limit=n)

    print(f"\n Sample of {n} enriched chunks \n")
    for doc, meta in zip(data["documents"], data["metadatas"]):
        print(f"  File      : {meta.get('file_name')}  Slide {meta.get('slide_number')}")
        print(f"  Topic     : {meta.get('topic')}  ->  {meta.get('subtopic')}")
        print(f"  Type      : {meta.get('concept_type')}  |  Difficulty: {meta.get('difficulty')}")
        print(f"  Keywords  : {meta.get('keywords')}")
        print(f"  Text      : {doc[:120]}...")
        print()


# Entry point

if __name__ == "__main__":
    enrich_metadata()
    preview(n=5)