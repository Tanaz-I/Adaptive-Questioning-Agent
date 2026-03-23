import chromadb
from chromadb.config import Settings
import json
import requests
from collections import Counter
# from Adaptation_RL.Agent import AdaptiveAgent

from NLP import knowledge_base_construction, enrich_metadata, rag_query_engine

DOCS_DIR = "./contents"
CHROMA_DB_DIR   = "./chroma_db"
COLLECTION_NAME = "rag_kb"

OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:1.5b-instruct"


#knowledge_base_construction.run_pipeline(DOCS_DIR)

#collection = enrich_metadata.enrich_metadata()

client     = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection(COLLECTION_NAME)
all_meta   = collection.get(include=["metadatas"])["metadatas"]

topics = sorted(set(
    m["topic"].lower()
    for m in all_meta
    if m.get("topic") not in (None, "", "Unknown")
))

print(topics)

course, level = enrich_metadata.detect_course(collection)   

# Step 1: get canonical topics AFTER normalization
all_meta = collection.get(include=["metadatas"])["metadatas"]
canonical_topics = sorted(set(
    m["topic"]
    for m in all_meta
    if m.get("topic") not in (None, "", "Unknown")
))

if not canonical_topics:
    print("[ERROR] No canonical topics found after normalization.")
    raise SystemExit(1)

print(f"Canonical topics: {canonical_topics}")

# Step 2: build topic->subtopics context
from collections import defaultdict
topic_subtopics = defaultdict(set)
for m in all_meta:
    t = m.get("topic", "")
    s = m.get("subtopic", "")
    if t and t != "Unknown" and s and s != "Unknown":
        topic_subtopics[t].add(s)

topics_with_context = {t: sorted(topic_subtopics[t]) for t in canonical_topics}

# Step 3: two-shot constrained prerequisite prompt
valid_topics_str = json.dumps(canonical_topics, indent=2)

prompt = f"""You are an expert in "{course}" at the {level} level.

These are the ONLY valid topic names:
{valid_topics_str}

Each topic covers these concepts:
{json.dumps(topics_with_context, indent=2)}

Task: For each topic, list which OTHER topics from the valid list above must be learned first.

Rules (STRICT):
- Prerequisite values MUST be copied EXACTLY from the valid topics list above
- Do NOT use subtopic names as prerequisites
- Do NOT invent new topic names
- Do NOT include a topic as its own prerequisite
- Only direct prerequisites (not transitive)
- If none, use []

Return ONLY a JSON object where every key is from the valid topics list.
Return ONLY valid JSON. No explanation, no markdown.

JSON:"""

response = requests.post(
    OLLAMA_URL,
    json={
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        },
    },
    timeout=60,
)

def safe_parse_json(raw: str, fallback):
    """Extract and parse the first JSON object or array from raw string."""
    raw = raw.strip()
    # Try object first
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    # Try array
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start != -1 and end > 0:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    print(f"  [WARN] JSON parse failed. Raw response was:\n{raw[:300]}")
    return fallback

raw  = response.json()["response"]
dependencies = safe_parse_json(raw, fallback={})

print(dependencies)

all_meta   = collection.get(include=["metadatas"])["metadatas"]

# Group difficulty values by topic
topic_difficulties = {}
for m in all_meta:
    topic      = m.get("topic", "")
    difficulty = m.get("difficulty", "")
    if not topic or topic == "Unknown" or not difficulty:
        continue
    topic_difficulties.setdefault(topic, []).append(difficulty)

# Compute mode for each topic
topic_modes = {
    topic: Counter(diffs).most_common(1)[0][0]
    for topic, diffs in topic_difficulties.items()
}

difficulty_list = dict(sorted(topic_modes.items()))
print(difficulty_list)

# RL_agent = AdaptiveAgent()
