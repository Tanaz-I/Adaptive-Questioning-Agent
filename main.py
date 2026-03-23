import chromadb
from chromadb.config import Settings
import json
import requests
from collections import Counter

from NLP import knowledge_base_construction, enrich_metadata, rag_query_engine

DOCS_DIR = "./contents"
CHROMA_DB_DIR   = "./chroma_db"
COLLECTION_NAME = "rag_kb"

OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:1.5b-instruct"


# knowledge_base_construction.run_pipeline(DOCS_DIR)

# collection = enrich_metadata.enrich_metadata()

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


prompt = f"""You are an expert in the following topics.

Given this list of topics:
{json.dumps(topics, indent=2)}

Return a JSON object where each topic is a key, and its value is a list
of topics from the same list that must be learned before it.
If a topic has no prerequisites, assign it an empty list.

Return ONLY valid JSON. No explanation, no markdown.
"""

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

raw  = response.json()["response"]
dependencies = json.loads(raw)

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
