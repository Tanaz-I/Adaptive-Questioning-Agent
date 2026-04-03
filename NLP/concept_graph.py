"""
Concept Graph Module
====================

• Extracts concepts from chunks
• Builds lightweight concept graph
• Expands topics for better retrieval

Used by:
- retrieval_engine.py
- question_generator.py (later)
"""

import json
import requests
from collections import defaultdict


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b-instruct"


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
    )

    response.raise_for_status()
    return response.json()["response"].strip()


# ─────────────────────────────────────────────
# SAFE JSON PARSER
# ─────────────────────────────────────────────

def parse_json(output):

    try:
        start = output.find("{")
        end = output.rfind("}") + 1

        if start != -1 and end > 0:
            return json.loads(output[start:end])
    except:
        pass

    return {"concepts": []}


# ─────────────────────────────────────────────
# 1. EXTRACT CONCEPTS FROM TEXT
# ─────────────────────────────────────────────

def extract_concepts(text):

    prompt = f"""
Extract key concepts from the following text.

Rules:
- Return ONLY important concepts
- Avoid generic words (like "example", "definition")
- Keep it domain-specific
- Max 5 concepts

Return JSON:
{{
  "concepts": ["...", "..."]
}}

Text:
{text[:500]}
"""

    output = call_llm(prompt)
    result = parse_json(output)

    return [c.lower() for c in result.get("concepts", [])]


# ─────────────────────────────────────────────
# 2. BUILD GRAPH
# ─────────────────────────────────────────────

def build_concept_graph(chunks):

    graph = defaultdict(set)

    for chunk in chunks:

        concepts = extract_concepts(chunk["text"])

        # connect all concepts in same chunk
        for c1 in concepts:
            for c2 in concepts:
                if c1 != c2:
                    graph[c1].add(c2)

    return graph


# ─────────────────────────────────────────────
# 3. EXPAND TOPIC
# ─────────────────────────────────────────────

def expand_topic(topic, graph, max_expansion=5):

    topic = topic.lower()

    expanded = set([topic])

    if topic in graph:
        related = list(graph[topic])[:max_expansion]
        expanded.update(related)

    return list(expanded)

if __name__ == "__main__":

    sample_chunks = [
        {"text": "Constructors initialize objects in a class."},
        {"text": "Destructors clean up memory when objects are destroyed."}
    ]

    graph = build_concept_graph(sample_chunks)

    print("\nGraph:")
    print(dict(graph))

    print("\nExpanded topic:")
    print(expand_topic("constructors", graph))