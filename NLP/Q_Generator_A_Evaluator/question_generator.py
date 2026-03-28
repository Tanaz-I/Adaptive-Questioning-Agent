"""
Question Generator (FINAL)
==========================

• Uses RAG (retrieval_engine)
• Cleans context
• Enforces question type
• Returns question_type for evaluator
"""

import json
import requests
from NLP.Q_Generator_A_Evaluator.retrieval_engine import retrieve_chunks


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b-instruct"


# ─────────────────────────────────────────────
# Clean Chunks
# ─────────────────────────────────────────────

def clean_chunks(chunks):
    cleaned = []
    seen = set()

    for c in chunks:
        text = " ".join(c["text"].split())

        if len(text) < 80:
            continue

        key = text[:60]
        if key in seen:
            continue
        seen.add(key)

        if any(word in text.lower() for word in ["inheritance", "class", "derived", "base"]):
            cleaned.append(text)

    return cleaned[:3]


# ─────────────────────────────────────────────
# Build Prompt
# ─────────────────────────────────────────────

def build_prompt(chunks, difficulty, question_type):

    cleaned = clean_chunks(chunks)

    if not cleaned:
        return None

    context = "\n\n".join(cleaned)

    return f"""
You are an expert educator.

Generate ONE {difficulty.upper()} level {question_type.upper()} question.

STRICT RULES:
- Do NOT generate incorrect or illogical questions
- Question MUST be answerable ONLY from the context

QUESTION TYPE RULES:
- Factual → direct recall
- Inferential → MUST combine at least TWO parts of context
- Evaluative → reasoning/judgment

IMPORTANT:
- If answer comes from one sentence → DO NOT use
- Focus on WHY / HOW
- Avoid trivial questions

GOOD example:
How does inheritance enable reuse and extension?

BAD example:
Which class inherits from X?

Context:
{context}

Return ONLY JSON:
{{
 "question": "...",
 "reference_answer": "..."
}}
"""


# ─────────────────────────────────────────────
# LLM Call
# ─────────────────────────────────────────────

def call_llm(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4}
        }
    )

    response.raise_for_status()
    return response.json()["response"].strip()


# ─────────────────────────────────────────────
# Parse JSON
# ─────────────────────────────────────────────

def parse_json(output):

    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        return json.loads(output[start:end])
    except:
        return {"question": "Error", "reference_answer": "Error"}


# ─────────────────────────────────────────────
# Main Function
# ─────────────────────────────────────────────

def generate_question(topic, difficulty, question_type):

    chunks = retrieve_chunks(topic, difficulty, question_type)

    if not chunks:
        return {"question": "No data", "reference_answer": "N/A", "question_type": question_type}

    prompt = build_prompt(chunks, difficulty, question_type)

    if not prompt:
        return {"question": "Insufficient data", "reference_answer": "N/A", "question_type": question_type}

    output = call_llm(prompt)
    result = parse_json(output)

    return {
        "question": result["question"],
        "reference_answer": result["reference_answer"],
        "question_type": question_type
    }


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    q = generate_question("Inheritance", "medium", "inferential")

    print("\nQuestion:\n", q["question"])
    print("\nReference Answer:\n", q["reference_answer"])