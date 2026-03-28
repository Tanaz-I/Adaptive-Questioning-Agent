"""
Advanced Question Generator (FINAL - MULTI-HOP VERSION)
======================================================

• Multi-hop question generation (factual → inferential → evaluative)
• Uses RAG (retrieval_engine)
• Chunk rotation for variety
• Avoids repeated questions
• Validation + fallback
"""

import json
import requests
from NLP.Q_Generator_A_Evaluator.retrieval_engine import retrieve_chunks


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b-instruct"


# ─────────────────────────────────────────────
# LLM Call
# ─────────────────────────────────────────────

def call_llm(prompt, temperature=0.6):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
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

        if start != -1 and end > 0:
            return json.loads(output[start:end])

    except:
        pass

    
    question = ""
    answer = ""

    lines = output.split("\n")

    for i, line in enumerate(lines):

        if "question" in line.lower():
            if ":" in line:
                question = line.split(":", 1)[1].strip()

        if "answer" in line.lower():
            if ":" in line:
                answer = line.split(":", 1)[1].strip()

    # fallback safety
    if not question:
        question = "Error"
    if not answer:
        answer = "Error"

    return {
        "question": question,
        "reference_answer": answer
    }


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate(result):

    if not result:
        return False

    q = result.get("question", "")
    a = result.get("reference_answer", "")

    if len(q.split()) < 5:
        return False

    if len(a.split()) < 8:
        return False

    if q == "Error" or a == "Error":
        return False

    return True


# ─────────────────────────────────────────────
# Avoid repetition helper
# ─────────────────────────────────────────────

def build_avoid_str(asked_questions):

    if not asked_questions:
        return ""

    last_qs = asked_questions[-5:]

    return "\nDo NOT generate any of these questions:\n" + "\n".join(f"- {q}" for q in last_qs)


# ─────────────────────────────────────────────
# 1. Factual Question
# ─────────────────────────────────────────────

def generate_factual(chunk, topic, difficulty, asked_questions):

    avoid = build_avoid_str(asked_questions)

    prompt = f"""
You are an expert teacher.

Topic: {topic}
Difficulty: {difficulty}

Content:
{chunk}

Generate ONE factual question.

Rules:
- Easy → direct definition
- Medium → concept understanding
- Hard → slightly tricky recall
- Answer must be strictly from content
- Do NOT use outside knowledge
- Do NOT add extra explanation

{avoid}

Return ONLY valid JSON.
No markdown, no text.

{{
 "question": "...",
 "reference_answer": "..."
}}
"""
    return parse_json(call_llm(prompt, temperature=0.7))

# ─────────────────────────────────────────────
# 2. Inferential Question (2-hop)
# ─────────────────────────────────────────────

def rewrite_inferential(q, a, chunk2, topic, difficulty, asked_questions):

    avoid = build_avoid_str(asked_questions)

    prompt = f"""
You are an expert teacher.

Topic: {topic}
Difficulty: {difficulty}

Original Question:
{q}

Original Answer:
{a}

Additional Content:
{chunk2}

Rewrite the question.

Rules:
- Must combine TWO concepts
- Medium → basic reasoning
- Hard → deeper reasoning
- Must require HOW or WHY
- Do NOT introduce new concepts
- Answer must be from given content only

{avoid}

Return ONLY valid JSON.

{{
 "question": "...",
 "reference_answer": "..."
}}
"""
    return parse_json(call_llm(prompt, temperature=0.7))

# ─────────────────────────────────────────────
# 3. Evaluative Question (3-hop)
# ─────────────────────────────────────────────

def rewrite_evaluative(q, a, chunk3, topic, difficulty, asked_questions):

    avoid = build_avoid_str(asked_questions)

    prompt = f"""
You are an expert teacher.

Topic: {topic}
Difficulty: {difficulty}

Current Question:
{q}

Current Answer:
{a}

Additional Content:
{chunk3}

Rewrite the question.

Rules:
- Must involve comparison / justification
- Medium → simple reasoning
- Hard → critical evaluation
- Combine ALL concepts
- Do NOT use external knowledge

{avoid}

Return ONLY valid JSON.

{{
 "question": "...",
 "reference_answer": "..."
}}
"""
    return parse_json(call_llm(prompt, temperature=0.7))

# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def generate_question(topic, difficulty, question_type,
                      question_count=0,
                      asked_questions=None):

    if asked_questions is None:
        asked_questions = []

    # ─────────────────────────────────────────────
    # Step 1: Retrieve chunks
    # ─────────────────────────────────────────────
    chunks = retrieve_chunks(topic, difficulty, question_type)

    if not chunks:
        return {"question": "No data", "reference_answer": "N/A", "question_type": question_type}

    # 🔥 FILTER BAD / IRRELEVANT CHUNKS
    texts = [
        c["text"] for c in chunks
        if len(c["text"]) > 50 and topic.lower() in c["text"].lower()
    ]

    if len(texts) < 2:
        texts = [c["text"] for c in chunks if len(c["text"]) > 50]

    if len(texts) < 2:
        return {"question": "Insufficient data", "reference_answer": "N/A", "question_type": question_type}

    # ─────────────────────────────────────────────
    # Step 2: Chunk rotation
    # ─────────────────────────────────────────────
    idx = question_count % len(texts)

    chunk1 = texts[idx]
    chunk2 = texts[(idx + 1) % len(texts)]
    chunk3 = texts[(idx + 2) % len(texts)]

    # ─────────────────────────────────────────────
    # Step 3: Factual
    # ─────────────────────────────────────────────
    result = generate_factual(chunk1, topic, difficulty, asked_questions)

    if not validate(result):
        result["question_type"] = "factual"
        return result

    if question_type == "factual":
        result["question_type"] = "factual"
        return result

    # ─────────────────────────────────────────────
    # Step 4: Inferential
    # ─────────────────────────────────────────────
    result_inf = rewrite_inferential(
        result["question"],
        result["reference_answer"],
        chunk2,
        topic,
        difficulty,
        asked_questions
    )

    if not validate(result_inf):
        result["question_type"] = "factual"
        return result

    if question_type == "inferential":
        result_inf["question_type"] = "inferential"
        return result_inf

    # ─────────────────────────────────────────────
    # Step 5: Evaluative
    # ─────────────────────────────────────────────
    result_eval = rewrite_evaluative(
        result_inf["question"],
        result_inf["reference_answer"],
        chunk3,
        topic,
        difficulty,
        asked_questions
    )

    if not validate(result_eval):
        # 🔥 FALLBACK → return inferential
        result_inf["question_type"] = "inferential"
        return result_inf

    result_eval["question_type"] = "evaluative"
    return result_eval

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    q = generate_question(
        topic="Constructors and destructors",
        difficulty="medium",
        question_type="inferential",
        question_count=0,
        asked_questions=[]
    )

    print("\nQuestion:\n", q["question"])
    print("\nAnswer:\n", q["reference_answer"])