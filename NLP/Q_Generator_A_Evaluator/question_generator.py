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
from NLP.Q_Generator_A_Evaluator.retrieval_engine import retrieve_chunks, get_neighbor_chunks
import numpy as np


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"


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

def validate(result, expected_type="factual"):

    if not result:
        return False

    q = result.get("question", "")
    a = result.get("reference_answer", "")

    if len(q.split()) < 5:
        return False

    if len(a.split()) < 5:
        return False

    if q == "Error" or a == "Error":
        return False

    if expected_type == "factual":
        if is_mcq_format(q):
            if not is_valid_mcq(q, result.get("correct_answer", "")):
                return False

    return True

# def contains_code(text):
#     return any(sym in text for sym in [";", "{", "}", "()", "class", "="])


def is_mcq_format(q):
    q = q.lower()

    patterns = [
        "a)", "b)", "c)", "d)",
        "a.", "b.", "c.", "d.",
        "1)", "2)", "3)", "4)"
    ]

    return sum(p in q for p in patterns) >= 3


def is_valid_mcq(result):

    q = result.get("question", "").lower()
    options = result.get("options", {})
    correct = result.get("correct_answer", "").lower()

    if not options or len(options) != 4:
        return False

    patterns = ["a)", "b)", "c)", "d)", "a.", "b.", "c.", "d."]
    count = sum(p in q for p in patterns)

    if count < 3:
        return False

    if correct not in ["a)", "b)", "c)", "d)", "a", "b", "c", "d"]:
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

def build_grounding_rule(meta):

    if meta.get("contains_code"):
        return "The question MUST refer to the code in the content."

    if meta.get("contains_example"):
        return "The question MUST refer to the example in the content."

    return ""
# ─────────────────────────────────────────────
# 1. Factual Question
# ─────────────────────────────────────────────

def generate_factual(chunk,meta, topic, difficulty, asked_questions):

    avoid = build_avoid_str(asked_questions)
    rule = build_grounding_rule(meta)

    prompt = f"""
    You are an expert teacher.

    Topic: {topic}
    Difficulty: {difficulty}

    Content:
    {chunk}

    Generate ONE factual question.
    STRICT RULES:
    - MUST NOT be a multiple choice question (MCQ)
    - MUST NOT start with "Which of the following"
    - MUST NOT include options like A), B), etc.
    - MUST be a direct question (short answer)
    - MUST be answerable in 1 to 3 sentences
    - MUST NOT require guessing
    Rules:
    - Easy → direct definition
    - Medium → concept understanding
    - Hard → slightly tricky recall
    - Question MUST be directly answerable from content
    - Do NOT use outside knowledge
    - Do NOT add extra explanation
    - Answer must be complete and specific (not one word)
    {rule}

    {avoid}

    Return ONLY valid JSON.
    No unnecessary statements. Give only the JSON.

    {{
    "question": "...",
    "reference_answer": "..."
    }}
    """
    return parse_json(call_llm(prompt, temperature=0.7))

def generate_mcq(context, meta, topic, difficulty, asked_questions):

    avoid = build_avoid_str(asked_questions)
    rule = build_grounding_rule(meta)

    prompt = f"""
You are an expert teacher.

Topic: {topic}
Difficulty: {difficulty}

Content:
{context}

Generate ONE MCQ question.

Rules:
- Question MUST be based on MAIN concept
- Do NOT use outside knowledge
- {rule}
- Question MUST use terms from content

MCQ Rules:
- 4 options (A, B, C, D)
- Only ONE correct answer
- Distractors must be realistic
- Answer must be from content
- If content is insufficient, DO NOT generate

{avoid}

Return ONLY JSON:

{{
 "question": "...",
 "options": {{
   "a)": "...",
   "b)": "...",
   "c)": "...",
   "d)": "..."
 }},
 "correct_answer": "a)"
}}
"""
    return parse_json(call_llm(prompt))

# ─────────────────────────────────────────────
# 2. Inferential Question (2-hop)
# ─────────────────────────────────────────────

def rewrite_inferential(q, a, chunk2, meta2, topic, difficulty, asked_questions):

    avoid = build_avoid_str(asked_questions)
    rule = build_grounding_rule(meta2)

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
    - MUST combine BOTH concepts
    - MUST require reasoning (How or Why)
    - Medium → basic reasoning
    - Hard → deeper reasoning
    - MUST NOT be answerable from a single sentence
    - MUST NOT introduce new concepts
    - Question must be concise (max 20 words)
    - Answer must combine BOTH contents clearly
    - Answer must be detailed (not short)
    {rule}

    {avoid}

    Return ONLY valid JSON.
    No unnecessary statements. Give only the JSON.

    {{
    "question": "...",
    "reference_answer": "..."
    }}
    """
    return parse_json(call_llm(prompt, temperature=0.7))

# ─────────────────────────────────────────────
# 3. Evaluative Question (3-hop)
# ─────────────────────────────────────────────

def rewrite_evaluative(q, a, chunk3, meta3, topic, difficulty, asked_questions):

    avoid = build_avoid_str(asked_questions)
    rule = build_grounding_rule(meta3)

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
    - MUST involve comparison / justification
    - MUST require reasoning
    - Medium → simple reasoning
    - Hard → critical evaluation
    - MUST combine ALL concepts
    - MUST NOT use external knowledge
    - Answer must justify clearly (not generic)
    {rule}

    {avoid}

    Return ONLY valid JSON.
    No unnecessary statements. Give only the JSON.

    {{
    "question": "...",
    "reference_answer": "..."
    }}
    """
    return parse_json(call_llm(prompt, temperature=0.7))

# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
def group_chunks_by_subtopic(chunks):

    groups = {}

    for c in chunks:
        sub = c.get("subtopic", "unknown")

        if sub not in groups:
            groups[sub] = []

        groups[sub].append(c)

    return groups

def generate_question(topic, difficulty, question_type,
                      question_count=0,
                      asked_questions=None,
                      prerequisites=None,
                      concept_graph=None):

    if asked_questions is None:
        asked_questions = []

    # ─────────────────────────────────────────────
    # Step 1: Retrieve chunks
    # ─────────────────────────────────────────────
    chunks = retrieve_chunks(
        topic,
        difficulty,
        question_type,
        prerequisites=prerequisites,
        concept_graph=concept_graph
    )

    if not chunks:
        return {"question": "No data", "reference_answer": "N/A", "question_type": question_type}

    
    # texts = [
    #     c["text"] for c in chunks
    #     if len(c["text"]) > 50 and topic.lower() in c["text"].lower()
    # ]

    # if len(texts) < 2:
    #     texts = [c["text"] for c in chunks if len(c["text"]) > 50]

    # if len(texts) < 2:
    #     return {"question": "Insufficient data", "reference_answer": "N/A", "question_type": question_type}

    # # ─────────────────────────────────────────────
    # # Step 2: Chunk rotation
    # # ─────────────────────────────────────────────
    # idx = question_count % len(texts)

    # chunk1 = texts[idx]
    # chunk2 = texts[(idx + 1) % len(texts)]
    # chunk3 = texts[(idx + 2) % len(texts)]
    
    groups = group_chunks_by_subtopic(chunks)
    subtopics = list(groups.keys())

    if len(subtopics) < 2:
        selected = chunks[:3]
    else:
        s1 = subtopics[question_count % len(subtopics)]
        s2 = subtopics[(question_count + 1) % len(subtopics)]
        s3 = subtopics[(question_count + 2) % len(subtopics)]

        selected = [
            groups[s1][0],
            groups[s2][0],
            groups[s3][0]
        ]

    # Extract text + metadata
    chunk1, chunk2, chunk3 = selected

    # PRIMARY chunks (decision making)
    primary1, primary2, primary3 = chunk1, chunk2, chunk3

    # GET NEIGHBORS (context only)
    neighbors1 = get_neighbor_chunks(primary1)
    neighbors2 = get_neighbor_chunks(primary2)
    neighbors3 = get_neighbor_chunks(primary3)

    # BUILD CONTEXT (merge)
    def build_context(primary, neighbors, k=2):
        texts = [primary["text"]]

        for n in neighbors[:k]:
            if n['text'] != primary['text']:
                texts.append(n["text"])

        return "\n\n".join(texts)

    text1 = build_context(primary1, neighbors1)
    text2 = build_context(primary2, neighbors2)
    text3 = build_context(primary3, neighbors3)

    # IMPORTANT: metadata only from primary
    meta1 = primary1
    meta2 = primary2
    meta3 = primary3
    # ─────────────────────────────────────────────
    # Step 3: Factual
    # ─────────────────────────────────────────────
    prob = np.random.uniform(0, 1)

    if prob < 0.3 and question_type == "factual":

        result = generate_mcq(text1, meta1, topic, difficulty, asked_questions)

        if not is_valid_mcq(result):
            result = generate_factual(text1, meta1, topic, difficulty, asked_questions)

    else:
        result = generate_factual(text1, meta1, topic, difficulty, asked_questions)
    
    if not validate(result):
        result = generate_factual(text1, meta1, topic, difficulty, asked_questions)

    if question_type == "factual":
        result["question_type"] = "factual"
        return result

    # ─────────────────────────────────────────────
    # Step 4: Inferential
    # ─────────────────────────────────────────────

    result_inf = rewrite_inferential(
        result["question"],
        result["reference_answer"],
        text2,
        meta2,
        topic,
        difficulty,
        asked_questions
    )

    # retry once
    if not validate(result_inf):
        result_inf = rewrite_inferential(
            result["question"],
            result["reference_answer"],
            text2,
            meta2,
            topic,
            difficulty,
            asked_questions
        )

    # ── FINAL DECISION FOR INFERENTIAL ──
    if question_type == "inferential":

        if validate(result_inf):
            result_inf["question_type"] = "inferential"
            return result_inf
        else:
            result["question_type"] = "factual"
            return result
    # ─────────────────────────────────────────────
    # Step 5: Evaluative
    # ─────────────────────────────────────────────

    result_eval = rewrite_evaluative(
        result_inf["question"],
        result_inf["reference_answer"],
        text3,
        meta3,
        topic,
        difficulty,
        asked_questions
    )

    # retry once
    if not validate(result_eval):
        result_eval = rewrite_evaluative(
            result_inf["question"],
            result_inf["reference_answer"],
            text3,
            meta3,
            topic,
            difficulty,
            asked_questions
        )

    # ── FINAL DECISION FOR EVALUATIVE ──
    if validate(result_eval):
        result_eval["question_type"] = "evaluative"
        return result_eval

    if validate(result_inf):
        result_inf["question_type"] = "inferential"
        return result_inf

    result["question_type"] = "factual"
    return result

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