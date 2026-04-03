"""
Advanced Question Generator (FINAL - MULTI-HOP VERSION)
======================================================

• Multi-hop question generation (factual → inferential → evaluative)
• Uses RAG (retrieval_engine)
• Chunk rotation for variety
• Avoids repeated questions
• Validation + fallback
"""
# NLP.Q_Generator_A_Evaluator.
import json
import requests
from retrieval_engine import retrieve_chunks, get_neighbor_chunks
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
    # print(output)

    # ---- Clean output ----
    output = output.replace("```", "")  # remove code fences

    # ---- Try JSON first ----
    try:
        start = output.find("{")
        end = output.rfind("}") + 1

        if start != -1 and end > 0:
            data = json.loads(output[start:end])

            question = data.get("question", "").strip()

            # 🔥 handle broken keys
            answer = data.get("reference_answer", "")

            # fallback: find any long string value
            if not answer:
                for k, v in data.items():
                    if isinstance(v, str) and len(v.split()) > 5:
                        if k != "question":
                            answer = v
                            break

            return {
                "question": question,
                "reference_answer": answer.strip()
            }
    except:
        pass

    # ---- Fallback extraction ----
    lines = [l.strip() for l in output.split("\n") if l.strip()]

    question = ""
    answer = ""

    # ---- Extract question ----
    for i, line in enumerate(lines):
        if "?" in line:
            # include previous lines if code exists
            start_idx = max(0, i - 8)
            question = "\n".join(lines[start_idx:i+1])
            break

    # ---- Extract answer ----
    for i, line in enumerate(lines):
        if "answer" in line.lower():
            answer = " ".join(lines[i: i+4])
            break

    # fallback if not found
    if not answer:
        answer = " ".join(lines[-4:])

    # safety
    if not question:
        question = "Error"
    if not answer:
        answer = "Error"

    # clean unwanted prefixes
    print(answer)
    print(question)
    question = question.replace('"question":', '').strip()
    answer = answer.replace('"reference_answer":', '').strip()

    return {
        "question": question,
        "reference_answer": answer
    }

# def parse_json(output):

#     try:
#         start = output.find("{")
#         end = output.rfind("}") + 1

#         if start != -1 and end > 0:
#             return json.loads(output[start:end])

#     except:
#         pass

    
#     question = ""
#     answer = ""

#     lines = output.split("\n")

#     for i, line in enumerate(lines):

#         if "question" in line.lower():
#             if ":" in line:
#                 question = line.split(":", 1)[1].strip()

#         if "answer" in line.lower():
#             if ":" in line:
#                 answer = line.split(":", 1)[1].strip()

#     # fallback safety
#     if not question:
#         question = "Error"
#     if not answer:
#         answer = "Error"

#     return {
#         "question": question,
#         "reference_answer": answer
#     }


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

def is_answerable(chunk, question):

    prompt = f"""
You are a strict evaluator.

Content:
{chunk[:500]}

Question:
{question}

Can the answer be found directly in the content WITHOUT reasoning?

Answer ONLY:
YES or NO
"""

    out = call_llm(prompt, temperature=0)

    return "YES" in out.upper()

def generate_answer_with_llm(context, question):

    prompt = f"""
You are an expert teacher.

Context:
{context}

Question:
{question}

Answer the question clearly.
Answer the question clearly.


Do NOT skip reasoning steps.

Rules:
- Use the context primarily
- You may apply reasoning if needed
- Do NOT say "not in context"
- Be precise and correct

Answer:
"""

    return call_llm(prompt, temperature=0.4)

# def contains_code(text):
#     return any(sym in text for sym in [";", "{", "}", "()", "class", "="])

def extract_code_block(text):
    CODE_SIGNALS = (
        '{', '}', ';', '::', '->', 'def ', 'class ',
        'public ', 'private ', 'return ', '#include',
        'void ', 'int ', 'bool ', 'string ', '()',
        '==', '!=', '+=', '-=', '<=', '>='
    )

    lines = text.split('\n')
    code_lines = []
    prose_lines = []

    for line in lines:
        is_code = (
            any(sig in line for sig in CODE_SIGNALS) or
            (len(line) > 2 and line[:2] == '  ' and line.strip())
        )

        if is_code:
            code_lines.append(line)
        else:
            prose_lines.append(line)

    return '\n'.join(code_lines).strip(), '\n'.join(prose_lines).strip()

def build_code_injection(chunk, meta):
    """
    If chunk has code, extracts it and builds injection strings for prompts.
    Returns (injection_block: str, instruction: str)
    """
    if not meta.get("contains_code") and meta.get("chunk_category") != "code":
        return "", ""

    code_block, _ = extract_code_block(chunk)
    if not code_block or len(code_block.split()) < 4:
        return "", ""

    injection = f"""
Code from the content (include this VERBATIM in the question):
{code_block}
"""
    instruction = """
    CRITICAL CODE RULES:
    - Copy the code block above EXACTLY into your question using triple backticks
    - Ask about a specific line, method, output, or behavior visible in the code
    - The student must be able to answer solely by reading the code you show them
    - NEVER describe code or ask about it without including it in the question text"""

    return injection, instruction



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

def generate_factual(chunk, meta, topic, difficulty, asked_questions):
    avoid         = build_avoid_str(asked_questions)
    code_inj, code_rule = build_code_injection(chunk, meta)

    # Difficulty-specific instruction
    diff_guide = {
        "easy":         "Ask for a direct definition or what something does.",
        "medium":       "Ask about how a concept works or why it behaves a certain way.",
        "hard":         "Ask about an edge case, a subtle distinction, or a tricky behavior.",
        "basic":        "Ask for a direct definition or what something does.",
        "intermediate": "Ask about how a concept works or why it behaves a certain way.",
        "advanced":     "Ask about an edge case, a subtle distinction, or a tricky behavior.",
    }.get(difficulty, "Ask about the main concept in the content.")

    prompt = f"""You are an expert teacher creating an exam question.

Topic: {topic}
Difficulty: {difficulty}

Content:
{chunk}
{code_inj}

Task: Generate ONE factual question and its reference answer.

Question rules:
- {diff_guide}
- MUST NOT be MCQ
- MUST end with a question mark
- MUST be directly answerable from the content above
- Do NOT use outside knowledge
{code_rule}

Answer rules:
- Must be 2-4 complete sentences minimum
- Must be specific, not generic ("it is used for...")
- Must fully answer the question

{avoid}

Return ONLY valid JSON, no explanation, no markdown:
{{"question": "...", "reference_answer": "..."}}"""

    return parse_json(call_llm(prompt, temperature=0.6))


def generate_mcq(context, meta, topic, difficulty, asked_questions):

    avoid = build_avoid_str(asked_questions)
    rule = build_grounding_rule(meta)
    code_inj, code_rule = build_code_injection(context, meta)

    prompt = f"""
You are an expert teacher.

Topic: {topic}
Difficulty: {difficulty}

Content:
{context}
{code_inj}

Generate ONE MCQ question.

Rules:
- Question MUST be based on MAIN concept
- Do NOT use outside knowledge
- {code_rule}
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
    code_inj, code_rule = build_code_injection(chunk2, meta2)

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
    {code_inj}

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
    {code_rule}

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
                      concept_graph=None,
                      used_chunk_ids = None):

    if asked_questions is None:
        asked_questions = []

    # ─────────────────────────────────────────────
    # Step 1: Retrieve chunks
    # ─────────────────────────────────────────────
    chunks, new_keys = retrieve_chunks(
        topic,
        difficulty,
        question_type,
        prerequisites=prerequisites,
        concept_graph=concept_graph,
        used_chunk_ids = used_chunk_ids
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
    def build_context(primary, neighbors):
        texts = [primary["text"]]

        for n in neighbors:
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

        if "options" in result:
            # convert MCQ → normal format for pipeline
            correct = result.get("correct_answer", "").lower()
            options = result.get("options", {})

            answer_text = options.get(correct, "")

            result["reference_answer"] = answer_text

        if not is_valid_mcq(result):
            result = generate_factual(text1, meta1, topic, difficulty, asked_questions)

    else:
        result = generate_factual(text1, meta1, topic, difficulty, asked_questions)
    
    if not validate(result):
        result = generate_factual(text1, meta1, topic, difficulty, asked_questions)

    if not is_answerable(text1, result["question"]):

        print("[INFO] Answer not directly in content → using reasoning")

        result["reference_answer"] = generate_answer_with_llm(
            text1,
            result["question"]
        )

    if question_type == "factual":
        result["question_type"] = "factual"
        return result, new_keys

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
    
    result_inf["reference_answer"] = generate_answer_with_llm(
        text1 + "\n\n" + text2,
        result_inf["question"]
    )

    # ── FINAL DECISION FOR INFERENTIAL ──
    if question_type == "inferential":

        if validate(result_inf):
            result_inf["question_type"] = "inferential"
            return result_inf, new_keys
        else:
            result["question_type"] = "factual"
            return result, new_keys
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

    
    result_eval["reference_answer"] = generate_answer_with_llm(
        text1 + "\n\n" + text2 + "\n\n" + text3,
        result_eval["question"]
    )

    # ── FINAL DECISION FOR EVALUATIVE ──
    if validate(result_eval):
        result_eval["question_type"] = "evaluative"
        return result_eval, new_keys

    if validate(result_inf):
        result_inf["question_type"] = "inferential"
        return result_inf, new_keys

    result["question_type"] = "factual"
    return result, new_keys

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    q, _ = generate_question(
        topic="Pointers to Class Members",
        difficulty="medium",
        question_type="factual",
        question_count=0,
        asked_questions=[]
    )

    print("\nQuestion:\n", q["question"])
    print("\nAnswer:\n", q["reference_answer"])