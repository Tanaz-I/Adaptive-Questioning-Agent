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
from NLP.Q_Generator_A_Evaluator.retrieval_engine import retrieve_chunks, get_neighbor_chunks
import numpy as np

from concurrent.futures import ThreadPoolExecutor


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

EVALUATIVE_FRAMES = [
    "Under what conditions would you prefer {A} over {B}? Justify your choice.",
    "A developer argues that {A} makes {B} unnecessary. Evaluate this claim.",
    "What is the most significant limitation of {A} that {B} addresses? Explain.",
    "When would using {A} actually worsen program behavior? Give a concrete scenario.",
    "Compare the design tradeoffs of {A} and {B} for a memory-constrained system.",
    "If you had to choose between {A} and {B} for a large codebase, which would you pick and why?",
]


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
            start_idx = 0  # include all previous lines
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
    # print(answer)
    # print(question)
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

# def is_answerable(chunk, question):

#     prompt = f"""
# You are a strict evaluator.

# Content:
# {chunk[:500]}

# Question:
# {question}

# Can the answer be found directly in the content WITHOUT reasoning?

# Answer ONLY:
# YES or NO
# """

#     out = call_llm(prompt, temperature=0)

#     return "YES" in out.upper()

# def generate_answer_with_llm(context, question):

#     prompt = f"""
# You are an expert teacher.

# Context:
# {context}

# Question:
# {question}

# Answer the question clearly.
# Answer the question clearly.


# Do NOT skip reasoning steps.

# Rules:
# - Use the context primarily
# - You may apply reasoning if needed
# - Do NOT say "not in context"
# - Be precise and correct

# CRITICAL:
# - The answer MUST be based ONLY on the given content
# - Use specific terms, variables, or examples from the content
# - DO NOT use general knowledge unless it appears in the content

# Answer:
# """

#     return call_llm(prompt, temperature=0.4)

# def contains_code(text):
#     return any(sym in text for sym in [";", "{", "}", "()", "class", "="])

def extract_code_block(text):
    # First, check for markdown code blocks
    if '```' in text:
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n?(.*?)\n?```', text, re.DOTALL)
        if code_blocks:
            # Join multiple code blocks if any
            code = '\n\n'.join(code_blocks).strip()
            # Remove code from text for prose
            prose = re.sub(r'```(?:\w+)?\n?.*?\n?```', '', text, flags=re.DOTALL).strip()
            return code, prose

    # Fallback to signal-based extraction
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
    """
    Validates the MCQ result dict.
    Expects keys: question, options (dict with a)-d) keys), correct_answer.
    Returns True only if all structural requirements are met.
    """
    if not result or not isinstance(result, dict):
        return False

    q       = result.get("question", "")
    options = result.get("options", {})
    correct = result.get("correct_answer", "").strip().lower()

    # must have a question
    if not q or len(q.split()) < 4:
        return False

    # normalise correct_answer format
    if not correct.endswith(")"):
        correct = correct + ")"

    # must have exactly 4 options with the right keys
    expected_keys = {"a)", "b)", "c)", "d)"}
    actual_keys   = {k.strip().lower() for k in options.keys()}
    if actual_keys != expected_keys:
        return False

    # all options must have non-empty text
    for v in options.values():
        if not v or not v.strip():
            return False

    # correct_answer must point to a valid key
    if correct not in expected_keys:
        return False

    # reference_answer must be resolvable
    resolved = options.get(correct, "")
    if not resolved or len(resolved.split()) < 2:
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

# def generate_factual(chunk, meta, topic, difficulty, asked_questions):
#     avoid         = build_avoid_str(asked_questions)
#     code_inj, code_rule = build_code_injection(chunk, meta)

#     # Difficulty-specific instruction
#     diff_guide = {
#         "easy":         "Ask for a direct definition or what something does.",
#         "medium":       "Ask about how a concept works or why it behaves a certain way.",
#         "hard":         "Ask about an edge case, a subtle distinction, or a tricky behavior.",
#         "basic":        "Ask for a direct definition or what something does.",
#         "intermediate": "Ask about how a concept works or why it behaves a certain way.",
#         "advanced":     "Ask about an edge case, a subtle distinction, or a tricky behavior.",
#     }.get(difficulty, "Ask about the main concept in the content.")

#     prompt = f"""You are an expert teacher creating an exam question.

# Topic: {topic}
# Difficulty: {difficulty}

# Content:
# {chunk}
# {code_inj}

# Task: Generate ONE factual question and its reference answer.

# Question rules:
# - {diff_guide}
# - MUST NOT be MCQ
# - MUST end with a question mark
# - MUST be directly answerable from the content above
# - Do NOT use outside knowledge
# {code_rule}

# Answer rules:
# - Must be 2-4 complete sentences minimum
# - Must be specific, not generic ("it is used for...")
# - Must fully answer the question

# {avoid}

# Return ONLY valid JSON, no explanation, no markdown:
# {{"question": "...", "reference_answer": "..."}}"""

#     return parse_json(call_llm(prompt, temperature=0.6))

def generate_factual_v2(chunk, meta, topic, difficulty, asked_questions):
    """
    Single LLM call.
    Generates the question AND the answer in one shot.
    The prompt instructs the model to:
      - extract the answer directly if it is present in the content
      - reason toward an answer using the content if it is not explicit
    Replaces: generate_factual + is_answerable + generate_answer_with_llm
    """
    avoid = build_avoid_str(asked_questions)
    code_inj, code_rule = build_code_injection(chunk, meta)

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

Your job has two parts:

PART 1 — Write ONE factual question.
Rules:
- {diff_guide}
- Must NOT be MCQ
- Must end with a question mark
- Must be rooted in the content above
{code_rule}

PART 2 — Write the reference answer.
Answer strategy (pick ONE):
  Strategy A — Direct extraction:
    If the answer is stated explicitly in the content, extract and paraphrase it.
  Strategy B — Reasoning from content:
    If the answer is NOT stated explicitly but can be inferred from the content,
    reason step by step using only what the content provides.
    Do NOT use outside knowledge.

Answer rules:
- 2 - 4 complete sentences minimum
- Use specific terms and variables from the content
- Do NOT say "the content does not mention" — always provide an answer
IMPORTANT:
- The answer must be COMPLETE and FINAL
- Do NOT describe what the answer will be
- Do NOT say "the answer is based on..."
- Directly give the final answer

For code questions:
- Include the exact output (if applicable)
- Then explain briefly why

STRICT OUTPUT FORMAT:
- Output ONLY valid JSON
-If you include anything outside JSON, the answer is WRONG
- Do NOT include any text before or after JSON
- Do NOT say "Here is the answer"
- Do NOT explain what you are doing
- Use double quotes only

Example:
{{"question": "...", "reference_answer": "..."}}

{avoid}

Return ONLY valid JSON, no markdown, no explanation:
{{"question": "...", "reference_answer": "..."}}"""

    return parse_json(call_llm(prompt, temperature=0.6))



# def generate_mcq(context, meta, topic, difficulty, asked_questions):

#     avoid = build_avoid_str(asked_questions)
#     rule = build_grounding_rule(meta)
#     code_inj, code_rule = build_code_injection(context, meta)

#     prompt = f"""
# You are an expert teacher.

# Topic: {topic}
# Difficulty: {difficulty}

# Content:
# {context}
# {code_inj}

# Generate ONE MCQ question.

# Rules:
# - Question MUST be based on MAIN concept
# - Do NOT use outside knowledge
# - {code_rule}
# - Question MUST use terms from content

# MCQ Rules:
# - 4 options (A, B, C, D)
# - Only ONE correct answer
# - Distractors must be realistic
# - Answer must be from content
# - If content is insufficient, DO NOT generate

# {avoid}

# Return ONLY JSON:

# {{
#  "question": "...",
#  "options": {{
#    "a)": "...",
#    "b)": "...",
#    "c)": "...",
#    "d)": "..."
#  }},
#  "correct_answer": "a)"
# }}
# """
#     return parse_json(call_llm(prompt))
def generate_mcq_v2(chunk, meta, topic, difficulty, asked_questions):
    """
    Single LLM call.
    Generates a 4-option MCQ and internally resolves the correct
    answer text into reference_answer so downstream code (validate,
    evaluate_answer) sees the same dict shape as factual questions.

    Returns:
        {
            "question"        : "<question stem with options embedded>",
            "reference_answer": "<full text of the correct option>",
            "options"         : {"a)": ..., "b)": ..., "c)": ..., "d)": ...},
            "correct_answer"  : "a)"   # kept for logging/transparency
        }
    or None if parsing fails, so the caller can fall back to generate_factual_v2.
    """
    avoid          = build_avoid_str(asked_questions)
    code_inj, code_rule = build_code_injection(chunk, meta)
    rule           = build_grounding_rule(meta)

    diff_guide = {
        "easy":         "Test recognition of a core definition or basic fact.",
        "medium":       "Test understanding of how or why something works.",
        "hard":         "Test ability to distinguish between closely related concepts or edge cases.",
        "basic":        "Test recognition of a core definition or basic fact.",
        "intermediate": "Test understanding of how or why something works.",
        "advanced":     "Test ability to distinguish between closely related concepts or edge cases.",
    }.get(difficulty, "Test understanding of the main concept.")

    prompt = f"""You are an expert teacher creating a multiple-choice exam question.

Topic: {topic}
Difficulty: {difficulty}

Content:
{chunk}
{code_inj}

Task: Generate ONE MCQ question with exactly 4 options.

Question rules:
- {diff_guide}
- Must be based on the MAIN concept in the content
- Do NOT use outside knowledge
- {rule}
{code_rule}

Option rules:
- Exactly 4 options labeled a), b), c), d)
- Only ONE option is correct
- The correct answer must be directly supported by the content
- The three distractors must be plausible but clearly wrong to someone who read the content
- Distractors must NOT be trick wording — they must represent genuinely different concepts

Answer rules:
- correct_answer must be exactly one of: "a)", "b)", "c)", "d)"
- reference_answer must be the FULL TEXT of the correct option (not just the label)

{avoid}
If you include anything outside JSON, the answer is WRONG
Return ONLY valid JSON, no markdown, no explanation:
{{
  "question": "<the question stem — do NOT embed options in the stem>",
  "options": {{
    "a)": "<option text>",
    "b)": "<option text>",
    "c)": "<option text>",
    "d)": "<option text>"
  }},
  "correct_answer": "a)",
  "reference_answer": "<full text of the correct option>"
}}"""

    raw  = call_llm(prompt, temperature=0.5)
    data = parse_json(raw)

    if not data:
        return None

    # ── Resolve reference_answer from options if LLM forgot to set it ──
    correct = data.get("correct_answer", "").strip().lower()
    options = data.get("options", {})

    # normalise key format — model sometimes returns "a" instead of "a)"
    normalised_options = {}
    for k, v in options.items():
        key = k.strip().lower()
        if not key.endswith(")"):
            key = key + ")"
        normalised_options[key] = v
    data["options"] = normalised_options

    if not correct.endswith(")"):
        correct = correct + ")"
    data["correct_answer"] = correct

    # always overwrite reference_answer with the resolved option text
    resolved_answer = normalised_options.get(correct, "")
    if resolved_answer:
        data["reference_answer"] = resolved_answer

    # validate the MCQ structure before returning
    if not is_valid_mcq(data):
        return None

    return data

# ─────────────────────────────────────────────
# 2. Inferential Question (2-hop)
# ─────────────────────────────────────────────
# def find_connection(text_a, text_b, topic):
#     """
#     Ask LLM to identify the causal/functional relationship between
#     two pieces of content before writing the question.
#     """
#     prompt = f"""You are an expert in {topic}.

# Concept A (what student already knows):
# {text_a}

# Concept B (new material):
# {text_b}

# In ONE sentence, describe the causal or functional relationship between
# these two concepts. Be specific — describe HOW or WHY one affects the other.
# Return ONLY the one sentence, no preamble or explanation.
# """
#     connection = call_llm(prompt, temperature=0.2)
#     return connection.strip().strip('"')


# def rewrite_inferential(q, a, chunk2, meta2, topic, difficulty, asked_questions):
#     avoid = build_avoid_str(asked_questions)

#     # Step 1: Find the conceptual bridge
#     connection = find_connection(a, chunk2, topic)

#     # Step 2: Code injection for chunk2 if applicable
#     code_inj, code_rule = build_code_injection(chunk2, meta2)

#     diff_guide = {
#         "easy":         "The reasoning should be straightforward — one step of logic.",
#         "medium":       "The reasoning should require connecting two ideas explicitly.",
#         "hard":         "The reasoning should involve a subtle or non-obvious connection.",
#         "basic":        "The reasoning should be straightforward — one step of logic.",
#         "intermediate": "The reasoning should require connecting two ideas explicitly.",
#         "advanced":     "The reasoning should involve a subtle or non-obvious connection.",
#     }.get(difficulty, "The reasoning should require connecting two ideas.")

#     prompt = f"""You are an expert teacher.

# Topic: {topic}
# Difficulty: {difficulty}

# What the student already knows:
# {a}

# Additional content:
# {chunk2}
# {code_inj}

# The key relationship between these two ideas is:
# "{connection}"

# Write ONE inferential question that REQUIRES the student to use this
# relationship in their answer.

# Question rules:
# CRITICAL REQUIREMENTS:
# - The question MUST require using this relationship:
#   "{connection}"
# - The question MUST NOT be answerable without understanding this relationship
# - The question MUST involve cause, effect, or mechanism
# - The question MUST NOT ask for steps or definition
# - Must start with "How" or "Why" or "Explain why" or "What happens when"
# - Must NOT be answerable from either piece of content alone
# - Must NOT be answerable with a simple definition
# - Maximum 25 words
# - {diff_guide}
# {code_rule}

# Answer rules:
# - Must explicitly state the connection: "{connection}"
# - Must be 3-5 sentences minimum
# - Must explain the mechanism, not just state the conclusion

# {avoid}

# Return ONLY valid JSON:
# {{"question": "...", "reference_answer": "..."}}"""

#     return parse_json(call_llm(prompt, temperature=0.65))
def generate_inferential_v2(chunk1, chunk2, meta2, topic, difficulty, asked_questions):
    """
    Single LLM call.
    Replaces: find_connection + rewrite_inferential + generate_answer_with_llm

    The prompt instructs the model to:
      1. Identify the causal/functional relationship between the factual answer
         and the new content (internal chain-of-thought, not returned)
      2. Write an inferential question that requires that relationship
      3. Write a reasoning-based answer that makes the connection explicit
    """
    avoid = build_avoid_str(asked_questions)
    code_inj, code_rule = build_code_injection(chunk2, meta2)

    diff_guide = {
        "easy":         "one step of logic connecting the two ideas.",
        "medium":       "explicitly connecting two ideas with a clear mechanism.",
        "hard":         "a subtle or non-obvious connection between the ideas.",
        "basic":        "one step of logic connecting the two ideas.",
        "intermediate": "explicitly connecting two ideas with a clear mechanism.",
        "advanced":     "a subtle or non-obvious connection between the ideas.",
    }.get(difficulty, "connecting two ideas with a clear mechanism.")

    prompt = f"""You are an expert teacher creating a multi-hop exam question.

Topic: {topic}
Difficulty: {difficulty}

--- Existing content ---
{chunk1}

--- New content to connect to ---
{chunk2}
{code_inj}

Your job has three parts:

PART 1 — Identify the connection (internal reasoning, do NOT include in output):
  Think: what is the causal or functional relationship between the student's
  existing knowledge (the factual answer above) and the new content?
  Be specific — how does one affect, enable, require, or constrain the other?

PART 2 — Write ONE inferential question.
Rules:
- Must start with: How, Why, Explain why, or What happens when
- Must REQUIRE the student to use the connection you identified in Part 1
- Must NOT be answerable from either piece of content alone
- Must NOT be answerable with a simple definition
- Maximum 25 words
- Reasoning required: {diff_guide}
{code_rule}

PART 3 — Write the reference answer.
Rules:
- First sentence must state the connection explicitly (the bridge between the two ideas)
- 3-5 sentences total
- Explain the mechanism, not just the conclusion
- Use specific terms from both content blocks

Answer rules:
- 2 - 4 complete sentences minimum
- Use specific terms and variables from the content
- Do NOT say "the content does not mention" — always provide an answer
IMPORTANT:
- The answer must be COMPLETE and FINAL
- TRY TO AVOID EXAMPLES not used in question
- Do NOT describe what the answer will be
- Do NOT say "the answer is based on..."
- Directly give the final answer

For code questions:
- Include the exact output (if applicable)
- Then explain briefly why

STRICT OUTPUT FORMAT:
- Output ONLY valid JSON
-If you include anything outside JSON, the answer is WRONG
- DO NOT include PART 1, PART 2, PART 3
- Do NOT include any text before or after JSON
- Do NOT say "Here is the answer"
- Do NOT explain what you are doing
- Use double quotes only

Example:
{{"question": "...", "reference_answer": "..."}}

{avoid}

Return ONLY valid JSON, no markdown, no explanation:
{{"question": "...", "reference_answer": "..."}}"""

    return parse_json(call_llm(prompt, temperature=0.65))


# ─────────────────────────────────────────────
# 3. Evaluative Question (3-hop)
# ─────────────────────────────────────────────
# def extract_concepts_from_answer(answer, topic):
#     """Extract two key technical concepts from the inferential answer."""
#     prompt = f"""From this explanation about {topic}:
# "{answer}"

# Name exactly two technical concepts being discussed.
# Return ONLY: ConceptA | ConceptB
# No explanation."""
#     raw    = call_llm(prompt, temperature=0.1)
#     parts  = raw.split("|")
#     concept_a = parts[0].strip() if len(parts) > 0 else topic
#     concept_b = parts[1].strip() if len(parts) > 1 else "the alternative"
#     return concept_a, concept_b


# def rewrite_evaluative(q, a, chunk3, meta3, topic, difficulty, asked_questions):
#     avoid = build_avoid_str(asked_questions)

#     concept_a, concept_b = extract_concepts_from_answer(a, topic)

#     import random
#     frame = random.choice(EVALUATIVE_FRAMES).format(
#         A=concept_a, B=concept_b
#     )

#     code_inj, code_rule = build_code_injection(chunk3, meta3)

#     diff_guide = {
#         "easy":         "The evaluation should be straightforward with one clear reason.",
#         "medium":       "The evaluation should weigh at least two considerations.",
#         "hard":         "The evaluation should involve nuanced tradeoffs and constraints.",
#         "basic":        "The evaluation should be straightforward with one clear reason.",
#         "intermediate": "The evaluation should weigh at least two considerations.",
#         "advanced":     "The evaluation should involve nuanced tradeoffs and constraints.",
#     }.get(difficulty, "The evaluation should weigh relevant tradeoffs.")

#     prompt = f"""You are an expert teacher.

# Topic: {topic}
# Difficulty: {difficulty}

# Background the student knows:
# {a}

# Additional context:
# {chunk3[:400]}
# {code_inj}

# Write ONE evaluative question using this structure as a guide:
# "{frame}"

# Question rules:
# - Must require the student to TAKE A POSITION, not just list facts
# - Must be answerable only with deep understanding of both concepts
# -DO NOT use general knowledge unless it appears in the content
# - Maximum 30 words
# - {diff_guide}
# {code_rule}

# Answer rules:
# - Must state a clear position in the first sentence
# - Must give at least 2 specific reasons supporting the position
# - Must acknowledge one limitation or counterpoint
# - Minimum 4 sentences

# {avoid}

# Return ONLY valid JSON:
# {{"question": "...", "reference_answer": "..."}}

# STRICT FORMAT:
# - No text before or after JSON
# - No "Here is the question"
# """

#     return parse_json(call_llm(prompt, temperature=0.7))
def generate_evaluative_v2(chunk1, chunk2, chunk3, meta3, topic, difficulty, asked_questions):
    """
    Single LLM call.
    Replaces: extract_concepts_from_answer + rewrite_evaluative + generate_answer_with_llm

    The prompt instructs the model to:
      1. Extract two key concepts from the inferential answer (internal)
      2. Write an evaluative question that requires taking a position on those concepts
      3. Write a justified answer with position + reasons + counterpoint
    """
    import random
    avoid = build_avoid_str(asked_questions)
    code_inj, code_rule = build_code_injection(chunk3, meta3)

    frame_hint = random.choice([
        "Under what conditions would you prefer one approach over the other?",
        "What is the most significant limitation of one concept that the other addresses?",
        "When would using one approach actually worsen program behavior?",
        "Compare the design tradeoffs of both concepts for a real system.",
        "If you had to choose between the two for a large codebase, which and why?",
        "A developer argues one makes the other unnecessary — evaluate this claim.",
    ])

    diff_guide = {
        "easy":         "one clear reason supporting the position.",
        "medium":       "at least two considerations weighed against each other.",
        "hard":         "nuanced tradeoffs, constraints, and a non-obvious counterpoint.",
        "basic":        "one clear reason supporting the position.",
        "intermediate": "at least two considerations weighed against each other.",
        "advanced":     "nuanced tradeoffs, constraints, and a non-obvious counterpoint.",
    }.get(difficulty, "relevant tradeoffs weighed against each other.")

    prompt = f"""You are an expert teacher creating a higher-order evaluation question.

Topic: {topic}
Difficulty: {difficulty}

--- Existing concepts(Assume students know these) ---
{chunk1} and
{chunk2}

--- Additional concepts ---
{chunk3}
{code_inj}

Your job has three parts:

PART 1 — Extract two key technical concepts (internal reasoning, do NOT include in output):
  From the existing and additional concepts above, identify exactly two technical concepts being
  discussed or compared. 

PART 2 — Write ONE evaluative question about those two concepts.
Use this style as inspiration: "{frame_hint}"
Rules:
- Must require the student to TAKE A POSITION, not just list facts
- Must be answerable only with deep understanding of both concepts
- Must NOT use outside knowledge beyond what appears in the content
- Maximum 30 words
- Evaluation depth required: {diff_guide}
{code_rule}

PART 3 — Write the reference answer.
Rules:
- First sentence: state a clear position
- Next 2-3 sentences: give specific reasons supporting the position,
  referencing both Concept A and Concept B
- Final sentence: acknowledge one meaningful limitation or counterpoint
- Minimum 4 sentences total
- Use specific terms from the content

Answer rules:
- 2 - 4 complete sentences minimum
- Use specific terms and variables from the content
- Do NOT say "the content does not mention" — always provide an answer

IMPORTANT:
- The answer must be COMPLETE and FINAL
- TRY TO AVOID EXAMPLES not used in question
- Do NOT describe what the answer will be
- Do NOT say "the answer is based on..."
- Directly give the final answer

For code questions:
- Include the exact output (if applicable)
- Then explain briefly why

STRICT OUTPUT FORMAT:
-If you include anything outside JSON, the answer is WRONG
- - DO NOT include PART 1, PART 2, PART 3
- DO NOT INCLUDE YOUR REASONING BEHIND QUESTION FRAMING IN THE OUTPUT
- Output ONLY valid JSON
- Do NOT include any text before or after JSON
- Do NOT say "Here is the answer"
- Do NOT explain what you are doing
- Use double quotes only

Example:
{{"question": "...", "reference_answer": "..."}}
{avoid}

Return ONLY valid JSON, no markdown, no explanation:
{{"question": "...", "reference_answer": "..."}}"""

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

# def generate_question(topic, difficulty, question_type,
#                       question_count=0,
#                       asked_questions=None,
#                       prerequisites=None,
#                       concept_graph=None,
#                       used_chunk_ids = None):

#     if asked_questions is None:
#         asked_questions = []

#     # ─────────────────────────────────────────────
#     # Step 1: Retrieve chunks
#     # ─────────────────────────────────────────────
#     chunks, new_keys = retrieve_chunks(
#         topic,
#         difficulty,
#         question_type,
#         prerequisites=prerequisites,
#         concept_graph=concept_graph,
#         used_chunk_ids = used_chunk_ids
#     )

#     if not chunks:
#         return {"question": "No data", "reference_answer": "N/A", "question_type": question_type}

    
#     # texts = [
#     #     c["text"] for c in chunks
#     #     if len(c["text"]) > 50 and topic.lower() in c["text"].lower()
#     # ]

#     # if len(texts) < 2:
#     #     texts = [c["text"] for c in chunks if len(c["text"]) > 50]

#     # if len(texts) < 2:
#     #     return {"question": "Insufficient data", "reference_answer": "N/A", "question_type": question_type}

#     # # ─────────────────────────────────────────────
#     # # Step 2: Chunk rotation
#     # # ─────────────────────────────────────────────
#     # idx = question_count % len(texts)

#     # chunk1 = texts[idx]
#     # chunk2 = texts[(idx + 1) % len(texts)]
#     # chunk3 = texts[(idx + 2) % len(texts)]
    
#     groups = group_chunks_by_subtopic(chunks)
#     subtopics = list(groups.keys())

#     if len(subtopics) < 2:
#         selected = chunks[:3]
#     else:
#         s1 = subtopics[question_count % len(subtopics)]
#         s2 = subtopics[(question_count + 1) % len(subtopics)]
#         s3 = subtopics[(question_count + 2) % len(subtopics)]

#         # selected = [
#         #     groups[s1][0],
#         #     groups[s2][0],
#         #     groups[s3][0]
#         # ]
#         def select_from_group(group, question_count, slot_idx):
#             if not group:
#                 return None

#             idx = (question_count + slot_idx * 3) % len(group)
#             return group[idx]
#         selected = [
#             select_from_group(groups[s1], question_count, 0),
#             select_from_group(groups[s2], question_count, 1),
#             select_from_group(groups[s3], question_count, 2)
#         ]

#     # Extract text + metadata
#     chunk1, chunk2, chunk3 = selected

#     # PRIMARY chunks (decision making)
#     primary1, primary2, primary3 = chunk1, chunk2, chunk3

#     # GET NEIGHBORS (context only)
#     neighbors1 = get_neighbor_chunks(primary1)
#     neighbors2 = get_neighbor_chunks(primary2)
#     neighbors3 = get_neighbor_chunks(primary3)

#     # BUILD CONTEXT (merge)
#     def build_context(primary, neighbors):
#         texts = [primary["text"]]

#         for n in neighbors:
#             if n['text'] != primary['text']:
#                 texts.append(n["text"])

#         return "\n\n".join(texts)

#     text1 = build_context(primary1, neighbors1)
#     text2 = build_context(primary2, neighbors2)
#     text3 = build_context(primary3, neighbors3)

#     # IMPORTANT: metadata only from primary
#     meta1 = primary1
#     meta2 = primary2
#     meta3 = primary3
#     # ─────────────────────────────────────────────
#     # Step 3: Factual
#     # ─────────────────────────────────────────────
#     prob = np.random.uniform(0, 1)

#     if prob < 0.3 and question_type == "factual":

#         result = generate_mcq(text1, meta1, topic, difficulty, asked_questions)

#         if "options" in result:
#             # convert MCQ → normal format for pipeline
#             correct = result.get("correct_answer", "").lower()
#             options = result.get("options", {})

#             answer_text = options.get(correct, "")

#             result["reference_answer"] = answer_text

#         if not is_valid_mcq(result):
#             result = generate_factual(text1, meta1, topic, difficulty, asked_questions)

#     else:
#         result = generate_factual(text1, meta1, topic, difficulty, asked_questions)
    
#     if not validate(result):
#         result = generate_factual(text1, meta1, topic, difficulty, asked_questions)

#     if not is_answerable(text1, result["question"]):

#         print("[INFO] Answer not directly in content → using reasoning")

#         result["reference_answer"] = generate_answer_with_llm(
#             text1,
#             result["question"]
#         )

#     if question_type == "factual":
#         result["question_type"] = "factual"
#         return result, new_keys

#     # ─────────────────────────────────────────────
#     # Step 4: Inferential
#     # ─────────────────────────────────────────────

#     result_inf = rewrite_inferential(
#         result["question"],
#         result["reference_answer"],
#         text2,
#         meta2,
#         topic,
#         difficulty,
#         asked_questions
#     )

#     # retry once
#     if not validate(result_inf):
#         result_inf = rewrite_inferential(
#             result["question"],
#             result["reference_answer"],
#             text2,
#             meta2,
#             topic,
#             difficulty,
#             asked_questions
#         )
    
#     result_inf["reference_answer"] = generate_answer_with_llm(
#         text1 + "\n\n" + text2,
#         result_inf["question"]
#     )

#     # ── FINAL DECISION FOR INFERENTIAL ──
#     if question_type == "inferential":

#         if validate(result_inf):
#             result_inf["question_type"] = "inferential"
#             return result_inf, new_keys
#         else:
#             result["question_type"] = "factual"
#             return result, new_keys
#     # ─────────────────────────────────────────────
#     # Step 5: Evaluative
#     # ─────────────────────────────────────────────

#     result_eval = rewrite_evaluative(
#         result_inf["question"],
#         result_inf["reference_answer"],
#         text3,
#         meta3,
#         topic,
#         difficulty,
#         asked_questions
#     )

#     # retry once
#     if not validate(result_eval):
#         result_eval = rewrite_evaluative(
#             result_inf["question"],
#             result_inf["reference_answer"],
#             text3,
#             meta3,
#             topic,
#             difficulty,
#             asked_questions
#         )

    
#     result_eval["reference_answer"] = generate_answer_with_llm(
#         text1 + "\n\n" + text2 + "\n\n" + text3,
#         result_eval["question"]
#     )

#     # ── FINAL DECISION FOR EVALUATIVE ──
#     if validate(result_eval):
#         result_eval["question_type"] = "evaluative"
#         return result_eval, new_keys

#     if validate(result_inf):
#         result_inf["question_type"] = "inferential"
#         return result_inf, new_keys

#     result["question_type"] = "factual"
#     return result, new_keys
def generate_question(topic, difficulty, question_type,
                      question_count=0,
                      asked_questions=None,
                      prerequisites=None,
                      concept_graph=None,
                      used_chunk_ids=None):

    if asked_questions is None:
        asked_questions = []

    # ── Step 1: Retrieve chunks ───────────────────────────────────
    chunks, new_keys = retrieve_chunks(
        topic, difficulty, question_type,
        prerequisites=prerequisites,
        concept_graph=concept_graph,
        used_chunk_ids=used_chunk_ids
    )

    if not chunks:
        return {"question": "No data", "reference_answer": "N/A",
                "question_type": question_type}, []

    # ── Step 2: Select chunks by subtopic ────────────────────────
    groups    = group_chunks_by_subtopic(chunks)
    subtopics = list(groups.keys())

    def select_from_group(group, qcount, slot_idx):
        if not group:
            return None
        return group[(qcount + slot_idx * 3) % len(group)]

    if len(subtopics) < 2:
        selected = (chunks + chunks + chunks)[:3]
    else:
        s1 = subtopics[question_count % len(subtopics)]
        s2 = subtopics[(question_count + 1) % len(subtopics)]
        s3 = subtopics[(question_count + 2) % len(subtopics)]
        selected = [
            select_from_group(groups[s1], question_count, 0),
            select_from_group(groups[s2], question_count, 1),
            select_from_group(groups[s3], question_count, 2),
        ]

    chunk1, chunk2, chunk3 = selected

    # ── Step 3: Get neighbors in parallel ────────────────────────
    with ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(get_neighbor_chunks, chunk1)
        f2 = ex.submit(get_neighbor_chunks, chunk2)
        f3 = ex.submit(get_neighbor_chunks, chunk3)
        n1, n2, n3 = f1.result(), f2.result(), f3.result()

    def build_context(primary, neighbors):
        texts = [primary["text"]]
        for n in neighbors:
            if n["text"] != primary["text"]:
                texts.append(n["text"])
        return "\n\n".join(texts)

    text1 = build_context(chunk1, n1)
    text2 = build_context(chunk2, n2)
    text3 = build_context(chunk3, n3)
    meta1, meta2, meta3 = chunk1, chunk2, chunk3

    # ── Step 4: Factual — 1 LLM call ─────────────────────────────
    if question_type == "factual":

        prob = np.random.uniform(0, 1)

        if prob < 0.3 and question_type == "factual":
            # ── MCQ path ──────────────────────────────────────────────
            result = generate_mcq_v2(text1, meta1, topic, difficulty, asked_questions)

            if result is None:
                # MCQ generation failed — fall back to open-ended factual
                print("[INFO] MCQ generation failed → falling back to generate_factual_v2")
                result = generate_factual_v2(text1, meta1, topic, difficulty, asked_questions)

            # result already has reference_answer set to correct option text
            # by generate_mcq_v2, so no post-processing needed here

        else:
            # ── Open-ended factual path ───────────────────────────────
            result = generate_factual_v2(text1, meta1, topic, difficulty, asked_questions)

        if not validate(result):
            result = generate_factual_v2(text1, meta1, topic, difficulty, asked_questions)

        result["question_type"] = "factual"
        return result, new_keys


    elif question_type == "inferential":

        result = generate_inferential_v2(
            text1,   # 🔥 no dependency on factual anymore
            text2, meta2, topic, difficulty, asked_questions
        )

        if not validate(result):
            result = generate_inferential_v2(
                text1,
                text2, meta2, topic, difficulty, asked_questions
            )

        result["question_type"] = "inferential"
        return result, new_keys


    elif question_type == "evaluative":

        result = generate_evaluative_v2(
            text1, text2,   # 🔥 no dependency on inferential
            text3, meta3, topic, difficulty, asked_questions
        )

        if not validate(result):
            result = generate_evaluative_v2(
                text1, text2,
                text3, meta3, topic, difficulty, asked_questions
            )

        result["question_type"] = "evaluative"
        return result, new_keys

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
import time
if __name__ == "__main__":
    start_time = time.time()

    q, _ = generate_question(
        topic="Pointers to Class Members",
        difficulty="medium",
        question_type="evaluative",
        question_count=0,
        asked_questions=[]
    )
    end_time = time.time()

    print(f"\nTotal Execution Time: {round(end_time - start_time, 2)} seconds")
    print("\nQuestion:\n", q["question"])
    print("\nAnswer:\n", q["reference_answer"])