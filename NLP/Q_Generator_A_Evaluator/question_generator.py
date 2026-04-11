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
import re

from concurrent.futures import ThreadPoolExecutor


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:instruct"

EVALUATIVE_FRAMES = [
    "Under what conditions would you prefer {A} over {B}? Justify your choice.",
    "A developer argues that {A} makes {B} unnecessary. Evaluate this claim.",
    "What is the most significant limitation of {A} that {B} addresses? Explain.",
    "When would using {A} actually worsen program behavior? Give a concrete scenario.",
    "Compare the design tradeoffs of {A} and {B} for a memory-constrained system.",
    "If you had to choose between {A} and {B} for a large codebase, which would you pick and why?",
]

def override_code_flag(text, meta):
    """
    Fix false positives in code detection
    """
    if not meta.get("contains_code"):
        return meta

    # Strong code indicators
    code_signals = ['{', '}', ';', '#include', 'class ', 'return', '::']
    score = sum(1 for s in code_signals if s in text)

    # If weak signal → downgrade
    if score < 3:
        meta["contains_code"] = False

    return meta

# ─────────────────────────────────────────────
# LLM Call
# ─────────────────────────────────────────────

def call_llm(prompt, temperature=0.6):
    import time
    start = time.time()

    print("\n[LLM CALL STARTED]")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
    )

    end = time.time()
    print(f"[LLM DONE] Time: {round(end - start, 2)} sec")


    response.raise_for_status()
    return response.json()["response"].strip()


# ─────────────────────────────────────────────
# Parse JSON
# ─────────────────────────────────────────────

def parse_json(output):
    # print(output)

    # ─────────────────────────────────────────────
    # Step 1: Extract ALL code blocks first
    # Code blocks belong in the QUESTION only, never in the answer.
    # ─────────────────────────────────────────────
    code_blocks_found = []

    def extract_and_replace_code(text):
        blocks = []
        pattern = re.compile(r'```(?:[a-zA-Z]*\n?)?(.*?)```', re.DOTALL)

        def replacer(match):
            code = match.group(1).strip()
            placeholder = f"__CODE_BLOCK_{len(blocks)}__"
            blocks.append(code)
            return placeholder

        cleaned = pattern.sub(replacer, text)
        return cleaned, blocks

    output_no_code, code_blocks_found = extract_and_replace_code(output)

    # ─────────────────────────────────────────────
    # Step 2: Restore helpers
    # Code is ONLY restored into the question, never the answer.
    # ─────────────────────────────────────────────

    def restore_code_in_question(text, blocks):
        """Restore code placeholders into question text with triple backticks."""
        for i, code in enumerate(blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            text = text.replace(placeholder, f"```\n{code}\n```")
        return text

    def strip_code_placeholders_from_answer(text, blocks):
        """
        Remove any code placeholders that leaked into the answer.
        The answer should be plain explanatory text only.
        """
        for i in range(len(blocks)):
            placeholder = f"__CODE_BLOCK_{i}__"
            text = text.replace(placeholder, "").strip()
        return text

    # ─────────────────────────────────────────────
    # Step 3: JSON parse on code-stripped output
    # ─────────────────────────────────────────────

    def try_parse_json_object(text):
        text = re.sub(r'^```json\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text.strip())

        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end <= 0:
            return None

        json_str = text[start:end]
        json_str = json_str.replace('"""', '"')

        def collapse_newlines_in_strings(s):
            result    = []
            in_string = False
            escape    = False
            for ch in s:
                if escape:
                    result.append(ch)
                    escape = False
                elif ch == '\\':
                    result.append(ch)
                    escape = True
                elif ch == '"' and not escape:
                    in_string = not in_string
                    result.append(ch)
                elif ch == '\n' and in_string:
                    result.append(' ')
                else:
                    result.append(ch)
            return ''.join(result)

        json_str = collapse_newlines_in_strings(json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Regex fallback for broken JSON structure
        q_match = re.search(
            r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str, re.DOTALL
        )
        a_match = re.search(
            r'"reference_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str, re.DOTALL
        )

        if q_match and a_match:
            return {
                "question"        : q_match.group(1).strip(),
                "reference_answer": a_match.group(1).strip()
            }

        return None

    # ── Attempt 1: parse with code replaced by placeholders ───────
    data = try_parse_json_object(output_no_code)

    if data:
        question = data.get("question", "").strip()
        answer   = data.get("reference_answer", "")
        options  = data.get("options", {})
        correct  = data.get("correct_answer", "")

        # Code goes into question ONLY
        question = restore_code_in_question(question, code_blocks_found)
        answer   = strip_code_placeholders_from_answer(answer, code_blocks_found)

        if not answer:
            for k, v in data.items():
                if isinstance(v, str) and len(v.split()) > 5 and k != "question":
                    answer += v
            # strip any code placeholders that came from fallback values too
            answer = strip_code_placeholders_from_answer(answer, code_blocks_found)

        result = {
            "question"        : question,
            "reference_answer": answer.strip()
        }
        for extra_key in ("options", "correct_answer"):
            if extra_key in data:
                result[extra_key] = data[extra_key]

        if options:
            result["options"]        = options
            result["correct_answer"] = correct

        return result

    # ── Attempt 2: parse raw output directly ──────────────────────
    data = try_parse_json_object(output)

    if data:
        question = data.get("question", "").strip()
        answer   = data.get("reference_answer", "")

        # Strip any raw backtick blocks from answer
        answer = re.sub(r'```(?:[a-zA-Z]*\n?)?.*?```', '', answer, flags=re.DOTALL).strip()

        if not answer:
            for k, v in data.items():
                if isinstance(v, str) and len(v.split()) > 5 and k != "question":
                    answer += re.sub(r'```(?:[a-zA-Z]*\n?)?.*?```', '', v, flags=re.DOTALL)
        result = {
            "question"        : question,
            "reference_answer": answer.strip()
        }
        for extra_key in ("options", "correct_answer"):
            if extra_key in data:
                result[extra_key] = data[extra_key]


        return result

    # ─────────────────────────────────────────────
    # Step 4: Full structural fallback
    # ─────────────────────────────────────────────

    print("[parse_json] JSON parse failed — using structural fallback extraction")

    question = ""
    answer   = ""

    # ── Extract question section ──────────────────────────────────
    q_section_match = re.search(
        r'"question"\s*:(.*?)"reference_answer"\s*:',
        output,
        re.DOTALL
    )

    if q_section_match:
        raw_q = q_section_match.group(1).strip()
        raw_q = re.sub(r'^"+', '', raw_q)
        raw_q = re.sub(r'"+\s*,?\s*$', '', raw_q).strip()

        # Restore code blocks into question
        raw_q = restore_code_in_question(raw_q, code_blocks_found)

        # Also catch raw code blocks still present in the question section
        # of the original output (case 2: code outside backticks)
        segment = q_section_match.group(1)
        inline_codes = re.findall(r'```(?:[a-zA-Z]*\n?)?(.*?)```', segment, re.DOTALL)
        for code in inline_codes:
            if code.strip() and f"```\n{code.strip()}\n```" not in raw_q:
                raw_q = f"```\n{code.strip()}\n```\n" + raw_q

        question = raw_q

    # ── Extract answer section ────────────────────────────────────
    a_section_match = re.search(
        r'"reference_answer"\s*:\s*"?(.*?)(?:"?\s*\}|$)',
        output,
        re.DOTALL
    )

    if a_section_match:
        raw_a = a_section_match.group(1).strip()
        raw_a = re.sub(r'^"+', '', raw_a)
        raw_a = re.sub(r'"+\s*\}?\s*$', '', raw_a).strip()

        # Strip ALL code blocks from answer — they don't belong here
        raw_a = re.sub(r'```(?:[a-zA-Z]*\n?)?.*?```', '', raw_a, flags=re.DOTALL).strip()
        raw_a = strip_code_placeholders_from_answer(raw_a, code_blocks_found)

        answer = raw_a

    # ── Final safety defaults ─────────────────────────────────────
    if not question:
        q_lines = [l.strip() for l in output.split('\n') if l.strip().endswith('?')]
        question = q_lines[0] if q_lines else "Error"
        if code_blocks_found and question != "Error":
            question = f"```\n{code_blocks_found[0]}\n```\n" + question

    if not answer:
        answer = "Error"

    return {
        "question"        : question.strip(),
        "reference_answer": answer.strip()
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
            if not is_valid_mcq(result):
                return False

    return True

  
def build_code_decision_instruction(chunk, meta):
    """
    Returns a prompt instruction block that tells the LLM to self-assess
    whether the content contains real executable code before using backticks.

    Replaces the old build_code_injection() approach for question prompts.
    The LLM decides — no hardcoded heuristics.
    """
    code_inj, code_rule = build_code_injection(chunk, meta)

    if not code_inj:
        # Metadata says no code — but still remind LLM not to wrap plain text
        return """
CODE DECISION (read before writing the question):
- Does the content contain actual executable code (C++, Python, Java etc.)? 
- If YES: include the relevant code snippet in the question using triple backticks.
- If NO (content has bullet points, rules, definitions, prose): 
  do NOT use triple backticks anywhere. Write a plain text question only.
""", ""

    # Metadata says code present — inject it but still let LLM verify
    return f"""
CODE DECISION (read before writing the question):
The content appears to contain code. Before using it:
- Ask yourself: is this actual executable/compilable code, or just
  formatted text (bullet points, rules, numbered lists)?
- If it is REAL CODE: include it verbatim in the question using triple backticks.
- If it is formatted TEXT mistakenly detected as code: 
  do NOT wrap it in backticks. Write a plain text question instead.

Candidate code block (use only if truly executable):
{code_inj}
""", code_rule

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
# def generate_chart_question(chunk_text: str, topic: str, difficulty: str, asked_questions: list) -> dict:
#     """
#     Generates a data-interpretation question from an extracted chart description.
#     """
#     avoid = build_avoid_str(asked_questions)

#     prompt = f"""You are an expert teacher creating a data interpretation question.

# Topic: {topic}
# Difficulty: {difficulty}

# Extracted chart data:
# {chunk_text}

# Write ONE question that requires the student to interpret the chart data.
# The question should:
# - Reference specific values, labels, or trends visible in the chart description
# - NOT be answerable without reading the chart data
# - Ask for interpretation, comparison, or trend analysis (not pure recall)

# Then write a reference answer using ONLY the information in the chart description above.

# {avoid}

# Return ONLY valid JSON:
# {{"question": "...", "reference_answer": "..."}}"""

#     return parse_json(call_llm(prompt, temperature=0.2))


# def generate_table_question(chunk_text: str, topic: str, difficulty: str, asked_questions: list) -> dict:
#     """
#     Generates a table-reading question from an extracted table (markdown format).
#     """
#     avoid = build_avoid_str(asked_questions)

#     prompt = f"""You are an expert teacher creating a question based on a data table.

# Topic: {topic}
# Difficulty: {difficulty}

# Extracted table:
# {chunk_text}

# Write ONE question that:
# - Asks the student to read, compare, or calculate from the table values
# - Cannot be answered without consulting the table
# - Is appropriate for difficulty level: {difficulty}

# Write a reference answer using the table data above.

# {avoid}

# Return ONLY valid JSON:
# {{"question": "...", "reference_answer": "..."}}"""

#     return parse_json(call_llm(prompt, temperature=0.2))

def generate_factual_v2(chunk, meta, topic, difficulty, asked_questions):
    avoid = build_avoid_str(asked_questions)
    code_decision, code_rule = build_code_decision_instruction(chunk, meta)

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

{code_decision}

Your job has two parts:

PART 1 — Write ONE factual question.
Rules:
- {diff_guide}
- Must NOT be MCQ
- Must end with a question mark
- Must be rooted in the content above
{code_rule}


- CRITICAL: Do NOT write "as shown in the code", "consider the following code",
  or any similar phrase UNLESS you are actually including a ``` code block
  in the question. If there is no real executable code to show, write the
  question in plain text without any reference to "the code".

PART 2 — Write the reference answer.
Answer strategy:
  Reason step by step using only what the content provides.
  Do NOT use outside knowledge.

Answer rules:
- 2-4 complete sentences minimum
- Use specific terms and variables from the content
- Do NOT say "the content does not mention" — always provide an answer
- The answer must be plain text only — NO code blocks, NO backticks in the answer
- The answer explains concepts in words, not by showing code

IMPORTANT:
- The answer must be COMPLETE and FINAL
- Do NOT describe what the answer will be
- Do NOT say "the answer is based on..."
- Directly give the final answer
- Generate exactly ONE question and its answer

Do NOT include: "Here is the question", "Here is the answer", "PART 1", "PART 2"

STRICT JSON RULES:
- Output ONLY valid JSON
- All values must be single-line strings
- Do NOT use triple quotes
- Do NOT include line breaks inside JSON values — replace with spaces
- The reference_answer value must NEVER contain backticks

{avoid}

Return ONLY valid JSON:
{{"question": "...", "reference_answer": "..."}}"""

    return parse_json(call_llm(prompt, temperature=0.1))

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

STRICT JSON RULES:
- All values must be single-line strings
- Do NOT use triple quotes
- Do NOT include line breaks inside JSON values
- Replace newlines with spaces
- Do NOT use ``` in output

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

    stem = data.get("question", "").strip()

    options_text = "\n".join(
        f"{key} {value}"
        for key, value in sorted(normalised_options.items())  # sorted: a) b) c) d)
    )

    data["question"] = f"{stem}\n\n{options_text}"
    return data

# ─────────────────────────────────────────────
# 2. Inferential Question (2-hop)
# ─────────────────────────────────────────────

def generate_inferential_v2(chunk1, chunk2, meta2, topic, difficulty, asked_questions):
    avoid = build_avoid_str(asked_questions)
    code_decision, code_rule = build_code_decision_instruction(chunk2, meta2)

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

{code_decision}

Your job has three parts:

PART 1 — Identify the connection (internal reasoning, do NOT include in output):
  What is the causal or functional relationship between the existing content
  and the new content? Be specific about how one affects, enables, requires,
  or constrains the other.

PART 2 — Write ONE inferential question.
Rules:
- Must start with: How, Why, Explain why, or What happens when
- Must REQUIRE the student to use the connection from Part 1
- Must NOT be answerable from either piece of content alone
- Must NOT be answerable with a simple definition
- Maximum 25 words
- Reasoning required: {diff_guide}
{code_rule}


- CRITICAL: Do NOT write "as shown in the code", "consider the following code",
  or any similar phrase UNLESS you are actually including a ``` code block
  in the question. If there is no real executable code to show, write the
  question in plain text without any reference to "the code".

PART 3 — Write the reference answer.
Rules:
- First sentence must state the connection explicitly
- 3-5 sentences total, explain the mechanism not just the conclusion
- Use specific terms from both content blocks
- The answer must be plain text only — NO code blocks, NO backticks in the answer
- The answer explains in words, not by showing code

Answer rules:
- 2-4 complete sentences minimum
- Do NOT say "the content does not mention" — always provide an answer
- The answer must be COMPLETE and FINAL
- Do NOT describe what the answer will be
- Generate exactly ONE question and its answer

Do NOT include: "Here is the question", "Here is the answer", "PART 1", "PART 2", "PART 3"

STRICT JSON RULES:
- Output ONLY valid JSON
- All values must be single-line strings
- Do NOT use triple quotes
- Do NOT include line breaks inside JSON values — replace with spaces
- The reference_answer value must NEVER contain backticks

{avoid}

Return ONLY valid JSON:
{{"question": "...", "reference_answer": "..."}}"""

    return parse_json(call_llm(prompt, temperature=0.15))

# ─────────────────────────────────────────────
# 3. Evaluative Question (3-hop)
# ─────────────────────────────────────────────

def generate_evaluative_v2(chunk1, chunk2, chunk3, meta3, topic, difficulty, asked_questions):
    import random
    avoid = build_avoid_str(asked_questions)
    code_decision, code_rule = build_code_decision_instruction(chunk3, meta3)

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

--- Existing concepts (assume students know these) ---
{chunk1}

{chunk2}

--- Additional concepts ---
{chunk3}

{code_decision}

Your job has three parts:

PART 1 — Extract two key technical concepts (internal reasoning, do NOT include in output):
  From the content above, identify exactly two technical concepts being
  discussed or compared. Call them Concept A and Concept B.

PART 2 — Write ONE evaluative question about those two concepts.
Use this style as inspiration: "{frame_hint}"
Rules:
- Must require the student to TAKE A POSITION, not just list facts
- Must be answerable only with deep understanding of both concepts
- Must NOT use outside knowledge beyond what appears in the content
- Maximum 30 words
- Evaluation depth required: {diff_guide}
{code_rule}


- CRITICAL: Do NOT write "as shown in the code", "consider the following code",
  or any similar phrase UNLESS you are actually including a ``` code block
  in the question. If there is no real executable code to show, write the
  question in plain text without any reference to "the code".


PART 3 — Write the reference answer.
Rules:
- First sentence: state a clear position
- Next 2-3 sentences: give specific reasons referencing both Concept A and Concept B
- Final sentence: acknowledge one meaningful limitation or counterpoint
- Minimum 4 sentences total
- Use specific terms from the content
- The answer must be plain text only — NO code blocks, NO backticks in the answer
- The answer explains in words, not by showing code

Answer rules:
- Do NOT say "the content does not mention" — always provide an answer
- The answer must be COMPLETE and FINAL
- Generate exactly ONE question and its answer

Do NOT include: "Here is the question", "Here is the answer", "PART 1", "PART 2", "PART 3"

STRICT JSON RULES:
- Output ONLY valid JSON
- All values must be single-line strings
- Do NOT use triple quotes
- Do NOT include line breaks inside JSON values — replace with spaces
- The reference_answer value must NEVER contain backticks

{avoid}

Return ONLY valid JSON:
{{"question": "...", "reference_answer": "..."}}"""

    return parse_json(call_llm(prompt, temperature=0.2))

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

    meta1 = override_code_flag(text1, meta1)
    meta2 = override_code_flag(text2, meta2)
    meta3 = override_code_flag(text3, meta3)

    print(meta2.get("contains_code"))

    # def get_chunk_type(chunk):
    #     return chunk.get("chunk_type") or chunk.get("meta", {}).get("chunk_type", "text")

    # top_type = get_chunk_type(chunk1)

    # if top_type == "chart" and question_type == "factual":
    #     result = generate_chart_question(text1, topic, difficulty, asked_questions)
    #     if validate(result):
    #         result["question_type"] = "factual"
    #         return result, new_keys

    # if top_type == "table" and question_type == "factual":
    #     result = generate_table_question(text1, topic, difficulty, asked_questions)
    #     if validate(result):
    #         result["question_type"] = "factual"
    #         return result, new_keys

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
            text1,   # no dependency on factual anymore
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
            text1, text2,   # no dependency on inferential
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
        question_type="inferential",
        question_count=0,
        asked_questions=[]
    )
    
    end_time = time.time()

    print(f"\nTotal Execution Time: {round(end_time - start_time, 2)} seconds")
    print("\nQuestion:\n", q["question"])
    print("\nAnswer:\n", q["reference_answer"])