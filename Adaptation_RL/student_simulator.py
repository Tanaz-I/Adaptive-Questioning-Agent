import requests
import random

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"


PERSONAS = {
    "strong": {
        "description": "a high-performing student who deeply understands the subject",
        "traits": [
            "gives precise, complete answers using correct technical terminology",
            "covers all key points from the question",
            "occasionally adds a relevant example or elaboration",
            "rarely makes conceptual errors",
        ],
        # probability of a good answer per difficulty
        "accuracy": {"easy": 0.95, "medium": 0.85, "hard": 0.70,
                     "basic": 0.95, "intermediate": 0.85, "advanced": 0.70},
        # how much of the reference answer to use as a basis (higher = closer match)
        "coverage": 0.90,
        "temperature": 0.5,
    },
    "weak": {
        "description": "a struggling student who has partial or surface-level understanding",
        "traits": [
            "gives vague or incomplete answers",
            "often misses key technical details",
            "sometimes confuses related concepts",
            "uses informal or imprecise language",
            "may answer a simpler version of the question",
        ],
        "accuracy": {"easy": 0.60, "medium": 0.35, "hard": 0.15,
                     "basic": 0.60, "intermediate": 0.35, "advanced": 0.15},
        "coverage": 0.40,
        "temperature": 0.75,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: call Ollama
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt, temperature = 0.6):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 300},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a "degraded" version of the reference for the weak student
# ─────────────────────────────────────────────────────────────────────────────

def _degrade_reference(reference_answer, coverage):
    """Return only the first `coverage` fraction of the reference answer sentences."""
    sentences = [s.strip() for s in reference_answer.replace("\n", " ").split(".") if s.strip()]
    keep = max(1, int(len(sentences) * coverage))
    return ". ".join(sentences[:keep]) + "."

# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedStudent:
    """
    Simulates a student (weak or strong) answering questions generated
    by the RL + RAG pipeline.

    Parameters
    ----------
    student_type : str
        "strong" or "weak"
    seed : int | None
        Random seed for reproducibility (optional)
    """

    def __init__(self, student_type = "strong", seed = None):
        #assert student_type in PERSONAS, f"student_type must be 'strong' or 'weak', got '{student_type}'"
        self.student_type = student_type
        self.persona      = PERSONAS[student_type]
        if seed is not None:
            random.seed(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def answer(
        self,
        question,
        reference_answer,
        topic,
        difficulty,
        question_type,
    ):
        """
        Generate a simulated student answer.

        Parameters
        ----------
        question         : the question text
        reference_answer : the correct reference (used as hidden context)
        topic            : topic name  (e.g. "Pointers")
        difficulty       : "easy" / "medium" / "hard"  (or basic/intermediate/advanced)
        question_type    : "factual" / "inferential" / "evaluative"

        Returns
        -------
        str : simulated student answer
        """
        accuracy_prob = self.persona["accuracy"].get(difficulty, 0.5)

        # decide whether this attempt is "good" or "bad" at a high level
        is_good_attempt = random.random() < accuracy_prob

        if is_good_attempt:
            return self._generate_good_answer(question, reference_answer, topic, difficulty, question_type)
        else:
            return self._generate_poor_answer(question, reference_answer, topic, difficulty, question_type)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal generators
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_good_answer(self, question, reference_answer, topic, difficulty, qtype):
        traits = "\n".join(f"- {t}" for t in self.persona["traits"])

        prompt = f"""You are simulating {self.persona['description']}.

Topic: {topic}
Difficulty: {difficulty}
Question Type: {qtype}

Question:
{question}

Reference (hidden from the student — use it as the knowledge source):
{reference_answer}

Student traits:
{traits}

Write the student's answer. It should reflect a {self.student_type} student
who {'understands most of this material' if self.student_type == 'strong' else 'has partial understanding'}.

Rules:
- Write ONLY the student's answer, no preamble
- Do NOT say "as a student" or refer to yourself
- Do NOT copy the reference verbatim — rephrase in the student's own words
- Length: 2-4 sentences
"""
        return _call_llm(prompt, temperature=self.persona["temperature"])

    def _generate_poor_answer(self, question, reference_answer, topic, difficulty, qtype):
        # give the weak student only a partial slice of the reference
        degraded = _degrade_reference(reference_answer, self.persona["coverage"])

        error_types = random.sample([
            "confuse a related but incorrect concept",
            "give only a surface-level definition without explanation",
            "omit the most important part of the answer",
            "use an incorrect example",
            "mix up cause and effect",
            "answer a simpler version of the question",
        ], k=2)

        error_instruction = " and ".join(error_types)

        prompt = f"""You are simulating {self.persona['description']}.

Topic: {topic}
Difficulty: {difficulty}
Question Type: {qtype}

Question:
{question}

Partial knowledge the student has:
{degraded}

The student should {error_instruction}.

Write the student's answer. It should reflect a struggling student.

Rules:
- Write ONLY the student's answer, no preamble
- Do NOT say "as a student" or refer to yourself
- Keep it short (1-3 sentences)
- Sound genuine, not deliberately wrong
"""
        return _call_llm(prompt, temperature=self.persona["temperature"])