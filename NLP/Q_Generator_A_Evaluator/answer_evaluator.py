"""
Answer Evaluator (GENERIC FINAL VERSION)
=======================================

• Works for ANY domain/topic
• Uses semantic similarity + keyword + NLI + completeness
• Uses NLTK stopwords
• No hardcoded concepts
"""

import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from math import log
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Auto-download stopwords if not present
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))
    
# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

print("Loading evaluation model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

STOPWORDS = set(stopwords.words('english'))


# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


# ─────────────────────────────────────────────
# 1. Semantic Similarity (CORE SIGNAL)
# ─────────────────────────────────────────────

def _semantic_score(student_emb, reference_emb):
    return float(cosine_similarity([student_emb], [reference_emb])[0][0])


# ─────────────────────────────────────────────
# 2. Keyword Coverage (GENERIC)
# ─────────────────────────────────────────────
def keyword_score(student, reference):
    ref_tokens = [w for w in preprocess(reference).split()
                  if w not in STOPWORDS and len(w) > 2]
    stu_words  = set(preprocess(student).split()) - STOPWORDS

    if not ref_tokens:
        return 0.0

    # Weight = inverse frequency within reference
    # Rare words in the reference are more "key" to its meaning
    freq   = Counter(ref_tokens)
    total  = len(ref_tokens)
    unique = set(ref_tokens)

    weights      = {w: log(total / freq[w] + 1) for w in unique}
    total_weight = sum(weights.values())

    if total_weight == 0:
        return 0.0

    covered_weight = sum(weights[w] for w in unique if w in stu_words)
    return min(covered_weight / total_weight, 1.0)

# ─────────────────────────────────────────────
# 3. Heuristic NLI (GENERIC)
# ─────────────────────────────────────────────

def nli_score(student, reference, student_emb=None, reference_emb=None):
    """
    Approximate NLI score.
    If embeddings are pre-computed, pass them in to avoid re-encoding.
    """
    NEGATION = {"not", "no", "never", "none", "incorrect", "wrong", "false"}
    s_words   = set(preprocess(student).split())
    r_words   = set(preprocess(reference).split()) - STOPWORDS
    r_negs    = r_words & NEGATION

    # Only flag contradiction if student negates a KEY reference term
    # (not just any word in the reference)
    if not r_negs:
        s_negs = s_words & NEGATION
        if s_negs:
            s_tokens = preprocess(student).split()
            for i, tok in enumerate(s_tokens):
                if tok in NEGATION and i + 1 < len(s_tokens):
                    if s_tokens[i + 1] in r_words:
                        return 0.25   # targeted contradiction

    # Use embedding similarity as entailment proxy
    if student_emb is None or reference_emb is None:
        embs = model.encode([student, reference])
        student_emb, reference_emb = embs[0], embs[1]

    sim = float(cosine_similarity([student_emb], [reference_emb])[0][0])

    if sim >= 0.75:   
        return 1.0
    elif sim >= 0.55: 
        return 0.75
    elif sim >= 0.40: 
        return 0.5
    elif sim >= 0.25: 
        return 0.3
    else:             
        return 0.15



# ─────────────────────────────────────────────
# 4. Semantic Completeness (GENERIC)
# ─────────────────────────────────────────────


def _completeness_score(student_emb, ref_sent_embs, threshold=0.50):
    if len(ref_sent_embs) == 0:
        return 0.0
    total = 0.0
    for ref_emb in ref_sent_embs:
        sim = float(cosine_similarity([student_emb], [ref_emb])[0][0])
        if sim >= 0.60:
            total += 1.0
        elif sim >= 0.45:
            total += 0.6    # partial credit
        elif sim >= 0.30:
            total += 0.2
    return total / len(ref_sent_embs)

def length_penalty(student, reference):
    """
    Multiplier in [0.55, 1.0] based on answer length relative to reference.
    Does not punish concise answers — only punishes very short ones.
    """
    s_words = len(student.split())
    r_words = len(reference.split())

    if r_words == 0:
        return 1.0

    ratio = s_words / r_words

    if ratio >= 0.40:    return 1.00   # wrote ≥40% as much → no penalty
    elif ratio >= 0.25:  return 0.85
    elif ratio >= 0.15:  return 0.70
    else:                return 0.55   # very short answer

def recalibrate(score, low=0.0, high=1.0):
    """
    Stretch scores from observed range [low, high] to [0, 1].
    Measure your actual score distribution first:
    - Run 30 evaluations manually, log final_score before recalibration
    - Set low = 10th percentile, high = 90th percentile of those scores
    Default values are starting estimates — tune after 30 real evaluations.
    """
    stretched = (score - low) / (high - low)
    return float(max(0.0, min(1.0, stretched)))

# ─────────────────────────────────────────────
# Error Taxonomy
# ─────────────────────────────────────────────

def _detect_code_in_answer(text: str) -> bool:
    CODE_SIGNALS = ['{', '}', ';', '::', '->', 'def ', 'class ', '#include', 'return ', 'void ']
    return sum(1 for s in CODE_SIGNALS if s in text) >= 2


def classify_error(student: str, reference: str, score: float,
                   semantic: float, keyword: float, nli: float) -> str:
    """
    Classify the student's error type using score thresholds and text signals.
    Returns one of: "none" | "off_topic" | "conceptual" | "partial" | "code_syntax" | "careless"
    """
    if score >= 0.75:
        return "none"

    s_words = set(preprocess(student).split()) - STOPWORDS
    r_words = set(preprocess(reference).split()) - STOPWORDS
    overlap = len(s_words & r_words) / max(len(r_words), 1)

    if semantic < 0.25 and overlap < 0.1:
        return "off_topic"

    if nli < 0.3 and overlap > 0.2:
        # Student uses right words but contradicts the reference
        return "conceptual"

    if _detect_code_in_answer(student) and score < 0.5:
        return "code_syntax"

    if keyword < 0.3 and semantic > 0.4:
        return "partial"   # understands but uses wrong terminology

    if score < 0.5:
        return "partial"

    return "careless"


def generate_feedback(student: str, reference: str,
                      error_type: str, question: str, score: float) -> str:
    """
    Generate concise textual feedback. Pure heuristic — no LLM call.
    """
    if score >= 0.85:
        return "Excellent — all key points covered."

    ref_words   = [w for w in preprocess(reference).split()
                   if w not in STOPWORDS and len(w) > 3]
    stu_words   = set(preprocess(student).split())
    missing     = [w for w in dict.fromkeys(ref_words) if w not in stu_words][:5]
    missing_str = ", ".join(missing) if missing else "key concepts"

    templates = {
        "off_topic":   f"Your answer doesn't address the question. Re-read: {question[:80]}",
        "conceptual":  f"Core concept misunderstood. Review: {missing_str}.",
        "partial":     f"Partially correct. Missing details on: {missing_str}.",
        "code_syntax": "Code logic may be right but contains syntax errors. Review carefully.",
        "careless":    f"Almost there — improve precision on: {missing_str}.",
    }
    return templates.get(error_type, f"Review the concept — missing: {missing_str}.")

# ─────────────────────────────────────────────
# Final Evaluation (GENERIC + ADAPTIVE)
# ─────────────────────────────────────────────

def evaluate_answer(student, reference, question_type, question):
    def is_question_rewrite(student: str, question: str) -> bool:
        s = set(student.lower().split())
        q = set(question.lower().split())

        overlap = len(s & q) / max(len(q), 1)

        return overlap > 0.8
    if is_question_rewrite(student, question):
        return {
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "nli_score": 0.0,
            "completeness_score": 0.0,
            "length_penalty": 0.0,
            "final_score": 0.0,
            "error_type": "conceptual",    
            "feedback": "Seems like question was paraphrased", 
        }
    # Tokenize reference into sentences
    ref_sentences = [s.strip() for s in reference.split(".")
                     if len(s.strip()) > 5]
    if not ref_sentences:
        ref_sentences = [reference]

    # ONE batch encode call for everything
    all_texts  = [student, reference] + ref_sentences
    embeddings = model.encode(all_texts, batch_size=32)

    student_emb   = embeddings[0]
    reference_emb = embeddings[1]
    ref_sent_embs = embeddings[2:]

    # Compute all scores from pre-computed embeddings
    s  = _semantic_score(student_emb, reference_emb)
    k  = keyword_score(student, reference)
    n  = nli_score(student, reference, student_emb, reference_emb)
    c  = _completeness_score(student_emb, ref_sent_embs)
    lp = length_penalty(student, reference)

    # Weights by question type
    W = {
        "factual":     (0.45, 0.30, 0.10, 0.15),
        "inferential": (0.35, 0.20, 0.15, 0.30),
        "evaluative":  (0.25, 0.15, 0.20, 0.40),
    }.get(question_type, (0.40, 0.20, 0.15, 0.25))

    raw   = W[0]*s + W[1]*k + W[2]*n + W[3]*c
    final = recalibrate(raw * lp)

    error_type = classify_error(student, reference, round(final, 3), s, k, n)
    feedback   = generate_feedback(student, reference, error_type, question, round(final, 3))

    return {
        "semantic_score":     round(s,  3),
        "keyword_score":      round(k,  3),
        "nli_score":          round(n,  3),
        "completeness_score": round(c,  3),
        "length_penalty":     round(lp, 3),
        "final_score":        round(final, 3),
        "error_type":         error_type,    
        "feedback":           feedback, 
    }



# ─────────────────────────────────────────────
# Integrated Pipeline Test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    from question_generator import generate_question

    q, _ = generate_question("Pointers to Class Members", "medium", "inferential")

    print("\nQuestion:\n", q["question"])

    print("\nEnter your answer:\n")
    student = input("> ")

    result = evaluate_answer(
        student,
        q["reference_answer"],
        q["question_type"],
        q["question"]
    )

    print("\nEvaluation Result:\n")

    for k, v in result.items():
        print(f"{k}: {v}")