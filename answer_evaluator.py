"""
Answer Evaluator (FINAL ADVANCED VERSION)
========================================

Includes:
✔ Semantic similarity
✔ Keyword overlap (with stopwords removal)
✔ NLI-style graded scoring
✔ Completeness check
✔ Copy-paste detection
"""

import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

try:
    STOPWORDS = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

model = SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────

def preprocess(text):
    words = text.lower().split()
    return [w for w in words if w not in STOPWORDS]


# ─────────────────────────────────────────────
# Keyword Score
# ─────────────────────────────────────────────

def keyword_score(student, reference):

    s_words = set(preprocess(student))
    r_words = set(preprocess(reference))

    if not r_words:
        return 0.0

    overlap = s_words.intersection(r_words)

    return len(overlap) / len(r_words)


# ─────────────────────────────────────────────
# Semantic Score
# ─────────────────────────────────────────────

def semantic_score(student, reference):

    emb1 = model.encode(student, convert_to_tensor=True)
    emb2 = model.encode(reference, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()

    return float(score)


# ─────────────────────────────────────────────
# NLI Score (Improved)
# ─────────────────────────────────────────────

def nli_score(semantic, keyword):

    # Combine signals instead of binary
    score = (0.7 * semantic) + (0.3 * keyword)

    return float(score)


# ─────────────────────────────────────────────
# Completeness Score
# ─────────────────────────────────────────────

def completeness_score(student, reference):

    student_words = student.strip().split()
    reference_words = reference.strip().split()

    # If too short → very low
    if len(student_words) < 3:
        return 0.1

    len_ratio = len(student_words) / max(len(reference_words), 1)

    # Check meaningful words (remove repeated chars)
    unique_words = set(student_words)

    # If all words same (like "fffff")
    if len(unique_words) == 1:
        return 0.1

    # Normal scoring
    if len_ratio > 0.9:
        return 1.0
    elif len_ratio > 0.7:
        return 0.8
    elif len_ratio > 0.5:
        return 0.6
    elif len_ratio > 0.3:
        return 0.4
    else:
        return 0.2


# ─────────────────────────────────────────────
# Copy Detection
# ─────────────────────────────────────────────

def detect_copy(student, reference, semantic):

    # Exact copy
    if student.strip().lower() == reference.strip().lower():
        return True, "⚠️ Exact copy detected!"

    # High similarity + very long answer
    if semantic > 0.95 and len(student) > 0.8 * len(reference):
        return True, "⚠️ Answer appears copied or paraphrased heavily!"

    return False, ""


# ─────────────────────────────────────────────
# Final Evaluation
# ─────────────────────────────────────────────

def evaluate_answer(student, reference, question_type):

    sem = semantic_score(student, reference)
    key = keyword_score(student, reference)
    nli = nli_score(sem, key)
    comp = completeness_score(student, reference)

    # Copy detection
    copied, warning = detect_copy(student, reference, sem)

    # ─────────────────────────────────────────
    # Dynamic Weighting by Question Type
    # ─────────────────────────────────────────

    if question_type == "factual":
        final = (0.4 * key) + (0.4 * sem) + (0.2 * comp)

    elif question_type == "inferential":
        final = (0.5 * sem) + (0.2 * key) + (0.3 * comp)

    elif question_type == "evaluative":
        final = (0.6 * sem) + (0.2 * key) + (0.2 * comp)

    else:
        final = (0.4 * sem) + (0.3 * key) + (0.3 * comp)

    # Penalize if copied
    if copied:
        final *= 0.6

    return {
        "semantic_score": float(round(sem, 3)),
        "keyword_score": float(round(key, 3)),
        "nli_score": float(round(nli, 3)),
        "completeness_score": float(round(comp, 3)),
        "final_score": float(round(final, 3)),
        "copied": copied,
        "warning": warning
    }