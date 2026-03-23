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

def semantic_score(student, reference):
    emb1 = model.encode([student])[0]
    emb2 = model.encode([reference])[0]
    return cosine_similarity([emb1], [emb2])[0][0]


# ─────────────────────────────────────────────
# 2. Keyword Coverage (GENERIC)
# ─────────────────────────────────────────────

def keyword_score(student, reference):

    ref_words = set(preprocess(reference).split()) - STOPWORDS
    stu_words = set(preprocess(student).split()) - STOPWORDS

    if not ref_words:
        return 0

    overlap = ref_words.intersection(stu_words)

    return (len(overlap) / len(ref_words)) ** 0.5


# ─────────────────────────────────────────────
# 3. Heuristic NLI (GENERIC)
# ─────────────────────────────────────────────

def nli_score(student, reference):

    negatives = {"not", "no", "never", "none"}

    student_words = set(preprocess(student).split())
    reference_words = set(preprocess(reference).split())

    # contradiction check
    if student_words.intersection(negatives) and not reference_words.intersection(negatives):
        return 0.3

    return 1.0


# ─────────────────────────────────────────────
# 4. Semantic Completeness (GENERIC)
# ─────────────────────────────────────────────

def completeness_score(student, reference):

    ref_sentences = [s.strip() for s in reference.split(".") if s.strip()]

    student_emb = model.encode([student])[0]

    covered = 0

    for sent in ref_sentences:
        sent_emb = model.encode([sent])[0]

        sim = cosine_similarity([student_emb], [sent_emb])[0][0]

        # semantic threshold
        if sim > 0.6:
            covered += 1

    return covered / max(len(ref_sentences), 1)


# ─────────────────────────────────────────────
# Final Evaluation (GENERIC + ADAPTIVE)
# ─────────────────────────────────────────────

def evaluate_answer(student, reference, question_type):

    s = semantic_score(student, reference)
    k = keyword_score(student, reference)
    n = nli_score(student, reference)
    c = completeness_score(student, reference)

    # Dynamic weights
    if question_type == "factual":
        weights = (0.5, 0.3, 0.1, 0.1)

    elif question_type == "inferential":
        weights = (0.4, 0.2, 0.2, 0.2)

    elif question_type == "evaluative":
        weights = (0.3, 0.1, 0.3, 0.3)

    else:
        weights = (0.4, 0.2, 0.2, 0.2)

    final = (
        weights[0] * s +
        weights[1] * k +
        weights[2] * n +
        weights[3] * c
    )

    return {
        "semantic_score": round(s, 3),
        "keyword_score": round(k, 3),
        "nli_score": round(n, 3),
        "completeness_score": round(c, 3),
        "final_score": round(final, 3)
    }


# ─────────────────────────────────────────────
# Integrated Pipeline Test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    from question_generator import generate_question

    q = generate_question("Inheritance", "medium", "inferential")

    print("\nQuestion:\n", q["question"])

    print("\nEnter your answer:\n")
    student = input("> ")

    result = evaluate_answer(
        student,
        q["reference_answer"],
        q["question_type"]
    )

    print("\nEvaluation Result:\n")

    for k, v in result.items():
        print(f"{k}: {v}")