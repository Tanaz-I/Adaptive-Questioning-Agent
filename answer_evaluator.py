"""
Answer Evaluator (FINAL STABLE VERSION)
======================================

Features:
• Semantic similarity (MiniLM)
• Smart keyword coverage
• Heuristic NLI (logic check)
• Semantic completeness (accurate)
• Concept bonus
• Dynamic weights per question type
• Fully integrated pipeline
"""

import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────

print("Loading evaluation model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────

def preprocess(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower())


# ─────────────────────────────────────────────
# 1. Semantic Similarity
# ─────────────────────────────────────────────

def semantic_score(student, reference):
    e1 = model.encode([student])[0]
    e2 = model.encode([reference])[0]
    return cosine_similarity([e1], [e2])[0][0]


# ─────────────────────────────────────────────
# 2. Keyword Score (Improved)
# ─────────────────────────────────────────────

def keyword_score(student, reference):

    stopwords = {
        "the","is","a","an","and","or","to","of","from","by","it","this","that",
        "as","with","for","on","in","at","be","are"
    }

    ref = set(preprocess(reference).split()) - stopwords
    stu = set(preprocess(student).split()) - stopwords

    if not ref:
        return 0

    overlap = ref.intersection(stu)

    return (len(overlap) / len(ref)) ** 0.5


# ─────────────────────────────────────────────
# 3. Heuristic NLI
# ─────────────────────────────────────────────

def nli_score(student, reference):

    negatives = ["not", "no", "never", "none"]

    student = preprocess(student)
    reference = preprocess(reference)

    if any(n in student for n in negatives):
        if not any(n in reference for n in negatives):
            return 0.3

    return 1.0


# ─────────────────────────────────────────────
# 4. Semantic Completeness (FINAL FIX)
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
# 5. Concept Bonus
# ─────────────────────────────────────────────

def concept_bonus(student):

    concepts = ["inherit", "base", "derived", "reuse", "extend"]

    student = preprocess(student)

    count = sum(1 for c in concepts if c in student)

    return min(count / len(concepts), 1.0)


# ─────────────────────────────────────────────
# Final Evaluation
# ─────────────────────────────────────────────

def evaluate_answer(student, reference, question_type):

    s = semantic_score(student, reference)
    k = keyword_score(student, reference)
    n = nli_score(student, reference)
    c = completeness_score(student, reference)
    b = concept_bonus(student)

    # Dynamic weights
    if question_type == "factual":
        w = (0.5, 0.3, 0.1, 0.1)

    elif question_type == "inferential":
        w = (0.35, 0.2, 0.2, 0.15)

    elif question_type == "evaluative":
        w = (0.3, 0.1, 0.3, 0.2)

    else:
        w = (0.4, 0.2, 0.2, 0.2)

    final = (
        w[0]*s +
        w[1]*k +
        w[2]*n +
        w[3]*c +
        0.1*b
    )

    return {
        "semantic_score": round(s, 3),
        "keyword_score": round(k, 3),
        "nli_score": round(n, 3),
        "completeness_score": round(c, 3),
        "concept_bonus": round(b, 3),
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