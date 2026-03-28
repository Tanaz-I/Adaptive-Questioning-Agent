from retrieval_engine import retrieve_chunks
import random

previous_questions = set()


# ─────────────────────────────────────────────
# CLEAN QUERY → EXTRACT TOPIC DYNAMICALLY
# ─────────────────────────────────────────────
def extract_query_parts(query: str):

    query = query.lower()

    # Difficulty
    if "easy" in query:
        difficulty = "easy"
    elif "hard" in query:
        difficulty = "hard"
    else:
        difficulty = "medium"

    # Question type
    if "inferential" in query:
        qtype = "inferential"
    elif "evaluative" in query:
        qtype = "evaluative"
    else:
        qtype = "factual"

    # Remove noise words → keep actual topic
    noise_words = [
        "easy", "medium", "hard",
        "question", "generate", "create",
        "inferential", "evaluative", "factual",
        "on", "about", "the", "of", "and", "its"
    ]

    words = query.split()
    topic_words = [w for w in words if w not in noise_words]

    topic = " ".join(topic_words)

    # fallback
    if not topic.strip():
        topic = query

    return topic, difficulty, qtype


# ─────────────────────────────────────────────
# GENERATE QUESTION
# ─────────────────────────────────────────────
def generate_question(query: str):

    topic, difficulty, qtype = extract_query_parts(query)

    # 🔥 PASS FULL TOPIC TO RETRIEVAL
    chunks = retrieve_chunks(topic, difficulty)

    # fallback 1
    if not chunks:
        chunks = retrieve_chunks(topic, "medium")

    # fallback 2
    if not chunks:
        chunks = retrieve_chunks(query, "medium")

    # final fallback
    if not chunks:
        return {
            "question": f"Explain {topic}.",
            "reference_answer": f"{topic} is an important concept.",
            "question_type": qtype,
            "difficulty": difficulty,
            "topic": topic
        }

    chunk = chunks[0]
    text = chunk["text"]

    reference_answer = text.strip().replace("\n", " ")[:400]

    # ───────── DYNAMIC QUESTION GENERATION ─────────

    if qtype == "factual":
        templates = [
            f"What is {topic}?",
            f"Define {topic}.",
            f"Explain {topic}.",
        ]

    elif qtype == "inferential":
        templates = [
            f"How does {topic} work?",
            f"Why is {topic} important?",
            f"Explain how {topic} is used in practice.",
        ]

    else:
        templates = [
            f"Evaluate the importance of {topic}.",
            f"Discuss advantages and limitations of {topic}.",
            f"Compare different aspects of {topic}.",
        ]

    question = random.choice(templates)

    # avoid duplicates
    if question in previous_questions:
        question += " (Explain briefly)"

    previous_questions.add(question)

    return {
        "question": question,
        "reference_answer": reference_answer,
        "question_type": qtype,
        "difficulty": difficulty,
        "topic": topic
    }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":

    query = "medium inferential question on graph traversal algorithms"

    q = generate_question(query)

    print("\nTopic:", q["topic"])
    print("\nQuestion:", q["question"])
    print("\nAnswer:", q["reference_answer"])