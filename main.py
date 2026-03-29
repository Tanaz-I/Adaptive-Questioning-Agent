import chromadb
from chromadb.config import Settings
import json
import requests
from collections import Counter, defaultdict
from Adaptation_RL.Agent import AdaptiveAgent
from NLP import knowledge_base_construction, enrich_metadata, rag_query_engine, topic_extraction
from NLP.Q_Generator_A_Evaluator.answer_evaluator import evaluate_answer
from NLP.Q_Generator_A_Evaluator.question_generator import generate_question
from NLP.concept_graph import build_concept_graph

DOCS_DIR        = "./contents"
CHROMA_DB_DIR   = "./chroma_db"
COLLECTION_NAME = "rag_kb"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llama3:8b"

# ─────────────────────────────────────────────
# Step 1 — Build Knowledge Base
# ─────────────────────────────────────────────

knowledge_base_construction.run_pipeline(DOCS_DIR)
enrich_metadata.enrich_metadata()
topic_extraction.run_global_topic_extraction()

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection(COLLECTION_NAME)
all_meta   = collection.get(include=["metadatas"])["metadatas"]

course, level = enrich_metadata.detect_course(collection)

# ─────────────────────────────────────────────
# Step 2 — Extract Canonical Topics
# ─────────────────────────────────────────────

all_meta = collection.get(include=["metadatas"])["metadatas"]
canonical_topics = sorted(set(
    m["topic"]
    for m in all_meta
    if m.get("topic") not in (None, "", "Unknown")
))

if not canonical_topics:
    print("[ERROR] No canonical topics found after normalization.")
    raise SystemExit(1)

print(f"Canonical topics: {canonical_topics}")

# ─────────────────────────────────────────────
# Step 3 — Build Topic → Subtopics Context
# ─────────────────────────────────────────────

topic_subtopics = defaultdict(set)
for m in all_meta:
    t = m.get("topic", "")
    s = m.get("subtopic", "")
    if t and t != "Unknown" and s and s != "Unknown":
        topic_subtopics[t].add(s)

topics_with_context = {t: sorted(topic_subtopics[t]) for t in canonical_topics}

# ─────────────────────────────────────────────
# Step 4 — LLM Infers Prerequisites
# ─────────────────────────────────────────────

valid_topics_str = json.dumps(canonical_topics, indent=2)

prompt = f"""You are an expert in "{course}" at the {level} level.

These are the ONLY valid topic names:
{valid_topics_str}

Each topic covers these concepts:
{json.dumps(topics_with_context, indent=2)}

Task: For each topic, list which OTHER topics from the valid list above must be learned first.

Rules (STRICT):
- Prerequisite values MUST be copied EXACTLY from the valid topics list above
- Do NOT use subtopic names as prerequisites
- Do NOT invent new topic names
- Do NOT include a topic as its own prerequisite
- Only direct prerequisites (not transitive)
- If none, use []

Return ONLY a JSON object where every key is from the valid topics list.
Return ONLY valid JSON. No explanation, no markdown.
No unnecessary statements. Give only the JSON.

JSON:"""

response = requests.post(
    OLLAMA_URL,
    json={
        "model" : OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        },
    },
    timeout=60,
)

def safe_parse_json(raw, fallback):
    raw   = raw.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start != -1 and end > 0:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    print(f"  [WARN] JSON parse failed. Raw:\n{raw[:300]}")
    return fallback

dependencies = safe_parse_json(response.json()["response"], fallback={})
print(f"Prerequisites: {dependencies}")

data = collection.get(include=["documents", "metadatas"])
filtered_chunks = []

for doc, meta in zip(data["documents"], data["metadatas"]):

    if meta.get("topic") in (None, "", "Unknown"):
        continue

    # prioritize useful chunks
    if meta.get("concept_type") in ["definition", "explanation", "example"]:
        filtered_chunks.append({
            "text": doc,
            "topic": meta.get("topic"),
            "subtopic": meta.get("subtopic")
        })

filtered_chunks = filtered_chunks[:150]

concept_graph = build_concept_graph(filtered_chunks)
print(f"[Graph] Nodes: {len(concept_graph)}")

sample_keys = list(concept_graph.keys())[:5]
print("[Graph Sample]:", sample_keys)

print("[Graph] Done.\n")

# ─────────────────────────────────────────────
# Step 5 — Compute Topic Difficulty (mode)
# ─────────────────────────────────────────────

topic_difficulties_raw = {}
for m in all_meta:
    topic      = m.get("topic", "")
    difficulty = m.get("difficulty", "")
    if not topic or topic == "Unknown" or not difficulty:
        continue
    topic_difficulties_raw.setdefault(topic, []).append(difficulty)

valid_topics = set(topic_difficulties_raw.keys())

clean_dependencies = {}

for topic, prereqs in dependencies.items():
    filtered = [p for p in prereqs if p in valid_topics and p != topic]
    clean_dependencies[topic] = filtered

dependencies = clean_dependencies

print("\n[Fixed Prerequisites]:", dependencies)

# map NLP difficulty labels → RL difficulty labels
diff_map_nlp_to_rl = {
    "easy"  : "basic",
    "medium": "intermediate",
    "hard"  : "advanced",
    # in case already in RL format
    "basic"       : "basic",
    "intermediate": "intermediate",
    "advanced"    : "advanced"
}

topics_difficulty = {
    topic: diff_map_nlp_to_rl.get(
        Counter(diffs).most_common(1)[0][0], "intermediate"
    )
    for topic, diffs in topic_difficulties_raw.items()
}
topics_difficulty = dict(sorted(topics_difficulty.items()))
print(f"Topic difficulties: {topics_difficulty}")

# ─────────────────────────────────────────────
# Step 6 — Initialize RL Agent (pretrains here)
# ─────────────────────────────────────────────

print("\nInitializing RL Agent (pretraining in progress)...")
rl_agent = AdaptiveAgent(
    topics_difficulty=topics_difficulty,
    prerequisites=dependencies,
    w1=0.4, w2=0.5, w3=0.1
)
print("RL Agent ready.\n")

# ─────────────────────────────────────────────
# Step 7 — Session Loop
# ─────────────────────────────────────────────

# reverse map for question generator
diff_map_rl_to_nlp = {
    'basic'        : 'easy',
    'intermediate' : 'medium',
    'advanced'     : 'hard'
}

N_QUESTIONS  = 10
session_log  = []
combo_question_count = {}
asked_questions_log = {}

for step in range(N_QUESTIONS):

    print(f"\n{'='*60}")
    print(f"Question {step + 1} / {N_QUESTIONS}")
    print(f"{'='*60}")

    # ── 7a. RL Agent selects action ──────────────────────────
    state_vector       = rl_agent.ks.get_state_vector()
    action_idx, _, _   = rl_agent.select_action(state_vector, training=False)
    topic, diff, qtype = rl_agent.mdp.decode(action_idx)

    print(f"Topic      : {topic}")
    print(f"Difficulty : {diff}")
    print(f"Type       : {qtype}")

    # ── 7b. RAG generates question ───────────────────────────
    combo_key = (topic, diff, qtype)

    question_count = combo_question_count.get(combo_key, 0)
    asked = asked_questions_log.get(topic, [])
    nlp_diff = diff_map_rl_to_nlp[diff]
    result = generate_question(
        topic,
        nlp_diff,
        qtype,
        question_count=question_count,
        asked_questions=asked,
        prerequisites=dependencies,
        concept_graph=concept_graph
    )

    question         = result['question']
    reference_answer = result['reference_answer']

    if question in ("No data", "Insufficient data", "Error"):
        print(f"[WARN] Could not generate question for {topic}/{diff}/{qtype}. Skipping.")
        continue

    print(f"\nQuestion:\n{question}")
    print(f"\n[Reference Answer]:\n{reference_answer}")

    # ── 7c. Get student answer ───────────────────────────────
    print("\nYour answer:")
    student_answer = input("> ").strip()

    if not student_answer:
        print("No answer provided. Skipping.")
        continue

    # ── 7d. NLP evaluates answer ─────────────────────────────
    eval_result = evaluate_answer(student_answer, reference_answer, qtype)
    score       = eval_result['final_score']

    print(f"\nEvaluation:")
    print(f"  Semantic    : {eval_result['semantic_score']}")
    print(f"  Keyword     : {eval_result['keyword_score']}")
    print(f"  NLI         : {eval_result['nli_score']}")
    print(f"  Completeness: {eval_result['completeness_score']}")
    print(f"  Final Score : {score}")

    # ── 7e. Update RL Agent with NLP score ───────────────────
    reward = rl_agent.update(topic, score, diff, qtype)
    print(f"  RL Reward   : {round(reward, 3)}")

    # ── 7f. Show mastery status ──────────────────────────────
    mastered   = [t for t in rl_agent.ks.topics if rl_agent.ks.is_mastered(t)]
    unlocked   = [t for t in rl_agent.ks.topics if rl_agent.ks.prerequisites_met(t)]
    print(f"\nUnlocked topics : {unlocked}")
    print(f"Mastered topics : {mastered if mastered else 'None yet'}")

    # ── 7g. Log ──────────────────────────────────────────────
    session_log.append({
        'step'          : step + 1,
        'topic'         : topic,
        'difficulty'    : diff,
        'question_type' : qtype,
        'question'      : question,
        'student_answer': student_answer,
        'score'         : score,
        'reward'        : reward,
        'mastered'      : mastered[:]
    })

    combo_question_count[combo_key] = question_count + 1

    if topic not in asked_questions_log:
        asked_questions_log[topic] = []

    asked_questions_log[topic].append(result["question"])
# ─────────────────────────────────────────────
# Step 8 — Session Summary
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print("SESSION COMPLETE")
print(f"{'='*60}")

for topic in rl_agent.ks.topics:
    ks = rl_agent.ks
    print(f"\n{topic}:")
    print(f"  Attempts   : {ks.attempts[topic]}")
    print(f"  Avg Score  : {round(ks.topic_score[topic], 3)}")
    print(f"  Mastered   : {ks.is_mastered(topic)}")
    
