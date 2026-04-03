"""
run_simulation.py
=================
Drop-in replacement for the session loop in main.py.
Runs N_QUESTIONS automatically using a simulated student (no manual input).

Usage:
    python run_simulation.py --student weak  --questions 500
    python run_simulation.py --student strong --questions 1000
    python run_simulation.py --student both   --questions 500   # runs both back-to-back
"""

import argparse
import json
import csv
import os
from collections import Counter, defaultdict

import chromadb
import requests
from chromadb.config import Settings

from Adaptation_RL.Agent import AdaptiveAgent
from NLP import knowledge_base_construction, enrich_metadata, rag_query_engine, topic_extraction
from NLP.Q_Generator_A_Evaluator.answer_evaluator import evaluate_answer
from NLP.Q_Generator_A_Evaluator.question_generator import generate_question
from NLP.concept_graph import build_concept_graph
from Adaptation_RL.student_simulator import SimulatedStudent          # ← new import

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DOCS_DIR        = "./contents"
CHROMA_DB_DIR   = "./chroma_db"
COLLECTION_NAME = "rag_kb"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llama3:8b"

diff_map_nlp_to_rl = {
    "easy": "basic", "medium": "intermediate", "hard": "advanced",
    "basic": "basic", "intermediate": "intermediate", "advanced": "advanced",
}
diff_map_rl_to_nlp = {"basic": "easy", "intermediate": "medium", "advanced": "hard"}

# ─────────────────────────────────────────────
# Pipeline setup (same as original main.py)
# ─────────────────────────────────────────────

def build_pipeline():
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

    canonical_topics = sorted(set(
        m["topic"] for m in all_meta
        if m.get("topic") not in (None, "", "Unknown")
    ))
    if not canonical_topics:
        raise SystemExit("[ERROR] No canonical topics found.")

    topic_subtopics = defaultdict(set)
    for m in all_meta:
        t, s = m.get("topic", ""), m.get("subtopic", "")
        if t and t != "Unknown" and s and s != "Unknown":
            topic_subtopics[t].add(s)

    topics_with_context = {t: sorted(topic_subtopics[t]) for t in canonical_topics}

    # LLM infers prerequisites
    valid_topics_str = json.dumps(canonical_topics, indent=2)
    prompt = f"""You are an expert in "{course}" at the {level} level.
These are the ONLY valid topic names:
{valid_topics_str}
Each topic covers these concepts:
{json.dumps(topics_with_context, indent=2)}
For each topic, list which OTHER topics from the valid list must be learned first.
Rules: values MUST be exact copies from the list. No invented names. No self-refs. Direct prereqs only.
Return ONLY a JSON object. No explanation, no markdown.
JSON:"""

    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.1, "num_predict": 200}},
        timeout=60,
    )

    def safe_parse(raw, fallback):
        raw = raw.strip()
        for s, e, cls in [(raw.find("{"), raw.rfind("}") + 1, dict),
                          (raw.find("["), raw.rfind("]") + 1, list)]:
            if s != -1 and e > 0:
                try:
                    return json.loads(raw[s:e])
                except json.JSONDecodeError:
                    pass
        return fallback

    dependencies = safe_parse(resp.json()["response"], fallback={})

    data = collection.get(include=["documents", "metadatas"])
    filtered_chunks = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        if meta.get("topic") in (None, "", "Unknown"):
            continue
        if meta.get("concept_type") in ["definition", "explanation", "example"]:
            filtered_chunks.append({"text": doc, "topic": meta.get("topic"),
                                    "subtopic": meta.get("subtopic")})
    filtered_chunks = filtered_chunks[:150]
    concept_graph = build_concept_graph(filtered_chunks)

    topic_difficulties_raw = {}
    for m in all_meta:
        t, d = m.get("topic", ""), m.get("difficulty", "")
        if not t or t == "Unknown" or not d:
            continue
        topic_difficulties_raw.setdefault(t, []).append(d)

    valid_topics = set(topic_difficulties_raw.keys())
    clean_deps   = {t: [p for p in prereqs if p in valid_topics and p != t]
                    for t, prereqs in dependencies.items()}

    topics_difficulty = {
        t: diff_map_nlp_to_rl.get(Counter(diffs).most_common(1)[0][0], "intermediate")
        for t, diffs in topic_difficulties_raw.items()
    }
    topics_difficulty = dict(sorted(topics_difficulty.items()))

    return topics_difficulty, clean_deps, concept_graph


# ─────────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────────

def run_simulation(
    rl_agent,
    student_type: str,
    n_questions: int,
    topics_difficulty: dict,
    dependencies: dict,
    concept_graph,
    seed: int | None = None,
    output_dir: str = "./simulation_results",
) -> list[dict]:

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting simulation: {student_type.upper()} student | {n_questions} questions")
    print(f"{'='*60}")

    student  = SimulatedStudent(student_type=student_type, seed=seed)
    """rl_agent = AdaptiveAgent(
        topics_difficulty=topics_difficulty,
        prerequisites=dependencies,
        w1=0.4, w2=0.5, w3=0.1,
    )"""

    session_log           = []
    combo_question_count  = {}
    asked_questions_log   = {}
    used_chunk_ids        = []
    MAX_MEM               = 10

    for step in range(n_questions):

        print(f"\n[{student_type.upper()}] Q {step + 1}/{n_questions}", end="  ")

        # ── RL selects action ─────────────────────────────────
        state_vector            = rl_agent.ks.get_state_vector()
        action_idx, _, _        = rl_agent.select_action(state_vector, training=False)
        topic, diff, qtype      = rl_agent.mdp.decode(action_idx)

        print(f"| {topic} | {diff} | {qtype}")

        # ── Generate question ─────────────────────────────────
        combo_key      = (topic, diff, qtype)
        question_count = combo_question_count.get(combo_key, 0)
        asked          = asked_questions_log.get(topic, [])
        nlp_diff       = diff_map_rl_to_nlp[diff]

        result, new_ids = generate_question(
            topic, nlp_diff, qtype,
            question_count=question_count,
            asked_questions=asked,
            prerequisites=dependencies,
            concept_graph=concept_graph,
            used_chunk_ids=used_chunk_ids,
        )

        used_chunk_ids.extend(new_ids)
        if len(used_chunk_ids) > MAX_MEM:
            used_chunk_ids = used_chunk_ids[-MAX_MEM:]

        question         = result["question"]
        reference_answer = result["reference_answer"]

        if question in ("No data", "Insufficient data", "Error"):
            print(f"  [SKIP] Could not generate question.")
            continue

        # ── Simulated student answers ─────────────────────────
        student_answer = student.answer(
            question=question,
            reference_answer=reference_answer,
            topic=topic,
            difficulty=diff,
            question_type=qtype,
        )
        print(f"  Answer (simulated): {student_answer[:80]}...")

        # ── Evaluate answer ───────────────────────────────────
        eval_result = evaluate_answer(student_answer, reference_answer, qtype)
        score       = eval_result["final_score"]

        # ── Update RL agent ───────────────────────────────────
        reward = rl_agent.update(topic, score, diff, qtype)

        mastered = [t for t in rl_agent.ks.topics if rl_agent.ks.is_mastered(t)]
        unlocked = [t for t in rl_agent.ks.topics if rl_agent.ks.prerequisites_met(t)]

        print(f"  Score: {round(score,3)}  Reward: {round(reward,3)}  "
              f"Mastered: {len(mastered)}/{len(rl_agent.ks.topics)}")

        # ── Log ───────────────────────────────────────────────
        session_log.append({
            "step"          : step + 1,
            "student_type"  : student_type,
            "topic"         : topic,
            "difficulty"    : diff,
            "question_type" : qtype,
            "question"      : question,
            "student_answer": student_answer,
            "reference_answer": reference_answer,
            "semantic_score": float(eval_result["semantic_score"]),
            "keyword_score": float(eval_result["keyword_score"]),
            "nli_score": float(eval_result["nli_score"]),
            "completeness": float(eval_result["completeness_score"]),
            "final_score": float(score),
            "reward": float(reward),
            "mastered_count": int(len(mastered)),
            "mastered_topics": mastered[:],
        })
        
        combo_question_count[combo_key] = question_count + 1
        asked_questions_log.setdefault(topic, []).append(question)

    # ── Save results ──────────────────────────────────────────
    _save_results(session_log, student_type, output_dir, rl_agent)

    return session_log


# ─────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────

def _save_results(log: list[dict], student_type: str, output_dir: str, rl_agent):

    base = os.path.join(output_dir, student_type)

    # JSON — full log
    with open(f"{base}_session_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # CSV — flat log for easy analysis
    if log:
        with open(f"{base}_session_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                k for k in log[0].keys() if k != "mastered_topics"
            ])
            writer.writeheader()
            for row in log:
                row_flat = {k: v for k, v in row.items() if k != "mastered_topics"}
                writer.writerow(row_flat)

    # Summary
    summary = _compute_summary(log, rl_agent)
    with open(f"{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Results saved to {output_dir}/]")
    _print_summary(summary, student_type)


def _compute_summary(log: list[dict], rl_agent) -> dict:
    if not log:
        return {}

    scores = [r["final_score"] for r in log]
    by_diff = defaultdict(list)
    by_type = defaultdict(list)
    by_topic = defaultdict(list)

    for r in log:
        by_diff[r["difficulty"]].append(r["final_score"])
        by_type[r["question_type"]].append(r["final_score"])
        by_topic[r["topic"]].append(r["final_score"])

    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "total_questions"    : len(log),
        "overall_avg_score"  : avg(scores),
        "avg_by_difficulty"  : {k: avg(v) for k, v in by_diff.items()},
        "avg_by_question_type": {k: avg(v) for k, v in by_type.items()},
        "avg_by_topic"       : {k: avg(v) for k, v in by_topic.items()},
        "final_mastered"     : log[-1]["mastered_topics"] if log else [],
        "topic_attempts": {
            t: rl_agent.ks.attempts[t] for t in rl_agent.ks.topics
        },
        "topic_avg_scores": {
            t: round(rl_agent.ks.topic_score[t], 3) for t in rl_agent.ks.topics
        },
    }


def _print_summary(summary: dict, student_type: str):
    print(f"\n{'='*60}")
    print(f"SUMMARY — {student_type.upper()} STUDENT")
    print(f"{'='*60}")
    print(f"Total Questions  : {summary['total_questions']}")
    print(f"Overall Avg Score: {summary['overall_avg_score']}")
    print(f"By Difficulty    : {summary['avg_by_difficulty']}")
    print(f"By Q-Type        : {summary['avg_by_question_type']}")
    print(f"Final Mastered   : {summary['final_mastered']}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":


    # Build shared pipeline once
    print("Building knowledge pipeline...")
    topics_difficulty, dependencies, concept_graph = build_pipeline()
    print(f"Topics: {list(topics_difficulty.keys())}")

    student_types = ["strong", "weak"] 

    print("Pretraining RL Agent")
    rl_agent = AdaptiveAgent(
        topics_difficulty=topics_difficulty,
        prerequisites=dependencies,
        w1=0.4, w2=0.5, w3=0.1,
    )
    
    all_logs = {}
    for stype in student_types:
        
        rl_agent.reset_rl_agent(topics_difficulty.keys())
        log = run_simulation(
            rl_agent,
            student_type=stype,
            n_questions=10,
            topics_difficulty=topics_difficulty,
            dependencies=dependencies,
            concept_graph=concept_graph,
            seed=42,
            output_dir="./simulation_results",
        )
        all_logs[stype] = log

    # If both were run, print a comparison
    if len(all_logs) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON: STRONG vs WEAK")
        print(f"{'='*60}")
        for stype, log in all_logs.items():
            scores = [r["final_score"] for r in log]
            avg    = round(sum(scores) / len(scores), 3) if scores else 0
            final_mastered = log[-1]["mastered_count"] if log else 0
            print(f"  {stype.upper():8s} | avg score: {avg} | topics mastered: {final_mastered}")