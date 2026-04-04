"""
run_simulation.py
=================
Drop-in replacement for the session loop in main.py.
Runs N_QUESTIONS automatically using a simulated student (no manual input).

Usage:
    python run_simulation.py --student weak  --questions 500
    python run_simulation.py --student strong --questions 1000
    python run_simulation.py --student both   --questions 500   # runs both back-to-back

CSV output (per student type):
    simulation_results/<student>_qa_summary.csv  ← written row-by-row during simulation
                                                    app.py /quit can reconstruct from this
    simulation_results/<student>_session_log.csv ← full log (written at end)
    simulation_results/<student>_session_log.json
    simulation_results/<student>_summary.json
"""

import json
import csv
import os
from collections import Counter, defaultdict

import chromadb
import requests
from chromadb.config import Settings

from Adaptation_RL.Agent import AdaptiveAgent
from NLP import knowledge_base_construction, enrich_metadata, topic_extraction
from NLP.Q_Generator_A_Evaluator.answer_evaluator import evaluate_answer
from NLP.Q_Generator_A_Evaluator.question_generator import generate_question
from NLP.concept_graph import build_concept_graph
from Adaptation_RL.student_simulator import SimulatedStudent

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DOCS_DIR        = "./contents"
CHROMA_DB_DIR   = "./chroma_db"
COLLECTION_NAME = "rag_kb"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llama3"

diff_map_nlp_to_rl = {
    "easy"        : "basic",
    "medium"      : "intermediate",
    "hard"        : "advanced",
    "basic"       : "basic",
    "intermediate": "intermediate",
    "advanced"    : "advanced",
}
diff_map_rl_to_nlp = {"basic": "easy", "intermediate": "medium", "advanced": "hard"}

# ─────────────────────────────────────────────
# CSV columns
# ─────────────────────────────────────────────
# Designed to mirror /quit in app.py exactly:
#
#   summary     → group by topic, take last row:
#                 topic | topic_attempts_so_far | topic_avg_score_so_far | is_mastered
#
#   weak_topics → (attempts > 0 AND avg_score < 0.5)
#                 OR (attempts == 0 AND topic_prereqs != "")
#
#   history     → all rows in order:
#                 question | final_score | reward   (same as state["history"])

SUMMARY_CSV_FIELDS = [
    # ── Identifiers ──────────────────────────────────────
    "step",
    "student_type",
    # ── RL action ─────────────────────────────────────────
    "topic",
    "difficulty",
    "question_type",
    # ── Q&A content ───────────────────────────────────────
    "question",
    "student_answer",
    "reference_answer",
    # ── NLP scorer breakdown ──────────────────────────────
    "semantic_score",
    "keyword_score",
    "nli_score",
    "completeness",
    "final_score",
    # ── RL signals ────────────────────────────────────────
    "reward",
    # ── Per-topic knowledge state snapshot ────────────────
    # Take the LAST row per topic to reconstruct /quit summary
    "topic_avg_score_so_far",   # ks.topic_score[topic]  → avg_score in /quit
    "topic_attempts_so_far",    # ks.attempts[topic]     → attempts in /quit
    "is_mastered",              # ks.is_mastered(topic)  → mastered in /quit
    "topic_prereqs",            # comma-separated prereqs → weak logic in /quit
    # ── Session-level ─────────────────────────────────────
    "mastered_count",
]


# ─────────────────────────────────────────────
# Pipeline setup
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
        timeout=6000,
    )

    def safe_parse(raw, fallback):
        raw = raw.strip()
        for s, e in [(raw.find("{"), raw.rfind("}") + 1),
                     (raw.find("["), raw.rfind("]") + 1)]:
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
            filtered_chunks.append({
                "text"    : doc,
                "topic"   : meta.get("topic"),
                "subtopic": meta.get("subtopic"),
            })
    filtered_chunks = filtered_chunks[:150]
    concept_graph = build_concept_graph(filtered_chunks)

    topic_difficulties_raw = {}
    for m in all_meta:
        t, d = m.get("topic", ""), m.get("difficulty", "")
        if not t or t == "Unknown" or not d:
            continue
        topic_difficulties_raw.setdefault(t, []).append(d)

    valid_topics = set(topic_difficulties_raw.keys())
    clean_deps   = {
        t: [p for p in prereqs if p in valid_topics and p != t]
        for t, prereqs in dependencies.items()
    }

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
    seed=None,
    output_dir: str = "./simulation_results",
) -> list[dict]:

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting simulation: {student_type.upper()} student | {n_questions} questions")
    print(f"{'='*60}")

    student = SimulatedStudent(student_type=student_type, seed=seed)

    session_log          = []
    combo_question_count = {}
    asked_questions_log  = {}
    used_chunk_ids       = []
    MAX_MEM              = 10

    # ── Open summary CSV for incremental writing ───────────────────────────
    # Flushed after every row — safe even if simulation crashes mid-run.
    # app.py reconstructs /quit summary by reading this file.
    summary_csv_path = os.path.join(output_dir, f"{student_type}_qa_summary.csv")
    summary_file     = open(summary_csv_path, "w", newline="", encoding="utf-8")
    summary_writer   = csv.DictWriter(summary_file, fieldnames=SUMMARY_CSV_FIELDS)
    summary_writer.writeheader()
    summary_file.flush()

    try:
        for step in range(n_questions):

            print(f"\n[{student_type.upper()}] Q {step + 1}/{n_questions}", end="  ")

            # ── RL selects action ──────────────────────────────────────────
            state_vector       = rl_agent.ks.get_state_vector()
            action_idx, _, _   = rl_agent.select_action(state_vector, training=False)
            topic, diff, qtype = rl_agent.mdp.decode(action_idx)

            print(f"| {topic} | {diff} | {qtype}")

            # ── Generate question ──────────────────────────────────────────
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

            print(question)

            if question in ("No data", "Insufficient data", "Error"):
                print(f"  [SKIP] Could not generate question.")
                continue

            # ── Simulated student answers ──────────────────────────────────
            student_answer = student.answer(
                question=question,
                reference_answer=reference_answer,
                topic=topic,
                difficulty=diff,
                question_type=qtype,
            )
            print(f"  Answer (simulated): {student_answer[:80]}...")

            # ── Evaluate answer ────────────────────────────────────────────
            eval_result = evaluate_answer(student_answer, reference_answer, qtype, question)
            score       = eval_result["final_score"]

            # ── Update RL agent ────────────────────────────────────────────
            reward   = rl_agent.update(topic, score, diff, qtype)
            mastered = [t for t in rl_agent.ks.topics if rl_agent.ks.is_mastered(t)]

            print(f"  Score: {round(score,3)}  Reward: {round(reward,3)}  "
                  f"Mastered: {len(mastered)}/{len(rl_agent.ks.topics)}")

            # ── Snapshot per-topic knowledge state (mirrors ks used in /quit) ──
            topic_avg_score_so_far = round(float(rl_agent.ks.topic_score[topic]), 3)
            topic_attempts_so_far  = int(rl_agent.ks.attempts[topic])
            is_mastered_now        = bool(rl_agent.ks.is_mastered(topic))
            topic_prereqs_str      = ",".join(dependencies.get(topic, []))

            # ── Build full log entry ───────────────────────────────────────
            log_entry = {
                # identifiers
                "step"                  : step + 1,
                "student_type"          : student_type,
                # RL action
                "topic"                 : topic,
                "difficulty"            : diff,
                "question_type"         : qtype,
                # Q&A content
                "question"              : question,
                "student_answer"        : student_answer,
                "reference_answer"      : reference_answer,
                # NLP scores
                "semantic_score"        : float(eval_result["semantic_score"]),
                "keyword_score"         : float(eval_result["keyword_score"]),
                "nli_score"             : float(eval_result["nli_score"]),
                "completeness"          : float(eval_result["completeness_score"]),
                "final_score"           : float(score),
                # RL signals
                "reward"                : float(reward),
                # Per-topic ks snapshot
                "topic_avg_score_so_far": topic_avg_score_so_far,
                "topic_attempts_so_far" : topic_attempts_so_far,
                "is_mastered"           : is_mastered_now,
                "topic_prereqs"         : topic_prereqs_str,
                # Session level
                "mastered_count"        : int(len(mastered)),
                # JSON-only (excluded from CSV)
                "mastered_topics"       : mastered[:],
            }
            session_log.append(log_entry)

            # ── Write CSV row immediately ──────────────────────────────────
            summary_writer.writerow({k: log_entry[k] for k in SUMMARY_CSV_FIELDS})
            summary_file.flush()

            # ── Update counters ────────────────────────────────────────────
            combo_question_count[combo_key] = question_count + 1
            asked_questions_log.setdefault(topic, []).append(question)

    finally:
        summary_file.close()
        print(f"\n[Summary CSV closed: {summary_csv_path}]")

    # ── Save full results ──────────────────────────────────────────────────
    _save_results(session_log, student_type, output_dir, rl_agent, dependencies)

    return session_log


# ─────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────

def _save_results(
    log: list[dict],
    student_type: str,
    output_dir: str,
    rl_agent,
    dependencies: dict,
):
    base = os.path.join(output_dir, student_type)

    # JSON — full log (includes mastered_topics list)
    with open(f"{base}_session_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # CSV — flat full log (all SUMMARY_CSV_FIELDS columns)
    if log:
        with open(f"{base}_session_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
            writer.writeheader()
            for row in log:
                writer.writerow({k: row[k] for k in SUMMARY_CSV_FIELDS})

    # Summary JSON — mirrors /quit response structure
    summary = _compute_summary(log, rl_agent, dependencies)
    with open(f"{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Results saved to {output_dir}/]")
    _print_summary(summary, student_type)


def _compute_summary(log: list[dict], rl_agent, dependencies: dict) -> dict:
    """
    Mirrors the /quit route in app.py exactly.

    /quit builds:
        summary     — per topic: topic, attempts, avg_score, mastered
        weak_topics — avg_score < 0.5 with attempts > 0,
                      OR attempts == 0 with prereqs
        history     — [{q, score, reward}, ...]

    This function reproduces all three from rl_agent.ks (same source as app.py)
    and also adds analytics for evaluation purposes.
    """
    if not log:
        return {}

    ks = rl_agent.ks

    # ── /quit: summary + weak_topics ──────────────────────────────────────
    summary_rows = []
    weak_topics  = []

    for topic in ks.topics:
        avg_score = round(float(ks.topic_score[topic]), 3)
        mastered  = bool(ks.is_mastered(topic))
        attempts  = int(ks.attempts[topic])
        prereqs   = dependencies.get(topic, [])

        summary_rows.append({
            "topic"    : topic,
            "attempts" : attempts,
            "avg_score": avg_score,
            "mastered" : mastered,
        })

        # Exact same condition as /quit
        if attempts > 0 and avg_score < 0.5:
            weak_topics.append(topic)
        elif attempts == 0 and prereqs:
            weak_topics.append(topic)

    # ── /quit: history ─────────────────────────────────────────────────────
    history = [
        {"q": r["question"], "score": r["final_score"], "reward": r["reward"]}
        for r in log
    ]

    # ── Analytics (for evaluation / paper metrics) ─────────────────────────
    scores   = [r["final_score"] for r in log]
    by_diff  = defaultdict(list)
    by_type  = defaultdict(list)
    by_topic = defaultdict(list)

    for r in log:
        by_diff[r["difficulty"]].append(r["final_score"])
        by_type[r["question_type"]].append(r["final_score"])
        by_topic[r["topic"]].append(r["final_score"])

    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        # /quit-compatible fields
        "summary"             : summary_rows,
        "weak_topics"         : weak_topics,
        "history"             : history,
        # analytics
        "total_questions"     : len(log),
        "overall_avg_score"   : avg(scores),
        "avg_by_difficulty"   : {k: avg(v) for k, v in by_diff.items()},
        "avg_by_question_type": {k: avg(v) for k, v in by_type.items()},
        "avg_by_topic"        : {k: avg(v) for k, v in by_topic.items()},
        "final_mastered"      : log[-1]["mastered_topics"] if log else [],
        "topic_attempts"      : {t: int(ks.attempts[t]) for t in ks.topics},
        "topic_avg_scores"    : {t: round(float(ks.topic_score[t]), 3) for t in ks.topics},
    }


def _print_summary(summary: dict, student_type: str):
    print(f"\n{'='*60}")
    print(f"SUMMARY — {student_type.upper()} STUDENT")
    print(f"{'='*60}")
    print(f"Total Questions  : {summary['total_questions']}")
    print(f"Overall Avg Score: {summary['overall_avg_score']}")
    print(f"By Difficulty    : {summary['avg_by_difficulty']}")
    print(f"By Q-Type        : {summary['avg_by_question_type']}")
    print(f"Weak Topics      : {summary['weak_topics']}")
    print(f"Final Mastered   : {summary['final_mastered']}")


# ─────────────────────────────────────────────
# Reconstructing /quit from CSV in app.py
# ─────────────────────────────────────────────
#
#   import pandas as pd
#
#   df = pd.read_csv("simulation_results/strong_qa_summary.csv")
#
#   # summary — last row per topic (= ks state at end of session)
#   last = df.groupby("topic").last().reset_index()
#   summary = last[["topic","topic_attempts_so_far","topic_avg_score_so_far","is_mastered"]] \
#               .rename(columns={
#                   "topic_attempts_so_far" : "attempts",
#                   "topic_avg_score_so_far": "avg_score",
#                   "is_mastered"           : "mastered",
#               }).to_dict("records")
#
#   # weak_topics — exact /quit condition
#   weak_topics = last[
#       ((last["topic_attempts_so_far"] > 0) & (last["topic_avg_score_so_far"] < 0.5)) |
#       ((last["topic_attempts_so_far"] == 0) & (last["topic_prereqs"] != ""))
#   ]["topic"].tolist()
#
#   # history — all rows in order (= state["history"] in app.py)
#   history = df[["question","final_score","reward"]] \
#               .rename(columns={"question":"q","final_score":"score"}) \
#               .to_dict("records")
#
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Building knowledge pipeline...")
    topics_difficulty, dependencies, concept_graph = build_pipeline()
    print(f"Topics: {list(topics_difficulty.keys())}")

    student_types = ["strong", "weak"]

    print("Pretraining RL Agent")
    rl_agent = AdaptiveAgent(
        topics_difficulty=topics_difficulty,
        prerequisites=dependencies,
        w1=0.35, w2=0.45, w3=0.2,
    )

    all_logs = {}
    for stype in student_types:
        rl_agent.reset_rl_agent(topics_difficulty.keys())
        log = run_simulation(
            rl_agent,
            student_type=stype,
            n_questions=5,
            topics_difficulty=topics_difficulty,
            dependencies=dependencies,
            concept_graph=concept_graph,
            seed=42,
            output_dir="./simulation_results",
        )
        all_logs[stype] = log

    if len(all_logs) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON: STRONG vs WEAK")
        print(f"{'='*60}")
        for stype, log in all_logs.items():
            scores = [r["final_score"] for r in log]
            avg    = round(sum(scores) / len(scores), 3) if scores else 0
            final_mastered = log[-1]["mastered_count"] if log else 0
            print(f"  {stype.upper():8s} | avg score: {avg} | topics mastered: {final_mastered}")