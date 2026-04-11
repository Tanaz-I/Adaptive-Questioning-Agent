from flask import Flask, render_template, request, jsonify
import threading
import os
import requests
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

state = {
    "ready": False,
    "rl": None,
    "current": {},
    "history": [],
    "topics_difficulty": {},
    "dependencies": {},
    "concept_graph": {},
    "used_chunk_ids": [],
    "asked_questions_log": {},
    "combo_question_count": {}
}

MAX_MEM = 10

original_post = requests.post

def safe_post(*args, **kwargs):
    try:
        res = original_post(*args, **kwargs)
        data = res.json()
        if "response" not in data:
            data["response"] = "{}"
        class SafeResponse:
            def json(self_inner): return data
            def raise_for_status(self_inner): return None
        return SafeResponse()
    except Exception:
        class DummyResponse:
            def json(self_inner): return {"response": "{}"}
            def raise_for_status(self_inner): return None
        return DummyResponse()

requests.post = safe_post


# ═════════════════════════════════════════════════════════════════════
#  HARDCODED TEST QUESTIONS  ←  edit these to change test content
#  Each dict must have: topic, difficulty, qtype, question, reference
# ═════════════════════════════════════════════════════════════════════
TEST_QUESTIONS = [
    {
        "topic"     : "Neural Networks",
        "difficulty": "intermediate",
        "qtype"     : "inferential",
        "question"  : "Explain how backpropagation works in a neural network and why the chain rule of calculus is essential to the process.",
        "reference" : (
            "Backpropagation is an algorithm used to train neural networks by computing the gradient "
            "of the loss function with respect to each weight using the chain rule of calculus. "
            "Starting from the output layer, it propagates the error signal backwards through the "
            "network layer by layer, calculating partial derivatives at each step. The chain rule "
            "allows these gradients to be composed across layers, enabling efficient computation of "
            "how much each weight contributed to the overall error. These gradients are then used by "
            "an optimiser (e.g. gradient descent) to update the weights and minimise the loss."
        )
    },
    {
        "topic"     : "Reinforcement Learning",
        "difficulty": "basic",
        "qtype"     : "factual",
        "question"  : "What is the difference between a policy and a value function in reinforcement learning?",
        "reference" : (
            "A policy is a mapping from states to actions — it defines the agent's behaviour by "
            "specifying which action to take in each state. A value function estimates the expected "
            "cumulative reward from a given state (V(s)) or from a state-action pair (Q(s,a)). "
            "The policy tells the agent what to do; the value function tells the agent how good a "
            "particular state or action is in terms of long-term reward."
        )
    },
    {
        "topic"     : "Transformers",
        "difficulty": "advanced",
        "qtype"     : "evaluative",
        "question"  : (
            "Critically evaluate the self-attention mechanism in Transformers. "
            "What are its strengths and limitations compared to recurrent architectures?"
        ),
        "reference" : (
            "Self-attention allows every token to directly attend to every other token, capturing "
            "long-range dependencies in O(1) sequential steps — a major advantage over RNNs which "
            "require O(n) steps and suffer from vanishing gradients. Transformers are also highly "
            "parallelisable during training. However, self-attention has O(n²) time and memory "
            "complexity, making it expensive for very long sequences. Unlike RNNs, Transformers have "
            "no inherent notion of order and require positional encodings. They also need large "
            "amounts of data and compute to train effectively."
        )
    },
]

# Cycles through TEST_QUESTIONS; resets to 0 on each new /start
_test_q_index = 0


def get_test_question():
    global _test_q_index
    q = TEST_QUESTIONS[_test_q_index % len(TEST_QUESTIONS)]
    _test_q_index += 1
    return q


# ═════════════════════════════════════════════════════════════════════
#  PIPELINE  —  fully commented out for frontend testing
#  To re-enable: uncomment the try/except block and delete the
#  two lines at the bottom of this function.
# ═════════════════════════════════════════════════════════════════════
def run_pipeline():
    print("run_pipeline() called — FRONTEND TEST MODE, skipping RL pipeline.")

    # ------------------------------------------------------------------
    # ORIGINAL PIPELINE (uncomment to restore)
    # ------------------------------------------------------------------
    # try:
    #     patch_knowledge_state()
    #
    #     from NLP import knowledge_base_construction, enrich_metadata, topic_extraction
    #     from NLP.concept_graph import build_concept_graph
    #     import chromadb
    #     from chromadb.config import Settings
    #     from collections import Counter, defaultdict
    #     import json
    #
    #     DOCS_DIR        = "./contents"
    #     CHROMA_DB_DIR   = "./chroma_db"
    #     COLLECTION_NAME = "rag_kb"
    #     OLLAMA_URL      = "http://localhost:11434/api/generate"
    #     OLLAMA_MODEL    = "llama3:8b"
    #
    #     knowledge_base_construction.run_pipeline(DOCS_DIR)
    #     client = chromadb.PersistentClient(
    #         path=CHROMA_DB_DIR, settings=Settings(anonymized_telemetry=False)
    #     )
    #     collection = client.get_collection(COLLECTION_NAME)
    #     enrich_metadata.enrich_metadata()
    #     topic_extraction.run_global_topic_extraction()
    #     all_meta = collection.get(include=["metadatas"])["metadatas"]
    #     course, level = enrich_metadata.detect_course(collection)
    #
    #     canonical_topics = sorted(set(
    #         m["topic"] for m in all_meta
    #         if m.get("topic") not in (None, "", "Unknown")
    #     ))
    #     if not canonical_topics:
    #         raise ValueError("No canonical topics found after normalization.")
    #
    #     topic_subtopics = defaultdict(set)
    #     for m in all_meta:
    #         t, s = m.get("topic", ""), m.get("subtopic", "")
    #         if t and t != "Unknown" and s and s != "Unknown":
    #             topic_subtopics[t].add(s)
    #     topics_with_context = {t: sorted(topic_subtopics[t]) for t in canonical_topics}
    #
    #     valid_topics_str = json.dumps(canonical_topics, indent=2)
    #     prompt = f"""..."""   # prerequisite-inference prompt
    #
    #     def safe_parse_json(raw, fallback):
    #         raw = raw.strip()
    #         start, end = raw.find("{"), raw.rfind("}") + 1
    #         if start != -1 and end > 0:
    #             try: return json.loads(raw[start:end])
    #             except json.JSONDecodeError: pass
    #         return fallback
    #
    #     resp = original_post(OLLAMA_URL, json={
    #         "model": OLLAMA_MODEL, "prompt": prompt,
    #         "stream": False, "options": {"temperature": 0.1, "num_predict": 200},
    #     }, timeout=60)
    #     dependencies = safe_parse_json(resp.json()["response"], fallback={})
    #
    #     data = collection.get(include=["documents", "metadatas"])
    #     filtered_chunks = []
    #     for doc, meta in zip(data["documents"], data["metadatas"]):
    #         if meta.get("topic") in (None, "", "Unknown"): continue
    #         if meta.get("concept_type") in ["definition", "explanation", "example"]:
    #             filtered_chunks.append({"text": doc, "topic": meta.get("topic"),
    #                                     "subtopic": meta.get("subtopic")})
    #     filtered_chunks = filtered_chunks[:150]
    #     concept_graph = build_concept_graph(filtered_chunks)
    #     state["concept_graph"] = concept_graph
    #
    #     topic_difficulties_raw = {}
    #     for m in all_meta:
    #         t, d = m.get("topic", ""), m.get("difficulty", "")
    #         if not t or t == "Unknown" or not d: continue
    #         topic_difficulties_raw.setdefault(t, []).append(d)
    #     valid_topics = set(topic_difficulties_raw.keys())
    #
    #     clean_dependencies = {
    #         topic: [p for p in prereqs if p in valid_topics and p != topic]
    #         for topic, prereqs in dependencies.items()
    #     }
    #     dependencies = clean_dependencies
    #     state["dependencies"] = dependencies
    #
    #     diff_map_nlp_to_rl = {
    #         "easy": "basic", "medium": "intermediate", "hard": "advanced",
    #         "basic": "basic", "intermediate": "intermediate", "advanced": "advanced"
    #     }
    #     topics_difficulty = {
    #         topic: diff_map_nlp_to_rl.get(
    #             Counter(diffs).most_common(1)[0][0], "intermediate"
    #         )
    #         for topic, diffs in topic_difficulties_raw.items()
    #     }
    #     topics_difficulty = dict(sorted(topics_difficulty.items()))
    #     state["topics_difficulty"] = topics_difficulty
    #
    #     from PPO_RL.PPOAgent import PPOAgent
    #     rl = PPOAgent(
    #         topics_difficulty=topics_difficulty, prerequisites=dependencies,
    #         w1=0.4, w2=0.5, w3=0.1, use_lstm=True
    #     )
    #     state["rl"] = rl
    #     state["used_chunk_ids"]       = []
    #     state["asked_questions_log"]  = {}
    #     state["combo_question_count"] = {}
    #
    #     state["current"] = get_question()   # ← swap back when RL is live
    #     state["ready"]   = True
    #     print("READY")
    #
    # except Exception as e:
    #     print(f"Pipeline error: {e}")
    #     import traceback; traceback.print_exc()
    #     state["ready"] = False
    # ------------------------------------------------------------------
    # END ORIGINAL PIPELINE
    # ------------------------------------------------------------------

    # ── FRONTEND TEST: simulate a brief loading delay then mark ready ──
    import time
    time.sleep(2)                            # mimic pipeline warmup (adjust freely)
    state["current"] = get_test_question()
    state["ready"]   = True
    print("READY (frontend test mode)")


# ═════════════════════════════════════════════════════════════════════
#  QUESTION GENERATION  —  returns hardcoded question in test mode
#  Uncomment the RL block and delete get_test_question() call to restore
# ═════════════════════════════════════════════════════════════════════
def get_question():
    # ------------------------------------------------------------------
    # ORIGINAL RL QUESTION GENERATION (uncomment to restore)
    # ------------------------------------------------------------------
    # diff_map_rl_to_nlp = {
    #     'basic': 'easy', 'intermediate': 'medium', 'advanced': 'hard'
    # }
    # rl = state["rl"]
    # from NLP.Q_Generator_A_Evaluator.question_generator import generate_question
    # for attempt in range(5):
    #     s = rl.ks.get_state_vector()
    #     a = rl.select_action(s, training=False)[0]
    #     topic, diff, qtype = rl.mdp.decode(a)
    #     combo_key      = (topic, diff, qtype)
    #     question_count = state["combo_question_count"].get(combo_key, 0)
    #     asked          = state["asked_questions_log"].get(topic, [])
    #     nlp_diff       = diff_map_rl_to_nlp[diff]
    #     result, new_ids = generate_question(
    #         topic, nlp_diff, qtype,
    #         question_count=question_count, asked_questions=asked,
    #         prerequisites=state["dependencies"], concept_graph=state["concept_graph"],
    #         used_chunk_ids=state["used_chunk_ids"]
    #     )
    #     state["used_chunk_ids"].extend(new_ids)
    #     if len(state["used_chunk_ids"]) > MAX_MEM:
    #         state["used_chunk_ids"] = state["used_chunk_ids"][-MAX_MEM:]
    #     question  = result.get("question", "")
    #     reference = result.get("reference_answer", "")
    #     if question and question not in (
    #         "No question generated", "No data", "Insufficient data", "Error", ""
    #     ):
    #         return {"topic": topic, "difficulty": diff, "qtype": qtype,
    #                 "question": question, "reference": reference}
    #     print(f"[WARN] Attempt {attempt+1}: Invalid question '{question}', retrying...")
    # return {"topic": topic, "difficulty": diff, "qtype": qtype,
    #         "question": "Could not generate a question. Please click Next.", "reference": ""}
    # ------------------------------------------------------------------
    # END ORIGINAL RL QUESTION GENERATION
    # ------------------------------------------------------------------

    # ── FRONTEND TEST: serve next hardcoded question ──
    return get_test_question()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    files = request.files.getlist("files")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded"}), 400

    os.makedirs("contents", exist_ok=True)
    for old in os.listdir("contents"):
        old_path = os.path.join("contents", old)
        if os.path.isfile(old_path):
            os.remove(old_path)

    saved = []
    for f in files:
        if f.filename:
            f.save(os.path.join("contents", f.filename))
            saved.append(f.filename)

    print(f"Saved {len(saved)} file(s): {saved}")

    state["ready"]   = False
    state["history"] = []

    # Reset test question rotation for fresh session
    global _test_q_index
    _test_q_index = 0

    threading.Thread(target=run_pipeline).start()
    return jsonify({"status": "processing", "files": saved})


@app.route("/status")
def status():
    return jsonify({
        "ready"   : state["ready"],
        "question": state["current"].get("question", "")
    })


@app.route("/submit", methods=["POST"])
def submit():
    try:
        data = request.json
        if not data or "answer" not in data:
            return jsonify({"error": "No answer provided"}), 400

        topic = state["current"]["topic"]
        diff  = state["current"]["difficulty"]
        ans   = data["answer"].strip()
        if not ans:
            return jsonify({"error": "Empty answer"}), 400

        ref   = state["current"]["reference"]
        qtype = state["current"]["qtype"]

        # ------------------------------------------------------------------
        # ORIGINAL ANSWER EVALUATION (uncomment to restore)
        # ------------------------------------------------------------------
        from NLP.Q_Generator_A_Evaluator.answer_evaluator import evaluate_answer
        question    = state["current"]["question"]
        eval_result = evaluate_answer(ans, ref, qtype, question)
        score       = eval_result["final_score"]
        print(f"  Semantic    : {eval_result.get('semantic_score')}")
        print(f"  Keyword     : {eval_result.get('keyword_score')}")
        print(f"  NLI         : {eval_result.get('nli_score')}")
        print(f"  Completeness: {eval_result.get('completeness_score')}")
        print(f"  Final Score : {score}")
        #rl     = state["rl"]
        #reward = rl.update(topic, score, diff, qtype)
        #print(f"  RL Reward   : {round(reward, 3)}")
        # ------------------------------------------------------------------
        # END ORIGINAL ANSWER EVALUATION
        # ------------------------------------------------------------------

        # ── FRONTEND TEST: dummy score based on answer word count ──
        # Scores 0.0–1.0 proportional to words written (max at ~80 words)
        # score  = round(min(1.0, len(ans.split()) / 80), 2)
        # reward = round(score * 0.9, 3)

        combo_key = (topic, diff, qtype)
        state["combo_question_count"][combo_key] = \
            state["combo_question_count"].get(combo_key, 0) + 1

        if topic not in state["asked_questions_log"]:
            state["asked_questions_log"][topic] = []
        state["asked_questions_log"][topic].append(state["current"]["question"])

        state["history"].append({
            "q"     : state["current"]["question"],
            "score" : float(score)
           # "reward": float(reward)
        })
        
        print(eval_result.get("feedback", ""))
        return jsonify({
            "score"    : float(score),
            #"reward"   : float(reward),
            "reference": ref,           # ← hardcoded ref answer shown to user
            "feedback"  : eval_result.get("feedback", ""),     
            "error_type": eval_result.get("error_type", "none")
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/next")
def next_q():
    try:
        state["current"] = get_question()
        return jsonify({
            "question": state["current"]["question"],
            "topic"   : state["current"]["topic"],
            "diff"    : state["current"]["difficulty"],
            "qtype"   : state["current"]["qtype"],
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/quit")
def quit_session():
    # ------------------------------------------------------------------
    # ORIGINAL QUIT / RECOMMENDATIONS (uncomment to restore)
    # ------------------------------------------------------------------
    # from NLP.recommend_material import get_weak_topic_material, recommend_courses
    # rl  = state["rl"]
    # ks  = rl.ks if rl else None
    # summary, weak_topics = [], []
    # dependencies = state.get("dependencies", {})
    # if ks:
    #     for topic in ks.topics:
    #         avg_score = round(float(ks.topic_score[topic]), 3)
    #         mastered  = ks.is_mastered(topic)
    #         attempts  = ks.attempts[topic]
    #         prereqs   = dependencies.get(topic, [])
    #         summary.append({"topic": topic, "attempts": attempts,
    #                          "avg_score": avg_score, "mastered": mastered})
    #         if attempts > 0 and avg_score < 0.5:  weak_topics.append(topic)
    #         elif attempts == 0 and prereqs:        weak_topics.append(topic)
    # weak_material = get_weak_topic_material(weak_topics)
    # course_recs   = recommend_courses(weak_topics, top_n=5)
    # ------------------------------------------------------------------
    # END ORIGINAL QUIT
    # ------------------------------------------------------------------

    # ── FRONTEND TEST: build summary from session history ──
    seen = {}
    for h in state["history"]:
        for tq in TEST_QUESTIONS:
            if tq["question"] == h["q"]:
                t = tq["topic"]
                seen.setdefault(t, []).append(h["score"])
                break

    summary = []
    for t, scores in seen.items():
        avg = round(sum(scores) / len(scores), 3)
        summary.append({
            "topic"    : t,
            "attempts" : len(scores),
            "avg_score": avg,
            "mastered" : avg >= 0.75,
        })

    weak_topics   = [s["topic"] for s in summary if s["avg_score"] < 0.5]
    weak_material = []                          # no material files in test mode
    course_recs   = [
        {
            "title": "Deep Learning Specialisation — Coursera",
            "url"  : "https://www.coursera.org/specializations/deep-learning"
        },
        {
            "title": "Reinforcement Learning — David Silver (UCL)",
            "url"  : "https://www.davidsilver.uk/teaching/"
        },
    ]

    state["summary"]       = summary
    state["weak_topics"]   = weak_topics
    state["weak_material"] = weak_material
    state["course_recs"]   = course_recs

    return jsonify({
        "history"      : state["history"],
        "summary"      : summary,
        "weak_topics"  : weak_topics,
        "weak_material": weak_material,
        "course_recs"  : course_recs,
    })


@app.route("/report")
def report():
    return render_template(
        "report.html",
        summary      = state.get("summary", []),
        history      = state.get("history", []),
        weak_topics  = state.get("weak_topics", []),
        weak_material= state.get("weak_material", []),
        course_recs  = state.get("course_recs",  []),
    )


if __name__ == "__main__":
    print("Flask running... (FRONTEND TEST MODE — RL pipeline disabled)")
    app.run(debug=True, use_reloader=False)