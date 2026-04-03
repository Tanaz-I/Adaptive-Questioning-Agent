from flask import Flask, render_template, request, jsonify
import threading
import os
import requests
import pytesseract

# ─────────────────────────────────────────────
# TESSERACT PATH (Windows)
# ─────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

state = {
    "ready": False,
    "rl": None,
    "current": {},
    "history": [],
    "topics_difficulty": {}
}

# ─────────────────────────────────────────────
# SAFE REQUEST FIX
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# PATCH RL (avoid crash)
# ─────────────────────────────────────────────
def patch_knowledge_state():
    from Adaptation_RL.knowledge_state import KnowledgeState

    original = KnowledgeState.get_valid_actions

    def safe(self, topic):
        valid = original(self, topic)
        if not valid:
            return [
                (d, q)
                for d in ['basic', 'intermediate', 'advanced']
                for q in ['factual', 'inferential', 'evaluative']
            ]
        return valid

    KnowledgeState.get_valid_actions = safe


# ─────────────────────────────────────────────
# PIPELINE (aligned with main.py)
# ─────────────────────────────────────────────
def run_pipeline():
    print("🚀 Starting pipeline...")

    try:
        patch_knowledge_state()

        # STEP 1: Build KB
        from NLP import knowledge_base_construction, enrich_metadata
        import chromadb
        from chromadb.config import Settings

        DOCS_DIR = "./contents"
        CHROMA_DB_DIR = "./chroma_db"
        COLLECTION_NAME = "rag_kb"

        knowledge_base_construction.run_pipeline(DOCS_DIR)

        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        collection = client.get_collection(COLLECTION_NAME)
        enrich_metadata.enrich_metadata()

        all_meta = collection.get(include=["metadatas"])["metadatas"]

        # STEP 2: Extract topics
        topics = sorted(set(
            m["topic"]
            for m in all_meta
            if m.get("topic") not in (None, "", "Unknown")
        ))

        # STEP 3: Difficulty (same as main.py)
        from collections import Counter

        diff_map_nlp_to_rl = {
            "easy"        : "basic",
            "medium"      : "intermediate",
            "hard"        : "advanced",
            "basic"       : "basic",
            "intermediate": "intermediate",
            "advanced"    : "advanced"
        }

        topic_difficulties_raw = {}
        for m in all_meta:
            t = m.get("topic", "")
            d = m.get("difficulty", "")
            if not t or t == "Unknown" or not d:
                continue
            topic_difficulties_raw.setdefault(t, []).append(d)

        topics_difficulty = {
            topic: diff_map_nlp_to_rl.get(
                Counter(diffs).most_common(1)[0][0], "intermediate"
            )
            for topic, diffs in topic_difficulties_raw.items()
        }
        topics_difficulty = dict(sorted(topics_difficulty.items()))

        state["topics_difficulty"] = topics_difficulty

        # STEP 4: RL Agent
        from Adaptation_RL.Agent import AdaptiveAgent

        rl = AdaptiveAgent(
            topics_difficulty=topics_difficulty,
            prerequisites={t: [] for t in topics_difficulty},
            w1=0.4, w2=0.5, w3=0.1,
            n_episodes=100
        )

        state["rl"] = rl

        # STEP 5: First question
        state["current"] = get_question()
        state["ready"] = True

        print("✅ READY")

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        state["ready"] = False


# ─────────────────────────────────────────────
# QUESTION GENERATION (aligned with main.py)
# ─────────────────────────────────────────────
def get_question():

    diff_map_rl_to_nlp = {
        'basic'       : 'easy',
        'intermediate': 'medium',
        'advanced'    : 'hard'
    }

    rl = state["rl"]

    from NLP.Q_Generator_A_Evaluator.question_generator import generate_question

    # Retry up to 5 times if no valid question is generated
    for attempt in range(5):
        s = rl.ks.get_state_vector()
        a, _, _ = rl.select_action(s, training=False)
        topic, diff, qtype = rl.mdp.decode(a)

        nlp_diff = diff_map_rl_to_nlp[diff]
        result, _ = generate_question(topic, nlp_diff, qtype)

        print(f"[DEBUG] generate_question result keys: {list(result.keys()) if isinstance(result, dict) else result}")

        print(result)
        question  = result.get("question", "")
        reference = result.get("reference_answer", "") or result.get("answer", "") or result.get("reference", "")

        # Valid question — return it
        if question and question not in ("No question generated", "No data", "Insufficient data", "Error", ""):
            return {
                "topic"     : topic,
                "difficulty": diff,
                "qtype"     : qtype,
                "question"  : question,
                "reference" : reference
            }

        print(f"[WARN] Attempt {attempt+1}: Invalid question '{question}', retrying...")

    # All retries failed — return best effort
    return {
        "topic"     : topic,
        "difficulty": diff,
        "qtype"     : qtype,
        "question"  : "Could not generate a question. Please click Next.",
        "reference" : ""
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    file = request.files["file"]

    os.makedirs("contents", exist_ok=True)
    file.save(f"contents/{file.filename}")

    state["ready"] = False
    state["history"] = []

    threading.Thread(target=run_pipeline).start()

    return jsonify({"status": "processing"})


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

        ans = data["answer"].strip()
        if not ans:
            return jsonify({"error": "Empty answer"}), 400

        from NLP.Q_Generator_A_Evaluator.answer_evaluator import evaluate_answer

        ref   = state["current"]["reference"]
        qtype = state["current"]["qtype"]
        topic = state["current"]["topic"]
        diff  = state["current"]["difficulty"]

        # Evaluate answer (same as main.py)
        eval_result = evaluate_answer(ans, ref, qtype)
        score = eval_result["final_score"]

        print(f"  Semantic    : {eval_result.get('semantic_score')}")
        print(f"  Keyword     : {eval_result.get('keyword_score')}")
        print(f"  NLI         : {eval_result.get('nli_score')}")
        print(f"  Completeness: {eval_result.get('completeness_score')}")
        print(f"  Final Score : {score}")

        # Update RL agent (same signature as main.py)
        rl = state["rl"]
        reward = rl.update(topic, score, diff, qtype)

        print(f"  RL Reward   : {round(reward, 3)}")

        # Log history
        state["history"].append({
            "q"     : state["current"]["question"],
            "score" : float(score),
            "reward": float(reward)
        })

        return jsonify({
            "score"    : float(score),
            "reward"   : float(reward),
            "reference": ref
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/next")
def next_q():
    try:
        state["current"] = get_question()
        return jsonify({"question": state["current"]["question"]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/quit")
def quit_session():
    return jsonify({"history": state["history"]})


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🌐 Flask running...")
    app.run(debug=True, use_reloader=False)