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
    "topics_difficulty": {},
    "dependencies": {},
    "concept_graph": {},
    "used_chunk_ids": [],
    "asked_questions_log": {},
    "combo_question_count": {}
}

MAX_MEM = 10

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

        from NLP import knowledge_base_construction, enrich_metadata, topic_extraction
        from NLP.concept_graph import build_concept_graph
        import chromadb
        from chromadb.config import Settings
        from collections import Counter, defaultdict
        import json

        DOCS_DIR        = "./contents"
        CHROMA_DB_DIR   = "./chroma_db"
        COLLECTION_NAME = "rag_kb"
        OLLAMA_URL      = "http://localhost:11434/api/generate"
        OLLAMA_MODEL    = "llama3:8b"

        # STEP 1: Build KB
        knowledge_base_construction.run_pipeline(DOCS_DIR)

        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(COLLECTION_NAME)

        enrich_metadata.enrich_metadata()
        topic_extraction.run_global_topic_extraction()  # NEW from main.py

        all_meta = collection.get(include=["metadatas"])["metadatas"]

        course, level = enrich_metadata.detect_course(collection)

        # STEP 2: Extract canonical topics
        canonical_topics = sorted(set(
            m["topic"]
            for m in all_meta
            if m.get("topic") not in (None, "", "Unknown")
        ))

        if not canonical_topics:
            raise ValueError("No canonical topics found after normalization.")

        print(f"Canonical topics: {canonical_topics}")

        # STEP 3: Build topic → subtopics context
        topic_subtopics = defaultdict(set)
        for m in all_meta:
            t = m.get("topic", "")
            s = m.get("subtopic", "")
            if t and t != "Unknown" and s and s != "Unknown":
                topic_subtopics[t].add(s)

        topics_with_context = {t: sorted(topic_subtopics[t]) for t in canonical_topics}

        # STEP 4: LLM infers prerequisites (from main.py)
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

        def safe_parse_json(raw, fallback):
            raw   = raw.strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > 0:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass
            print(f"  [WARN] JSON parse failed. Raw:\n{raw[:300]}")
            return fallback

        resp = original_post(
            OLLAMA_URL,
            json={
                "model" : OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 200},
            },
            timeout=60,
        )
        dependencies = safe_parse_json(resp.json()["response"], fallback={})
        print(f"Prerequisites: {dependencies}")

        # STEP 4b: Build concept graph (NEW from main.py)
        data = collection.get(include=["documents", "metadatas"])
        filtered_chunks = []
        for doc, meta in zip(data["documents"], data["metadatas"]):
            if meta.get("topic") in (None, "", "Unknown"):
                continue
            if meta.get("concept_type") in ["definition", "explanation", "example"]:
                filtered_chunks.append({
                    "text"    : doc,
                    "topic"   : meta.get("topic"),
                    "subtopic": meta.get("subtopic")
                })
        filtered_chunks = filtered_chunks[:150]

        concept_graph = build_concept_graph(filtered_chunks)
        print(f"[Graph] Nodes: {len(concept_graph)}")
        print("[Graph] Done.\n")

        state["concept_graph"] = concept_graph

        # STEP 5: Compute topic difficulty
        topic_difficulties_raw = {}
        for m in all_meta:
            t = m.get("topic", "")
            d = m.get("difficulty", "")
            if not t or t == "Unknown" or not d:
                continue
            topic_difficulties_raw.setdefault(t, []).append(d)

        valid_topics = set(topic_difficulties_raw.keys())

        # Clean dependencies to only valid topics
        clean_dependencies = {}
        for topic, prereqs in dependencies.items():
            clean_dependencies[topic] = [p for p in prereqs if p in valid_topics and p != topic]
        dependencies = clean_dependencies
        print(f"\n[Fixed Prerequisites]: {dependencies}")

        state["dependencies"] = dependencies

        diff_map_nlp_to_rl = {
            "easy"        : "basic",
            "medium"      : "intermediate",
            "hard"        : "advanced",
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

        state["topics_difficulty"] = topics_difficulty

        # STEP 6: RL Agent
        from Adaptation_RL.Agent import AdaptiveAgent

        rl = AdaptiveAgent(
            topics_difficulty=topics_difficulty,
            prerequisites=dependencies,
            w1=0.4, w2=0.5, w3=0.1
        )
        state["rl"] = rl

        # Reset session tracking
        state["used_chunk_ids"]       = []
        state["asked_questions_log"]  = {}
        state["combo_question_count"] = {}

        # STEP 7: First question
        state["current"] = get_question()
        state["ready"]   = True

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

    for attempt in range(5):
        s = rl.ks.get_state_vector()
        a, _, _ = rl.select_action(s, training=False)
        topic, diff, qtype = rl.mdp.decode(a)

        combo_key      = (topic, diff, qtype)
        question_count = state["combo_question_count"].get(combo_key, 0)
        asked          = state["asked_questions_log"].get(topic, [])
        nlp_diff       = diff_map_rl_to_nlp[diff]

        # generate_question returns (result, new_ids) per main.py
        result, new_ids = generate_question(
            topic,
            nlp_diff,
            qtype,
            question_count=question_count,
            asked_questions=asked,
            prerequisites=state["dependencies"],
            concept_graph=state["concept_graph"],
            used_chunk_ids=state["used_chunk_ids"]
        )

        # Rolling used_chunk_ids memory (same as main.py)
        state["used_chunk_ids"].extend(new_ids)
        if len(state["used_chunk_ids"]) > MAX_MEM:
            state["used_chunk_ids"] = state["used_chunk_ids"][-MAX_MEM:]

        question  = result.get("question", "")
        reference = result.get("reference_answer", "")

        if question and question not in ("No question generated", "No data", "Insufficient data", "Error", ""):
            return {
                "topic"     : topic,
                "difficulty": diff,
                "qtype"     : qtype,
                "question"  : question,
                "reference" : reference
            }

        print(f"[WARN] Attempt {attempt+1}: Invalid question '{question}', retrying...")

    # All retries failed
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

        topic = state["current"]["topic"]
        diff  = state["current"]["difficulty"]
        ans = data["answer"].strip()
        if not ans:
            return jsonify({"error": "Empty answer"}), 400

        from NLP.Q_Generator_A_Evaluator.answer_evaluator import evaluate_answer

        ref   = state["current"]["reference"]
        qtype = state["current"]["qtype"]
        question = state["current"]["question"]

        eval_result = evaluate_answer(ans, ref, qtype,question)
        score = eval_result["final_score"]

        print(f"  Semantic    : {eval_result.get('semantic_score')}")
        print(f"  Keyword     : {eval_result.get('keyword_score')}")
        print(f"  NLI         : {eval_result.get('nli_score')}")
        print(f"  Completeness: {eval_result.get('completeness_score')}")
        print(f"  Final Score : {score}")

        rl = state["rl"]
        reward = rl.update(topic, score, diff, qtype)
        print(f"  RL Reward   : {round(reward, 3)}")

        # Update combo + asked logs (same as main.py)
        combo_key = (topic, diff, qtype)
        state["combo_question_count"][combo_key] = \
            state["combo_question_count"].get(combo_key, 0) + 1

        if topic not in state["asked_questions_log"]:
            state["asked_questions_log"][topic] = []
        state["asked_questions_log"][topic].append(state["current"]["question"])

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

"""
@app.route("/quit")
def quit_session():
    return jsonify({"history": state["history"]})
"""

@app.route("/quit")
def quit_session():
    rl  = state["rl"]
    ks  = rl.ks if rl else None
    summary = []
    weak_topics = []
    
    dependencies = state.get("dependencies", {})

    if ks:
        for topic in ks.topics:
            avg_score = round(float(ks.topic_score[topic]), 3)
            mastered  = ks.is_mastered(topic)
            attempts  = ks.attempts[topic]
            prereqs=dependencies.get(topic,[])
            summary.append({
                "topic"    : topic,
                "attempts" : attempts,
                "avg_score": avg_score,
                "mastered" : mastered,
            })
            
            if attempts>0 and avg_score < 0.5:
                weak_topics.append(topic)
                
            elif attempts==0 and prereqs:
                weak_topics.append(topic)

    state["summary"]     = summary
    state["weak_topics"] = weak_topics

    return jsonify({
        "history"    : state["history"],
        "summary"    : summary,
        "weak_topics": weak_topics,
    })


@app.route("/report")
def report():
    return render_template(
        "report.html",
        summary     = state.get("summary", []),
        history     = state.get("history", []),
        weak_topics = state.get("weak_topics", []),
    )

# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🌐 Flask running...")
    app.run(debug=True, use_reloader=False)