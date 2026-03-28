import streamlit as st
import streamlit.components.v1 as components
from question_generator import generate_question
from answer_evaluator import evaluate_answer
from reward_module import compute_reward

st.set_page_config(page_title="Adaptive Questioning", layout="centered")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>🎯 Adaptive Questioning System</h1>
<p style='text-align: center; color: gray;'>AI-powered learning with evaluation & reward</p>
<hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "question_data" not in st.session_state:
    st.session_state.question_data = None

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "show_details" not in st.session_state:
    st.session_state.show_details = False

if "prev_score" not in st.session_state:
    st.session_state.prev_score = 0.0

if "result" not in st.session_state:
    st.session_state.result = None

if "reward" not in st.session_state:
    st.session_state.reward = None


# ─────────────────────────────────────────────
# GENERATE QUESTION
# ─────────────────────────────────────────────

if st.button("🚀 Start / Next Question", use_container_width=True):

    query = "medium inferential question on inheritance"

    q = generate_question(query)

    st.session_state.question_data = q
    st.session_state.submitted = False
    st.session_state.show_details = False
    st.session_state.result = None
    st.session_state.reward = None


# ─────────────────────────────────────────────
# DISPLAY QUESTION
# ─────────────────────────────────────────────

if st.session_state.question_data:

    q = st.session_state.question_data

    st.markdown("### 📌 Question")
    st.info(q["question"])

    student_answer = st.text_area(
        "✍️ Enter your answer:",
        height=150,
        disabled=st.session_state.submitted
    )

    # ───── BLOCK PASTE ─────
    components.html("""
    <script>
    setTimeout(() => {
        const textarea = window.parent.document.querySelector('textarea');
        if (!textarea) return;

        textarea.addEventListener("paste", e => {
            e.preventDefault();
            alert("Paste disabled!");
        });

        textarea.addEventListener("keydown", e => {
            if ((e.ctrlKey || e.metaKey) &&
                ['v','c','x'].includes(e.key.toLowerCase())) {
                e.preventDefault();
            }
        });

        textarea.addEventListener("contextmenu", e => e.preventDefault());
    }, 500);
    </script>
    """, height=0)

    # ─────────────────────────────────────────
    # SUBMIT
    # ─────────────────────────────────────────

    if not st.session_state.submitted:

        if st.button("✅ Submit Answer", use_container_width=True):

            if not student_answer.strip():
                st.error("Please enter an answer.")
                st.stop()

            with st.spinner("Evaluating..."):

                result = evaluate_answer(
                    student_answer,
                    q["reference_answer"],
                    q["question_type"]
                )

                final_score = float(result["final_score"])

                reward = compute_reward(
                    final_score,
                    st.session_state.prev_score,
                    q.get("difficulty", "medium")
                )

                st.session_state.prev_score = final_score
                st.session_state.submitted = True
                st.session_state.result = result
                st.session_state.reward = reward


    # ─────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────

    if st.session_state.submitted:

        result = st.session_state.result
        final_score = float(result["final_score"])
        reward = st.session_state.reward

        st.markdown("### 📊 Final Score")
        st.success(f"{round(final_score, 3)}")
        st.progress(final_score)

        if st.button("📖 View Detailed Analysis"):
            st.session_state.show_details = True

        if st.session_state.show_details:

            st.markdown("### 🔍 Score Breakdown")
            st.json({
                "semantic_score": float(result["semantic_score"]),
                "keyword_score": float(result["keyword_score"]),
                "nli_score": float(result["nli_score"]),
                "completeness_score": float(result["completeness_score"])
            })

            st.markdown("### 🎯 Reward")
            st.success(round(reward, 3))

            with st.expander("📖 Reference Answer"):
                st.write(q["reference_answer"])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("➡️ Next Question"):
                st.session_state.question_data = None
                st.rerun()

        with col2:
            if st.button("❌ Exit"):
                st.session_state.clear()
                st.rerun()