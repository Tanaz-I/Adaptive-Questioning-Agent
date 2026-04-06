"""
llm_judge_eval.py
=================
LLM-as-a-Judge evaluation of the Adaptive Learning System.

Reads:   simulation_results/strong_qa_summary.csv
         simulation_results/weak_qa_summary.csv

Outputs: simulation_results/llm_judge_results.csv   ← per-row judge scores
         simulation_results/llm_judge_report.json    ← aggregated metrics

Four evaluation dimensions:
  1. Question Quality     — is the question clear, relevant, appropriately difficult?
  2. Answer Score Audit   — does the LLM judge agree with your NLP scorer's final_score?
  3. Multi-hop Depth      — does the question actually require the claimed reasoning type?
  4. RL Adaptation        — did the system correctly adapt to weak vs strong student?

Usage:
    python llm_judge_eval.py
    python llm_judge_eval.py --sample 50      # sample 50 rows per student type
    python llm_judge_eval.py --student strong # only one file
"""

import json
import csv
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
JUDGE_MODEL  = "llama3"          # use your best available model
RESULTS_DIR  = Path("./simulation_results")
OUTPUT_CSV   = RESULTS_DIR / "llm_judge_results.csv"
OUTPUT_JSON  = RESULTS_DIR / "llm_judge_report.json"

RETRY_LIMIT  = 3
RETRY_DELAY  = 2


# ─────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────

def call_llm(prompt: str, temperature: float = 0.1) -> str:
    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": JUDGE_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": 400},
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [LLM ERROR] {e}")
                return ""
    return ""


def safe_parse_json(raw: str) -> dict:
    raw = raw.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {}


# ─────────────────────────────────────────────
# Dimension 1 — Question Quality
# ─────────────────────────────────────────────

def judge_question_quality(row: dict) -> dict:
    """
    Asks the LLM to rate the question on 4 criteria, each 1-5.
    Returns scores + a brief rationale.
    """
    prompt = f"""You are an expert educator evaluating questions in an intelligent tutoring system.

Topic: {row['topic']}
Difficulty level: {row['difficulty']}
Question type: {row['question_type']}

Question:
{row['question']}

Reference answer:
{row['reference_answer']}

Rate the question on each criterion from 1 (very poor) to 5 (excellent):

1. clarity        — Is the question clearly worded with no ambiguity?
2. relevance      — Does it genuinely test knowledge of the topic?
3. difficulty_fit — Does the difficulty match the stated level ({row['difficulty']})?
4. type_fit       — Does the question actually require {row['question_type']} reasoning?
                    (factual = recall, inferential = connecting ideas, evaluative = judgement/tradeoff)

Return ONLY valid JSON. No explanation outside the JSON.
{{
  "clarity":        <1-5>,
  "relevance":      <1-5>,
  "difficulty_fit": <1-5>,
  "type_fit":       <1-5>,
  "rationale":      "<one sentence>"
}}"""

    raw    = call_llm(prompt)
    result = safe_parse_json(raw)

    return {
        "q_clarity":        int(result.get("clarity",        0)),
        "q_relevance":      int(result.get("relevance",      0)),
        "q_difficulty_fit": int(result.get("difficulty_fit", 0)),
        "q_type_fit":       int(result.get("type_fit",       0)),
        "q_rationale":      result.get("rationale", ""),
        "q_avg":            round(np.mean([
            int(result.get("clarity",        0)),
            int(result.get("relevance",      0)),
            int(result.get("difficulty_fit", 0)),
            int(result.get("type_fit",       0)),
        ]), 2) if result else 0.0,
    }


# ─────────────────────────────────────────────
# Dimension 2 — Answer Score Audit
# ─────────────────────────────────────────────

def judge_answer_score(row: dict) -> dict:
    """
    Asks the LLM to independently score the student answer 0.0-1.0,
    then we measure agreement with the system's final_score.
    """
    prompt = f"""You are an expert educator grading a student's answer.

Topic: {row['topic']}
Question: {row['question']}

Reference answer (correct answer):
{row['reference_answer']}

Student answer:
{row['student_answer']}

Grade the student answer on a scale from 0.0 to 1.0:
- 1.0 = completely correct, covers all key points
- 0.7 = mostly correct, minor gaps
- 0.5 = partially correct, some key points missing
- 0.3 = mostly wrong but shows some understanding
- 0.0 = completely wrong or irrelevant

Return ONLY valid JSON. No explanation outside the JSON.
{{
  "judge_score": <0.0 to 1.0>,
  "verdict": "<correct|partial|incorrect>",
  "missing_points": "<key points missing from student answer, or 'none'>"
}}"""

    raw    = call_llm(prompt)
    result = safe_parse_json(raw)

    judge_score  = float(result.get("judge_score", -1))
    system_score = float(row["final_score"])

    agreement    = round(1.0 - abs(judge_score - system_score), 3) if judge_score >= 0 else -1
    score_gap    = round(judge_score - system_score, 3)            if judge_score >= 0 else 0

    return {
        "judge_score":    judge_score,
        "system_score":   system_score,
        "score_gap":      score_gap,       # positive = system underscored, negative = overscored
        "score_agreement":agreement,
        "verdict":        result.get("verdict", ""),
        "missing_points": result.get("missing_points", ""),
    }


# ─────────────────────────────────────────────
# Dimension 3 — Multi-hop Depth Check
# ─────────────────────────────────────────────

def judge_multihop_depth(row: dict) -> dict:
    """
    Only runs for inferential and evaluative questions.
    Checks whether the question actually requires the claimed depth.
    """
    if row["question_type"] == "factual":
        return {"hop_depth_score": -1, "hop_comment": "n/a (factual)"}

    if row["question_type"] == "inferential":
        requirement = "requires connecting two distinct ideas — the answer cannot be found in a single sentence of the reference"
    else:  # evaluative
        requirement = "requires the student to take a position, compare tradeoffs, or make a judgment — not just recall facts"

    prompt = f"""You are an expert educator checking whether a question requires deep reasoning.

Topic: {row['topic']}
Claimed question type: {row['question_type']}
Requirement for this type: {requirement}

Question:
{row['question']}

Reference answer:
{row['reference_answer']}

Does this question actually satisfy the requirement above?
Score 1-5:
  5 = fully satisfies — genuinely requires {row['question_type']} reasoning
  3 = partially — could be answered with shallow reasoning
  1 = fails — this is essentially a factual recall question regardless of its label

Return ONLY valid JSON.
{{
  "hop_depth_score": <1-5>,
  "hop_comment": "<one sentence explaining why>"
}}"""

    raw    = call_llm(prompt)
    result = safe_parse_json(raw)

    return {
        "hop_depth_score": int(result.get("hop_depth_score", 0)),
        "hop_comment":     result.get("hop_comment", ""),
    }


# ─────────────────────────────────────────────
# Dimension 4 — RL Adaptation Check
# (session-level, not per-row)
# ─────────────────────────────────────────────

def judge_rl_adaptation(strong_df: pd.DataFrame, weak_df: pd.DataFrame) -> dict:
    """
    Feeds the LLM a compact session summary for both students
    and asks it to evaluate whether the system adapted correctly.
    This is a single call covering the whole session.
    """

    def make_summary(df: pd.DataFrame, student_type: str) -> str:
        # topic-level summary: attempts, avg score, mastered
        last = df.groupby("topic").last().reset_index()
        lines = [f"Student type: {student_type}"]
        lines.append(f"Total questions: {len(df)}")
        lines.append(f"Overall avg score: {round(df['final_score'].mean(), 3)}")
        lines.append(f"Topics covered: {df['topic'].nunique()}")
        lines.append(f"Topics mastered: {last['is_mastered'].sum()}")
        lines.append("")
        lines.append("Difficulty distribution (% of questions):")
        for d, g in df.groupby("difficulty"):
            lines.append(f"  {d}: {round(100*len(g)/len(df))}%")
        lines.append("")
        lines.append("Question type distribution (% of questions):")
        for qt, g in df.groupby("question_type"):
            lines.append(f"  {qt}: {round(100*len(g)/len(df))}%")
        lines.append("")
        lines.append("Per-topic avg scores (last 5 topics by attempts):")
        top5 = last.nlargest(5, "topic_attempts_so_far")[
            ["topic", "topic_attempts_so_far", "topic_avg_score_so_far", "is_mastered"]
        ]
        for _, r in top5.iterrows():
            lines.append(f"  {r['topic']}: avg={r['topic_avg_score_so_far']}, "
                         f"attempts={r['topic_attempts_so_far']}, mastered={r['is_mastered']}")
        return "\n".join(lines)

    strong_summary = make_summary(strong_df, "strong")
    weak_summary   = make_summary(weak_df,   "weak")

    prompt = f"""You are an expert evaluating an adaptive intelligent tutoring system (ITS).

The system uses a PPO reinforcement learning agent to select which topic, difficulty level,
and question type to ask next, based on a student's performance history.

Below are session summaries for two simulated students — one strong, one weak.

=== STRONG STUDENT ===
{strong_summary}

=== WEAK STUDENT ===
{weak_summary}

Evaluate the RL agent's adaptation on these criteria, each scored 1-5:

1. difficulty_progression — Did the system appropriately give harder questions to the strong student
   and easier/more basic questions to the weak student?

2. topic_coverage — Did the system cover a reasonable spread of topics, or did it get stuck
   on one topic?

3. weak_student_support — For the weak student, did the system revisit topics with low scores
   rather than abandoning them?

4. strong_student_challenge — For the strong student, did the system push toward advanced
   difficulty and evaluative question types?

5. overall_adaptation — Overall, did the system behave differently and appropriately
   for the two student types?

Return ONLY valid JSON.
{{
  "difficulty_progression": <1-5>,
  "topic_coverage":         <1-5>,
  "weak_student_support":   <1-5>,
  "strong_student_challenge":<1-5>,
  "overall_adaptation":     <1-5>,
  "summary": "<2-3 sentence overall assessment>"
}}"""

    raw    = call_llm(prompt, temperature=0.2)
    result = safe_parse_json(raw)

    return {
        "rl_difficulty_progression":  int(result.get("difficulty_progression",   0)),
        "rl_topic_coverage":          int(result.get("topic_coverage",           0)),
        "rl_weak_student_support":    int(result.get("weak_student_support",     0)),
        "rl_strong_student_challenge":int(result.get("strong_student_challenge", 0)),
        "rl_overall_adaptation":      int(result.get("overall_adaptation",       0)),
        "rl_summary":                 result.get("summary", ""),
    }


# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────

def evaluate(sample_n: int = None, student_filter: str = None):

    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Load CSVs ─────────────────────────────────────────────
    files = {}
    for stype in ["strong", "weak"]:
        if student_filter and stype != student_filter:
            continue
        path = RESULTS_DIR / f"{stype}_qa_summary.csv"
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        if sample_n:
            df = df.sample(min(sample_n, len(df)), random_state=42).reset_index(drop=True)
        files[stype] = df
        print(f"Loaded {stype}: {len(df)} rows")

    if not files:
        print("[ERROR] No CSV files found in simulation_results/")
        return

    all_rows     = pd.concat(files.values(), ignore_index=True)
    total        = len(all_rows)
    output_rows  = []

    # ── Per-row evaluation (Dims 1, 2, 3) ────────────────────
    print(f"\nRunning per-row evaluation ({total} rows) ...")
    print("Dimensions: Question Quality | Answer Score Audit | Multi-hop Depth\n")

    for i, row in all_rows.iterrows():
        print(f"  [{i+1}/{total}] step={row['step']} | {row['student_type']} | "
              f"{row['topic'][:30]} | {row['difficulty']} | {row['question_type']}")

        q_scores   = judge_question_quality(row)
        ans_scores = judge_answer_score(row)
        hop_scores = judge_multihop_depth(row)

        output_row = {
            "step":         row["step"],
            "student_type": row["student_type"],
            "topic":        row["topic"],
            "difficulty":   row["difficulty"],
            "question_type":row["question_type"],
            **q_scores,
            **ans_scores,
            **hop_scores,
        }
        output_rows.append(output_row)

        # Save incrementally — safe if run crashes mid-way
        if (i + 1) % 10 == 0 or (i + 1) == total:
            tmp_df = pd.DataFrame(output_rows)
            tmp_df.to_csv(OUTPUT_CSV, index=False)
            print(f"    [Checkpoint saved: {OUTPUT_CSV}]")

    results_df = pd.DataFrame(output_rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPer-row results saved → {OUTPUT_CSV}")

    # ── Session-level RL evaluation (Dim 4) ──────────────────
    rl_scores = {}
    if "strong" in files and "weak" in files:
        print("\nRunning RL adaptation evaluation (session-level) ...")
        rl_scores = judge_rl_adaptation(files["strong"], files["weak"])
        print(f"  RL scores: {rl_scores}")
    else:
        print("\n[SKIP] RL adaptation requires both strong and weak CSVs.")

    # ── Aggregate metrics ─────────────────────────────────────
    report = build_report(results_df, rl_scores, files)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved → {OUTPUT_JSON}")

    print_report(report)


# ─────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────

def build_report(df: pd.DataFrame, rl_scores: dict, files: dict) -> dict:

    def avg(series):
        s = series.replace(-1, float("nan")).dropna()
        return round(s.mean(), 3) if len(s) else None

    report = {}

    # ── Dimension 1: Question Quality ────────────────────────
    report["question_quality"] = {
        "overall_avg":       avg(df["q_avg"]),
        "clarity":           avg(df["q_clarity"]),
        "relevance":         avg(df["q_relevance"]),
        "difficulty_fit":    avg(df["q_difficulty_fit"]),
        "type_fit":          avg(df["q_type_fit"]),
        "by_difficulty": {
            d: round(grp["q_avg"].mean(), 3)
            for d, grp in df.groupby("difficulty")
        },
        "by_question_type": {
            qt: round(grp["q_avg"].mean(), 3)
            for qt, grp in df.groupby("question_type")
        },
    }

    # ── Dimension 2: Answer Score Audit ──────────────────────
    valid = df[df["judge_score"] >= 0]
    report["answer_score_audit"] = {
        "avg_judge_score":       avg(valid["judge_score"]),
        "avg_system_score":      avg(valid["system_score"]),
        "avg_score_gap":         avg(valid["score_gap"]),        # + = system underscore
        "avg_agreement":         avg(valid["score_agreement"]),
        "correlation":           round(
            valid["judge_score"].corr(valid["system_score"]), 3
        ) if len(valid) > 1 else None,
        "system_overscored_pct": round(
            100 * (valid["score_gap"] < -0.15).mean(), 1
        ),  # system gave > 0.15 more than judge
        "system_underscored_pct": round(
            100 * (valid["score_gap"] > 0.15).mean(), 1
        ),
        "by_student": {
            st: {
                "avg_judge":    round(grp["judge_score"].mean(), 3),
                "avg_system":   round(grp["system_score"].mean(), 3),
                "avg_gap":      round(grp["score_gap"].mean(), 3),
                "correlation":  round(grp["judge_score"].corr(grp["system_score"]), 3)
                                if len(grp) > 1 else None,
            }
            for st, grp in valid.groupby("student_type")
        },
    }

    # ── Dimension 3: Multi-hop Depth ─────────────────────────
    hop_df = df[df["hop_depth_score"] > 0]
    report["multihop_depth"] = {
        "avg_depth_score":        avg(hop_df["hop_depth_score"]),
        "pct_genuinely_inferential": round(
            100 * (hop_df[hop_df["question_type"] == "inferential"]["hop_depth_score"] >= 4).mean(), 1
        ) if len(hop_df[hop_df["question_type"] == "inferential"]) else None,
        "pct_genuinely_evaluative": round(
            100 * (hop_df[hop_df["question_type"] == "evaluative"]["hop_depth_score"] >= 4).mean(), 1
        ) if len(hop_df[hop_df["question_type"] == "evaluative"]) else None,
        "by_question_type": {
            qt: round(grp["hop_depth_score"].mean(), 3)
            for qt, grp in hop_df.groupby("question_type")
        },
    }

    # ── Dimension 4: RL Adaptation ───────────────────────────
    report["rl_adaptation"] = rl_scores

    # ── Quick per-student summary ────────────────────────────
    if files:
        report["student_comparison"] = {}
        for st, sdf in files.items():
            last = sdf.groupby("topic").last().reset_index()
            report["student_comparison"][st] = {
                "total_questions":   len(sdf),
                "avg_system_score":  round(sdf["final_score"].mean(), 3),
                "topics_mastered":   int(last["is_mastered"].sum()),
                "pct_advanced_diff": round(100*(sdf["difficulty"]=="advanced").mean(), 1),
                "pct_evaluative":    round(100*(sdf["question_type"]=="evaluative").mean(), 1),
            }

    return report


# ─────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────

def print_report(report: dict):
    print("\n" + "="*60)
    print("LLM-AS-A-JUDGE EVALUATION REPORT")
    print("="*60)

    qq = report.get("question_quality", {})
    print(f"\n[1] QUESTION QUALITY  (1-5 scale)")
    print(f"    Overall avg     : {qq.get('overall_avg')}")
    print(f"    Clarity         : {qq.get('clarity')}")
    print(f"    Relevance       : {qq.get('relevance')}")
    print(f"    Difficulty fit  : {qq.get('difficulty_fit')}")
    print(f"    Type fit        : {qq.get('type_fit')}")
    print(f"    By difficulty   : {qq.get('by_difficulty')}")
    print(f"    By q-type       : {qq.get('by_question_type')}")

    au = report.get("answer_score_audit", {})
    print(f"\n[2] ANSWER SCORE AUDIT")
    print(f"    Judge avg score : {au.get('avg_judge_score')}")
    print(f"    System avg score: {au.get('avg_system_score')}")
    print(f"    Avg gap         : {au.get('avg_score_gap')}  (+= system underscore)")
    print(f"    Correlation     : {au.get('correlation')}")
    print(f"    Overscored  >0.15: {au.get('system_overscored_pct')}%")
    print(f"    Underscored >0.15: {au.get('system_underscored_pct')}%")
    for st, s in au.get("by_student", {}).items():
        print(f"    [{st}] judge={s['avg_judge']} system={s['avg_system']} "
              f"gap={s['avg_gap']} corr={s['correlation']}")

    mh = report.get("multihop_depth", {})
    print(f"\n[3] MULTI-HOP DEPTH  (1-5 scale)")
    print(f"    Avg depth score           : {mh.get('avg_depth_score')}")
    print(f"    % genuinely inferential   : {mh.get('pct_genuinely_inferential')}%")
    print(f"    % genuinely evaluative    : {mh.get('pct_genuinely_evaluative')}%")
    print(f"    By q-type                 : {mh.get('by_question_type')}")

    rl = report.get("rl_adaptation", {})
    print(f"\n[4] RL ADAPTATION  (1-5 scale)")
    print(f"    Difficulty progression  : {rl.get('rl_difficulty_progression')}")
    print(f"    Topic coverage          : {rl.get('rl_topic_coverage')}")
    print(f"    Weak student support    : {rl.get('rl_weak_student_support')}")
    print(f"    Strong student challenge: {rl.get('rl_strong_student_challenge')}")
    print(f"    Overall adaptation      : {rl.get('rl_overall_adaptation')}")
    print(f"    Summary: {rl.get('rl_summary')}")

    sc = report.get("student_comparison", {})
    if sc:
        print(f"\n[5] STUDENT COMPARISON")
        for st, s in sc.items():
            print(f"    [{st}] questions={s['total_questions']}  "
                  f"avg_score={s['avg_system_score']}  "
                  f"mastered={s['topics_mastered']}  "
                  f"advanced%={s['pct_advanced_diff']}  "
                  f"evaluative%={s['pct_evaluative']}")

    print("\n" + "="*60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",  type=int, default=100,
                        help="Sample N rows per student type (default: all)")
    parser.add_argument("--student", type=str, default=None,
                        choices=["strong", "weak"],
                        help="Evaluate only one student type")
    args = parser.parse_args()

    evaluate(sample_n=args.sample, student_filter=args.student)