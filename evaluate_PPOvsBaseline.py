import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from PPOAgent_1 import PPOAgent
from Agent import AdaptiveAgent
from rule_based_agent import RuleBasedAgent


def run_agent_session(agent, simulator, topics, n_questions, is_rl=True, student_nos=0):
    """
    Runs one session for either agent.
    Returns:
        score_progression      - list of scores per question
        topics_mastered        - number of topics mastered at end
        mastery_steps          - dict of topic -> question number when mastered (-1 if never)
        per_topic_qtype_scores - nested dict [topic][qtype] -> list of scores
    """
    score_progression      = []
    mastery_steps          = {t: -1 for t in topics}
    per_topic_qtype_scores = defaultdict(lambda: defaultdict(list))

    for step in range(n_questions):
        if is_rl:
            state_vector = agent.ks.get_state_vector()
            result = agent.select_action(state_vector, training=False)
            action_idx = result[0]
            topic, diff, qtype = agent.mdp.decode(action_idx)
        else:
            topic, diff, qtype = agent.select_action()

        score = simulator.get_score(topic, diff, qtype)
        if is_rl and student_nos == 0:
            print(f"Step {step}: {topic} | {diff} | {qtype} | score: {score:.2f}")
        agent.update(topic, score, diff, qtype)

        score_progression.append(score)
        per_topic_qtype_scores[topic][qtype].append(score)

        if mastery_steps[topic] == -1 and agent.ks.is_mastered(topic):
            mastery_steps[topic] = step + 1

    topics_mastered = sum(1 for v in mastery_steps.values() if v != -1)
    return score_progression, topics_mastered, mastery_steps, per_topic_qtype_scores


def reset_rl_agent(agent, topics):
    for topic in topics:
        agent.ks.topic_score[topic]    = 0.0
        agent.ks.attempts[topic]       = 0
        agent.ks.recent_scores[topic].clear()
        agent.ks.combo_scores[topic]   = defaultdict(list)
        agent.ks.prev_qtype[topic]     = (None, None)
        agent.ks.current_level[topic]  = {
            'diff_idx'        : 0,
            'qtype_idx'       : 0,
            'earned_diff_idx' : 0,
            'earned_qtype_idx': 0
        }
    agent.ks.prev_topic = None


def print_per_topic_report(topics, topics_difficulty,
                            ppo_per_topic_all, baseline_per_topic_all,
                            ppo_mastered_per_topic, baseline_mastered_per_topic,
                            n_students):
    print("\n" + "="*70)
    print("PER TOPIC BREAKDOWN")
    print("="*70)

    for topic in topics:
        diff = topics_difficulty[topic]
        print(f"\nTopic: {topic} ({diff})")
        print(f"  {'Question Type':<15} {'Baseline':>10} {'PPO':>10}")
        print(f"  {'-'*38}")

        for qtype in question_types:
            p_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in ppo_per_topic_all]
            b_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in baseline_per_topic_all]
            print(f"  {qtype:<15} {np.mean(b_scores):>10.3f} {np.mean(p_scores):>10.3f}")

        p_mastery_rate = ppo_mastered_per_topic[topic]      / n_students
        b_mastery_rate = baseline_mastered_per_topic[topic] / n_students
        print(f"  {'Mastery Rate':<15} {b_mastery_rate:>10.1%} {p_mastery_rate:>10.1%}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(topics_difficulty, prerequisites, w1=0.4, w2=0.5, w3=0.1, n_students=10, n_questions=500):
    """
    Compare PPO agent vs rule-based baseline on n_students simulated students.
    Both agents face identical simulated students for fair comparison.
    """
    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)

    baseline_agent = RuleBasedAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)
    print("Baseline agent ready.\n")

    print("Pretraining PPO agent...")
    ppo_agent = PPOAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)
    print("Pretraining done.\n")

    # storage
    baseline_scores_all         = []
    baseline_mastered_all       = []
    baseline_per_topic_all      = []
    baseline_mastered_per_topic = defaultdict(int)

    ppo_scores_all         = []
    ppo_mastered_all       = []
    ppo_per_topic_all      = []
    ppo_mastered_per_topic = defaultdict(int)

    for student in range(n_students):

        # generate fresh student and save state for fair comparison
        simulator.reset_mastery_scores()
        saved_mastery = simulator.mastery_topic.copy()

        # --- Baseline ---
        baseline_agent.reset(topics_difficulty)
        simulator.mastery_topic = saved_mastery.copy()
        b_scores, b_mastered, b_mastery_steps, b_per_topic = run_agent_session(
            baseline_agent, simulator, topics, n_questions, is_rl=False, student_nos=student
        )

        # --- PPO ---
        reset_rl_agent(ppo_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        p_scores, p_mastered, p_mastery_steps, p_per_topic = run_agent_session(
            ppo_agent, simulator, topics, n_questions, is_rl=True, student_nos=student
        )

        # store results
        baseline_scores_all.append(b_scores)
        baseline_mastered_all.append(b_mastered)
        baseline_per_topic_all.append(b_per_topic)

        ppo_scores_all.append(p_scores)
        ppo_mastered_all.append(p_mastered)
        ppo_per_topic_all.append(p_per_topic)

        # track per-topic mastery counts
        for topic in topics:
            if b_mastery_steps[topic] != -1:
                baseline_mastered_per_topic[topic] += 1
            if p_mastery_steps[topic] != -1:
                ppo_mastered_per_topic[topic] += 1

        print(f"Student {student+1:02d} | "
              f"PPO mastered: {p_mastered}/{len(topics)} | "
              f"Baseline mastered: {b_mastered}/{len(topics)}")

    # --- Summary ---
    ppo_avg_score      = np.mean([np.mean(s) for s in ppo_scores_all])
    baseline_avg_score = np.mean([np.mean(s) for s in baseline_scores_all])

    print(f"\n{'Metric':<30} {'Baseline':>10} {'PPO':>10}")
    print("="*52)
    print(f"{'Avg Final Score':<30} {baseline_avg_score:>10.3f} {ppo_avg_score:>10.3f}")
    print(f"{'Avg Topics Mastered':<30} "
          f"{np.mean(baseline_mastered_all):>10.2f}   "
          f"{np.mean(ppo_mastered_all):>10.2f}")

    ppo_mean_curve      = np.mean(ppo_scores_all, axis=0)
    baseline_mean_curve = np.mean(baseline_scores_all, axis=0)

    # per topic breakdown
    print_per_topic_report(
        topics, topics_difficulty,
        ppo_per_topic_all, baseline_per_topic_all,
        ppo_mastered_per_topic, baseline_mastered_per_topic,
        n_students
    )

    # --- Plot 1: Score Progression ---
    plt.figure(figsize=(10, 5))
    plt.plot(ppo_mean_curve,      label='PPO',      linewidth=2)
    plt.plot(baseline_mean_curve, label='Baseline', linewidth=2, linestyle='--')
    plt.xlabel('Question Number')
    plt.ylabel('Score')
    plt.title('Score Progression — PPO Agent vs Rule-Based Baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Images/eval_6.png', dpi=150)

    # --- Plot 2: Per-topic mastery rate bar chart ---
    ppo_mastery_rates = [ppo_mastered_per_topic[t]      / n_students for t in topics]
    b_mastery_rates   = [baseline_mastered_per_topic[t] / n_students for t in topics]
    x = np.arange(len(topics))

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.125, b_mastery_rates,   0.25, label='Baseline')
    plt.bar(x + 0.125, ppo_mastery_rates, 0.25, label='PPO')
    plt.xticks(x, [t[:20] for t in topics], rotation=15, ha='right')
    plt.ylabel('Mastery Rate')
    plt.title('Per-Topic Mastery Rate — PPO Agent vs Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Images/mastery_6.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    topics_difficulty = {
        # Level 0
        "Linear Algebra"   : "basic",
        "Calculus"         : "basic",
        # Level 1
        "Probability"      : "basic",
        "Optimization"     : "intermediate",
        # Level 2
        "Loss Functions"   : "intermediate",
        "Gradient Descent" : "intermediate",
        # Level 3
        "Neural Networks"  : "advanced"
    }

    prerequisites = {
        "Linear Algebra"   : [],
        "Calculus"         : [],
        "Probability"      : ["Linear Algebra"],
        "Optimization"     : ["Calculus"],
        "Loss Functions"   : ["Probability", "Calculus"],
        "Gradient Descent" : ["Optimization"],
        "Neural Networks"  : ["Loss Functions", "Gradient Descent"]
    }

    evaluate(topics_difficulty, prerequisites, w1=0.4, w2=0.5, w3=0.1, n_students=50, n_questions=2000)