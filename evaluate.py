import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from Agent import AdaptiveAgent


class RuleBasedAgent:
    
    def __init__(self, topics_difficulty, w1, w2, w3):
        self.ks  = KnowledgeState(topics_difficulty=topics_difficulty, window_size=10)
        self.mdp = MDP(
            list(topics_difficulty.keys()),
            difficulty_types=['basic', 'intermediate', 'advanced'],
            q_types=['factual', 'inferential', 'evaluative'],
            w1=w1, w2=w2, w3=w3
        )
        self.topics      = list(topics_difficulty.keys())
        self.topic_idx   = 0
        self.last_score  = defaultdict(float)
        self.curr_diff   = {t: 'basic' for t in self.topics}

    def select_action(self):
        topic = self.topics[self.topic_idx % len(self.topics)]
        self.topic_idx += 1

        last = self.last_score[topic]
        if last > 0.8 and self.curr_diff[topic] != 'advanced':
            self.curr_diff[topic] = difficulty_level[
                min(difficulty_level.index(self.curr_diff[topic]) + 1, 2)
            ]
        elif last < 0.4 and self.curr_diff[topic] != 'basic':
            self.curr_diff[topic] = difficulty_level[
                max(difficulty_level.index(self.curr_diff[topic]) - 1, 0)
            ]

        return topic, self.curr_diff[topic], 'factual'

    def update(self, topic, score, difficulty, question_type):
        self.last_score[topic] = score
        self.ks.update(topic, score, difficulty, question_type)

    def reset(self, topics_difficulty):
        self.topic_idx  = 0
        self.last_score = defaultdict(float)
        self.curr_diff  = {t: 'basic' for t in self.topics}
        for topic in self.topics:
            self.ks.topic_score[topic]  = 0.0
            self.ks.attempts[topic]     = 0
            self.ks.recent_scores[topic].clear()
            self.ks.prev_qtype[topic]   = (None, None)


def run_agent_session(agent, simulator, topics, n_questions, is_rl=True):
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
            state_vector        = agent.ks.get_state_vector()
            action_idx, _       = agent.select_action(state_vector, training=False)
            topic, diff, qtype  = agent.mdp.decode(action_idx)
        else:
            topic, diff, qtype  = agent.select_action()

        score = simulator.get_score(topic, diff, qtype)
        agent.update(topic, score, diff, qtype)

        score_progression.append(score)
        per_topic_qtype_scores[topic][qtype].append(score)

        if mastery_steps[topic] == -1 and agent.ks.is_mastered(topic):
            mastery_steps[topic] = step + 1

    topics_mastered = sum(1 for v in mastery_steps.values() if v != -1)
    return score_progression, topics_mastered, mastery_steps, per_topic_qtype_scores


def reset_rl_agent(agent, topics):
    """Reset RL agent knowledge state between sessions."""
    for topic in topics:
        agent.ks.topic_score[topic]  = 0.0
        agent.ks.attempts[topic]     = 0
        agent.ks.recent_scores[topic].clear()
        agent.ks.prev_qtype[topic]   = (None, None)


def print_per_topic_report(topics, topics_difficulty,
                            rl_per_topic_all, baseline_per_topic_all,
                            rl_mastered_per_topic, baseline_mastered_per_topic,
                            n_students):
    print("\n" + "="*70)
    print("PER TOPIC BREAKDOWN")
    print("="*70)

    for topic in topics:
        diff = topics_difficulty[topic]
        print(f"\nTopic: {topic} ({diff})")
        print(f"  {'Question Type':<15} {'Baseline':>10} {'RL Agent':>10}")
        print(f"  {'-'*35}")

        for qtype in question_types:
            rl_scores = [
                np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0
                for s in rl_per_topic_all
            ]
            b_scores = [
                np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0
                for s in baseline_per_topic_all
            ]
            print(f"  {qtype:<15} {np.mean(b_scores):>10.3f} {np.mean(rl_scores):>10.3f}")

        rl_mastery_rate = rl_mastered_per_topic[topic] / n_students
        b_mastery_rate  = baseline_mastered_per_topic[topic] / n_students
        print(f"  {'Mastery Rate':<15} {b_mastery_rate:>10.1%} {rl_mastery_rate:>10.1%}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(topics_difficulty, w1=0.6, w2=0.3, w3=0.1, n_students=20, n_questions=25):
    """
    Compare RL agent vs rule-based baseline on n_students simulated students.
    Both agents face identical simulated students for fair comparison.
    """
    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)

    print("Pretraining RL agent...")
    rl_agent       = AdaptiveAgent(topics_difficulty, w1=w1, w2=w2, w3=w3)
    baseline_agent = RuleBasedAgent(topics_difficulty, w1=w1, w2=w2, w3=w3)
    print("Pretraining done.\n")

    # storage
    rl_scores_all               = []
    baseline_scores_all         = []
    rl_mastered_all             = []
    baseline_mastered_all       = []
    rl_per_topic_all            = []
    baseline_per_topic_all      = []
    rl_mastered_per_topic       = defaultdict(int)
    baseline_mastered_per_topic = defaultdict(int)

    for student in range(n_students):

        # generate fresh student and save state for fair comparison
        simulator.reset_mastery_scores()
        saved_mastery = simulator.mastery_topic.copy()

        # --- run baseline ---
        baseline_agent.reset(topics_difficulty)
        simulator.mastery_topic = saved_mastery.copy()
        b_scores, b_mastered, b_mastery_steps, b_per_topic = run_agent_session(
            baseline_agent, simulator, topics, n_questions, is_rl=False
        )

        # --- run RL agent (same student) ---
        reset_rl_agent(rl_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        rl_scores, rl_mastered, rl_mastery_steps, rl_per_topic = run_agent_session(
            rl_agent, simulator, topics, n_questions, is_rl=True
        )

        # store results
        rl_scores_all.append(rl_scores)
        baseline_scores_all.append(b_scores)
        rl_mastered_all.append(rl_mastered)
        baseline_mastered_all.append(b_mastered)
        rl_per_topic_all.append(rl_per_topic)
        baseline_per_topic_all.append(b_per_topic)

        # track per-topic mastery counts
        for topic in topics:
            if rl_mastery_steps[topic] != -1:
                rl_mastered_per_topic[topic] += 1
            if b_mastery_steps[topic] != -1:
                baseline_mastered_per_topic[topic] += 1

        print(f"Student {student+1:02d} | "
              f"RL mastered: {rl_mastered}/{len(topics)} | "
              f"Baseline mastered: {b_mastered}/{len(topics)}")


    rl_avg_score          = np.mean([np.mean(s) for s in rl_scores_all])
    baseline_avg_score    = np.mean([np.mean(s) for s in baseline_scores_all])
    rl_avg_mastered       = np.mean(rl_mastered_all)
    baseline_avg_mastered = np.mean(baseline_mastered_all)

    print("\n" + "="*55)
    print(f"{'Metric':<30} {'Baseline':>10} {'RL Agent':>10}")
    print("="*55)
    print(f"{'Avg Final Score':<30} {baseline_avg_score:>10.3f} {rl_avg_score:>10.3f}")
    print(f"{'Avg Topics Mastered':<30} {baseline_avg_mastered:>10.2f} {rl_avg_mastered:>10.2f}")
    print("="*55)

    # per topic breakdown
    print_per_topic_report(
        topics, topics_difficulty,
        rl_per_topic_all, baseline_per_topic_all,
        rl_mastered_per_topic, baseline_mastered_per_topic,
        n_students
    )

    
    rl_mean_curve       = np.mean(rl_scores_all, axis=0)
    baseline_mean_curve = np.mean(baseline_scores_all, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(rl_mean_curve,       label='RL Agent', linewidth=2)
    plt.plot(baseline_mean_curve, label='Baseline', linewidth=2, linestyle='--')
    plt.xlabel('Question Number')
    plt.ylabel('Score')
    plt.title('Score Progression — RL Agent vs Rule-Based Baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('evaluation_plot.png', dpi=150)
    plt.show()

    # 2. Per-topic mastery rate bar chart
    rl_mastery_rates  = [rl_mastered_per_topic[t]       / n_students for t in topics]
    b_mastery_rates   = [baseline_mastered_per_topic[t] / n_students for t in topics]
    x = np.arange(len(topics))

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, b_mastery_rates,  0.4, label='Baseline')
    plt.bar(x + 0.2, rl_mastery_rates, 0.4, label='RL Agent')
    plt.xticks(x, [t[:20] for t in topics], rotation=15, ha='right')
    plt.ylabel('Mastery Rate')
    plt.title('Per-Topic Mastery Rate — RL Agent vs Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mastery_plot.png', dpi=150)
    plt.show()

    print("\nPlots saved to evaluation_plot.png and mastery_plot.png")


if __name__ == "__main__":

    topics_difficulty = {
        "Neural Networks"     : "advanced",
        "Gradient Descent"    : "intermediate",
        "Backpropagation"     : "advanced",
        "Activation Functions": "basic",
        "Overfitting"         : "intermediate"
    }

    evaluate(
        topics_difficulty,
        w1=0.6, w2=0.3, w3=0.1,
        n_students=20,
        n_questions=25
    )