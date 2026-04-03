import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from DQNAgent import DQNAgent
from Agent import AdaptiveAgent


class RuleBasedAgent:
    
    def __init__(self, topics_difficulty, prerequisites, w1, w2, w3):
        self.ks  = KnowledgeState(topics_difficulty=topics_difficulty, prerequisites=prerequisites, window_size=10)
        self.mdp = MDP(
            list(topics_difficulty.keys()),
            difficulty_types=['basic', 'intermediate', 'advanced'],
            q_types=['factual', 'inferential', 'evaluative'],
            w1=w1, w2=w2, w3=w3
        )
        self.topics    = list(topics_difficulty.keys())
        self.topic_idx = 0
        self.last_score = defaultdict(float)
        self.curr_diff  = {t: 'basic' for t in self.topics}

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

        qtype_cycle = ['factual', 'inferential', 'evaluative']
        qtype = qtype_cycle[self.topic_idx % 3]
        return topic, self.curr_diff[topic], qtype

    def update(self, topic, score, difficulty, question_type):
        self.last_score[topic] = score
        self.ks.update(topic, score, difficulty, question_type)

    def reset(self, topics_difficulty):
        self.topic_idx  = 0
        self.last_score = defaultdict(float)
        self.curr_diff  = {t: 'basic' for t in self.topics}
        for topic in self.topics:
            self.ks.topic_score[topic]   = 0.0
            self.ks.attempts[topic]      = 0
            self.ks.recent_scores[topic].clear()
            self.ks.combo_scores[topic]  = defaultdict(list)
            self.ks.prev_qtype[topic]    = (None, None)
            self.ks.current_level[topic] = {
                'diff_idx': 0, 'qtype_idx': 0,
                'earned_diff_idx': 0, 'earned_qtype_idx': 0
            }
        self.ks.prev_topic = None


def run_agent_session(agent, simulator, topics, n_questions, is_rl=True, student_nos=0):
    score_progression      = []
    mastery_steps          = {t: -1 for t in topics}
    per_topic_qtype_scores = defaultdict(lambda: defaultdict(list))

    for step in range(n_questions):
        if is_rl:
            state_vector = agent.ks.get_state_vector()
            # DQN returns int; REINFORCE returns (int, log_prob, entropy)
            result     = agent.select_action(state_vector, training=False)
            action_idx = result[0] if isinstance(result, tuple) else result
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
        agent.ks.topic_score[topic]   = 0.0
        agent.ks.attempts[topic]      = 0
        agent.ks.recent_scores[topic].clear()
        agent.ks.combo_scores[topic]  = defaultdict(list)
        agent.ks.prev_qtype[topic]    = (None, None)
        agent.ks.current_level[topic] = {
            'diff_idx': 0, 'qtype_idx': 0,
            'earned_diff_idx': 0, 'earned_qtype_idx': 0
        }
    agent.ks.prev_topic = None

    # DQN-specific: clear replay buffer so the previous student's
    # experience doesn't bleed into the next student's evaluation
    if hasattr(agent, 'replay'):
        agent.replay.buffer.clear()


def print_per_topic_report(topics, topics_difficulty,
                            reinforce_per_topic_all, dqn_per_topic_all, baseline_per_topic_all,
                            reinforce_mastered_per_topic, dqn_mastered_per_topic,
                            baseline_mastered_per_topic, n_students):
    print("\n" + "=" * 80)
    print("PER TOPIC BREAKDOWN")
    print("=" * 80)

    for topic in topics:
        diff = topics_difficulty[topic]
        print(f"\nTopic: {topic} ({diff})")
        print(f"  {'Question Type':<15} {'Baseline':>10} {'REINFORCE':>12} {'DQN':>10}")
        print(f"  {'-' * 50}")

        for qtype in question_types:
            r_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in reinforce_per_topic_all]
            d_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in dqn_per_topic_all]
            b_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in baseline_per_topic_all]
            print(f"  {qtype:<15} {np.mean(b_scores):>10.3f} {np.mean(r_scores):>12.3f} {np.mean(d_scores):>10.3f}")

        r_mastery_rate = reinforce_mastered_per_topic[topic] / n_students
        d_mastery_rate = dqn_mastered_per_topic[topic]       / n_students
        b_mastery_rate = baseline_mastered_per_topic[topic]  / n_students
        print(f"  {'Mastery Rate':<15} {b_mastery_rate:>10.1%} {r_mastery_rate:>12.1%} {d_mastery_rate:>10.1%}")


def evaluate(topics_difficulty, prerequisites, w1=0.4, w2=0.5, w3=0.1, n_students=10, n_questions=500):
    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)

    print("Pretraining DQN agent...")
    dqn_agent = DQNAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    print("Pretraining REINFORCE agent...")
    reinforce_agent = AdaptiveAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    baseline_agent = RuleBasedAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)
    print("Pretraining done.\n")

    reinforce_scores_all, reinforce_mastered_all, reinforce_per_topic_all = [], [], []
    dqn_scores_all,       dqn_mastered_all,       dqn_per_topic_all       = [], [], []
    baseline_scores_all,  baseline_mastered_all,  baseline_per_topic_all  = [], [], []

    reinforce_mastered_per_topic = defaultdict(int)
    dqn_mastered_per_topic       = defaultdict(int)
    baseline_mastered_per_topic  = defaultdict(int)

    for student in range(n_students):
        simulator.reset_mastery_scores()
        saved_mastery = simulator.mastery_topic.copy()

        # Baseline
        baseline_agent.reset(topics_difficulty)
        simulator.mastery_topic = saved_mastery.copy()
        b_scores, b_mastered, b_mastery_steps, b_per_topic = run_agent_session(
            baseline_agent, simulator, topics, n_questions, is_rl=False, student_nos=student
        )

        # REINFORCE
        reset_rl_agent(reinforce_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        r_scores, r_mastered, r_mastery_steps, r_per_topic = run_agent_session(
            reinforce_agent, simulator, topics, n_questions, is_rl=True
        )

        # DQN — keep a small epsilon so the agent visits all (diff, qtype)
        # combos that is_mastered() requires; pure greedy misses rare combos
        reset_rl_agent(dqn_agent, topics)
        dqn_agent.eps = 0.1
        simulator.mastery_topic = saved_mastery.copy()
        d_scores, d_mastered, d_mastery_steps, d_per_topic = run_agent_session(
            dqn_agent, simulator, topics, n_questions, is_rl=True
        )

        reinforce_scores_all.append(r_scores)
        reinforce_mastered_all.append(r_mastered)
        reinforce_per_topic_all.append(r_per_topic)

        dqn_scores_all.append(d_scores)
        dqn_mastered_all.append(d_mastered)
        dqn_per_topic_all.append(d_per_topic)

        baseline_scores_all.append(b_scores)
        baseline_mastered_all.append(b_mastered)
        baseline_per_topic_all.append(b_per_topic)

        for topic in topics:
            if r_mastery_steps[topic] != -1:
                reinforce_mastered_per_topic[topic] += 1
            if d_mastery_steps[topic] != -1:
                dqn_mastered_per_topic[topic] += 1
            if b_mastery_steps[topic] != -1:
                baseline_mastered_per_topic[topic] += 1

        print(f"Student {student+1:02d} | "
              f"REINFORCE mastered: {r_mastered}/{len(topics)} | "
              f"DQN mastered: {d_mastered}/{len(topics)} | "
              f"Baseline mastered: {b_mastered}/{len(topics)}")

    reinforce_avg_score = np.mean([np.mean(s) for s in reinforce_scores_all])
    dqn_avg_score       = np.mean([np.mean(s) for s in dqn_scores_all])
    baseline_avg_score  = np.mean([np.mean(s) for s in baseline_scores_all])

    print(f"\n{'Metric':<30} {'Baseline':>10} {'REINFORCE':>12} {'DQN':>10}")
    print("=" * 65)
    print(f"{'Avg Final Score':<30} {baseline_avg_score:>10.3f} {reinforce_avg_score:>12.3f} {dqn_avg_score:>10.3f}")
    print(f"{'Avg Topics Mastered':<30} "
          f"{np.mean(baseline_mastered_all):>10.2f}   "
          f"{np.mean(reinforce_mastered_all):>12.2f}   "
          f"{np.mean(dqn_mastered_all):>10.2f}")

    print_per_topic_report(
        topics, topics_difficulty,
        reinforce_per_topic_all, dqn_per_topic_all, baseline_per_topic_all,
        reinforce_mastered_per_topic, dqn_mastered_per_topic, baseline_mastered_per_topic,
        n_students
    )

    reinforce_mean_curve = np.mean(reinforce_scores_all, axis=0)
    dqn_mean_curve       = np.mean(dqn_scores_all,       axis=0)
    baseline_mean_curve  = np.mean(baseline_scores_all,  axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(reinforce_mean_curve, label='REINFORCE', linewidth=2, linestyle='-.')
    plt.plot(dqn_mean_curve,       label='DQN',       linewidth=2)
    plt.plot(baseline_mean_curve,  label='Baseline',  linewidth=2, linestyle='--')
    plt.xlabel('Question Number')
    plt.ylabel('Score')
    plt.title('Score Progression — RL Agents vs Rule-Based Baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Images/eval_dqn.png', dpi=150)
    plt.show()

    reinforce_mastery_rates = [reinforce_mastered_per_topic[t] / n_students for t in topics]
    dqn_mastery_rates       = [dqn_mastered_per_topic[t]       / n_students for t in topics]
    b_mastery_rates         = [baseline_mastered_per_topic[t]  / n_students for t in topics]
    x = np.arange(len(topics))

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.25, b_mastery_rates,        0.25, label='Baseline')
    plt.bar(x,        reinforce_mastery_rates, 0.25, label='REINFORCE')
    plt.bar(x + 0.25, dqn_mastery_rates,       0.25, label='DQN')
    plt.xticks(x, [t[:20] for t in topics], rotation=15, ha='right')
    plt.ylabel('Mastery Rate')
    plt.title('Per-Topic Mastery Rate — RL Agents vs Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Images/mastery_dqn.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    topics_difficulty = {
        "Neural Networks"     : "advanced",
        "Gradient Descent"    : "intermediate",
        "Backpropagation"     : "advanced",
        "Activation Functions": "basic",
        "Overfitting"         : "intermediate"
    }

    prerequisites = {
        "Neural Networks"     : [],
        "Gradient Descent"    : [],
        "Activation Functions": [],
        "Overfitting"         : ["Neural Networks"],
        "Backpropagation"     : ["Neural Networks", "Gradient Descent"]
    }

    evaluate(topics_difficulty, prerequisites, w1=0.4, w2=0.5, w3=0.1, n_students=50, n_questions=1000)