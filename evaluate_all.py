import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from PPO_Agent import PPOAgent
from Agent import AdaptiveAgent
from DQNAgent import DQNAgent


class RuleBasedAgent:

    def __init__(self, topics_difficulty, prerequisites, w1, w2, w3):
        self.ks  = KnowledgeState(topics_difficulty=topics_difficulty, prerequisites=prerequisites, window_size=10)
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


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def run_agent_session(agent, simulator, topics, n_questions, is_rl=True, student_nos=0):
    score_progression      = []
    mastery_steps          = {t: -1 for t in topics}
    per_topic_qtype_scores = defaultdict(lambda: defaultdict(list))

    for step in range(n_questions):
        if is_rl:
            state_vector = agent.ks.get_state_vector()
            result       = agent.select_action(state_vector, training=False)
            # REINFORCE/PPO return tuple; DQN returns plain int
            action_idx   = result[0] if isinstance(result, tuple) else result
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
    # Keep DQN replay buffer across students so it accumulates
    # diverse experience — clearing it hurts DQN performance


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_per_topic_report(topics, topics_difficulty,
                            reinforce_per_topic_all, ppo_per_topic_all,
                            dqn_per_topic_all, baseline_per_topic_all,
                            reinforce_mastered_per_topic, ppo_mastered_per_topic,
                            dqn_mastered_per_topic, baseline_mastered_per_topic,
                            n_students):
    print("\n" + "=" * 80)
    print("PER TOPIC BREAKDOWN")
    print("=" * 80)

    for topic in topics:
        diff = topics_difficulty[topic]
        print(f"\nTopic: {topic} ({diff})")
        print(f"  {'Question Type':<15} {'Baseline':>10} {'REINFORCE':>12} {'PPO':>10} {'DQN':>10}")
        print(f"  {'-' * 60}")

        for qtype in question_types:
            r_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in reinforce_per_topic_all]
            p_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in ppo_per_topic_all]
            d_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in dqn_per_topic_all]
            b_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in baseline_per_topic_all]
            print(f"  {qtype:<15} {np.mean(b_scores):>10.3f} {np.mean(r_scores):>12.3f} "
                  f"{np.mean(p_scores):>10.3f} {np.mean(d_scores):>10.3f}")

        r_rate = reinforce_mastered_per_topic[topic] / n_students
        p_rate = ppo_mastered_per_topic[topic]       / n_students
        d_rate = dqn_mastered_per_topic[topic]       / n_students
        b_rate = baseline_mastered_per_topic[topic]  / n_students
        print(f"  {'Mastery Rate':<15} {b_rate:>10.1%} {r_rate:>12.1%} {p_rate:>10.1%} {d_rate:>10.1%}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_progression(reinforce_curve, ppo_curve, dqn_curve, baseline_curve, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(baseline_curve,   label='Baseline',   linewidth=2, linestyle='--',  color='gray')
    plt.plot(reinforce_curve,  label='REINFORCE',  linewidth=2, linestyle='-.',  color='steelblue')
    plt.plot(ppo_curve,        label='PPO',        linewidth=2,                  color='darkorange')
    plt.plot(dqn_curve,        label='DQN',        linewidth=2, linestyle=':',   color='green')
    plt.xlabel('Question Number')
    plt.ylabel('Score')
    plt.title('Score Progression — All Agents vs Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_mastery_rates(topics,
                       baseline_rates, reinforce_rates, ppo_rates, dqn_rates,
                       save_path):
    x     = np.arange(len(topics))
    width = 0.2

    plt.figure(figsize=(13, 5))
    plt.bar(x - 1.5 * width, baseline_rates,  width, label='Baseline',  color='gray')
    plt.bar(x - 0.5 * width, reinforce_rates, width, label='REINFORCE', color='steelblue')
    plt.bar(x + 0.5 * width, ppo_rates,       width, label='PPO',       color='darkorange')
    plt.bar(x + 1.5 * width, dqn_rates,       width, label='DQN',       color='green')
    plt.xticks(x, [t[:20] for t in topics], rotation=15, ha='right')
    plt.ylabel('Mastery Rate')
    plt.title('Per-Topic Mastery Rate — All Agents vs Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_avg_mastered_bar(agent_names, avg_mastered, save_path):
    colors = ['gray', 'steelblue', 'darkorange', 'green']
    plt.figure(figsize=(7, 5))
    bars = plt.bar(agent_names, avg_mastered, color=colors, width=0.5)
    for bar, val in zip(bars, avg_mastered):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    plt.ylabel('Avg Topics Mastered')
    plt.title('Average Topics Mastered per Student')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(topics_difficulty, prerequisites, w1=0.4, w2=0.5, w3=0.1, n_students=10, n_questions=500):
    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)

    print("Pretraining PPO agent...")
    ppo_agent = PPOAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    print("Pretraining REINFORCE agent...")
    reinforce_agent = AdaptiveAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    print("Pretraining DQN agent...")
    dqn_agent = DQNAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    baseline_agent = RuleBasedAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)
    print("Pretraining done.\n")

    # storage
    reinforce_scores_all, reinforce_mastered_all, reinforce_per_topic_all = [], [], []
    ppo_scores_all,       ppo_mastered_all,       ppo_per_topic_all       = [], [], []
    dqn_scores_all,       dqn_mastered_all,       dqn_per_topic_all       = [], [], []
    baseline_scores_all,  baseline_mastered_all,  baseline_per_topic_all  = [], [], []

    reinforce_mastered_per_topic = defaultdict(int)
    ppo_mastered_per_topic       = defaultdict(int)
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

        # PPO
        reset_rl_agent(ppo_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        p_scores, p_mastered, p_mastery_steps, p_per_topic = run_agent_session(
            ppo_agent, simulator, topics, n_questions, is_rl=True
        )

        # DQN — fixed eval_eps handles exploration internally
        reset_rl_agent(dqn_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        d_scores, d_mastered, d_mastery_steps, d_per_topic = run_agent_session(
            dqn_agent, simulator, topics, n_questions, is_rl=True
        )

        reinforce_scores_all.append(r_scores);  reinforce_mastered_all.append(r_mastered);  reinforce_per_topic_all.append(r_per_topic)
        ppo_scores_all.append(p_scores);        ppo_mastered_all.append(p_mastered);        ppo_per_topic_all.append(p_per_topic)
        dqn_scores_all.append(d_scores);        dqn_mastered_all.append(d_mastered);        dqn_per_topic_all.append(d_per_topic)
        baseline_scores_all.append(b_scores);   baseline_mastered_all.append(b_mastered);   baseline_per_topic_all.append(b_per_topic)

        for topic in topics:
            if r_mastery_steps[topic] != -1: reinforce_mastered_per_topic[topic] += 1
            if p_mastery_steps[topic] != -1: ppo_mastered_per_topic[topic]       += 1
            if d_mastery_steps[topic] != -1: dqn_mastered_per_topic[topic]       += 1
            if b_mastery_steps[topic] != -1: baseline_mastered_per_topic[topic]  += 1

        print(f"Student {student+1:02d} | "
              f"Baseline: {b_mastered}/{len(topics)} | "
              f"REINFORCE: {r_mastered}/{len(topics)} | "
              f"PPO: {p_mastered}/{len(topics)} | "
              f"DQN: {d_mastered}/{len(topics)}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    b_avg_score = np.mean([np.mean(s) for s in baseline_scores_all])
    r_avg_score = np.mean([np.mean(s) for s in reinforce_scores_all])
    p_avg_score = np.mean([np.mean(s) for s in ppo_scores_all])
    d_avg_score = np.mean([np.mean(s) for s in dqn_scores_all])

    b_avg_mastered = np.mean(baseline_mastered_all)
    r_avg_mastered = np.mean(reinforce_mastered_all)
    p_avg_mastered = np.mean(ppo_mastered_all)
    d_avg_mastered = np.mean(dqn_mastered_all)

    print(f"\n{'Metric':<30} {'Baseline':>10} {'REINFORCE':>12} {'PPO':>10} {'DQN':>10}")
    print("=" * 75)
    print(f"{'Avg Final Score':<30} {b_avg_score:>10.3f} {r_avg_score:>12.3f} {p_avg_score:>10.3f} {d_avg_score:>10.3f}")
    print(f"{'Avg Topics Mastered':<30} {b_avg_mastered:>10.2f} {r_avg_mastered:>12.2f} {p_avg_mastered:>10.2f} {d_avg_mastered:>10.2f}")

    print_per_topic_report(
        topics, topics_difficulty,
        reinforce_per_topic_all, ppo_per_topic_all, dqn_per_topic_all, baseline_per_topic_all,
        reinforce_mastered_per_topic, ppo_mastered_per_topic, dqn_mastered_per_topic, baseline_mastered_per_topic,
        n_students
    )

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    reinforce_curve = np.mean(reinforce_scores_all, axis=0)
    ppo_curve       = np.mean(ppo_scores_all,       axis=0)
    dqn_curve       = np.mean(dqn_scores_all,       axis=0)
    baseline_curve  = np.mean(baseline_scores_all,  axis=0)

    plot_score_progression(reinforce_curve, ppo_curve, dqn_curve, baseline_curve,
                           save_path='Images/eval_all.png')

    reinforce_rates = [reinforce_mastered_per_topic[t] / n_students for t in topics]
    ppo_rates       = [ppo_mastered_per_topic[t]       / n_students for t in topics]
    dqn_rates       = [dqn_mastered_per_topic[t]       / n_students for t in topics]
    b_rates         = [baseline_mastered_per_topic[t]  / n_students for t in topics]

    plot_mastery_rates(topics, b_rates, reinforce_rates, ppo_rates, dqn_rates,
                       save_path='Images/mastery_all.png')

    plot_avg_mastered_bar(
        agent_names  = ['Baseline', 'REINFORCE', 'PPO', 'DQN'],
        avg_mastered = [b_avg_mastered, r_avg_mastered, p_avg_mastered, d_avg_mastered],
        save_path    = 'Images/avg_mastered_all.png'
    )


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