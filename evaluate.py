import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from PPOAgent_1 import PPOAgent
from Agent import AdaptiveAgent
from rule_based_agent import RuleBasedAgent


"""class RuleBasedAgent:
    
    def __init__(self, topics_difficulty, prerequisites, w1, w2, w3):
        self.ks  = KnowledgeState(topics_difficulty=topics_difficulty, prerequisites = prerequisites, window_size=10)
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
            self.ks.combo_scores[topic] = defaultdict(list)
            self.ks.prev_qtype[topic]    = (None, None)
            self.ks.current_level[topic] = {
                'diff_idx'        : 0,
                'qtype_idx'       : 0,
                'earned_diff_idx' : 0,
                'earned_qtype_idx': 0
            }
        self.ks.prev_topic = None"""


def run_agent_session(agent, simulator, topics, n_questions, is_rl=True, student_nos = 0):
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
            action_idx = result[0]  # just take first element regardless of how many values returned
            topic, diff, qtype = agent.mdp.decode(action_idx)
        else:
            topic, diff, qtype  = agent.select_action()

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
        agent.ks.combo_scores[topic] = defaultdict(list)
        agent.ks.prev_qtype[topic]    = (None, None)
        agent.ks.current_level[topic] = {
            'diff_idx'        : 0,
            'qtype_idx'       : 0,
            'earned_diff_idx' : 0,
            'earned_qtype_idx': 0
        }
    agent.ks.prev_topic = None


def print_per_topic_report(topics, topics_difficulty,
                            reinforce_per_topic_all, ppo_per_topic_all, baseline_per_topic_all,
                            reinforce_mastered_per_topic, ppo_mastered_per_topic, baseline_mastered_per_topic,
                            n_students):
    print("\n" + "="*80)
    print("PER TOPIC BREAKDOWN")
    print("="*80)

    for topic in topics:
        diff = topics_difficulty[topic]
        print(f"\nTopic: {topic} ({diff})")
        print(f"  {'Question Type':<15} {'Baseline':>10} {'REINFORCE':>12} {'PPO':>10}")
        print(f"  {'-'*50}")

        for qtype in question_types:
            r_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in reinforce_per_topic_all]
            p_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in ppo_per_topic_all]
            b_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in baseline_per_topic_all]
            print(f"  {qtype:<15} {np.mean(b_scores):>10.3f} {np.mean(r_scores):>12.3f} {np.mean(p_scores):>10.3f}")

        r_mastery_rate = reinforce_mastered_per_topic[topic] / n_students
        p_mastery_rate = ppo_mastered_per_topic[topic]       / n_students
        b_mastery_rate = baseline_mastered_per_topic[topic]  / n_students
        print(f"  {'Mastery Rate':<15} {b_mastery_rate:>10.1%} {r_mastery_rate:>12.1%} {p_mastery_rate:>10.1%}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(topics_difficulty, prerequisites, w1=0.4, w2=0.5, w3=0.1, n_students=10, n_questions=500):
    """
    Compare RL agent vs rule-based baseline on n_students simulated students.
    Both agents face identical simulated students for fair comparison.
    """
    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)
    
    baseline_agent = RuleBasedAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)
    print("Pretraining done.\n")
    
    print("Pretraining PPO agent...")
    ppo_agent = PPOAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)
    
    print("Pretraining REINFORCE agent...")
    reinforce_agent = AdaptiveAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)
    
    

    

    # storage
    baseline_scores_all         = []
    baseline_mastered_all       = []
    baseline_per_topic_all      = []
    baseline_mastered_per_topic = defaultdict(int)
    
    reinforce_scores_all, reinforce_mastered_all, reinforce_per_topic_all = [], [], []
    ppo_scores_all, ppo_mastered_all, ppo_per_topic_all = [], [], []
    reinforce_mastered_per_topic = defaultdict(int)
    ppo_mastered_per_topic = defaultdict(int)

    for student in range(n_students):

        # generate fresh student and save state for fair comparison
        simulator.reset_mastery_scores()
        saved_mastery = simulator.mastery_topic.copy()

        # --- run baseline ---
        baseline_agent.reset(topics_difficulty)
        simulator.mastery_topic = saved_mastery.copy()
        b_scores, b_mastered, b_mastery_steps, b_per_topic = run_agent_session(
            baseline_agent, simulator, topics, n_questions, is_rl=False, student_nos = student
        )
        
         # REINFORCE
        reset_rl_agent(reinforce_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        r_scores, r_mastered, r_mastery_steps, r_per_topic = run_agent_session(
            reinforce_agent, simulator, topics, n_questions, is_rl=True)

        # PPO
        reset_rl_agent(ppo_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        p_scores, p_mastered, p_mastery_steps, p_per_topic = run_agent_session(
            ppo_agent, simulator, topics, n_questions, is_rl=True)

        """# --- run RL agent (same student) ---
        reset_rl_agent(rl_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        rl_scores, rl_mastered, rl_mastery_steps, rl_per_topic = run_agent_session(
            rl_agent, simulator, topics, n_questions, is_rl=True, student_nos = student
        )"""

        # store results
        
        reinforce_scores_all.append(r_scores)
        reinforce_mastered_all.append(r_mastered)
        reinforce_per_topic_all.append(r_per_topic)

        ppo_scores_all.append(p_scores)
        ppo_mastered_all.append(p_mastered)
        ppo_per_topic_all.append(p_per_topic)

        baseline_scores_all.append(b_scores)
        baseline_mastered_all.append(b_mastered)
        baseline_per_topic_all.append(b_per_topic)

        # track per-topic mastery counts
        for topic in topics:
            if r_mastery_steps[topic] != -1:
                reinforce_mastered_per_topic[topic] += 1
            if p_mastery_steps[topic] != -1:
                ppo_mastered_per_topic[topic] += 1
            if b_mastery_steps[topic] != -1:
                baseline_mastered_per_topic[topic] += 1

        print(f"Student {student+1:02d} | "
            f"REINFORCE mastered: {r_mastered}/{len(topics)} | "
            f"PPO mastered: {p_mastered}/{len(topics)} | "
            f"Baseline mastered: {b_mastered}/{len(topics)}")


    reinforce_avg_score = np.mean([np.mean(s) for s in reinforce_scores_all])
    ppo_avg_score       = np.mean([np.mean(s) for s in ppo_scores_all])
    baseline_avg_score  = np.mean([np.mean(s) for s in baseline_scores_all])

    print(f"{'Metric':<30} {'Baseline':>10} {'REINFORCE':>12} {'PPO':>10}")
    print("="*65)
    print(f"{'Avg Final Score':<30} {baseline_avg_score:>10.3f} {reinforce_avg_score:>12.3f} {ppo_avg_score:>10.3f}")
    print(f"{'Avg Topics Mastered':<30} "
        f"{np.mean(baseline_mastered_all):>10.2f}   "
        f"{np.mean(reinforce_mastered_all):>12.2f}   "
        f"{np.mean(ppo_mastered_all):>10.2f}")
    
    reinforce_mean_curve = np.mean(reinforce_scores_all, axis=0)
    ppo_mean_curve       = np.mean(ppo_scores_all, axis=0)
    baseline_mean_curve  = np.mean(baseline_scores_all, axis=0)

    # per topic breakdown
    print_per_topic_report(
        topics, topics_difficulty,
        reinforce_per_topic_all, ppo_per_topic_all, baseline_per_topic_all,
        reinforce_mastered_per_topic, ppo_mastered_per_topic, baseline_mastered_per_topic,
        n_students
    )

    


    plt.figure(figsize=(10, 5))
    plt.plot(reinforce_mean_curve, label='REINFORCE', linewidth=2, linestyle='-.')
    plt.plot(ppo_mean_curve,       label='PPO',       linewidth=2)
    plt.plot(baseline_mean_curve,  label='Baseline',  linewidth=2, linestyle='--')
    #plt.plot(rl_mean_curve,       label='RL Agent', linewidth=2)
    #plt.plot(baseline_mean_curve, label='Baseline', linewidth=2, linestyle='--')
    plt.xlabel('Question Number')
    plt.ylabel('Score')
    plt.title('Score Progression — RL Agent vs Rule-Based Baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Images/eval_6.png', dpi=150)
    #plt.show()

    # 2. Per-topic mastery rate bar chart
    reinforce_mastery_rates = [reinforce_mastered_per_topic[t] / n_students for t in topics]
    ppo_mastery_rates       = [ppo_mastered_per_topic[t]       / n_students for t in topics]
    b_mastery_rates         = [baseline_mastered_per_topic[t]  / n_students for t in topics]
    x = np.arange(len(topics))



    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.25, b_mastery_rates,         0.25, label='Baseline')
    plt.bar(x,        reinforce_mastery_rates,  0.25, label='REINFORCE')
    plt.bar(x + 0.25, ppo_mastery_rates,        0.25, label='PPO')
    plt.xticks(x, [t[:20] for t in topics], rotation=15, ha='right')
    plt.ylabel('Mastery Rate')
    plt.title('Per-Topic Mastery Rate — RL Agent vs Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Images/mastery_6.png', dpi=150)
    plt.show()



if __name__ == "__main__":
    """topics_difficulty = {
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
    }"""
    
    topics_difficulty = {

    # Level 0
    "Linear Algebra"     : "basic",
    "Calculus"           : "basic",

    # Level 1
    "Probability"        : "basic",
    "Optimization"       : "intermediate",

    # Level 2
    "Loss Functions"     : "intermediate",
    "Gradient Descent"   : "intermediate",

    # Level 3
    "Neural Networks"    : "advanced"
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

    evaluate(topics_difficulty, prerequisites, w1=0.4, w2=0.5, w3=0.2, n_students=50, n_questions=2000)