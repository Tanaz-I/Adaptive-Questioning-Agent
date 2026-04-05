import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from PPOAgent import PPOAgent
from REINFORCEAgent import AdaptiveAgent
from rule_based_agent import RuleBasedAgent



def run_agent_session(agent, simulator, topics, n_questions, is_rl=True, student_nos=0):

    score_progression      = []
    mastery_steps          = {t: -1 for t in topics}
    per_topic_qtype_scores = defaultdict(lambda: defaultdict(list))

    for step in range(n_questions):
        if is_rl:
            state_vector = agent.ks.get_state_vector()
            result       = agent.select_action(state_vector, training=False)
            action_idx   = result[0]
            topic, diff, qtype = agent.mdp.decode(action_idx)
        else:
            topic, diff, qtype = agent.select_action()

        score = simulator.get_score(topic, diff, qtype)
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
            'earned_qtype_idx': 0,
        }
    agent.ks.prev_topic = None

    if hasattr(agent, 'reset_hidden'):
        agent.reset_hidden()


def print_per_topic_report(topics, topics_difficulty, agent_labels,
                            per_topic_all_list, mastered_per_topic_list, n_students):
    print("\n" + "=" * 90)
    print("PER TOPIC BREAKDOWN")
    print("=" * 90)

    col_w = 12
    header_agents = "".join(f"{lbl:>{col_w}}" for lbl in agent_labels)
    for topic in topics:
        diff = topics_difficulty[topic]
        print(f"\nTopic: {topic}  ({diff})")
        print(f"  {'Question Type':<15}" + header_agents)
        print(f"  {'-' * (15 + col_w * len(agent_labels))}")

        for qtype in question_types:
            row = f"  {qtype:<15}"
            for per_topic_all in per_topic_all_list:
                scores = [
                    np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0
                    for s in per_topic_all
                ]
                row += f"{np.mean(scores):>{col_w}.3f}"
            print(row)

        mastery_row = f"  {'Mastery Rate':<15}"
        for mastered_per_topic in mastered_per_topic_list:
            mastery_row += f"{mastered_per_topic[topic] / n_students:>{col_w}.1%}"
        print(mastery_row)

def print_summary_table(agent_labels, scores_all_list, mastered_all_list):
    col_w = 16
    header = f"{'Metric':<30}" + "".join(f"{lbl:>{col_w}}" for lbl in agent_labels)
    print("\n" + "=" * (30 + col_w * len(agent_labels)))
    print(header)
    print("=" * (30 + col_w * len(agent_labels)))

    avg_scores_row = f"{'Avg Final Score':<30}"
    avg_master_row = f"{'Avg Topics Mastered':<30}"
    for scores_all, mastered_all in zip(scores_all_list, mastered_all_list):
        avg_score  = np.mean([np.mean(s) for s in scores_all])
        avg_master = np.mean(mastered_all)
        avg_scores_row += f"{avg_score:>{col_w}.3f}"
        avg_master_row += f"{avg_master:>{col_w}.2f}"

    print(avg_scores_row)
    print(avg_master_row)


STYLE = {
    'REINFORCE'      : dict(linestyle='-.', linewidth=2, color='steelblue'),
    'REINFORCE+LSTM' : dict(linestyle=':',  linewidth=2, color='dodgerblue'),
    'PPO'            : dict(linestyle='-',  linewidth=2, color='tomato'),
    'PPO+LSTM'       : dict(linestyle='--', linewidth=2, color='firebrick'),
}


def plot_score_progression(agent_labels, scores_all_list, save_path='Results/eval_compare.png'):
    plt.figure(figsize=(11, 5))
    for label, scores_all in zip(agent_labels, scores_all_list):
        mean_curve = np.mean(scores_all, axis=0)
        style      = STYLE.get(label, {})
        plt.plot(mean_curve, label=label, **style)
    plt.xlabel('Question Number')
    plt.ylabel('Score')
    plt.title('Score Progression — REINFORCE vs REINFORCE+LSTM vs PPO vs PPO+LSTM')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")


def plot_mastery_rates(topics, agent_labels, mastered_per_topic_list, n_students,
                       save_path='Results/mastery_compare.png'):
    n_agents = len(agent_labels)
    x        = np.arange(len(topics))
    width    = 0.8 / n_agents          

    plt.figure(figsize=(13, 5))
    colors = ['steelblue', 'dodgerblue', 'tomato', 'firebrick']

    for i, (label, mastered_per_topic) in enumerate(zip(agent_labels, mastered_per_topic_list)):
        rates  = [mastered_per_topic[t] / n_students for t in topics]
        offset = (i - n_agents / 2 + 0.5) * width
        plt.bar(x + offset, rates, width, label=label,
                color=colors[i % len(colors)], alpha=0.85)

    plt.xticks(x, [t[:20] for t in topics], rotation=15, ha='right')
    plt.ylabel('Mastery Rate')
    plt.title('Per-Topic Mastery Rate — All Agents')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")
    plt.show()


def plot_smoothed_progression(agent_labels, scores_all_list, window=50,
                               save_path='Images/Results_smoothed_compare.png'):
    """Rolling-average smoothed score curves."""
    plt.figure(figsize=(11, 5))
    for label, scores_all in zip(agent_labels, scores_all_list):
        mean_curve = np.mean(scores_all, axis=0)
        smoothed   = np.convolve(mean_curve, np.ones(window) / window, mode='valid')
        style      = STYLE.get(label, {})
        plt.plot(smoothed, label=label, **style)
    plt.xlabel(f'Question Number (smoothed, window={window})')
    plt.ylabel('Score')
    plt.title('Smoothed Score Progression — All Agents')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(topics_difficulty, prerequisites,
             w1=0.4, w2=0.5, w3=0.1,
             n_students=10, n_questions=500):
    
    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)

    # ── Instantiate agents ────────────────────────────────────────────────────
    print("Initialising REINFORCE+LSTM agent …")
    reinforce_lstm_agent = AdaptiveAgent(topics_difficulty, prerequisites,
                                                  w1=w1, w2=w2, w3=w3,use_lstm=True)
    
    print("Initialising REINFORCE agent …")
    reinforce_agent = AdaptiveAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    print("Initialising PPO agent …")
    ppo_agent = PPOAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    

    print("Initialising PPO+LSTM agent …")
    ppo_lstm_agent = PPOAgent(topics_difficulty, prerequisites,
                                       w1=w1, w2=w2, w3=w3,use_lstm=True)

    
    active_agents = [
        ('REINFORCE',      reinforce_agent,      True),
        ('REINFORCE+LSTM', reinforce_lstm_agent, True),
        ('PPO',            ppo_agent,            True),
        ('PPO+LSTM',       ppo_lstm_agent,       True),
    ]
    active_agents = [(lbl, ag, is_rl)
                     for lbl, ag, is_rl in active_agents if ag is not None]

    agent_labels = [lbl for lbl, _, _ in active_agents]

    scores_all_list         = [[] for _ in active_agents]
    mastered_all_list       = [[] for _ in active_agents]
    per_topic_all_list      = [[] for _ in active_agents]
    mastered_per_topic_list = [defaultdict(int) for _ in active_agents]

    for student in range(n_students):
        simulator.reset_mastery_scores()
        saved_mastery = simulator.mastery_topic.copy()

        row_parts = [f"Student {student+1:02d}"]

        for idx, (label, agent, is_rl) in enumerate(active_agents):
           
            simulator.mastery_topic = saved_mastery.copy()

            if is_rl:
                reset_rl_agent(agent, topics)
            else:
                agent.reset(topics_difficulty)

            scores, mastered, mastery_steps, per_topic = run_agent_session(
                agent, simulator, topics, n_questions, is_rl=is_rl, student_nos=student
            )

            scores_all_list[idx].append(scores)
            mastered_all_list[idx].append(mastered)
            per_topic_all_list[idx].append(per_topic)

            for topic in topics:
                if mastery_steps[topic] != -1:
                    mastered_per_topic_list[idx][topic] += 1

            row_parts.append(f"{label} mastered: {mastered}/{len(topics)}")

        print(" | ".join(row_parts))

    
    print_summary_table(agent_labels, scores_all_list, mastered_all_list)

  
    print_per_topic_report(
        topics, topics_difficulty,
        agent_labels,
        per_topic_all_list,
        mastered_per_topic_list,
        n_students
    )

    plot_score_progression(agent_labels, scores_all_list)
    plot_smoothed_progression(agent_labels, scores_all_list)
    plot_mastery_rates(topics, agent_labels, mastered_per_topic_list, n_students)




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
        "Neural Networks"  : "advanced",
    }

    prerequisites = {
        "Linear Algebra"   : [],
        "Calculus"         : [],
        "Probability"      : ["Linear Algebra"],
        "Optimization"     : ["Calculus"],
        "Loss Functions"   : ["Probability", "Calculus"],
        "Gradient Descent" : ["Optimization"],
        "Neural Networks"  : ["Loss Functions", "Gradient Descent"],
    }

    evaluate(
        topics_difficulty,
        prerequisites,
        w1=0.35, w2=0.45, w3=0.2,
        n_students=50,
        n_questions=5000,
    )