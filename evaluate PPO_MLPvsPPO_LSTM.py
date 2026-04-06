import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from PPOAgent1 import PPOAgent
from rule_based_agent import RuleBasedAgent


"""
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
    return score_progression, topics_mastered, mastery_steps, per_topic_qtype_scores"""

def run_agent_session(agent, simulator, topics, n_questions, is_rl=True):
    score_progression      = []
    mastery_steps          = {t: -1 for t in topics}
    per_topic_qtype_scores = defaultdict(lambda: defaultdict(list))

    for step in range(n_questions):
        if is_rl:
            state_vector = agent.ks.get_state_vector()
            action_idx   = agent.select_action_online(state_vector)   # [CHANGED] online path
            topic, diff, qtype = agent.mdp.decode(action_idx)
        else:
            topic, diff, qtype = agent.select_action()

        score = simulator.get_score(topic, diff, qtype)

        if is_rl:
            agent.record_student_response(topic, score, diff, qtype)  # [CHANGED] triggers mid-session PPO update
        else:
            agent.update(topic, score, diff, qtype)

        score_progression.append(score)
        per_topic_qtype_scores[topic][qtype].append(score)

        if mastery_steps[topic] == -1 and agent.ks.is_mastered(topic):
            mastery_steps[topic] = step + 1

    if is_rl:
        agent.end_session()   # [CHANGED] flush remaining buffer, final update

    topics_mastered = sum(1 for v in mastery_steps.values() if v != -1)
    return score_progression, topics_mastered, mastery_steps, per_topic_qtype_scores

"""
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
    agent.ks.prev_topic = None"""

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
    agent.ks.prev_topic    = None
    agent.online_hidden    = None      # [ADDED]
    agent.session_step     = 0         # [ADDED]
    agent.online_buf       = defaultdict(list)  # [ADDED]
    
def print_per_topic_report(topics, topics_difficulty,
                            ppo_mlp_per_topic_all,
                            ppo_lstm_per_topic_all,
                            baseline_per_topic_all,
                            ppo_mlp_mastered_per_topic,
                            ppo_lstm_mastered_per_topic,
                            baseline_mastered_per_topic,
                            n_students):
    print("\n" + "="*75)
    print("PER TOPIC BREAKDOWN")
    print("="*75)

    for topic in topics:
        diff = topics_difficulty[topic]
        print(f"\nTopic: {topic} ({diff})")
        print(f"  {'Question Type':<15} {'Baseline':>10} {'PPO-MLP':>10} {'PPO-LSTM':>10}")
        print(f"  {'-'*48}")

        for qtype in question_types:
            b_scores    = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0
                           for s in baseline_per_topic_all]
            mlp_scores  = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0
                           for s in ppo_mlp_per_topic_all]
            lstm_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0
                           for s in ppo_lstm_per_topic_all]
            print(f"  {qtype:<15} {np.mean(b_scores):>10.3f} "
                  f"{np.mean(mlp_scores):>10.3f} {np.mean(lstm_scores):>10.3f}")

        b_rate    = baseline_mastered_per_topic[topic]    / n_students
        mlp_rate  = ppo_mlp_mastered_per_topic[topic]     / n_students
        lstm_rate = ppo_lstm_mastered_per_topic[topic]    / n_students
        print(f"  {'Mastery Rate':<15} {b_rate:>10.1%} {mlp_rate:>10.1%} {lstm_rate:>10.1%}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(topics_difficulty, prerequisites,
             w1=0.4, w2=0.5, w3=0.1,
             n_students=10, n_questions=500):

    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)

    print("Initialising Baseline agent...")
    baseline_agent = RuleBasedAgent(topics_difficulty, prerequisites,
                                    w1=w1, w2=w2, w3=w3)
    print("Baseline agent ready.\n")
    
    print("Pretraining PPO-LSTM agent...")
    ppo_lstm_agent = PPOAgent(topics_difficulty, prerequisites,
                              w1=w1, w2=w2, w3=w3, use_lstm=True)
    print("PPO-LSTM pretraining done.\n")

    print("Pretraining PPO-MLP agenta...")
    ppo_mlp_agent = PPOAgent(topics_difficulty, prerequisites,
                             w1=w1, w2=w2, w3=w3, use_lstm=False)
    print("PPO-MLP pretraining done.\n")

    

    # storage
    baseline_scores_all          = []
    baseline_mastered_all        = []
    baseline_per_topic_all       = []
    baseline_mastered_per_topic  = defaultdict(int)

    ppo_mlp_scores_all           = []
    ppo_mlp_mastered_all         = []
    ppo_mlp_per_topic_all        = []
    ppo_mlp_mastered_per_topic   = defaultdict(int)

    ppo_lstm_scores_all          = []
    ppo_lstm_mastered_all        = []
    ppo_lstm_per_topic_all       = []
    ppo_lstm_mastered_per_topic  = defaultdict(int)

    for student in range(n_students):

        simulator.reset_mastery_scores()
        saved_mastery = simulator.mastery_topic.copy()

        # --- Baseline ---
        baseline_agent.reset(topics_difficulty)
        simulator.mastery_topic = saved_mastery.copy()
        b_scores, b_mastered, b_mastery_steps, b_per_topic = run_agent_session(
            baseline_agent, simulator, topics, n_questions, is_rl=False)

        # --- PPO-MLP ---
        reset_rl_agent(ppo_mlp_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        mlp_scores, mlp_mastered, mlp_mastery_steps, mlp_per_topic = run_agent_session(
            ppo_mlp_agent, simulator, topics, n_questions, is_rl=True)

        # --- PPO-LSTM ---
        reset_rl_agent(ppo_lstm_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        lstm_scores, lstm_mastered, lstm_mastery_steps, lstm_per_topic = run_agent_session(
            ppo_lstm_agent, simulator, topics, n_questions, is_rl=True)

        # store
        baseline_scores_all.append(b_scores)
        baseline_mastered_all.append(b_mastered)
        baseline_per_topic_all.append(b_per_topic)

        ppo_mlp_scores_all.append(mlp_scores)
        ppo_mlp_mastered_all.append(mlp_mastered)
        ppo_mlp_per_topic_all.append(mlp_per_topic)

        ppo_lstm_scores_all.append(lstm_scores)
        ppo_lstm_mastered_all.append(lstm_mastered)
        ppo_lstm_per_topic_all.append(lstm_per_topic)

        for topic in topics:
            if b_mastery_steps[topic]    != -1:
                baseline_mastered_per_topic[topic]   += 1
            if mlp_mastery_steps[topic]  != -1:
                ppo_mlp_mastered_per_topic[topic]    += 1
            if lstm_mastery_steps[topic] != -1:
                ppo_lstm_mastered_per_topic[topic]   += 1

        print(f"Student {student+1:02d} | "
              f"Baseline: {b_mastered}/{len(topics)} | "
              f"PPO-MLP: {mlp_mastered}/{len(topics)} | "
              f"PPO-LSTM: {lstm_mastered}/{len(topics)}")

    # --- Summary ---
    b_avg    = np.mean([np.mean(s) for s in baseline_scores_all])
    mlp_avg  = np.mean([np.mean(s) for s in ppo_mlp_scores_all])
    lstm_avg = np.mean([np.mean(s) for s in ppo_lstm_scores_all])

    print(f"\n{'Metric':<30} {'Baseline':>10} {'PPO-MLP':>10} {'PPO-LSTM':>10}")
    print("="*62)
    print(f"{'Avg Final Score':<30} {b_avg:>10.3f} {mlp_avg:>10.3f} {lstm_avg:>10.3f}")
    print(f"{'Avg Topics Mastered':<30} "
          f"{np.mean(baseline_mastered_all):>10.2f} "
          f"{np.mean(ppo_mlp_mastered_all):>10.2f} "
          f"{np.mean(ppo_lstm_mastered_all):>10.2f}")

    print_per_topic_report(
        topics, topics_difficulty,
        ppo_mlp_per_topic_all, ppo_lstm_per_topic_all, baseline_per_topic_all,
        ppo_mlp_mastered_per_topic, ppo_lstm_mastered_per_topic,
        baseline_mastered_per_topic, n_students
    )

    # --- Plot 1: Score Progression ---
    b_curve    = np.mean(baseline_scores_all,  axis=0)
    mlp_curve  = np.mean(ppo_mlp_scores_all,   axis=0)
    lstm_curve = np.mean(ppo_lstm_scores_all,  axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(b_curve,    label='Baseline',  linewidth=2, linestyle='--')
    plt.plot(mlp_curve,  label='PPO-MLP',   linewidth=2, linestyle='-.')
    plt.plot(lstm_curve, label='PPO-LSTM',  linewidth=2)
    plt.xlabel('Question Number')
    plt.ylabel('Score')
    plt.title('Score Progression — PPO-LSTM vs PPO-MLP vs Baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Images/eval_lstm.png', dpi=150)

    # --- Plot 2: Per-topic mastery rate ---
    b_rates    = [baseline_mastered_per_topic[t]   / n_students for t in topics]
    mlp_rates  = [ppo_mlp_mastered_per_topic[t]    / n_students for t in topics]
    lstm_rates = [ppo_lstm_mastered_per_topic[t]   / n_students for t in topics]
    x = np.arange(len(topics))

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.25, b_rates,    0.25, label='Baseline')
    plt.bar(x,        mlp_rates,  0.25, label='PPO-MLP')
    plt.bar(x + 0.25, lstm_rates, 0.25, label='PPO-LSTM')
    plt.xticks(x, [t[:20] for t in topics], rotation=15, ha='right')
    plt.ylabel('Mastery Rate')
    plt.title('Per-Topic Mastery Rate — PPO-LSTM vs PPO-MLP vs Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Images/mastery_lstm.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    
    
    
    topics_difficulty = {
        "Linear Algebra"   : "basic",
        "Calculus"         : "basic",
        "Probability"      : "basic",
        "Optimization"     : "intermediate",
        "Loss Functions"   : "intermediate",
        "Gradient Descent" : "intermediate",
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

    evaluate(topics_difficulty, prerequisites,
             w1=0.35, w2=0.45, w3=0.2,
             n_students=50, n_questions=1000)