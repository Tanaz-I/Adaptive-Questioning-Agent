import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import defaultdict
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from PPOAgent import PPOAgent
from REINFORCEAgent import AdaptiveAgent
from rule_based_agent import RuleBasedAgent
from DQNAgent import DQNAgent


def run_agent_session(agent, simulator, topics, n_questions, is_rl=True, student_nos=0):

    score_progression      = []
    mastery_steps          = {t: -1 for t in topics}
    per_topic_qtype_scores = defaultdict(lambda: defaultdict(list))

    for step in range(n_questions):
        if is_rl:
            state_vector = agent.ks.get_state_vector()
            result       = agent.select_action(state_vector, training=False)
            action_idx   = result[0] if isinstance(result, tuple) else result
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
        agent.ks.topic_score[topic]   = 0.0
        agent.ks.attempts[topic]      = 0
        agent.ks.recent_scores[topic].clear()
        agent.ks.combo_scores[topic]  = defaultdict(list)
        agent.ks.prev_qtype[topic]    = (None, None)
        agent.ks.current_level[topic] = {
            'diff_idx'        : 0,
            'qtype_idx'       : 0,
            'earned_diff_idx' : 0,
            'earned_qtype_idx': 0,
        }
    agent.ks.prev_topic = None


def print_per_topic_report(
    topics, topics_difficulty,
    reinforce_per_topic_all, ppo_per_topic_all, dqn_per_topic_all, baseline_per_topic_all,
    reinforce_mastered_per_topic, ppo_mastered_per_topic,
    dqn_mastered_per_topic, baseline_mastered_per_topic,
    n_students,
    csv_path='ComparisonPlots/per_topic_report.csv',
):
    print("\n" + "=" * 90)
    print("PER TOPIC BREAKDOWN")
    print("=" * 90)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Topic', 'Difficulty', 'Question Type',
                         'Baseline', 'REINFORCE', 'PPO', 'DQN'])

        for topic in topics:
            diff = topics_difficulty[topic]
            print(f"\nTopic: {topic} ({diff})")
            print(f"  {'Question Type':<15} {'Baseline':>10} {'REINFORCE':>12} {'PPO':>10} {'DQN':>10}")
            print(f"  {'-'*60}")

            for qtype in question_types:
                r_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in reinforce_per_topic_all]
                p_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in ppo_per_topic_all]
                d_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in dqn_per_topic_all]
                b_scores = [np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0 for s in baseline_per_topic_all]

                b_mean = np.mean(b_scores)
                r_mean = np.mean(r_scores)
                p_mean = np.mean(p_scores)
                d_mean = np.mean(d_scores)

                print(
                    f"  {qtype:<15} {b_mean:>10.3f} "
                    f"{r_mean:>12.3f} {p_mean:>10.3f} {d_mean:>10.3f}"
                )
                writer.writerow([
                    topic, diff, qtype,
                    round(b_mean, 3), round(r_mean, 3),
                    round(p_mean, 3), round(d_mean, 3),
                ])

            r_mastery_rate = reinforce_mastered_per_topic[topic] / n_students
            p_mastery_rate = ppo_mastered_per_topic[topic]       / n_students
            d_mastery_rate = dqn_mastered_per_topic[topic]       / n_students
            b_mastery_rate = baseline_mastered_per_topic[topic]  / n_students

            print(
                f"  {'Mastery Rate':<15} {b_mastery_rate:>10.1%} "
                f"{r_mastery_rate:>12.1%} {p_mastery_rate:>10.1%} {d_mastery_rate:>10.1%}"
            )
            writer.writerow([
                topic, diff, 'Mastery Rate',
                round(b_mastery_rate, 3), round(r_mastery_rate, 3),
                round(p_mastery_rate, 3), round(d_mastery_rate, 3),
            ])

    print(f"\nSaved → {csv_path}")


def evaluate(topics_difficulty, prerequisites, w1=0.4, w2=0.5, w3=0.1, n_students=10, n_questions=500):
    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)

    print("Setting up Baseline agent...")
    baseline_agent = RuleBasedAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    print("Pretraining PPO agent...")
    ppo_agent = PPOAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    print("Pretraining REINFORCE agent...")
    reinforce_agent = AdaptiveAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    print("Pretraining DQN agent...")
    dqn_agent = DQNAgent(topics_difficulty, prerequisites, w1=w1, w2=w2, w3=w3)

    print("Pretraining done.\n")

    baseline_scores_all, baseline_mastered_all, baseline_per_topic_all = [], [], []
    reinforce_scores_all, reinforce_mastered_all, reinforce_per_topic_all = [], [], []
    ppo_scores_all, ppo_mastered_all, ppo_per_topic_all = [], [], []
    dqn_scores_all, dqn_mastered_all, dqn_per_topic_all = [], [], []

    baseline_mastered_per_topic  = defaultdict(int)
    reinforce_mastered_per_topic = defaultdict(int)
    ppo_mastered_per_topic       = defaultdict(int)
    dqn_mastered_per_topic       = defaultdict(int)

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

        # DQN
        reset_rl_agent(dqn_agent, topics)
        simulator.mastery_topic = saved_mastery.copy()
        d_scores, d_mastered, d_mastery_steps, d_per_topic = run_agent_session(
            dqn_agent, simulator, topics, n_questions, is_rl=True
        )

        baseline_scores_all.append(b_scores);  baseline_mastered_all.append(b_mastered);  baseline_per_topic_all.append(b_per_topic)
        reinforce_scores_all.append(r_scores); reinforce_mastered_all.append(r_mastered); reinforce_per_topic_all.append(r_per_topic)
        ppo_scores_all.append(p_scores);       ppo_mastered_all.append(p_mastered);       ppo_per_topic_all.append(p_per_topic)
        dqn_scores_all.append(d_scores);       dqn_mastered_all.append(d_mastered);       dqn_per_topic_all.append(d_per_topic)

        for topic in topics:
            if r_mastery_steps[topic] != -1: reinforce_mastered_per_topic[topic] += 1
            if p_mastery_steps[topic] != -1: ppo_mastered_per_topic[topic]       += 1
            if d_mastery_steps[topic] != -1: dqn_mastered_per_topic[topic]       += 1
            if b_mastery_steps[topic] != -1: baseline_mastered_per_topic[topic]  += 1

        print(
            f"Student {student+1:02d} | "
            f"REINFORCE: {r_mastered}/{len(topics)} | "
            f"PPO: {p_mastered}/{len(topics)} | "
            f"DQN: {d_mastered}/{len(topics)} | "
            f"Baseline: {b_mastered}/{len(topics)}"
        )

    # Summary table
    reinforce_avg = np.mean([np.mean(s) for s in reinforce_scores_all])
    ppo_avg       = np.mean([np.mean(s) for s in ppo_scores_all])
    dqn_avg       = np.mean([np.mean(s) for s in dqn_scores_all])
    baseline_avg  = np.mean([np.mean(s) for s in baseline_scores_all])

    reinforce_avg_mastered = np.mean(reinforce_mastered_all)
    ppo_avg_mastered       = np.mean(ppo_mastered_all)
    dqn_avg_mastered       = np.mean(dqn_mastered_all)
    baseline_avg_mastered  = np.mean(baseline_mastered_all)
    
    print(f"\n{'Metric':<30} {'Baseline':>10} {'REINFORCE':>12} {'PPO':>10} {'DQN':>10}")
    print("=" * 75)
    print(f"{'Avg Final Score':<30} {baseline_avg:>10.3f} {reinforce_avg:>12.3f} {ppo_avg:>10.3f} {dqn_avg:>10.3f}")
    print(
        f"{'Avg Topics Mastered':<30} "
        f"{baseline_avg_mastered:>10.2f}   "
        f"{reinforce_avg_mastered:>12.2f}   "
        f"{ppo_avg_mastered:>10.2f}   "
        f"{dqn_avg_mastered:>10.2f}"
    )
    
    with open('ComparisonPlots/summary_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Baseline', 'REINFORCE', 'PPO', 'DQN'])
        writer.writerow([
            'Avg Final Score',
            round(baseline_avg,  3), round(reinforce_avg, 3),
            round(ppo_avg,       3), round(dqn_avg,       3),
        ])
        writer.writerow([
            'Avg Topics Mastered',
            round(baseline_avg_mastered,  2), round(reinforce_avg_mastered, 2),
            round(ppo_avg_mastered,       2), round(dqn_avg_mastered,       2),
        ])
    print("Saved → Results/summary_metrics.csv")

    print_per_topic_report(
        topics, topics_difficulty,
        reinforce_per_topic_all, ppo_per_topic_all, dqn_per_topic_all, baseline_per_topic_all,
        reinforce_mastered_per_topic, ppo_mastered_per_topic,
        dqn_mastered_per_topic, baseline_mastered_per_topic,
        n_students,
    )
    
    reinforce_curve  = np.mean(reinforce_scores_all, axis=0)
    ppo_curve        = np.mean(ppo_scores_all,       axis=0)
    dqn_curve        = np.mean(dqn_scores_all,       axis=0)
    baseline_curve   = np.mean(baseline_scores_all,  axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(reinforce_curve, label='REINFORCE', linewidth=2, linestyle='-.')
    plt.plot(ppo_curve,       label='PPO',       linewidth=2)
    plt.plot(dqn_curve,       label='DQN',       linewidth=2, linestyle=':')
    plt.plot(baseline_curve,  label='Baseline',  linewidth=2, linestyle='--')
    plt.xlabel('Question Number')
    plt.ylabel('Score')
    plt.title('Score Progression — All Agents vs Baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ComparisonPlots/score_all_agents.png', dpi=150)

    reinforce_mastery_rates = [reinforce_mastered_per_topic[t] / n_students for t in topics]
    ppo_mastery_rates       = [ppo_mastered_per_topic[t]       / n_students for t in topics]
    dqn_mastery_rates       = [dqn_mastered_per_topic[t]       / n_students for t in topics]
    b_mastery_rates         = [baseline_mastered_per_topic[t]  / n_students for t in topics]

    x     = np.arange(len(topics))
    width = 0.2

    plt.figure(figsize=(13, 5))
    plt.bar(x - 1.5 * width, b_mastery_rates,        width, label='Baseline')
    plt.bar(x - 0.5 * width, reinforce_mastery_rates, width, label='REINFORCE')
    plt.bar(x + 0.5 * width, ppo_mastery_rates,       width, label='PPO')
    plt.bar(x + 1.5 * width, dqn_mastery_rates,       width, label='DQN')
    plt.xticks(x, [t[:20] for t in topics], rotation=15, ha='right')
    plt.ylabel('Mastery Rate')
    plt.title('Per-Topic Mastery Rate — All Agents vs Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ComparisonPlots/mastery_all_agents.png', dpi=150)
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
    
    evaluate(topics_difficulty, prerequisites, w1=0.35, w2=0.45, w3=0.2, n_students=50, n_questions=1000)