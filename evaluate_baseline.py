import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from Simulator import Simulator
from rulebased_1 import RuleBasedAgent
from knowledge_state import question_types


def run_baseline_session(agent, simulator, topics, n_questions):

    score_progression      = []
    mastery_steps          = {t: -1 for t in topics}
    per_topic_qtype_scores = defaultdict(lambda: defaultdict(list))

    for step in range(n_questions):

        topic, diff, qtype = agent.select_action() 

        score = simulator.get_score(topic, diff, qtype)

        agent.update(topic, score, diff, qtype)

        score_progression.append(score)
        per_topic_qtype_scores[topic][qtype].append(score)

        if mastery_steps[topic] == -1 and agent.ks.is_mastered(topic):
            mastery_steps[topic] = step + 1

    topics_mastered = sum(1 for v in mastery_steps.values() if v != -1)

    return score_progression, topics_mastered, mastery_steps, per_topic_qtype_scores



def evaluate_baseline(
    topics_difficulty,
    prerequisites,
    w1=0.4,
    w2=0.5,
    w3=0.1,
    n_students=10,
    n_questions=500
):

    topics    = list(topics_difficulty.keys())
    simulator = Simulator(topic_difficulty=topics_difficulty)

    baseline_agent = RuleBasedAgent(
        topics_difficulty,
        prerequisites,
        w1=w1,
        w2=w2,
        w3=w3
    )

    baseline_scores_all         = []
    baseline_mastered_all       = []
    baseline_per_topic_all      = []
    baseline_mastered_per_topic = defaultdict(int)


    for student in range(n_students):

        simulator.reset_mastery_scores()

        baseline_agent.reset(topics_difficulty)

        scores, mastered, mastery_steps, per_topic = run_baseline_session(
            baseline_agent,
            simulator,
            topics,
            n_questions
        )

        baseline_scores_all.append(scores)
        baseline_mastered_all.append(mastered)
        baseline_per_topic_all.append(per_topic)

        for topic in topics:
            if mastery_steps[topic] != -1:
                baseline_mastered_per_topic[topic] += 1

        print(
            f"Student {student+1:02d} | "
            f"Baseline mastered: {mastered}/{len(topics)}"
        )


    # -----------------------
    # summary metrics
    # -----------------------

    avg_score = np.mean([np.mean(s) for s in baseline_scores_all])
    avg_mastered = np.mean(baseline_mastered_all)

    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)

    print(f"Average Score       : {avg_score:.3f}")
    print(f"Avg Topics Mastered : {avg_mastered:.2f}")



    # -----------------------
    # per-topic breakdown
    # -----------------------

    print("\nPER TOPIC BREAKDOWN")
    print("="*60)

    for topic in topics:

        diff = topics_difficulty[topic]

        print(f"\nTopic: {topic} ({diff})")

        print(f"{'Question Type':<15} {'Score':>10}")

        for qtype in question_types:

            scores = [
                np.mean(s[topic][qtype]) if s[topic][qtype] else 0.0
                for s in baseline_per_topic_all
            ]

            print(f"{qtype:<15} {np.mean(scores):>10.3f}")


        mastery_rate = baseline_mastered_per_topic[topic] / n_students

        print(f"{'Mastery Rate':<15} {mastery_rate:>10.1%}")



    # -----------------------
    # learning curve plot
    # -----------------------

    mean_curve = np.mean(baseline_scores_all, axis=0)

    plt.figure(figsize=(10,5))

    plt.plot(mean_curve, linewidth=2)

    plt.xlabel("Question Number")

    plt.ylabel("Score")

    plt.title("Baseline Score Progression")

    plt.grid(True)

    plt.tight_layout()

    plt.show()



    # -----------------------
    # mastery bar chart
    # -----------------------

    mastery_rates = [
        baseline_mastered_per_topic[t] / n_students
        for t in topics
    ]

    x = np.arange(len(topics))

    plt.figure(figsize=(10,5))

    plt.bar(x, mastery_rates)

    plt.xticks(x, topics, rotation=15)

    plt.ylabel("Mastery Rate")

    plt.title("Baseline Per-Topic Mastery Rate")

    plt.tight_layout()

    plt.show()



# --------------------------------------------------


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


    evaluate_baseline(

        topics_difficulty,

        prerequisites,

        w1=0.4,

        w2=0.5,

        w3=0.1,

        n_students=50,

        n_questions=1000

    )