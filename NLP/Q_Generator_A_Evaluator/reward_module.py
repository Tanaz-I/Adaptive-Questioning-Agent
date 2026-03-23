"""
Reward Module (Improved)
=======================

• Uses evaluation score
• Considers improvement
• Considers difficulty
• Produces smooth reward signal
"""

def compute_reward(current_score, prev_score=0.0, difficulty="medium"):

    # 1. Base reward (absolute performance)
    base = current_score

    # 2. Improvement reward
    improvement = current_score - prev_score

    # 3. Difficulty weight
    if difficulty == "easy":
        diff_weight = 0.8
    elif difficulty == "medium":
        diff_weight = 1.0
    else:  # hard
        diff_weight = 1.2

    # 4. Final reward
    reward = (
        0.6 * base +
        0.4 * improvement
    ) * diff_weight

    return round(reward, 3)


# ─────────────────────────────────────────────
# Test Mode
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Example values (replace with evaluator output)
    current_score = 0.76
    previous_score = 0.65
    difficulty = "medium"

    reward = compute_reward(current_score, previous_score, difficulty)

    print("\nReward:", reward)