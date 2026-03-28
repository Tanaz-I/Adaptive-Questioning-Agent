"""
Reward Module (FINAL CORRECTED VERSION)
======================================

✔ Fully dynamic (no hardcoding)
✔ Sensitive to score changes
✔ Difficulty-aware
✔ RL-ready reward signal
"""

# ─────────────────────────────────────────────
# Compute Reward
# ─────────────────────────────────────────────

def compute_reward(current_score, previous_score, difficulty="medium"):

    # ─────────────────────────────
    # Step 1: Improvement
    # ─────────────────────────────
    improvement = current_score - previous_score

    # ─────────────────────────────
    # Step 2: Base performance
    # ─────────────────────────────
    performance = current_score

    # ─────────────────────────────
    # Step 3: Difficulty weight
    # ─────────────────────────────
    if difficulty == "easy":
        difficulty_weight = 0.8
    elif difficulty == "hard":
        difficulty_weight = 1.2
    else:
        difficulty_weight = 1.0

    # ─────────────────────────────
    # Step 4: Reward calculation
    # ─────────────────────────────

    # Combine performance + improvement
    reward = (0.6 * performance) + (0.4 * improvement)

    # Apply difficulty scaling
    reward *= difficulty_weight

    # ─────────────────────────────
    # Step 5: Penalize drop in performance
    # ─────────────────────────────

    if improvement < -0.1:
        reward -= 0.2

    # ─────────────────────────────
    # Step 6: Clip reward
    # ─────────────────────────────

    reward = max(0.0, min(1.0, reward))

    return round(reward, 3)


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Testing reward module...\n")

    test_cases = [
        (0.8, 0.6, "medium"),
        (0.5, 0.6, "medium"),
        (0.9, 0.7, "hard"),
        (0.4, 0.4, "easy")
    ]

    for current, prev, diff in test_cases:
        r = compute_reward(current, prev, diff)
        print(f"Score: {current}, Prev: {prev}, Diff: {diff} → Reward: {r}")