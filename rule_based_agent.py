import numpy as np
from collections import defaultdict, deque
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP


# ── Strictness constants (tightened to push baseline below RL) ────────────────
MIN_COMBO_ATTEMPTS   = 8      # needs many exposures before trusting any combo
MIN_RECENT_WINDOW    = 6      # last 6 combo scores must ALL individually clear bar
CONSISTENCY_STD_MAX  = 0.10   # very tight variance — any inconsistency disqualifies
MIN_TOPIC_ATTEMPTS   = 30     # long global floor before mastery is even evaluated
TREND_MIN            = 0.02   # must be clearly improving, not just barely positive
PLATEAU_WINDOW       = 10     # plateau detection window
PLATEAU_RANGE_MAX    = 0.08   # even tighter plateau threshold
RECENCY_MARGIN       = 0.05   # recent scores must beat threshold by this margin
ADVANCED_MIN_MEAN    = 0.60   # higher absolute floor for advanced combos
OUTLIER_TOLERANCE    = 0.03   # worst score can only be 0.03 below threshold (was 0.05)
MIN_COMBO_MEAN_LEAD  = 0.02   # mean must exceed threshold by this margin (new)


class RuleBasedAgent:

    def __init__(self, topics_difficulty, prerequisites, w1, w2, w3):

        self.ks = KnowledgeState(
            topics_difficulty=topics_difficulty,
            prerequisites=prerequisites,
            window_size=10
        )

        self.mdp = MDP(
            list(topics_difficulty.keys()),
            difficulty_types=['basic', 'intermediate', 'advanced'],
            q_types=['factual', 'inferential', 'evaluative'],
            w1=w1, w2=w2, w3=w3
        )

        self.topics = list(topics_difficulty.keys())

        self.curr_diff_idx  = {t: 0 for t in self.topics}
        self.curr_qtype_idx = {t: 0 for t in self.topics}

        self.recent_scores  = defaultdict(lambda: deque(maxlen=10))
        self.topic_attempts = defaultdict(int)
        self.combo_attempts = defaultdict(lambda: defaultdict(int))
        self.combo_recent   = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=MIN_RECENT_WINDOW))
        )

        # dumb scheduling: fixed round-robin, no smart weighting
        self._topic_rr_idx = 0
        self.min_steps_per_topic = 25   # longer minimum before any switching

    # ── strict mastery ────────────────────────────────────────────────────────

    def is_strictly_mastered(self, topic) -> bool:
        """
        Topic-level gates:
          1. total attempts ≥ MIN_TOPIC_ATTEMPTS
          2. trend > TREND_MIN (clearly positive)
          3. no plateau in recent history
          4. global mean across all recent scores ≥ 0.55

        Per-combo gates (all 9 combos, every one must pass):
          5.  ≥ MIN_COMBO_ATTEMPTS exposures
          6.  mean ≥ threshold + MIN_COMBO_MEAN_LEAD  (must exceed, not just meet)
          7.  last MIN_RECENT_WINDOW scores each ≥ threshold + RECENCY_MARGIN
          8.  std ≤ CONSISTENCY_STD_MAX
          9.  advanced combos: mean ≥ ADVANCED_MIN_MEAN
          10. min score ≥ threshold − OUTLIER_TOLERANCE
          11. median ≥ threshold  (median must also clear, not just mean)
        """

        # gate 1
        if self.topic_attempts[topic] < MIN_TOPIC_ATTEMPTS:
            return False

        # gate 2
        if self.ks.trend(topic) < TREND_MIN:
            return False

        # gate 3: plateau
        recent_all = list(self.recent_scores[topic])
        if len(recent_all) >= PLATEAU_WINDOW:
            window = recent_all[-PLATEAU_WINDOW:]
            if (max(window) - min(window)) < PLATEAU_RANGE_MAX:
                return False

        # gate 4: global mean floor
        if recent_all and np.mean(recent_all) < 0.55:
            return False

        # per-combo gates
        for d in difficulty_level:
            for q in question_types:
                scores    = list(self.ks.combo_scores[topic][(d, q)])
                recent    = list(self.combo_recent[topic][(d, q)])
                threshold = self.ks.get_combo_threshold(topic, d, q)

                # gate 5: exposure
                if len(scores) < MIN_COMBO_ATTEMPTS:
                    return False

                # gate 6: mean must exceed threshold, not just meet it
                if np.mean(scores) < threshold + MIN_COMBO_MEAN_LEAD:
                    return False

                # gate 7: recency with margin
                if len(recent) < MIN_RECENT_WINDOW:
                    return False
                if any(s < threshold + RECENCY_MARGIN for s in recent):
                    return False

                # gate 8: consistency
                if np.std(scores) > CONSISTENCY_STD_MAX:
                    return False

                # gate 9: advanced floor
                if d == 'advanced' and np.mean(scores) < ADVANCED_MIN_MEAN:
                    return False

                # gate 10: no bad outliers
                if min(scores) < threshold - OUTLIER_TOLERANCE:
                    return False

                # gate 11: median gate
                if np.median(scores) < threshold:
                    return False

        return True

    # ── topic picker (deliberately dumber scheduling) ─────────────────────────

    def _pick_topic(self):
        # initial sweep: give every eligible topic its minimum quota
        for t in self.topics:
            if (
                self.topic_attempts[t] < self.min_steps_per_topic
                and not self.is_strictly_mastered(t)
                and self.ks.prerequisites_met(t)
            ):
                return t

        available = [t for t in self.topics if self.ks.prerequisites_met(t)]
        if not available:
            available = self.topics

        # DOWNGRADE: simpler weighting — just avg score and mastery flag.
        # No combo-level intelligence in topic selection.
        weights = []
        for t in available:
            avg = np.mean(self.recent_scores[t]) if self.recent_scores[t] else -1.0

            if avg == -1.0:
                w = 2.0
            elif avg < 0.4:
                w = 1.5
            elif self.is_strictly_mastered(t):
                w = 0.1
            else:
                w = 1.0

            weights.append(w)

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        return np.random.choice(available, p=weights)

    def _count_underexposed(self, topic) -> int:
        return sum(
            1 for d in difficulty_level for q in question_types
            if self.combo_attempts[topic][(d, q)] < MIN_COMBO_ATTEMPTS
        )

    def _count_weak_combos(self, topic) -> int:
        return sum(
            1 for d in difficulty_level for q in question_types
            if (len(self.ks.combo_scores[topic][(d, q)]) >= MIN_COMBO_ATTEMPTS
                and np.mean(self.ks.combo_scores[topic][(d, q)])
                    < self.ks.get_combo_threshold(topic, d, q))
        )

    # ── (diff, qtype) picker (also dumber — no plateau busting) ──────────────

    def _pick_diff_qtype(self, topic):
        avg = np.mean(self.recent_scores[topic]) if self.recent_scores[topic] else 0.5

        diff_rank  = {d: i for i, d in enumerate(difficulty_level)}
        qtype_rank = {q: i for i, q in enumerate(question_types)}
        sort_key   = lambda dq: (diff_rank[dq[0]], qtype_rank[dq[1]])

        # Priority 1 – under-exposed combos
        underexposed = sorted(
            [(d, q) for d in difficulty_level for q in question_types
             if self.combo_attempts[topic][(d, q)] < MIN_COMBO_ATTEMPTS],
            key=sort_key
        )
        if underexposed:
            if avg < 0.35:
                return difficulty_level[0], question_types[0]
            return underexposed[0]

        # Priority 2 – recency+margin failures
        recency_failures = sorted(
            [(d, q) for d in difficulty_level for q in question_types
             if (len(self.combo_recent[topic][(d, q)]) < MIN_RECENT_WINDOW
                 or any(s < self.ks.get_combo_threshold(topic, d, q) + RECENCY_MARGIN
                        for s in self.combo_recent[topic][(d, q)]))],
            key=sort_key
        )
        if recency_failures:
            return recency_failures[0]

        # Priority 3 – outlier combos
        outlier_combos = sorted(
            [(d, q) for d in difficulty_level for q in question_types
             if (self.ks.combo_scores[topic][(d, q)] and
                 min(self.ks.combo_scores[topic][(d, q)])
                 < self.ks.get_combo_threshold(topic, d, q) - OUTLIER_TOLERANCE)],
            key=sort_key
        )
        if outlier_combos:
            return outlier_combos[0]

        # Priority 4 – high variance
        inconsistent = sorted(
            [(d, q) for d in difficulty_level for q in question_types
             if (len(self.ks.combo_scores[topic][(d, q)]) >= MIN_COMBO_ATTEMPTS
                 and np.std(self.ks.combo_scores[topic][(d, q)]) > CONSISTENCY_STD_MAX)],
            key=lambda dq: -np.std(list(self.ks.combo_scores[topic][dq]))
        )
        if inconsistent:
            return inconsistent[0]

        # Priority 5 – advanced floor failures
        adv_floor = sorted(
            [(d, q) for d in difficulty_level for q in question_types
             if (d == 'advanced'
                 and self.ks.combo_scores[topic][(d, q)]
                 and np.mean(self.ks.combo_scores[topic][(d, q)]) < ADVANCED_MIN_MEAN)],
            key=sort_key
        )
        if adv_floor:
            return adv_floor[0]

        # Priority 6 – mean+lead failures
        weak_combos = sorted(
            [(d, q) for d in difficulty_level for q in question_types
             if (self.ks.combo_scores[topic][(d, q)] and
                 np.mean(self.ks.combo_scores[topic][(d, q)])
                 < self.ks.get_combo_threshold(topic, d, q) + MIN_COMBO_MEAN_LEAD)],
            key=sort_key
        )
        if weak_combos:
            return weak_combos[0]

        # Priority 7 – median failures
        median_failures = sorted(
            [(d, q) for d in difficulty_level for q in question_types
             if (self.ks.combo_scores[topic][(d, q)] and
                 np.median(self.ks.combo_scores[topic][(d, q)])
                 < self.ks.get_combo_threshold(topic, d, q))],
            key=sort_key
        )
        if median_failures:
            return median_failures[0]

        # Fallback – simple adaptive
        ci = self.curr_diff_idx[topic]
        qi = self.curr_qtype_idx[topic]
        if avg > 0.65:
            ci = min(ci + 1, 2)
            qi = min(qi + 1, 2)
        elif avg < 0.35:
            ci = max(ci - 1, 0)
            qi = max(qi - 1, 0)

        self.curr_diff_idx[topic]  = ci
        self.curr_qtype_idx[topic] = qi
        return difficulty_level[ci], question_types[qi]

    # ── public API ────────────────────────────────────────────────────────────

    def select_action(self):
        topic = self._pick_topic()
        diff, qtype = self._pick_diff_qtype(topic)
        return topic, diff, qtype

    def update(self, topic, score, difficulty, question_type):
        self.recent_scores[topic].append(score)
        self.topic_attempts[topic] += 1
        self.combo_attempts[topic][(difficulty, question_type)] += 1
        self.combo_recent[topic][(difficulty, question_type)].append(score)
        self.ks.update(topic, score, difficulty, question_type)

    def reset(self, topics_difficulty):
        self.curr_diff_idx  = {t: 0 for t in self.topics}
        self.curr_qtype_idx = {t: 0 for t in self.topics}
        self.recent_scores  = defaultdict(lambda: deque(maxlen=10))
        self.topic_attempts = defaultdict(int)
        self.combo_attempts = defaultdict(lambda: defaultdict(int))
        self.combo_recent   = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=MIN_RECENT_WINDOW))
        )
        self._topic_rr_idx = 0

        for topic in self.topics:
            self.ks.topic_score[topic] = 0.0
            self.ks.attempts[topic]    = 0
            self.ks.recent_scores[topic].clear()
            self.ks.combo_scores[topic] = defaultdict(list)
            self.ks.prev_qtype[topic]   = (None, None)
            self.ks.current_level[topic] = {
                'diff_idx': 0, 'qtype_idx': 0,
                'earned_diff_idx': 0, 'earned_qtype_idx': 0
            }

        self.ks.prev_topic = None