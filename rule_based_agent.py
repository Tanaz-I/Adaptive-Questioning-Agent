import numpy as np
from collections import defaultdict, deque
from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP


MIN_COMBO_EXPOSURE = 2


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
        self.curr_diff_idx = {t: 0 for t in self.topics}
        self.curr_qtype_idx = {t: 0 for t in self.topics}
        self.recent_scores = defaultdict(lambda: deque(maxlen=6))
        self.topic_attempts = defaultdict(int)
        self.total_steps = 0



    def is_mastered(self, topic):
        return self.ks.is_mastered(topic)

    def _topic_priority(self, topic):

        avg_score = (
            np.mean(self.recent_scores[topic])
            if self.recent_scores[topic]
            else 0.5
        )
        attempts = self.topic_attempts[topic]
        spacing_penalty = 0.1 if topic == self.ks.prev_topic else 0.0
        priority = (
            0.6 * (1 - avg_score) +
            0.3 * (1 / (1 + attempts)) +
            spacing_penalty
        )
        return priority


    def _pick_topic(self):

        eligible_topics = [
            t for t in self.topics
            if self.ks.prerequisites_met(t)
            and not self.is_mastered(t)
        ]

        if not eligible_topics:

            eligible_topics = [
                t for t in self.topics
                if self.ks.prerequisites_met(t)
            ]

        best_topic = max(
            eligible_topics,
            key=self._topic_priority
        )

        explore_prob = max(
            0.05,
            0.3 * np.exp(-0.0008 * self.total_steps)
        )

        if np.random.rand() < explore_prob:
            return np.random.choice(eligible_topics)
        return best_topic


    def _pick_diff_qtype(self, topic):

        avg_score = (
            np.mean(self.recent_scores[topic])
            if self.recent_scores[topic]
            else 0.5
        )

        scores = list(self.recent_scores[topic])
        if len(scores) >= 4:

            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            trend = second_half - first_half

        else:
            trend = 0.0


        
        valid_combos = self.ks.get_valid_actions(topic)

        if not valid_combos:

            valid_combos = [
                (d, q)
                for d in difficulty_level
                for q in question_types
            ]


        combo_counts = {
            combo: len(self.ks.combo_scores[topic][combo])
            for combo in valid_combos
        }


        least_practiced = min(
            combo_counts,
            key=combo_counts.get
        )


        if combo_counts[least_practiced] < MIN_COMBO_EXPOSURE:

            diff_idx = difficulty_level.index(least_practiced[0])
            qtype_idx = question_types.index(least_practiced[1])

            self.curr_diff_idx[topic] = diff_idx
            self.curr_qtype_idx[topic] = qtype_idx

            return least_practiced

        topic_level = self.ks.topics_difficulty[topic]

        master_thresh = {
            'basic': 0.75,
            'intermediate': 0.65,
            'advanced': 0.55
        }

        increase_thresh = master_thresh[topic_level]
        decrease_thresh = increase_thresh - 0.25
        diff_idx = self.curr_diff_idx[topic]
        qtype_idx = self.curr_qtype_idx[topic]

        if avg_score > increase_thresh and trend > 0:
            diff_idx = min(diff_idx + 1, 2)

        elif avg_score < decrease_thresh:
            diff_idx = max(diff_idx - 1, 0)

        qtype_idx = (qtype_idx + 1) % 3

        candidate_combo = (
            difficulty_level[diff_idx],
            question_types[qtype_idx]
        )

        if candidate_combo not in valid_combos:
            idx = np.random.randint(len(valid_combos))
            candidate_combo = valid_combos[idx]

        diff_idx = difficulty_level.index(candidate_combo[0])
        qtype_idx = question_types.index(candidate_combo[1])

        self.curr_diff_idx[topic] = diff_idx
        self.curr_qtype_idx[topic] = qtype_idx
        
        return candidate_combo



    def select_action(self):
        topic = self._pick_topic()
        difficulty, question_type = self._pick_diff_qtype(topic)
        self.total_steps += 1
        return topic, difficulty, question_type


    def update(self, topic, score, difficulty, question_type):
        self.topic_attempts[topic] += 1
        self.recent_scores[topic].append(score)
        self.ks.update(topic, score, difficulty, question_type)


    def reset(self, topics_difficulty):
        self.curr_diff_idx = {t: 0 for t in self.topics}
        self.curr_qtype_idx = {t: 0 for t in self.topics}
        self.recent_scores = defaultdict(lambda: deque(maxlen=6))
        self.topic_attempts = defaultdict(int)
        self.total_steps = 0

        for topic in self.topics:
            self.ks.topic_score[topic] = 0.0
            self.ks.attempts[topic] = 0
            self.ks.recent_scores[topic].clear()
            self.ks.combo_scores[topic] = defaultdict(list)
            self.ks.prev_qtype[topic] = (None, None)
            self.ks.current_level[topic] = {
                'diff_idx': 0,
                'qtype_idx': 0,
                'earned_diff_idx': 0,
                'earned_qtype_idx': 0
            }
        self.ks.prev_topic = None