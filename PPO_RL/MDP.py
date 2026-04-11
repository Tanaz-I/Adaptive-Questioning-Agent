
class MDP:
    def __init__(self, topics, difficulty_types, q_types, w1, w2, w3, w4 = 0.1, w5 = 0.2):
        self.int_to_actions = {}
        self.actions_to_int = {}
        self.n_actions = len(topics) * len(difficulty_types) * len(q_types)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        for topic in topics:
            for diff_type in difficulty_types:
                for q_type in q_types:
                    combo = (topic, diff_type, q_type)
                    val = len(self.int_to_actions)
                    self.actions_to_int[combo] = val
                    self.int_to_actions[val] = combo

    def encode(self, action_commb):
        return self.actions_to_int.get(action_commb, - 1)
    
    def decode(self, action_idx):
        return self.int_to_actions.get(action_idx, -1)
    
    """def compute_reward(self, ks, topic, score, prev_score, old_earned_diff, old_earned_qtype):
    
        improvement = score - prev_score
        if score < 0.3 and ks.attempts[topic] > 3:
            improvement -= 0.2

        if ks.is_neglected(topic):
            coverage_bonus = 1.0
        elif ks.attempts[topic] > 5 and ks.topic_score[topic] < 0.2:
            coverage_bonus = -0.5
        elif ks.attempts[topic] < 5:
            coverage_bonus = 0.7
        elif ks.trend(topic) < 0:
            coverage_bonus = 0.4
        else:
            coverage_bonus = 0.0

        prereq_bonus = 0.0
        for other_topic in ks.topics:
            prereqs = ks.prerequisites_topic.get(other_topic, [])
            if topic in prereqs and not ks.prerequisites_met(other_topic):
                prereq_bonus += 0.2 * ks.topic_score[topic]

        unlock_bonus = 0.0
        if ks.prerequisites_met(topic) and ks.attempts[topic] == 1:
            unlock_bonus = 0.8

        new_earned_diff  = ks.current_level[topic]['earned_diff_idx']
        new_earned_qtype = ks.current_level[topic]['earned_qtype_idx']
        advancement_bonus = 0.2 if (new_earned_diff > old_earned_diff or 
                                    new_earned_qtype > old_earned_qtype) else 0.0

        mastery_penalty = 1.0 if ks.is_mastered(topic) else 0.0

        return (self.w1 * (0.6 * score + 0.4 * improvement)
            + self.w2 * coverage_bonus
            + self.w4 * advancement_bonus
            + self.w5 * prereq_bonus
            + self.w2 * unlock_bonus
            - self.w3 * mastery_penalty)"""
            
    def compute_reward(self, ks, topic, score, prev_score, old_earned_diff, old_earned_qtype):
        # Reward = improvement in overall knowledge state
        new_mastered   = sum(1 for t in ks.topics if ks.is_mastered(t))
        total_topics   = len(ks.topics)
        prev_avg_score = ks.topic_score[topic]
        avg_score_gain = score - prev_avg_score  # improvement this step
        
        return (0.6 * avg_score_gain) + (0.4 * new_mastered / total_topics)