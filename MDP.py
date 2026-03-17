
class MDP:
    def __init__(self, topics, difficulty_types, q_types, w1, w2, w3):
        self.int_to_actions = {}
        self.actions_to_int = {}
        self.n_actions = len(topics) * len(difficulty_types) * len(q_types)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

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
    
    def compute_reward(self, ks, topic, score, prev_score):
        #ks -> obj of KnowledgeState
        improvement = score - prev_score
        if score < 0.3 and ks.attempts[topic] > 3:
            improvement -= 0.2
        if ks.is_neglected(topic):
            coverage_bonus = 1.0
        elif ks.attempts[topic] < 5:
            coverage_bonus = 0.7
        elif ks.trend(topic) < 0:
            coverage_bonus = 0.4
        if ks.attempts[topic] > 5 and ks.topic_score[topic] < 0.2:
            coverage_bonus = -0.5
        else:
            coverage_bonus = 0.0
        
        
        mastery_penalty = 1.0 if ks.is_mastered(topic) else 0.0
        return self.w1 * improvement + self.w2 * coverage_bonus - self.w3 * mastery_penalty