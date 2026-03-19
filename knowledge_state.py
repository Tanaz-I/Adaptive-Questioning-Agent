import numpy as np
from collections import defaultdict, deque


class KnowledgeState:
    
    def __init__(self, topics_difficulty, window_size, prerequisites):        
        self.topics_difficulty = topics_difficulty # dictionary of topic:difficulty
        self.prerequisites_topic = prerequisites
        self.topics = list(self.topics_difficulty.keys())
        self.window_size  = window_size  #window size for the trend
        self.num_topics = len(topics_difficulty)
        self.topic_score = defaultdict(float)
        self.recent_scores = defaultdict(lambda: deque(maxlen=window_size))  
        self.attempts = defaultdict(int)
        self.prev_qtype = defaultdict(lambda: (None,None))
        self.combo_scores = defaultdict(lambda: defaultdict(list))
        self.current_level = {topic: {'diff_idx': 0, 'qtype_idx': 0,'earned_diff_idx': 0, 'earned_qtype_idx': 0} for topic in self.topics }
        self.prev_topic = None

        self.master_thresh = {'basic' : 0.75, 'intermediate' : 0.65, 'advanced':0.55}
        self.ordinal_order_difficulty = {'basic' : 0.2, 'intermediate' : 0.6, 'advanced': 1}
        self.question_type_encoding = {'factual' : 0.2, 'inferential' : 0.6, 'evaluative': 1}
        self.diff_order    = {'basic' : 0, 'intermediate' : 1, 'advanced': 2}
        self.qtype_order   = {'factual' : 0, 'inferential' : 1, 'evaluative': 2}
        self.max_attempts = 50

    def prerequisites_met(self, topic):
        for prereq in self.prerequisites_topic.get(topic, []):
            if not self.is_sufficiently_understood(prereq):
                return False
        return True
    
    def is_sufficiently_understood(self, topic):
        topic_difficulty = self.topics_difficulty[topic]
        sufficiency_thresh = {'basic' : 0.65, 'intermediate' : 0.55, 'advanced' : 0.45}
        threshold = sufficiency_thresh[topic_difficulty]
        return (self.attempts[topic] >= 5 and self.topic_score[topic] >= threshold and self.trend(topic) >= 0)

    def update(self, topic, score, difficulty,question_type):
        
        self.attempts[topic]  += 1      
        self.topic_score[topic] += (score - self.topic_score[topic]) / self.attempts[topic]
        self.recent_scores[topic].append(score)       
        self.prev_qtype[topic] = (difficulty, question_type)
        self.prev_topic = topic
        self.combo_scores[topic][(difficulty, question_type)].append(score)
        self.update_current_level(topic, difficulty, question_type)

    def get_combo_threshold(self, topic, difficulty, question_type):
        topic_base = self.master_thresh[self.topics_difficulty[topic]]
        
        # adjustment based on question difficulty
        diff_adjustment = {
            'basic'        :  0.10,   
            'intermediate' :  0.05,
            'advanced'     :  0.00    
        }
        
        # adjustment based on question type
        qtype_adjustment = {
            'factual'     :  0.05,   
            'inferential' :  0.02,
            'evaluative'  :  0.00    
        }
        
        return (topic_base + diff_adjustment[difficulty] + qtype_adjustment[question_type])

    def update_current_level(self, topic, difficulty, question_type):
        diff_idx  = self.diff_order[difficulty]
        qtype_idx = self.qtype_order[question_type]
        
        self.current_level[topic]['diff_idx']  = diff_idx
        self.current_level[topic]['qtype_idx'] = qtype_idx
        
        self.current_level[topic]['earned_diff_idx'] = max(self.current_level[topic]['earned_diff_idx'], diff_idx)
        self.current_level[topic]['earned_qtype_idx'] = max(self.current_level[topic]['earned_qtype_idx'], qtype_idx)

    def get_valid_actions(self, topic):
        # block entire topic if prerequisites not met
        if not self.prerequisites_met(topic):
            return []
        
        curr_diff_idx  = self.current_level[topic]['earned_diff_idx']
        curr_qtype_idx = self.current_level[topic]['earned_qtype_idx']
        
        valid = []
        
        # all levels at or below current
        for d in range(curr_diff_idx + 1):
            for q in range(curr_qtype_idx + 1):
                valid.append((difficulty_level[d], question_types[q]))
        
        # one step ahead — qtype first, then difficulty
        if curr_qtype_idx < 2:
            valid.append(( difficulty_level[curr_diff_idx], question_types[curr_qtype_idx + 1]))
        elif curr_diff_idx < 2:
            valid.append((difficulty_level[curr_diff_idx + 1], question_types[0]))
        
        return valid

    def trend(self, topic):        
        recent = list(self.recent_scores[topic])
        if len(recent) < 2:
            return 0.0
        x = np.arange(len(recent), dtype=np.float32)
        slope = float(np.polyfit(x, recent, 1)[0])
        return float(np.clip(slope, -1.0, 1.0))

    def is_mastered(self, topic, min_attempts=3):
        topic_difficulty = self.topics_difficulty[topic]
        threshold        = self.master_thresh[topic_difficulty]
        trend_eval       = self.trend(topic)

        if self.attempts[topic] < min_attempts or trend_eval < 0:
            return False

        all_combos = {
            (d, q)
            for d in difficulty_level
            for q in question_types
        }

        
        for combo in all_combos:
            diff, qtype = combo
            scores = self.combo_scores[topic][combo]
            if not scores:                          
                return False
            threshold = self.get_combo_threshold(topic, diff, qtype)
            if np.mean(scores) < threshold:         
                return False

        return True

    def is_neglected(self, topic):
        return self.attempts[topic] == 0
    
    def get_state_vector(self):
        vector = []
        for topic in self.topics:
            vector.append(self.topic_score[topic])
            vector.append(min(self.attempts[topic] / self.max_attempts, 1.0))
            
            prev_diff, prev_qtype = self.prev_qtype[topic]
            if prev_diff is None:
                vector.append(0.0)
            else:
                vector.append(self.ordinal_order_difficulty[prev_diff])
                
            if prev_qtype is None:
                vector.append(0.0)
            else:
                vector.append(self.question_type_encoding[prev_qtype])
                
            vector.append(self.trend(topic))
            vector.append(float(self.is_mastered(topic)))
            vector.append(float(self.is_neglected(topic)))
            vector.append(1.0 if topic == self.prev_topic else 0.0)

        return np.array(vector, dtype=np.float32)
            

    
question_types = ['factual', 'inferential', 'evaluative']
difficulty_level  = ['basic', 'intermediate', 'advanced']

