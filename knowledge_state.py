import numpy as np
from collections import defaultdict, deque


class KnowledgeState:
    
    def __init__(self, topics, window_size):        
        self.topics  = topics
        self.window_size  = window_size  #window size for the trend
        self.num_topics = len(topics)
        self.topic_score = defaultdict(float)
        self.recent_scores = defaultdict(lambda: deque(maxlen=window_size))  
        self.attempts = defaultdict(int)
        self.prev_qtype = defaultdict(lambda: (None,None))

    def update(self, topic, score, difficulty,question_type):
        
        self.attempts[topic]  += 1      
        self.topic_score[topic] += (score - self.topic_score[topic]) / self.attempts[topic]
        self.recent_scores[topic].append(score)       
        self.prev_qtype[topic] = (difficulty,question_type)

    def trend(self, topic):        
        recent = list(self.recent_scores[topic])
        if len(recent) < 2:
            return 0.0
        x = np.arange(len(recent), dtype=np.float32)
        slope = float(np.polyfit(x, recent, 1)[0])
        return slope

    def is_mastered(self, topic, threshold = 0.85,min_attempts= 3):
        return (self.attempts[topic] >= min_attempts and
                self.topic_score[topic] >= threshold)

    def is_neglected(self, topic):
        return self.attempts[topic] == 0
    
question_types = ['factual', 'inferential', 'evaluative']
difficulty_level  = ['basic', 'intermediate', 'advanced']

