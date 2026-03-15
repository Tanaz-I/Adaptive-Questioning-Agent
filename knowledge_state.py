import numpy as np
from collections import defaultdict, deque


class KnowledgeState:
    
    def __init__(self, topics_difficulty, window_size):        
        self.topics_difficulty = topics_difficulty # dictionary of topic:difficulty
        self.topics = list(self.topics_difficulty.keys())
        self.window_size  = window_size  #window size for the trend
        self.num_topics = len(topics_difficulty)
        self.topic_score = defaultdict(float)
        self.recent_scores = defaultdict(lambda: deque(maxlen=window_size))  
        self.attempts = defaultdict(int)
        self.prev_qtype = defaultdict(lambda: (None,None))
        self.master_thresh = {'basic' : 0.90, 'intermediate' : 0.85, 'advanced':0.75}

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

    def is_mastered(self, topic,min_attempts= 3):
        topic_difficulty = self.topics_difficulty[topic]
        threshold = self.master_thresh[topic_difficulty]
        trend_eval = self.trend(topic)
        return (self.attempts[topic] >= min_attempts and self.topic_score[topic] >= threshold and self.trend_eval >= 0)

    def is_neglected(self, topic):
        return self.attempts[topic] == 0
    
question_types = ['factual', 'inferential', 'evaluative']
difficulty_level  = ['basic', 'intermediate', 'advanced']

