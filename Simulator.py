import numpy as np


class Simulator:

    def __init__(self, topic_difficulty):
        self.topic_difficulty = topic_difficulty
        self.difficulties  = ['basic', 'intermediate', 'advanced']
        self.question_types = ['factual', 'inferential', 'evaluative']

        self.question_difficulty_penalty = {'basic' : 0, 'intermediate' : 0.05, 'advanced' : 0.1}
        self.question_type_penalty = {'factual' : 0, 'inferential' : 0.02, 'evaluative' : 0.05}
        self.mastery_topic = {topic: np.random.uniform(0.3, 1) for topic in topic_difficulty.keys()}

    def reset_mastery_scores(self):
        self.mastery_topic = { topic: np.random.uniform(0, 1) for topic in self.topic_difficulty.keys()}

    def get_score(self, topic, difficulty, question_type):
        """
        Returns a simulated score for a student answer.
        Higher mastery + easier question = higher score.
        """
        base       = self.mastery_topic[topic]
        diff_pen   = self.question_difficulty_penalty[difficulty]
        qtype_pen  = self.question_type_penalty[question_type]
        noise      = np.random.normal(0, 0.05)
        score      = base - diff_pen - qtype_pen + noise
        return float(np.clip(score, 0, 1))