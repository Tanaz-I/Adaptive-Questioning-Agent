import torch
import torch.nn.functional as F
import numpy as np
from Adaptation_RL.knowledge_state import KnowledgeState, difficulty_level, question_types
from Adaptation_RL.MDP import MDP
from Adaptation_RL.policy_network import PolicyNetwork
from Adaptation_RL.Simulator import Simulator
import matplotlib.pyplot as plt
from collections import defaultdict, deque

class AdaptiveAgent:

    def __init__(self, topics_difficulty, prerequisites, w1, w2, w3, n_episodes=1000, n_questions=100):
        self.mdp = MDP(list(topics_difficulty.keys()), difficulty_types = ['basic', 'intermediate', 'advanced'], q_types = ['factual', 'inferential', 'evaluative'], w1 = w1, w2 = w2, w3 = w3)
        self.policy_network  = PolicyNetwork(num_topics = len(topics_difficulty), num_actions = len(topics_difficulty) * 3 * 3)
        self.simulator = Simulator(topic_difficulty = topics_difficulty)
        self.ks = KnowledgeState(topics_difficulty = topics_difficulty, prerequisites = prerequisites, window_size = 10)
        self.optimizer = torch.optim.AdamW( self.policy_network.parameters(), lr = 1e-4)
        self.pretrain(n_episodes =n_episodes, n_questions = n_questions)
        self.reset_rl_agent(list(topics_difficulty.keys()))

    def _build_action_mask(self):
        mask = torch.full((self.mdp.n_actions,), float('-inf'))
        for idx in range(self.mdp.n_actions):
            topic, diff, qtype = self.mdp.decode(idx)
            if not self.ks.prerequisites_met(topic):
                continue
            valid = self.ks.get_valid_actions(topic)
            if (diff, qtype) in valid:
                mask[idx] = 0.0
        return mask
    
    def select_action(self, state_vector, training=False):
        state  = torch.FloatTensor(state_vector)
        logits = self.policy_network(state)
        mask   = self._build_action_mask()
        logits = logits + mask
        temperature = 1.0 if training else 0.7
        probs  = F.softmax(logits / temperature, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def run_episode(self, n_questions=25):
        """
        Runs one full simulated session.
        Returns log_probs and rewards collected at each step.
        """
        
        self.simulator.reset_mastery_scores()
        for topic in self.ks.topics:
            self.ks.topic_score[topic] = 0.0
            self.ks.attempts[topic] = 0
            self.ks.recent_scores[topic].clear()
            self.ks.prev_qtype[topic] = (None, None)
            self.ks.combo_scores[topic] = defaultdict(list)
            self.ks.current_level[topic] = { 'diff_idx' : 0, 'qtype_idx' : 0, 'earned_diff_idx' : 0, 'earned_qtype_idx': 0}
        self.ks.prev_topic = None

        log_probs = []
        rewards   = []
        entropies = []

        for _ in range(n_questions):
            state_vector = self.ks.get_state_vector()
            action_idx, log_prob, entropy = self.select_action(state_vector, training = True)
            entropies.append(entropy)
            topic, difficulty, question_type = self.mdp.decode(action_idx)

            old_earned_diff  = self.ks.current_level[topic]['earned_diff_idx']
            old_earned_qtype = self.ks.current_level[topic]['earned_qtype_idx']

            prev_score = self.ks.topic_score[topic]
            score = self.simulator.get_score(topic, difficulty, question_type)
            self.ks.update(topic, score, difficulty, question_type)

            reward = self.mdp.compute_reward(self.ks, topic, score, prev_score, old_earned_diff, old_earned_qtype)

            log_probs.append(log_prob)
            rewards.append(reward)

        return log_probs, rewards, entropies

    def update_policy_pretrain(self, log_probs, rewards, entropies, gamma=0.99, entropy_coef = 0.1):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        entropy_loss = [e for e in entropies]

        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)
        
        loss = torch.stack(policy_loss).sum() - entropy_coef * torch.stack(entropy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update(self, topic, score, difficulty, question_type):
        prev_score = self.ks.topic_score[topic]
        old_earned_diff  = self.ks.current_level[topic]['earned_diff_idx']
        old_earned_qtype = self.ks.current_level[topic]['earned_qtype_idx']
        self.ks.update(topic, score, difficulty, question_type)
        return self.mdp.compute_reward(self.ks, topic, score, prev_score, old_earned_diff, old_earned_qtype)

    def pretrain(self, n_episodes=3000, n_questions=100, gamma=0.99):
        total_losses = []

        for episode in range(n_episodes):
            log_probs, rewards, entropies = self.run_episode(n_questions)
            loss = self.update_policy_pretrain(log_probs, rewards, entropies, gamma)
            total_losses.append(loss)

            if (episode + 1) % 50 == 0:
                avg_loss   = np.mean(total_losses[-50:])
                avg_reward = np.mean(rewards)
                print(f"Episode {episode+1}/{n_episodes} | " f"Avg Loss: {avg_loss:.4f} | " f"Avg Reward: {avg_reward:.4f}")
        
        window   = 50
        smoothed = np.convolve(total_losses, np.ones(window)/window, mode='valid')
        plt.plot(smoothed)
        plt.title('Smoothed Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.show()

        return total_losses
    
    def reset_rl_agent(self, topics):
        for topic in topics:
            self.ks.topic_score[topic]   = 0.0
            self.ks.attempts[topic]      = 0
            self.ks.recent_scores[topic].clear()
            self.ks.combo_scores[topic] = defaultdict(list)
            self.ks.prev_qtype[topic]    = (None, None)
            self.ks.current_level[topic] = {
                'diff_idx'        : 0,
                'qtype_idx'       : 0,
                'earned_diff_idx' : 0,
                'earned_qtype_idx': 0
            }
        self.ks.prev_topic = None