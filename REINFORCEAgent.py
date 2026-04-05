import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from policy_network import PolicyNetworkMLP, PolicyNetworkLSTM


class AdaptiveAgent:

    def __init__(self, topics_difficulty, prerequisites,
                 w1, w2, w3,
                 use_lstm= False,
                 hidden_size= 128,
                 num_layers = 1,
                 n_episodes = 1000,
                 n_questions = 500):

        self.use_lstm  = use_lstm
        num_topics     = len(topics_difficulty)
        num_actions    = num_topics * 3 * 3   # topics × difficulties × qtypes

        self.mdp = MDP(
            list(topics_difficulty.keys()),
            difficulty_types=['basic', 'intermediate', 'advanced'],
            q_types=['factual', 'inferential', 'evaluative'],
            w1=w1, w2=w2, w3=w3,
        )
        self.simulator = Simulator(topic_difficulty=topics_difficulty)
        self.ks = KnowledgeState(
            topics_difficulty=topics_difficulty,
            prerequisites=prerequisites,
            window_size=10,
        )

       
        if use_lstm:
            self.policy_network = PolicyNetworkLSTM(
                num_topics=num_topics,
                num_actions=num_actions,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
        else:
            self.policy_network = PolicyNetworkMLP(
                num_topics=num_topics,
                num_actions=num_actions,
            )

        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=1e-4)
        self.pretrain(n_episodes=n_episodes, n_questions=n_questions)

    
    def _build_action_mask(self):
        mask = torch.full((self.mdp.n_actions,), float('-inf'))
        for idx in range(self.mdp.n_actions):
            topic, diff, qtype = self.mdp.decode(idx)
            if not self.ks.prerequisites_met(topic):
                continue
            if (diff, qtype) in self.ks.get_valid_actions(topic):
                mask[idx] = 0.0
        return mask

    def _reset_knowledge_state(self):
       
        for topic in self.ks.topics:
            self.ks.topic_score[topic]   = 0.0
            self.ks.attempts[topic]      = 0
            self.ks.recent_scores[topic].clear()
            self.ks.prev_qtype[topic]    = (None, None)
            self.ks.combo_scores[topic]  = defaultdict(list)
            self.ks.current_level[topic] = {
                'diff_idx'        : 0,
                'qtype_idx'       : 0,
                'earned_diff_idx' : 0,
                'earned_qtype_idx': 0,
            }
        self.ks.prev_topic = None
        self.policy_network.reset_hidden()   


    def select_action(self, state_vector, training: bool = False):
        state         = torch.FloatTensor(state_vector)
        logits, _     = self.policy_network(state)
        mask          = self._build_action_mask()
        logits        = logits + mask
        temperature   = 1.0 if training else 0.7
        probs         = F.softmax(logits / temperature, dim=-1)
        dist          = torch.distributions.Categorical(probs)
        action        = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


    def run_episode(self, n_questions: int = 25):
        self.simulator.reset_mastery_scores()
        self._reset_knowledge_state()

        log_probs = []
        rewards   = []
        entropies = []

        for _ in range(n_questions):
            state_vector                  = self.ks.get_state_vector()
            action_idx, log_prob, entropy = self.select_action(state_vector, training=True)

            topic, difficulty, question_type = self.mdp.decode(action_idx)

            old_earned_diff  = self.ks.current_level[topic]['earned_diff_idx']
            old_earned_qtype = self.ks.current_level[topic]['earned_qtype_idx']
            prev_score       = self.ks.topic_score[topic]

            score = self.simulator.get_score(topic, difficulty, question_type)
            self.ks.update(topic, score, difficulty, question_type)

            reward = self.mdp.compute_reward(
                self.ks, topic, score, prev_score,
                old_earned_diff, old_earned_qtype,
            )

            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

        return log_probs, rewards, entropies

  
    def update_policy_pretrain(self, log_probs, rewards, entropies,
                               gamma: float = 0.99, entropy_coef: float = 0.1):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = [-lp * G_t for lp, G_t in zip(log_probs, returns)]
        loss = (torch.stack(policy_loss).sum()
                - entropy_coef * torch.stack(entropies).sum())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def update(self, topic, score, difficulty, question_type):
        prev_score       = self.ks.topic_score[topic]
        old_earned_diff  = self.ks.current_level[topic]['earned_diff_idx']
        old_earned_qtype = self.ks.current_level[topic]['earned_qtype_idx']
        self.ks.update(topic, score, difficulty, question_type)
        return self.mdp.compute_reward(
            self.ks, topic, score, prev_score,
            old_earned_diff, old_earned_qtype,
        )

    def reset_hidden(self):
        self.policy_network.reset_hidden()


    def pretrain(self, n_episodes: int = 1000, n_questions: int = 100,
                 gamma: float = 0.99):
        backbone = "LSTM" if self.use_lstm else "MLP"
        print(f"Pre-training REINFORCE ({backbone}) …")

        total_losses = []
        for episode in range(n_episodes):
            log_probs, rewards, entropies = self.run_episode(n_questions)
            loss = self.update_policy_pretrain(log_probs, rewards, entropies, gamma)
            total_losses.append(loss)

            if (episode + 1) % 50 == 0:
                print(f"  Episode {episode+1}/{n_episodes} | "
                      f"Avg Loss: {np.mean(total_losses[-50:]):.4f} | "
                      f"Avg Reward: {np.mean(rewards):.4f}")

        window   = min(50, len(total_losses))
        smoothed = np.convolve(total_losses, np.ones(window) / window, mode='valid')
        plt.figure()
        plt.plot(smoothed)
        plt.title(f'Smoothed Training Loss — REINFORCE ({backbone})')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.savefig(f"Results/TrainingLoss_REINFORCE_{backbone}.png")

        return total_losses