import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import random
import matplotlib.pyplot as plt

from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from dqn_network import DQNNetwork


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(
        self,
        topics_difficulty,
        prerequisites,
        w1, w2, w3,
        n_episodes=1000,
        n_questions=100,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_capacity=10_000,
        target_update_freq=10,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        eval_eps=0.15,          # exploration used at evaluation time
    ):
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.eps                = eps_start
        self.eps_end            = eps_end
        self.eps_decay          = eps_decay
        self.eval_eps           = eval_eps  # fixed epsilon used when training=False

        self.mdp = MDP(
            list(topics_difficulty.keys()),
            difficulty_types=['basic', 'intermediate', 'advanced'],
            q_types=['factual', 'inferential', 'evaluative'],
            w1=w1, w2=w2, w3=w3,
        )
        n_actions = len(topics_difficulty) * 3 * 3
        n_topics  = len(topics_difficulty)

        self.online_net = DQNNetwork(num_topics=n_topics, num_actions=n_actions)
        self.target_net = DQNNetwork(num_topics=n_topics, num_actions=n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer(buffer_capacity)

        self.simulator = Simulator(topic_difficulty=topics_difficulty)
        self.ks        = KnowledgeState(
            topics_difficulty=topics_difficulty,
            prerequisites=prerequisites,
            window_size=10,
        )

        self.pretrain(n_episodes=n_episodes, n_questions=n_questions)

    def _build_action_mask(self):
        mask = torch.zeros(self.mdp.n_actions, dtype=torch.bool)
        for idx in range(self.mdp.n_actions):
            topic, diff, qtype = self.mdp.decode(idx)
            if not self.ks.prerequisites_met(topic):
                continue
            if (diff, qtype) in self.ks.get_valid_actions(topic):
                mask[idx] = True
        return mask

    def _reset_knowledge_state(self):
        self.simulator.reset_mastery_scores()
        for topic in self.ks.topics:
            self.ks.topic_score[topic]   = 0.0
            self.ks.attempts[topic]      = 0
            self.ks.recent_scores[topic].clear()
            self.ks.prev_qtype[topic]    = (None, None)
            self.ks.combo_scores[topic]  = defaultdict(list)
            self.ks.current_level[topic] = {
                'diff_idx': 0, 'qtype_idx': 0,
                'earned_diff_idx': 0, 'earned_qtype_idx': 0,
            }
        self.ks.prev_topic = None

    def select_action(self, state_vector, training=False):
        mask = self._build_action_mask()
        valid_indices = mask.nonzero(as_tuple=False).squeeze(1).tolist()

        # During training use decaying eps; during evaluation use fixed eval_eps
        # so the agent still explores enough to unlock all (diff, qtype) combos
        # that is_mastered() requires — pure greedy never gets there.
        active_eps = self.eps if training else self.eval_eps
        if random.random() < active_eps:
            return random.choice(valid_indices)

        with torch.no_grad():
            state    = torch.FloatTensor(state_vector).unsqueeze(0)
            q_vals   = self.online_net(state).squeeze(0)

        masked_q = q_vals.clone()
        masked_q[~mask] = float('-inf')
        return int(masked_q.argmax().item())

    def run_episode(self, n_questions=25):
        self._reset_knowledge_state()
        episode_rewards = []

        for _ in range(n_questions):
            state_vector = self.ks.get_state_vector()
            action_idx   = self.select_action(state_vector, training=True)

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

            next_state_vector = self.ks.get_state_vector()
            self.replay.push(state_vector, action_idx, reward, next_state_vector, 0.0)
            episode_rewards.append(reward)

        # Mark last transition as terminal
        if self.replay.buffer:
            s, a, r, ns, _ = self.replay.buffer[-1]
            self.replay.buffer[-1] = (s, a, r, ns, 1.0)

        return episode_rewards

    def update_network(self):
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets      = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def pretrain(self, n_episodes=1000, n_questions=100):
        total_losses        = []
        episode_rewards_log = []

        for episode in range(1, n_episodes + 1):
            ep_rewards = self.run_episode(n_questions)
            loss       = self.update_network()

            self.eps = max(self.eps_end, self.eps * self.eps_decay)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            if loss is not None:
                total_losses.append(loss)
            episode_rewards_log.append(np.mean(ep_rewards))

            if episode % 50 == 0:
                avg_loss   = np.mean(total_losses[-50:]) if total_losses else float('nan')
                avg_reward = np.mean(episode_rewards_log[-50:])
                print(
                    f"Episode {episode}/{n_episodes} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Epsilon: {self.eps:.4f}"
                )

        window   = 50
        smoothed = np.convolve(total_losses, np.ones(window) / window, mode='valid')
        plt.plot(smoothed)
        plt.title('Smoothed DQN Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.show()

        return total_losses

    def update(self, topic, score, difficulty, question_type):
        prev_score       = self.ks.topic_score[topic]
        old_earned_diff  = self.ks.current_level[topic]['earned_diff_idx']
        old_earned_qtype = self.ks.current_level[topic]['earned_qtype_idx']
        self.ks.update(topic, score, difficulty, question_type)
        return self.mdp.compute_reward(
            self.ks, topic, score, prev_score,
            old_earned_diff, old_earned_qtype,
        )