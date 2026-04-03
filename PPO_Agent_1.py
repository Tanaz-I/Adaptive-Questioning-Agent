import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

from knowledge_state import KnowledgeState, difficulty_level, question_types
from MDP import MDP
from Simulator import Simulator
from actor_critic_network import ActorCriticNetwork


class PPOAgent:

    def __init__(self, topics_difficulty, prerequisites, w1, w2, w3,
                 n_episodes=1000, n_questions=1000,
                 gamma=0.99,
                 lam=0.9,
                 clip_eps=0.3,        # Fix 3: tightened from 0.2
                 ppo_epochs=6,         # Fix 3: increased from 4
                 mini_batch=128,       # Fix 3: increased from 64
                 entropy_coef=0.12,    # Fix 3: start higher
                 value_coef=0.5,      # Fix 3: increased from 0.5
                 rollout_episodes=8, use_lstm=False):

        self.gamma            = gamma
        self.lam              = lam
        self.clip_eps         = clip_eps
        self.ppo_epochs       = ppo_epochs
        self.mini_batch       = mini_batch
        self.entropy_coef     = entropy_coef
        self.value_coef       = value_coef
        self.rollout_episodes = rollout_episodes

        self.mdp        = MDP(list(topics_difficulty.keys()),
                              difficulty_types=['basic', 'intermediate', 'advanced'],
                              q_types=['factual', 'inferential', 'evaluative'],
                              w1=w1, w2=w2, w3=w3)
        self.ac_network = ActorCriticNetwork(num_topics=len(topics_difficulty),
                                             num_actions=len(topics_difficulty) * 3 * 3)
        self.simulator  = Simulator(topic_difficulty=topics_difficulty)
        self.ks         = KnowledgeState(topics_difficulty=topics_difficulty,
                                         prerequisites=prerequisites, window_size=10)
        self.optimizer  = torch.optim.AdamW(self.ac_network.parameters(), lr=1e-4)
        

        self.pretrain(n_episodes=n_episodes, n_questions=n_questions)

    # ------------------------------------------------------------------
    def _build_action_mask(self):
        mask = torch.full((self.mdp.n_actions,), float('-inf'))
        for idx in range(self.mdp.n_actions):
            topic, diff, qtype = self.mdp.decode(idx)
            if not self.ks.prerequisites_met(topic):
                continue
            if (diff, qtype) in self.ks.get_valid_actions(topic):
                mask[idx] = 0.0
        return mask

    def select_action(self, state_vector, training=False):
        state  = torch.FloatTensor(state_vector)
        logits, value = self.ac_network(state)
        mask   = self._build_action_mask()
        logits = logits + mask
        temperature = 1.0 if training else 0.7
        probs  = F.softmax(logits / temperature, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), value, mask

    # ------------------------------------------------------------------
    # Fix 1 + Fix 4: GAE reset at episode boundaries, correct bootstrapping
    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae        = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            if dones[t]:              # Fix 4: reset at every episode boundary
                next_value = 0.0
                gae        = 0.0

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]

        advantages = torch.FloatTensor(advantages)
        returns    = advantages + torch.FloatTensor(values)
        return advantages, returns

    # ------------------------------------------------------------------
    def run_episode(self, n_questions=25):
        self.simulator.reset_mastery_scores()
        for topic in self.ks.topics:
            self.ks.topic_score[topic]    = 0.0
            self.ks.attempts[topic]       = 0
            self.ks.recent_scores[topic].clear()
            self.ks.prev_qtype[topic]     = (None, None)
            self.ks.combo_scores[topic]   = defaultdict(list)
            self.ks.current_level[topic]  = {
                'diff_idx': 0, 'qtype_idx': 0,
                'earned_diff_idx': 0, 'earned_qtype_idx': 0
            }
        self.ks.prev_topic = None

        buf = defaultdict(list)

        for _ in range(n_questions):
            state_vector = self.ks.get_state_vector()
            action_idx, log_prob, entropy, value, mask = self.select_action(
                state_vector, training=True)
            topic, difficulty, question_type = self.mdp.decode(action_idx)

            old_earned_diff  = self.ks.current_level[topic]['earned_diff_idx']
            old_earned_qtype = self.ks.current_level[topic]['earned_qtype_idx']
            prev_score       = self.ks.topic_score[topic]

            score = self.simulator.get_score(topic, difficulty, question_type)
            self.ks.update(topic, score, difficulty, question_type)

            reward = self.mdp.compute_reward(
                self.ks, topic, score, prev_score,
                old_earned_diff, old_earned_qtype)

            buf['states'].append(state_vector)
            buf['actions'].append(action_idx)
            buf['log_probs'].append(log_prob.item())
            buf['rewards'].append(reward)
            buf['values'].append(value.item())
            buf['entropies'].append(entropy.item())
            buf['dones'].append(0.0)
            buf['masks'].append(mask)

        buf['dones'][-1] = 1.0   # mark episode end
        return buf

    # ------------------------------------------------------------------
    def ppo_update(self, buf):
        states    = torch.FloatTensor(np.array(buf['states']))
        actions   = torch.LongTensor(buf['actions'])
        old_lp    = torch.FloatTensor(buf['log_probs'])
        masks     = torch.stack(buf['masks'])

        # Fix 1: advantages already computed per-episode before merging
        advantages = torch.FloatTensor(buf['advantages'])
        returns    = torch.FloatTensor(buf['returns'])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_val = 0.0
        n = len(actions)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.mini_batch):
                idx = indices[start: start + self.mini_batch]

                s_batch   = states[idx]
                a_batch   = actions[idx]
                olp_batch = old_lp[idx]
                adv_batch = advantages[idx]
                ret_batch = returns[idx]
                mask_batch = masks[idx]

                logits, values = self.ac_network(s_batch)
                logits = logits + mask_batch
                dist      = torch.distributions.Categorical(F.softmax(logits, dim=-1))
                new_lp    = dist.log_prob(a_batch)
                entropy   = dist.entropy().mean()

                ratio       = torch.exp(new_lp - olp_batch)
                surr1       = ratio * adv_batch
                surr2       = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(values, ret_batch)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_network.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss_val += loss.item()

        return total_loss_val / (self.ppo_epochs * max(1, n // self.mini_batch))

    # ------------------------------------------------------------------
    def pretrain(self, n_episodes=2000, n_questions=500):
        total_losses = []

        base_entropy = 0.25   # Fix 2: higher starting entropy
        min_entropy  = 0.1   # Fix 2: higher floor

        episode = 0
        while episode < n_episodes:

            # Fix 2: delayed + slower entropy decay (only starts after 30% of training)
            progress = max(0.0, (episode - 0.3 * n_episodes) / (0.7 * n_episodes))
            self.entropy_coef = max(min_entropy, base_entropy * (1.0 - progress))
            self.clip_eps     = max(0.10, 0.30 * (1 - episode / n_episodes)) 

            # Fix 1: compute GAE per episode BEFORE merging into combined buffer
            combined_buf = defaultdict(list)
            for _ in range(self.rollout_episodes):
                buf = self.run_episode(n_questions)
                adv, ret = self._compute_gae(buf['rewards'], buf['values'], buf['dones'])
                buf['advantages'] = adv.tolist()
                buf['returns']    = ret.tolist()
                for key in buf:
                    combined_buf[key].extend(buf[key])

            loss = self.ppo_update(combined_buf)
            total_losses.append(loss)
            episode += self.rollout_episodes

            if episode % 50 == 0:
                avg_loss   = np.mean(total_losses[-50:])
                avg_reward = np.mean(combined_buf['rewards'])
                print(f"Episode {episode}/{n_episodes} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Avg Reward: {avg_reward:.4f} | "
                      f"Entropy Coef: {self.entropy_coef:.4f}")

        window   = 50
        smoothed = np.convolve(total_losses, np.ones(window) / window, mode='valid')
        plt.plot(smoothed)
        plt.title('Smoothed PPO Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        return total_losses

    # ------------------------------------------------------------------
    def update(self, topic, score, difficulty, question_type):
        prev_score       = self.ks.topic_score[topic]
        old_earned_diff  = self.ks.current_level[topic]['earned_diff_idx']
        old_earned_qtype = self.ks.current_level[topic]['earned_qtype_idx']
        self.ks.update(topic, score, difficulty, question_type)
        return self.mdp.compute_reward(
            self.ks, topic, score, prev_score,
            old_earned_diff, old_earned_qtype)