import torch
import torch.nn.functional as F
import numpy as np


class Agent:

    def __init__(self, mdp, policy_network, simulator, knowledge_state, optimizer):
        self.mdp = mdp
        self.policy_network  = policy_network
        self.simulator = simulator
        self.ks = knowledge_state
        self.optimizer = optimizer

    def select_action(self, state_vector):
        state  = torch.FloatTensor(state_vector)
        logits = self.policy_network(state)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

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

        log_probs = []
        rewards   = []

        for _ in range(n_questions):
            state_vector = self.ks.get_state_vector()
            action_idx, log_prob = self.select_action(state_vector)
            topic, difficulty, question_type = self.mdp.decode(action_idx)

            prev_score = self.ks.topic_score[topic]
            score = self.simulator.get_score(topic, difficulty, question_type)
            self.ks.update(topic, score, difficulty, question_type)

            reward = self.mdp.compute_reward(self.ks, topic, score, prev_score)

            log_probs.append(log_prob)
            rewards.append(reward)

        return log_probs, rewards

    def update_policy(self, log_probs, rewards, gamma=0.99):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = []
        for log_prob, G_t in zip(log_probs, returns):
            loss.append(-log_prob * G_t)
        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def pretrain(self, n_episodes=500, n_questions=25, gamma=0.99):
        total_losses = []

        for episode in range(n_episodes):
            log_probs, rewards = self.run_episode(n_questions)
            loss = self.update_policy(log_probs, rewards, gamma)
            total_losses.append(loss)

            if (episode + 1) % 50 == 0:
                avg_loss   = np.mean(total_losses[-50:])
                avg_reward = np.mean(rewards)
                print(f"Episode {episode+1}/{n_episodes} | " f"Avg Loss: {avg_loss:.4f} | " f"Avg Reward: {avg_reward:.4f}")

        return total_losses