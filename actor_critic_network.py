import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, num_topics: int, num_actions: int, dropout: float = 0.2):
        super(ActorCriticNetwork, self).__init__()
        self.input_size = num_topics * 8
        hidden_size1    = self.input_size * 2
        hidden_size2    = (self.input_size + num_actions) // 2

        self.fc1     = nn.Linear(self.input_size, hidden_size1)
        self.fc2     = nn.Linear(hidden_size1, hidden_size2)
        self.dropout = nn.Dropout(dropout)

        self.actor_head  = nn.Linear(hidden_size2, num_actions)  
        self.critic_head = nn.Linear(hidden_size2, 1)           

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.actor_head(x)
        value  = self.critic_head(x).squeeze(-1)
        return logits, value
