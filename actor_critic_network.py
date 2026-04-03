import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, num_topics, num_actions, dropout = 0.1):
        super(ActorCriticNetwork, self).__init__()
        self.input_size = num_topics * 8
        hidden_size1    = self.input_size * 2
        hidden_size2    = (self.input_size + num_actions) // 2

        self.fc1     = nn.Linear(self.input_size, hidden_size1)
        #self.fc2     = nn.Linear(hidden_size1, hidden_size2)
        self.dropout = nn.Dropout(dropout)
        
        self.actor_fc = nn.Linear(hidden_size1,hidden_size2)
        self.actor_head  = nn.Linear(hidden_size2, num_actions)  
        
        self.critic_fc = nn.Linear(hidden_size1, hidden_size2)
        self.critic_head = nn.Linear(hidden_size2, 1) 
        self._init_weights() 
        
    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight,      gain=np.sqrt(2)); nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.actor_fc.weight,  gain=np.sqrt(2)); nn.init.constant_(self.actor_fc.bias, 0.0)
        nn.init.orthogonal_(self.critic_fc.weight, gain=np.sqrt(2)); nn.init.constant_(self.critic_fc.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01); nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0);  nn.init.constant_(self.critic_head.bias, 0.0)
         

    def forward(self, state: torch.Tensor):
        """x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.actor_head(x)
        value  = self.critic_head(x).squeeze(-1)"""
        
        shared = self.dropout(F.relu(self.fc1(state)))
        actor_x = self.dropout(F.relu(self.actor_fc(shared)))
        logits  = self.actor_head(actor_x)
        
        critic_x = self.dropout(F.relu(self.critic_fc(shared)))
        value    = self.critic_head(critic_x).squeeze(-1)
        return logits, value
