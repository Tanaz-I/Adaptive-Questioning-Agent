import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNNetwork(nn.Module):
    def __init__(self, num_topics, num_actions, dropout=0.1):
        super(DQNNetwork, self).__init__()
        self.input_size  = num_topics * 8
        hidden_size1     = self.input_size * 2
        hidden_size2     = (self.input_size + num_actions) // 2

        self.fc1     = nn.Linear(self.input_size, hidden_size1)
        self.dropout = nn.Dropout(dropout)

        # Dueling streams
        self.value_fc   = nn.Linear(hidden_size1, hidden_size2)
        self.value_head = nn.Linear(hidden_size2, 1)

        self.advantage_fc   = nn.Linear(hidden_size1, hidden_size2)
        self.advantage_head = nn.Linear(hidden_size2, num_actions)

        #self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight,           gain=np.sqrt(2)); nn.init.constant_(self.fc1.bias,           0.0)
        nn.init.orthogonal_(self.value_fc.weight,      gain=np.sqrt(2)); nn.init.constant_(self.value_fc.bias,      0.0)
        nn.init.orthogonal_(self.advantage_fc.weight,  gain=np.sqrt(2)); nn.init.constant_(self.advantage_fc.bias,  0.0)
        nn.init.orthogonal_(self.value_head.weight,    gain=1.0);        nn.init.constant_(self.value_head.bias,    0.0)
        nn.init.orthogonal_(self.advantage_head.weight,gain=0.01);       nn.init.constant_(self.advantage_head.bias,0.0)

    def forward(self, state: torch.Tensor):
        shared = self.dropout(F.relu(self.fc1(state)))

        val = self.dropout(F.relu(self.value_fc(shared)))
        val = self.value_head(val)                          # (B, 1)

        adv = self.dropout(F.relu(self.advantage_fc(shared)))
        adv = self.advantage_head(adv)                      # (B, A)

        # Dueling combination: Q = V + (A - mean(A))
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q