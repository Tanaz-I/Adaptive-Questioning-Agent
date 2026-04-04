import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNNetwork(nn.Module):
    def __init__(self, num_topics, num_actions, dropout=0.1):
        super(DQNNetwork, self).__init__()
        input_size   = num_topics * 8
        hidden_large = 256
        hidden_small = 128

        # Shared trunk — two layers before the dueling split
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_large),
            nn.LayerNorm(hidden_large),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_large, hidden_large),
            nn.LayerNorm(hidden_large),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_large, hidden_small),
            nn.ReLU(),
            nn.Linear(hidden_small, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_large, hidden_small),
            nn.ReLU(),
            nn.Linear(hidden_small, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Value stream — standard gain
        for m in self.value_stream:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

        # Advantage head — small gain to keep initial Q-values near zero
        for m in self.advantage_stream:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        shared = self.shared(state)           # dropout only active in train mode

        val = self.value_stream(shared)       # (B, 1)
        adv = self.advantage_stream(shared)   # (B, A)

        # Dueling: Q = V + (A - mean(A))
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q