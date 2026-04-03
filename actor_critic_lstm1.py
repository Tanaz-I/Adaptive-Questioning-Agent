import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticLSTM(nn.Module):
    def __init__(self, num_topics, num_actions, hidden_size=128, num_layers=1, dropout=0.1):
        super(ActorCriticLSTM, self).__init__()

        self.input_size  = num_topics * 8
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # shared LSTM trunk
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # separate actor and critic heads
        self.actor_fc   = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_head = nn.Linear(hidden_size // 2, num_actions)

        self.critic_fc   = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_head = nn.Linear(hidden_size // 2, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.orthogonal_(self.actor_fc.weight,    gain=np.sqrt(2))
        nn.init.orthogonal_(self.critic_fc.weight,   gain=np.sqrt(2))
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state, hidden=None):
        """
        Args:
            state  : (batch, input_size)          — single step (action selection)
                     (batch, seq_len, input_size) — full sequence (PPO update)
            hidden : LSTM hidden state tuple or None

        Returns:
            logits  : (batch, num_actions)          for single step
                      (batch, seq_len, num_actions) for sequence
            values  : (batch,)                      for single step
                      (batch, seq_len)              for sequence
            hidden  : updated LSTM hidden state
        """
        single_step = (state.dim() == 2)
        if single_step:
            state = state.unsqueeze(1)          # (batch, 1, input_size)

        lstm_out, hidden = self.lstm(state, hidden)
        # lstm_out: (batch, seq_len, hidden_size)

        # ---- Actor: always use only the LAST timestep for action logits ----
        actor_x = F.relu(self.actor_fc(lstm_out[:, -1, :]))   # (batch, H//2)
        logits  = self.actor_head(actor_x)                     # (batch, num_actions)

        # ---- Critic: use ALL timesteps so we get a value per step ----
        critic_x = F.relu(self.critic_fc(lstm_out))            # (batch, seq_len, H//2)
        values   = self.critic_head(critic_x).squeeze(-1)      # (batch, seq_len)

        if single_step:
            # Squeeze seq_len=1 away for both outputs
            logits = logits                                     # already (batch, num_actions)
            values = values.squeeze(1)                         # (batch,)

        return logits, values, hidden