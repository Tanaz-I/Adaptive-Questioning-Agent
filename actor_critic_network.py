import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCriticMLP(nn.Module):
    def __init__(self, num_topics, num_actions, dropout = 0.1):
        super(ActorCriticMLP, self).__init__()
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
        nn.init.orthogonal_(self.actor_fc.weight,   gain=np.sqrt(2))
        nn.init.orthogonal_(self.critic_fc.weight,  gain=np.sqrt(2))
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state, hidden=None):
        # state shape: (batch, input_size) or (batch, seq_len, input_size)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # add seq_len=1 dim

        out, hidden = self.lstm(state, hidden)
        out = out[:, -1, :]  # take last timestep

        actor_x = F.relu(self.actor_fc(out))
        logits  = self.actor_head(actor_x)

        critic_x = F.relu(self.critic_fc(out))
        value    = self.critic_head(critic_x).squeeze(-1)

        return logits, value, hidden
        
    """ def forward(self, state, hidden=None):

        # state shape:
        # (batch, input_size) OR (batch, seq_len, input_size)

        if state.dim() == 2:
            state = state.unsqueeze(1)

        lstm_out, hidden = self.lstm(state, hidden)

        # -------- ACTOR --------
        # use only last timestep for action selection
        actor_input = lstm_out[:, -1, :]

        actor_x = F.relu(self.actor_fc(actor_input))
        logits  = self.actor_head(actor_x)

        # -------- CRITIC --------
        # use ALL timesteps for value prediction
        critic_x = F.relu(self.critic_fc(lstm_out))

        values = self.critic_head(critic_x).squeeze(-1)

        return logits, values, hidden"""
