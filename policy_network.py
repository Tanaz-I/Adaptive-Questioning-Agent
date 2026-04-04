import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetworkMLP(nn.Module):
    def __init__(self, num_topics: int, num_actions: int, dropout: float = 0.2):
        super().__init__()
        input_size   = num_topics * 8
        hidden_size1 = input_size * 2
        hidden_size2 = (input_size + num_actions) // 2

        self.net = nn.Sequential(
            nn.Linear(input_size,   hidden_size1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size2, num_actions),
        )

    def forward(self, state, hidden=None):
        return self.net(state), None

    def reset_hidden(self):
        pass 


class PolicyNetworkLSTM(nn.Module):
    """
    LSTM-based policy network.

    The hidden state (h, c) persists across steps within one episode so the
    network conditions on the full question history.
    Call reset_hidden() at the start of every new episode / student session.
    """

    def __init__(self, num_topics: int, num_actions: int,
                 hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        input_size       = num_topics * 8

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_actions),
        )

        self.hidden = None 

    def reset_hidden(self):
        self.hidden = None

    def forward(self, state, hidden=None):
        """
        state  : (input_size,) or (batch, input_size)
        hidden : optional (h, c) override; falls back to self.hidden
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)     
        state = state.unsqueeze(1)          

        x = self.input_proj(state)         

        h = hidden if hidden is not None else self.hidden
        lstm_out, new_hidden = self.lstm(x, h)
        self.hidden = (new_hidden[0].detach(),
                       new_hidden[1].detach())

        out    = lstm_out.squeeze(0).squeeze(0)  
        logits = self.output_head(out)             
        return logits, new_hidden