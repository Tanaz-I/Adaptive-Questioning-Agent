import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, num_topics, num_actions, dropout=0.2):
        super(PolicyNetwork, self).__init__()
        self.input_size  = num_topics * 8
        hidden_size1     = self.input_size * 2
        hidden_size2     = (self.input_size + num_actions) // 2
        
        self.fc1     = nn.Linear(self.input_size, hidden_size1)
        self.fc2     = nn.Linear(hidden_size1, hidden_size2)
        self.fc3     = nn.Linear(hidden_size2, num_actions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x