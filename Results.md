<b> <h2> Results for REINFORCE with regularization : 
<h3> Policy network
```python 
self.input_size  = num_topics * 7 
self.fc1         = nn.Linear(self.input_size, hidden_size1)
self.fc2         = nn.Linear(hidden_size1, hidden_size2)
self.fc3         = nn.Linear(hidden_size2, num_actions)
self.dropout     = nn.Dropout(dropout)
```
<b> <h3> Reward_weights
weight_improvement = 0.4, weight_coverage_penalty = 0.5, weight_mastery_penalty = 0.1

<b> <h3>Results:
