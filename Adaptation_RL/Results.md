## Results for REINFORCE with regularization : 
- ## Policy network
    ```python 
    self.input_size  = num_topics * 7 
    self.fc1         = nn.Linear(self.input_size, hidden_size1)
    self.fc2         = nn.Linear(hidden_size1, hidden_size2)
    self.fc3         = nn.Linear(hidden_size2, num_actions)
    self.dropout     = nn.Dropout(dropout)
    ```
-  ## Reward_weights</b> </h3>
    weight_improvement = 0.4, weight_coverage_penalty = 0.5, weight_mastery_penalty = 0.1

- ## Results:
    ## Attempt 1:
    ![Score_plot_attempt 1 ](Images/evaluation_plot_1.png)
    ![Mastery_plot_attempt 1 ](Images/mastery_plot_1.png)

    ## Attempt 2:
    ![Score_plot_attempt 2 ](Images/evaluation_plot_2.png)
    ![Mastery_plot_attempt 2 ](Images/mastery_plot_2.png)

    ## Attempt 3:
    ![Score_plot_attempt 3 ](Images/evaluation_plot_3.png)
    ![Mastery_plot_attempt 3 ](Images/mastery_plot_3.png)

    ## Summary 
    | Metric               | Baseline        | RL Agent        |
    |----------------------|-----------------|-----------------|
    | Avg Final Score      | 2.60            | 3.20  (+23%)    |
    | Avg Topics Mastered  | 0.820           | 0.851 (+3.8%)   |

## Results for REINFORCE + Ordering constraint + strict mastery constraint + loose sufficiency constraint: 
- RL agent stores previously questioned topic. 
- Also the agent is allowed to switch to lower levels but max level is stored so that user need not re-unlock it again. 
- Also topics with uncompleted prerequisites are masked. 
- Higher levels are also masked(like difficult qns etc) untill current level(basic etc) is mastered which the agent decides. i.e one step upgrade is always given as a possible action to agent but any level above that is masked
- Input topic based policy network
- Mastery is done only when all kinds of questions with different difficulty each has score above a certain threshold.
- Separate loose criteria to see if a topic that is prerequisite of topic B is sufficient to unlock topic B. 
- New rewards one for choosing topics that are prerequisites of locked topic and another for advancing to higher difficulty  is defined.
- ## Policy network
    ```python 
    self.input_size  = num_topics * 8
    hidden_size1     = self.input_size * 2
    hidden_size2     = (self.input_size + num_actions) // 2
    
    self.fc1     = nn.Linear(self.input_size, hidden_size1)
    self.fc2     = nn.Linear(hidden_size1, hidden_size2)
    self.fc3     = nn.Linear(hidden_size2, num_actions)
    self.dropout = nn.Dropout(dropout)
    ```
-  ## Reward_weights</b> </h3>
    weight_improvement = 0.4, weight_coverage_penalty = 0.5, weight_mastery_penalty = 0.1, weight_prereq_bonus = 0.2, weight_advancement_bonus = 0.1

- ## Results:
    ## Attempt 1:
    ![Score_plot_attempt 1 ](Images/evaluation_plot_4.png)
    ![Mastery_plot_attempt 1 ](Images/mastery_plot_4.png)

    ## Summary 
    | Metric               | Baseline        | RL Agent        |
    |----------------------|-----------------|-----------------|
    | Avg Final Score      | 0.661           | 0.759           |
    | Avg Topics Mastered  | 0.24 ± 0.43     | 2.22 ± 1.14     |


