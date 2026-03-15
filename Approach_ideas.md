# Adaptive Questioning Agent — RL Component Design Summary

## `knowledge_state.py`

### Core Design Decisions
- `topics_difficulty` as a single dict instead of separate `topics` + `difficulty` — cleaner
- `self.topics = list(topics_difficulty.keys())` — needed for consistent iteration order
- `max_attempts` as a configurable parameter for normalization

### `is_mastered()`
- Difficulty-based thresholds instead of fixed:
  ```python
  {'basic': 0.90, 'intermediate': 0.85, 'advanced': 0.75}
  ```
- Three conditions — min attempts + score threshold + trend >= 0
- **Future work** — second RL model updates difficulty per user (bi-level optimization)

### `get_state_vector()`
- 7 features per topic → fixed size = `7 × num_topics`

| Feature | Details |
|---|---|
| `avg_score` | rolling average score `[0, 1]` |
| `attempts_norm` | `min(attempts / max_attempts, 1.0)` — stable, clipped |
| `trend` | slope over recent window `[-1, 1]` |
| `is_mastered` | binary float `{0.0, 1.0}` |
| `is_neglected` | binary float `{0.0, 1.0}` |
| `last_difficulty` | ordinal encoded `{0.0, 0.2, 0.6, 1.0}` |
| `last_question_type` | ordinal encoded `{0.0, 0.2, 0.6, 1.0}` |

- Attempts normalized by fixed `max_attempts` with clip — stable across steps
- Ordinal encoding using `0.0, 0.2, 0.6, 1.0` — preserves order and `None = 0.0` is distinguishable from real values
- Returns `np.float32` array for PyTorch compatibility

---

## `mdp.py`

### Action Space
- All `(topic, difficulty, question_type)` combinations enumerated in fixed order
- Two dicts for O(1) lookup both ways:
  ```python
  int_to_actions = {0: ("topic_1", "basic", "factual"), ...}
  actions_to_int = {("topic_1", "basic", "factual"): 0, ...}
  ```
- Total actions = `num_topics × 3 × 3`

### Reward Function (inside MDP class)
- `prev_score` passed as argument — read from `ks` **before** calling `update()`
- `score` is the only external input — comes from NLP team

| Component | Formula | Range |
|---|---|---|
| Improvement | `score - prev_score` | `[-1, 1]` |
| Coverage bonus | `1.0` if neglected, `0.5` if trend < 0, `0.0` otherwise | `[0, 1]` |
| Mastery penalty | `1.0` if mastered, `0.0` otherwise | `{0, 1}` |

```python
reward = w1 * improvement + w2 * coverage_bonus - w3 * mastery_penalty
```
- Weights `w1, w2, w3` configurable in `__init__` (defaults: `0.6, 0.3, 0.1`)

### Why Reward Function is Inside MDP (not separate class)
- Action space and reward are both core MDP components
- Keeps everything in one place — cleaner and reflects the theory better

---

## Pretraining Strategy

### The Problem
- Topics change per document upload — can't pretrain on fixed topics
- Different students upload different documents
- Variable number of topics breaks fixed MLP input size

### Solution — Session-Level Pretraining
```
User uploads documents
        ↓
NLP team extracts topics + difficulties
        ↓
Simulate 500-1000 sessions with actual extracted topics  ← pretrain here
        ↓
Real session begins with informed weights
        ↓
Agent continues adapting online during real session
```

- No `MAX_TOPICS` padding needed — train fresh per document upload
- Simulating 500-1000 sessions takes only a few seconds on CPU
- Agent learns topic-agnostic strategies (neglect handling, mastery detection)
- Specific to this document's topic count and difficulty distribution

### Why This Works Across Different Students
- MLP never sees topic **names** — only numeric features
- State vector structure is identical regardless of topic content
- Pretrained weights encode **strategies**, not content knowledge

### Future Work — Bi-level Optimization
- Inner loop — current RL agent (question by question)
- Outer loop — second RL model updates topic difficulty per user (session by session)
- Moves from document-centric difficulty to learner-relative difficulty