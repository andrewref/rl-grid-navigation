# RL Grid Navigation

Obstacle-avoiding robot navigation in a 2D grid world using tabular reinforcement learning.

A mobile robot learns to navigate from a start cell to a goal cell while avoiding obstacles, using **Value Iteration** and **Q-Learning**. The environment is modelled as a discrete Markov Decision Process (MDP) and implemented from scratch in Python (no external RL libraries).

## Algorithms

- **Value Iteration** — model-based dynamic programming using the Bellman optimality equation.
- **Q-Learning** — model-free, off-policy temporal difference learning with ε-greedy exploration.

## Environment

| Property | Value |
|---|---|
| State space | every free cell on an `N × N` grid |
| Action space | up, down, left, right |
| Goal reward | `+100` |
| Obstacle penalty | `−10` (replaces the step penalty) |
| Step penalty | `−1` per transition |
| Transitions | deterministic; invalid moves leave the agent in place |

## Project Structure

```
rl-grid-navigation/
├── src/
│   ├── __init__.py
│   ├── environment.py      # GridEnvironment (MDP)
│   ├── value_iteration.py  # ValueIteration agent
│   ├── q_learning.py       # QLearningAgent
│   ├── visualize.py        # plotting helpers
│   └── compare.py          # orchestration script
├── tests/
│   ├── test_environment.py
│   ├── test_value_iteration.py
│   └── test_q_learning.py
├── results/                # generated plots (gitignored except .gitkeep)
├── main.py                 # thin entry point
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/andrewref/rl-grid-navigation.git
cd rl-grid-navigation

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
# Run both algorithms and save plots to results/
py main.py

# Custom grid size, episode count, and random seed
py main.py --grid-size 7 --episodes 2000 --seed 0

# Run only the comparison module directly
py -m src.compare --episodes 1000 --seed 42

# Run tests
py -m pytest
```

## Evaluation Metrics

Both algorithms are compared on:

- **Convergence speed** — iterations (VI) or episodes (Q-Learning) to convergence
- **Policy quality** — does the greedy policy find the shortest path?
- **Greedy trajectory** — path from start to goal under the final policy
- **Training curve** — episode reward and smoothed moving average (Q-Learning)

## Results (5 × 5 grid, default obstacles)

| Algorithm | Convergence | Greedy steps | Greedy reward |
|---|---|---|---|
| Value Iteration | 9 iterations | 8 | 93.0 |
| Q-Learning | 1 000 episodes | 8 | 93.0 |

Both algorithms recover the same 8-step optimal path.
Plots are saved to `results/` after each run.

## Team

| Name | ID |
|---|---|
| Youssef Medhat Asly | 55-0482 |
| Andrew Refaat | 55-11692 |
| Omar Ezzat | 55-4315 |
| Omar Sherif Haridy | 55-2010 |
| Mohamad Hamam | 55-10030 |

## References

1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
2. S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Pearson, 2020.
3. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, 2015.
