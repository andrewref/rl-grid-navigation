# RL Grid Navigation

Obstacle-avoiding robot navigation in a 2D grid world using tabular reinforcement learning.

A mobile robot learns to navigate from a start cell to a goal cell while avoiding obstacles, using **Value Iteration** and **Q-Learning**. The environment is modeled as a discrete Markov Decision Process (MDP) and implemented from scratch in Python (no external RL libraries).

## Algorithms

- **Value Iteration** — model-based dynamic programming using the Bellman optimality equation.
- **Q-Learning** — model-free, off-policy temporal difference learning with ε-greedy exploration.

## Environment

- **State space:** each free cell on an `N x N` grid.
- **Action space:** up, down, left, right.
- **Rewards:** `+100` for reaching the goal, `-10` for hitting an obstacle, `-1` per step.
- **Transitions:** deterministic; invalid moves (into walls or obstacles) leave the agent in place.

## Project Structure

```
rl-grid-navigation/
├── src/                 # source code (environment, agents, training loops)
├── results/             # training curves, trajectory plots, saved policies
├── notebooks/           # exploratory analysis and visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/rl-grid-navigation.git
cd rl-grid-navigation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run Value Iteration
python src/value_iteration.py

# Train Q-Learning agent
python src/q_learning.py

# Compare both algorithms
python src/compare.py
```

## Evaluation

Both algorithms are compared on:

- Convergence speed (iterations / episodes to convergence)
- Quality of the learned policy (does it find the shortest path?)
- Trajectory from start to goal under the learned policy
- Episode reward and length curves (Q-Learning)

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
