"""
RL Grid Navigation — main entry point.

Trains and evaluates a robot navigating a 2D grid world with obstacles
using Value Iteration and Q-Learning.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class GridEnvironment:
    """Discrete N x N grid world with obstacles, a start cell, and a goal cell.

    Actions: 0 = up, 1 = down, 2 = left, 3 = right.
    Rewards: +100 goal, -10 obstacle attempt, -1 per step.
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def __init__(
        self,
        grid_size: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: list[tuple[int, int]] | None = None,
        max_steps: int = 200,
    ) -> None:
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.max_steps = max_steps
        self.state: tuple[int, int] = start
        self.steps: int = 0

    def reset(self) -> tuple[int, int]:
        """Reset the agent to the start state and return it."""
        # TODO: reset state and step counter
        raise NotImplementedError

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        """Apply an action; return (next_state, reward, done)."""
        # TODO: compute next position, handle walls/obstacles, assign reward
        raise NotImplementedError

    def is_valid(self, pos: tuple[int, int]) -> bool:
        """Return True if pos is inside the grid and not an obstacle."""
        # TODO: bounds + obstacle check
        raise NotImplementedError

    def render(self) -> None:
        """Print a simple ASCII view of the grid (S=start, G=goal, #=obstacle, A=agent)."""
        # TODO: pretty-print the grid
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Value Iteration (model-based)
# ---------------------------------------------------------------------------
class ValueIteration:
    """Model-based dynamic programming using the Bellman optimality equation."""

    def __init__(self, env: GridEnvironment, gamma: float = 0.95, theta: float = 1e-4) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V: np.ndarray | None = None
        self.policy: np.ndarray | None = None

    def run(self) -> np.ndarray:
        """Iterate Bellman updates until value changes drop below theta. Return policy."""
        # TODO: iterate over states, apply Bellman update, track max delta
        raise NotImplementedError

    def extract_policy(self) -> np.ndarray:
        """Greedy policy from the converged value function."""
        # TODO: argmax over actions for each state
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Q-Learning (model-free)
# ---------------------------------------------------------------------------
class QLearningAgent:
    """Off-policy temporal difference learning with epsilon-greedy exploration."""

    def __init__(
        self,
        env: GridEnvironment,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q: np.ndarray = np.zeros((env.grid_size, env.grid_size, len(env.ACTIONS)))

    def select_action(self, state: tuple[int, int]) -> int:
        """Epsilon-greedy action selection."""
        # TODO: with prob epsilon explore, else argmax over Q
        raise NotImplementedError

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> None:
        """Apply the Q-Learning update rule."""
        # TODO: Q[s,a] += alpha * (r + gamma * max_a' Q[s', a'] - Q[s,a])
        raise NotImplementedError

    def train(self, num_episodes: int = 1000) -> list[float]:
        """Run training for num_episodes; return list of total rewards per episode."""
        # TODO: episode loop -> step loop -> update -> decay epsilon
        raise NotImplementedError

    def extract_policy(self) -> np.ndarray:
        """Greedy policy from the learned Q-table."""
        # TODO: argmax over actions for each state
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def plot_training_curve(rewards: list[float], path: str = "results/training_curve.png") -> None:
    """Plot total reward per episode and save to disk."""
    # TODO: matplotlib line plot
    raise NotImplementedError


def plot_trajectory(env: GridEnvironment, policy: np.ndarray, path: str) -> None:
    """Plot the trajectory the agent takes under the given policy."""
    # TODO: roll out policy from start, draw on grid
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Build the environment
    env = GridEnvironment(
        grid_size=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=[(1, 1), (2, 2), (3, 1)],
    )

    # 2. Run Value Iteration
    print("Running Value Iteration...")
    vi = ValueIteration(env)
    vi_policy = vi.run()
    plot_trajectory(env, vi_policy, "results/value_iteration_trajectory.png")

    # 3. Train Q-Learning
    print("Training Q-Learning agent...")
    ql = QLearningAgent(env)
    rewards = ql.train(num_episodes=1000)
    ql_policy = ql.extract_policy()
    plot_training_curve(rewards)
    plot_trajectory(env, ql_policy, "results/q_learning_trajectory.png")

    # 4. Compare
    print("Done. See results/ for plots.")


if __name__ == "__main__":
    main()
