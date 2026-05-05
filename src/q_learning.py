"""Model-free Q-Learning with epsilon-greedy exploration."""
from __future__ import annotations

import numpy as np

from src.environment import GridEnvironment


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
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((env.grid_size, env.grid_size, len(env.ACTIONS)))

    def select_action(self, state: tuple[int, int]) -> int:
        """Epsilon-greedy action selection; random tie-breaking on greedy ties."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.env.ACTIONS)))
        q = self.Q[state[0], state[1]]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> None:
        """Q-Learning TD update; no bootstrap on terminal transitions."""
        current = self.Q[state[0], state[1], action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state[0], next_state[1]].max()
        self.Q[state[0], state[1], action] += self.alpha * (target - current)

    def train(self, num_episodes: int = 1000) -> tuple[list[float], list[int]]:
        """Train for num_episodes; return (rewards_per_episode, steps_per_episode)."""
        rewards: list[float] = []
        steps_list: list[int] = []

        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            while True:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards.append(total_reward)
            steps_list.append(steps)

        return rewards, steps_list

    def extract_policy(self) -> np.ndarray:
        """Greedy policy from the learned Q-table with random tie-breaking."""
        n = self.env.grid_size
        policy = np.zeros((n, n), dtype=int)
        for r in range(n):
            for c in range(n):
                q = self.Q[r, c]
                policy[r, c] = int(self.rng.choice(np.flatnonzero(q == q.max())))
        return policy
