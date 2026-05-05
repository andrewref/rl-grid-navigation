"""Model-based Value Iteration using the Bellman optimality equation."""
from __future__ import annotations

import numpy as np

from src.environment import GridEnvironment


class ValueIteration:
    """Synchronous Value Iteration over the known deterministic MDP."""

    def __init__(self, env: GridEnvironment, gamma: float = 0.95, theta: float = 1e-4) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V: np.ndarray | None = None
        self.policy: np.ndarray | None = None

    def _transition(
        self, state: tuple[int, int], action: int
    ) -> tuple[tuple[int, int], float]:
        """Pure transition model: (next_state, reward) without touching env state."""
        dr, dc = self.env.ACTIONS[action]
        nr, nc = state[0] + dr, state[1] + dc
        new_pos = (nr, nc)
        if not (0 <= nr < self.env.grid_size and 0 <= nc < self.env.grid_size):
            return state, -1.0
        if new_pos in self.env.obstacles:
            return state, -10.0
        if new_pos == self.env.goal:
            return new_pos, 100.0
        return new_pos, -1.0

    def run(self) -> tuple[np.ndarray, int, list[float]]:
        """Iterate Bellman updates until convergence.

        Returns (policy, iterations, delta_history).
        """
        n = self.env.grid_size
        V = np.zeros((n, n))
        free_states = [
            (r, c)
            for r in range(n)
            for c in range(n)
            if (r, c) not in self.env.obstacles
        ]
        delta_history: list[float] = []
        iteration = 0

        while True:
            V_new = V.copy()
            V_new[self.env.goal] = 0.0  # anchor terminal state
            max_delta = 0.0

            for state in free_states:
                if state == self.env.goal:
                    continue
                old_val = V[state]
                best = float("-inf")
                for action in range(len(self.env.ACTIONS)):
                    next_s, reward = self._transition(state, action)
                    val = reward + self.gamma * V[next_s]
                    if val > best:
                        best = val
                V_new[state] = best
                max_delta = max(max_delta, abs(best - old_val))

            V = V_new
            delta_history.append(max_delta)
            iteration += 1
            if max_delta < self.theta:
                break

        self.V = V
        self.policy = self.extract_policy()
        return self.policy, iteration, delta_history

    def extract_policy(self) -> np.ndarray:
        """Greedy policy via one-step lookahead from the converged value function."""
        assert self.V is not None, "call run() before extract_policy()"
        n = self.env.grid_size
        policy = np.zeros((n, n), dtype=int)
        for r in range(n):
            for c in range(n):
                state = (r, c)
                if state in self.env.obstacles or state == self.env.goal:
                    continue
                best_action, best_val = 0, float("-inf")
                for action in range(len(self.env.ACTIONS)):
                    next_s, reward = self._transition(state, action)
                    val = reward + self.gamma * self.V[next_s]
                    if val > best_val:
                        best_val = val
                        best_action = action
                policy[r, c] = best_action
        return policy
