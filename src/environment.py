"""Discrete N×N grid world MDP for robot navigation."""
from __future__ import annotations

import numpy as np


class GridEnvironment:
    """Discrete N×N grid world with obstacles, a start cell, and a goal cell.

    Actions: 0=up, 1=down, 2=left, 3=right.
    Rewards: +100 goal, -10 obstacle attempt (replaces step penalty), -1 per step.
    Obstacle penalty replaces the step penalty (agent receives -10, not -11).
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def __init__(
        self,
        grid_size: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: list[tuple[int, int]] | None = None,
        max_steps: int = 200,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        assert 0 <= start[0] < grid_size and 0 <= start[1] < grid_size, "start out of bounds"
        assert 0 <= goal[0] < grid_size and 0 <= goal[1] < grid_size, "goal out of bounds"
        assert start not in self.obstacles, "start cell is an obstacle"
        assert goal not in self.obstacles, "goal cell is an obstacle"

        self.state: tuple[int, int] = start
        self.steps: int = 0

    def reset(self) -> tuple[int, int]:
        """Reset agent to start; return start state."""
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        """Apply action; return (next_state, reward, done)."""
        dr, dc = self.ACTIONS[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        new_pos = (nr, nc)

        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            reward = -1.0  # wall: stay, step penalty only
        elif new_pos in self.obstacles:
            reward = -10.0  # obstacle: stay, obstacle penalty replaces step penalty
        elif new_pos == self.goal:
            self.state = new_pos
            self.steps += 1
            return self.state, 100.0, True
        else:
            self.state = new_pos
            reward = -1.0

        self.steps += 1
        done = self.steps >= self.max_steps
        return self.state, reward, done

    def is_valid(self, pos: tuple[int, int]) -> bool:
        """True if pos is inside the grid and not an obstacle."""
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size and pos not in self.obstacles

    def render(self) -> None:
        """Print ASCII grid: S=start, G=goal, #=obstacle, A=agent, .=free."""
        for r in range(self.grid_size):
            row = ""
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == self.state:
                    row += "A "
                elif pos == self.goal:
                    row += "G "
                elif pos == self.start:
                    row += "S "
                elif pos in self.obstacles:
                    row += "# "
                else:
                    row += ". "
            print(row)
        print()
