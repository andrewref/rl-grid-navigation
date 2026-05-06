"""Visualization utilities: training curves, trajectories, value heatmaps, policy arrows."""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from src.environment import GridEnvironment


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_training_curve(
    rewards: list[float],
    path: str = "results/training_curve.png",
) -> None:
    """Plot episode rewards with a 50-episode rolling mean and save to disk."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")

    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color="steelblue",
            linewidth=2,
            label=f"{window}-episode moving avg",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning Training Curve")
    ax.legend()
    _ensure_dir(path)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(
    env: GridEnvironment,
    policy: np.ndarray,
    path: str,
    title: str = "Policy Trajectory",
) -> None:
    """Roll out policy from start and plot the path overlaid on the grid."""
    state = env.reset()
    trajectory = [state]
    for _ in range(env.max_steps):
        action = int(policy[state[0], state[1]])
        state, _, done = env.step(action)
        trajectory.append(state)
        if done:
            break

    n = env.grid_size
    grid = np.zeros((n, n))
    for obs in env.obstacles:
        grid[obs] = 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="upper")

    rows = [s[0] for s in trajectory]
    cols = [s[1] for s in trajectory]
    ax.plot(cols, rows, "b-o", markersize=5, linewidth=1.5, zorder=3, label="Path")
    ax.plot(cols[0], rows[0], "go", markersize=12, zorder=4, label="Start")
    ax.plot(cols[-1], rows[-1], "r*", markersize=14, zorder=4, label="Goal")

    for obs in env.obstacles:
        ax.text(obs[1], obs[0], "#", ha="center", va="center", color="white", fontsize=14)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_title(title)
    ax.legend()
    _ensure_dir(path)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_value_heatmap(
    V: np.ndarray,
    env: GridEnvironment,
    path: str = "results/value_heatmap.png",
) -> None:
    """2D color map of the value function; obstacle cells shown as NaN."""
    display = V.copy().astype(float)
    for obs in env.obstacles:
        display[obs] = np.nan

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(display, cmap="hot", origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="V(s)")

    n = env.grid_size
    for r in range(n):
        for c in range(n):
            pos = (r, c)
            if pos in env.obstacles:
                ax.text(c, r, "#", ha="center", va="center", color="grey", fontsize=12)
            elif pos == env.start:
                ax.text(c, r, "S", ha="center", va="center", color="cyan", fontsize=12, fontweight="bold")
            elif pos == env.goal:
                ax.text(c, r, "G", ha="center", va="center", color="cyan", fontsize=12, fontweight="bold")

    ax.set_title("Value Function (Value Iteration)")
    _ensure_dir(path)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_policy_arrows(
    policy: np.ndarray,
    env: GridEnvironment,
    path: str = "results/policy_arrows.png",
    title: str = "Learned Policy",
) -> None:
    """Draw one Unicode arrow per free cell showing the greedy action."""
    ARROW_LABELS = ["↑", "↓", "←", "→"]
    n = env.grid_size

    grid = np.zeros((n, n))
    for obs in env.obstacles:
        grid[obs] = 0.7

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="upper")

    for r in range(n):
        for c in range(n):
            pos = (r, c)
            if pos in env.obstacles:
                ax.text(c, r, "#", ha="center", va="center", color="white", fontsize=14)
            elif pos == env.goal:
                ax.text(c, r, "G", ha="center", va="center", color="green", fontsize=14, fontweight="bold")
            else:
                label = ("S\n" + ARROW_LABELS[policy[r, c]]) if pos == env.start else ARROW_LABELS[policy[r, c]]
                ax.text(c, r, label, ha="center", va="center", fontsize=16)

    ax.set_xticks(np.arange(-0.5, n, 1))
    ax.set_yticks(np.arange(-0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linewidth=0.5)
    ax.set_title(title)
    _ensure_dir(path)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
