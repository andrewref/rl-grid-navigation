"""Orchestrate and compare Value Iteration against Q-Learning."""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.environment import GridEnvironment
from src.value_iteration import ValueIteration
from src.q_learning import QLearningAgent
from src.visualize import (
    plot_training_curve,
    plot_trajectory,
    plot_value_heatmap,
    plot_policy_arrows,
)

_DEFAULT_OBSTACLES = [(1, 1), (2, 2), (3, 1)]


def _greedy_rollout(env: GridEnvironment, policy: np.ndarray) -> tuple[float, int]:
    """Run one deterministic episode under policy; return (total_reward, steps)."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    for _ in range(env.max_steps):
        action = int(policy[state[0], state[1]])
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    return total_reward, steps


def run_comparison(grid_size: int = 5, num_episodes: int = 1000, seed: int = 42) -> None:
    """Run both algorithms on the same environment and save all plots."""
    obstacles = _DEFAULT_OBSTACLES if grid_size == 5 else []
    env_kwargs: dict = dict(
        grid_size=grid_size,
        start=(0, 0),
        goal=(grid_size - 1, grid_size - 1),
        obstacles=obstacles,
    )

    # --- Value Iteration ---
    print("Running Value Iteration...")
    vi_env = GridEnvironment(**env_kwargs, seed=seed)
    vi = ValueIteration(vi_env)
    vi_policy, vi_iters, _ = vi.run()
    vi_reward, vi_steps = _greedy_rollout(GridEnvironment(**env_kwargs, seed=seed), vi_policy)

    # --- Q-Learning ---
    print("Training Q-Learning agent...")
    ql_env = GridEnvironment(**env_kwargs, seed=seed)
    agent = QLearningAgent(ql_env, seed=seed)
    rewards, _ = agent.train(num_episodes)
    ql_policy = agent.extract_policy()
    ql_reward, ql_steps = _greedy_rollout(GridEnvironment(**env_kwargs, seed=seed), ql_policy)

    # --- Summary ---
    print("\n=== Results ===")
    print(
        f"Value Iteration : converged in {vi_iters:4d} iterations | "
        f"greedy steps: {vi_steps:3d} | reward: {vi_reward:7.1f}"
    )
    print(
        f"Q-Learning      : {num_episodes} episodes trained    | "
        f"greedy steps: {ql_steps:3d} | reward: {ql_reward:7.1f} | "
        f"avg last-50 ep: {np.mean(rewards[-50:]):.2f}"
    )

    # --- Plots ---
    os.makedirs("results", exist_ok=True)
    plot_training_curve(rewards, "results/ql_training_curve.png")
    plot_trajectory(
        GridEnvironment(**env_kwargs, seed=seed), vi_policy,
        "results/vi_trajectory.png", "Value Iteration Trajectory",
    )
    plot_trajectory(
        GridEnvironment(**env_kwargs, seed=seed), ql_policy,
        "results/ql_trajectory.png", "Q-Learning Trajectory",
    )
    plot_value_heatmap(vi.V, vi_env, "results/vi_value_heatmap.png")
    plot_policy_arrows(vi_policy, vi_env, "results/vi_policy_arrows.png", "Value Iteration Policy")
    plot_policy_arrows(ql_policy, ql_env, "results/ql_policy_arrows.png", "Q-Learning Policy")
    print("Plots saved to results/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VI and Q-Learning on grid navigation")
    parser.add_argument("--grid-size", type=int, default=5, help="grid side length (default 5)")
    parser.add_argument("--episodes", type=int, default=1000, help="Q-Learning training episodes")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    run_comparison(args.grid_size, args.episodes, args.seed)


if __name__ == "__main__":
    main()
