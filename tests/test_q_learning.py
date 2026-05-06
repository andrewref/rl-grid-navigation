import pytest
import numpy as np
from src.environment import GridEnvironment
from src.q_learning import QLearningAgent


def _make_env(seed: int = 0) -> GridEnvironment:
    return GridEnvironment(grid_size=3, start=(0, 0), goal=(2, 2), obstacles=[(1, 1)], seed=seed)


def test_greedy_action_selection():
    env = GridEnvironment(grid_size=3, start=(0, 0), goal=(2, 2), obstacles=[], seed=0)
    agent = QLearningAgent(env, epsilon=0.0)
    agent.Q[0, 0] = [-1.0, 10.0, -1.0, -1.0]  # action 1 (down) is uniquely best
    assert agent.select_action((0, 0)) == 1


def test_update_non_terminal():
    env = GridEnvironment(grid_size=3, start=(0, 0), goal=(2, 2), obstacles=[], seed=0)
    agent = QLearningAgent(env, alpha=0.5, gamma=0.9, epsilon=0.0)
    # Q[0,0,1]=0; next_max=0; target = -1 + 0.9*0 = -1; new = 0 + 0.5*(-1-0) = -0.5
    agent.update((0, 0), 1, -1.0, (1, 0), False)
    assert agent.Q[0, 0, 1] == pytest.approx(-0.5)


def test_update_terminal():
    env = GridEnvironment(grid_size=3, start=(0, 0), goal=(2, 2), obstacles=[], seed=0)
    agent = QLearningAgent(env, alpha=0.5, gamma=0.9, epsilon=0.0)
    # terminal: target=100; new = 0 + 0.5*(100-0) = 50
    agent.update((2, 1), 1, 100.0, (2, 2), True)
    assert agent.Q[2, 1, 1] == pytest.approx(50.0)


def test_epsilon_decays_after_training():
    env = _make_env()
    agent = QLearningAgent(env, epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.05, seed=0)
    agent.train(num_episodes=5)
    assert agent.epsilon < 1.0


def test_training_returns_correct_length():
    env = _make_env(seed=0)
    agent = QLearningAgent(env, seed=0)
    rewards, steps = agent.train(num_episodes=100)
    assert len(rewards) == 100
    assert len(steps) == 100


def test_trained_agent_reaches_goal():
    env = _make_env(seed=42)
    agent = QLearningAgent(env, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, seed=42)
    agent.train(num_episodes=3000)

    agent.epsilon = 0.0
    state = env.reset()
    for _ in range(50):
        action = agent.select_action(state)
        state, _, done = env.step(action)
        if done:
            break
    assert state == env.goal
