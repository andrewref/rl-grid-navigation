import pytest
from src.environment import GridEnvironment


@pytest.fixture
def env():
    return GridEnvironment(
        grid_size=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=[(1, 1), (2, 2)],
        seed=0,
    )


def test_reset_returns_start(env):
    assert env.reset() == (0, 0)


def test_reset_clears_steps(env):
    env.reset()
    env.step(1)
    env.reset()
    assert env.steps == 0


def test_step_valid_move(env):
    env.reset()
    state, reward, done = env.step(1)  # down: (0,0) -> (1,0)
    assert state == (1, 0)
    assert reward == pytest.approx(-1.0)
    assert not done


def test_step_into_obstacle(env):
    env.reset()
    env.step(1)  # down to (1,0)
    state, reward, done = env.step(3)  # right into (1,1) obstacle
    assert state == (1, 0)
    assert reward == pytest.approx(-10.0)
    assert not done


def test_step_out_of_bounds(env):
    env.reset()
    state, reward, done = env.step(0)  # up from (0,0) — out of bounds
    assert state == (0, 0)
    assert reward == pytest.approx(-1.0)
    assert not done


def test_step_reaches_goal():
    env = GridEnvironment(grid_size=2, start=(0, 0), goal=(0, 1), obstacles=[], seed=0)
    env.reset()
    state, reward, done = env.step(3)  # right: (0,0) -> (0,1) = goal
    assert state == (0, 1)
    assert reward == pytest.approx(100.0)
    assert done


def test_max_steps_terminates():
    env = GridEnvironment(grid_size=5, start=(0, 0), goal=(4, 4), obstacles=[], max_steps=3, seed=0)
    env.reset()
    for i in range(3):
        _, _, done = env.step(0)  # keep hitting top wall
        assert done == (i == 2)


def test_init_rejects_start_on_obstacle():
    with pytest.raises(AssertionError):
        GridEnvironment(grid_size=5, start=(1, 1), goal=(4, 4), obstacles=[(1, 1)])
