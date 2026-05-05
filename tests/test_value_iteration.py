import pytest
from src.environment import GridEnvironment
from src.value_iteration import ValueIteration


@pytest.fixture
def small_env():
    """3×3 grid, start (0,0), goal (2,2), obstacle at (1,1)."""
    return GridEnvironment(grid_size=3, start=(0, 0), goal=(2, 2), obstacles=[(1, 1)], seed=0)


def test_goal_value_is_zero(small_env):
    vi = ValueIteration(small_env)
    vi.run()
    assert vi.V[2, 2] == pytest.approx(0.0)


def test_adjacent_cell_has_higher_value_than_distant(small_env):
    vi = ValueIteration(small_env)
    vi.run()
    # Cell (2,1) is one step from goal; (0,0) is far — its value must be higher (less negative)
    assert vi.V[2, 1] > vi.V[0, 0]


def test_policy_reaches_goal(small_env):
    vi = ValueIteration(small_env)
    policy, _, _ = vi.run()
    state = small_env.reset()
    for _ in range(20):
        action = policy[state[0], state[1]]
        state, _, done = small_env.step(action)
        if done:
            break
    assert state == small_env.goal


def test_convergence_within_bound(small_env):
    vi = ValueIteration(small_env)
    _, iters, _ = vi.run()
    assert iters < 500


def test_extract_policy_requires_run():
    env = GridEnvironment(grid_size=3, start=(0, 0), goal=(2, 2), obstacles=[])
    vi = ValueIteration(env)
    with pytest.raises(AssertionError):
        vi.extract_policy()
