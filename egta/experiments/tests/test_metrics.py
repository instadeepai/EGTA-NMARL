import numpy as np
import pytest

from egta.experiments.evaluation_metrics import Utilitarian, Equality, Sustainability, Aggression, Peace

T = 5
dummy_set_of_rewards_two_agents = np.array([
    [[1.0, 2.0] for i in range(T)],
    [[1.0, 1.0] for i in range(T)]
])

set_of_rewards_test_data = [
    np.array([[[1.0]]]),
    [dummy_set_of_rewards_two_agents[0]],
    dummy_set_of_rewards_two_agents
]

dummy_set_of_observations_two_agents = np.array([
    [[(1.0 * (0.9 ** i), 0.0), (1.0 * (0.9 ** i), 0.0)] for i in range(T)],
    [[(1.0 * (0.9 ** i), 0.5), (1.0 * (0.9 ** i), 0.0)] for i in range(T)],
])

set_of_observations_test_data = [
    np.array([[[(1.0, 0.0)]]]),
    np.array([[[(0.8, 1.0)]]]),
    [dummy_set_of_observations_two_agents[0]],
    [dummy_set_of_observations_two_agents[1]],
    dummy_set_of_observations_two_agents
]

utilitarian_expected_results = [
    1.0,
    3.0,
    2.5,
]


@pytest.mark.parametrize("set_of_rewards, expected_result",
                         list(zip(set_of_rewards_test_data, utilitarian_expected_results)))
def test_utilitarian(set_of_rewards, expected_result):
    T = len(set_of_rewards[0])
    number_of_episodes = len(set_of_rewards)
    result = Utilitarian(T, number_of_episodes, set_of_rewards)
    assert result == expected_result


equality_expected_results = [
    1.0,
    5.0 / 6.0,
    11.0 / 12.0
]


@pytest.mark.parametrize("set_of_rewards, expected_result",
                         list(zip(set_of_rewards_test_data, equality_expected_results)))
def test_equality(set_of_rewards, expected_result):
    N = len(set_of_rewards[0][0])
    number_of_episodes = len(set_of_rewards)
    result = Equality(N, number_of_episodes, set_of_rewards)
    assert result == pytest.approx(expected_result)


sustainability_test_data = [
    1.0,
    5.0,
    5.0
]


@pytest.mark.parametrize("set_of_rewards, expected_result",
                         list(zip(set_of_rewards_test_data, sustainability_test_data)))
def test_sustainability(set_of_rewards, expected_result):
    N = len(set_of_rewards[0][0])
    number_of_episodes = len(set_of_rewards)
    result = Sustainability(N, number_of_episodes, set_of_rewards)
    assert result == pytest.approx(expected_result)


aggression_test_data = [
    0.0,
    1.0,
    0.0,
    1.0,
    0.5
]


@pytest.mark.parametrize("set_of_observations, expected_result",
                         list(zip(set_of_observations_test_data, aggression_test_data)))
def test_aggression(set_of_observations, expected_result):
    number_of_episodes = len(set_of_observations)
    T = len(set_of_observations[0])
    N = len(set_of_observations[0][0])
    result = Aggression(N, T, number_of_episodes, set_of_observations)
    assert result == pytest.approx(expected_result)


peace_test_data = [
    1.0,
    0.0,
    2.0,
    1.0,
    1.5
]


@pytest.mark.parametrize("set_of_observations, expected_result",
                         list(zip(set_of_observations_test_data, peace_test_data)))
def test_peace(set_of_observations, expected_result):
    number_of_episodes = len(set_of_observations)
    T = len(set_of_observations[0])
    N = len(set_of_observations[0][0])
    result = Peace(N, T, number_of_episodes, set_of_observations)
    assert result == pytest.approx(expected_result)
