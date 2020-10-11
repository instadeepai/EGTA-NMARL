import numpy as np


def get_total_returns(set_of_rewards_for_eps):
    total_returns = np.nansum(set_of_rewards_for_eps, axis=0)
    return total_returns


def Utilitarian(T, number_of_episodes, set_of_rewards):
    total_for_all_eps = 0.0
    for eps in range(number_of_episodes):
        utility_for_each_episode = get_total_returns(set_of_rewards[eps]).sum() / T
        total_for_all_eps += utility_for_each_episode
    return total_for_all_eps / number_of_episodes


def Equality(N, number_of_episodes, set_of_rewards):
    total_for_all_eps = 0.0
    for eps in range(number_of_episodes):
        total = 0.0
        d = 0.0
        total_returns_each_agent = get_total_returns(set_of_rewards[eps])
        for i in range(N):
            for j in range(N):
                total += np.abs(total_returns_each_agent[i] - total_returns_each_agent[j])
            d += 2.0 * N * total_returns_each_agent[i]
        total_for_all_eps += 1.0 - total / d
    return total_for_all_eps / number_of_episodes


def Sustainability(N, number_of_episodes, set_of_rewards):
    total_for_all_eps = 0.0
    for eps in range(number_of_episodes):
        rewards_per_episode = set_of_rewards[eps]
        rewards_per_episode[np.isnan(rewards_per_episode)] = 0.0
        positive_rewards_per_episode = np.count_nonzero(rewards_per_episode > 0)
        total_for_all_eps += positive_rewards_per_episode / N
    return total_for_all_eps / number_of_episodes


# TODO: Update due to removal of tagging
def Peace(N, T, number_of_episodes, set_of_observations):
    total_for_all_eps = 0.0
    for eps in range(number_of_episodes):
        tagged_agent_steps = 0.0
        for t in range(T):
            for n in range(N):
                if set_of_observations[eps][t][n][1] > 0.0:
                    tagged_agent_steps += 1
        total_for_all_eps += (((N * T) - tagged_agent_steps) / T)
    return total_for_all_eps / number_of_episodes


# TODO: Update due to removal of tagging
def Aggression(N, T, number_of_episodes, set_of_observations):
    total_for_all_eps = 0.0
    for eps in range(number_of_episodes):
        tagged_agent_steps = 0.0
        for t in range(T):
            for n in range(N):
                if set_of_observations[eps][t][n][1] > 0.0:
                    tagged_agent_steps += 1
        total_for_all_eps += tagged_agent_steps / T
    return total_for_all_eps / number_of_episodes


def Restraint(N, number_of_episodes, set_of_actions):
    average_restraint = np.zeros((number_of_episodes, N))
    for eps in range(number_of_episodes):
        action = set_of_actions[eps]
        T = np.count_nonzero(~np.isnan(action[:, 0]))
        average_restraint[eps, :] = ((T - np.nansum(action, axis=0)) / T)
    return np.mean(average_restraint, axis=0)
