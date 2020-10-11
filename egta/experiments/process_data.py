import argparse
import os
import pickle

import numpy as np

from egta.experiments.utils import ALGORITHMS, REGEN_RATES, SEEDS, N_EPISODES, MAX_N_STEPS, N_AGENTS, \
    DEFECTORS_COUNT, ROOT_DATA_FOLDER


# file structure for each type (actions, rewards, observations)
# { 'algo': {regen_rate: {'seed:Array of shape (N_EPISODES,MAX_N_STEPS, N_AGENTS)}} }

def process_evaluation_data(folder_path, save_file_name):
    base_shape = (N_EPISODES, MAX_N_STEPS, N_AGENTS)
    obs_shape = (*base_shape, 2)

    observations = {}
    rewards = {}
    actions = {}

    for algo in ALGORITHMS:
        algo_path = algo
        observations[algo] = {}
        rewards[algo] = {}
        actions[algo] = {}

        for regen_rate in REGEN_RATES:
            regan_path = "regen_rate=" + "{:.6f}".format(regen_rate)
            observations[algo][regen_rate] = {}
            rewards[algo][regen_rate] = {}
            actions[algo][regen_rate] = {}

            for seed in SEEDS:
                seed_path = "seed=" + str(seed)
                full_path = os.path.join(folder_path, seed_path, regan_path, algo_path, 'eva_data', '')
                rewards[algo][regen_rate][seed] = np.zeros((base_shape))
                rewards[algo][regen_rate][seed][:] = np.nan
                actions[algo][regen_rate][seed] = np.zeros((base_shape))
                actions[algo][regen_rate][seed][:] = np.nan

                for eps in range(N_EPISODES):
                    file_name = full_path + "episode%s.npz" % (eps)
                    data = np.load(file_name, allow_pickle=True)
                    print(algo, regen_rate, seed, eps)
                    # ACTIONS
                    action_steps = min(MAX_N_STEPS, len(data['actions']))
                    actions[algo][regen_rate][seed][eps, :action_steps, :] = data['actions'][:action_steps, :]
                    # REWARDS
                    rewards_steps = min(MAX_N_STEPS, len(data['rewards']))
                    rewards_shape = data['rewards'].shape
                    if len(rewards_shape) == 3:
                        rewards[algo][regen_rate][seed][eps, :rewards_steps, :] = data['rewards'][:rewards_steps, 0, :]
                        print(algo, regen_rate, seed, data['rewards'].shape)
                    else:
                        rewards[algo][regen_rate][seed][eps, :rewards_steps, :] = data['rewards'][:rewards_steps, :]

    print('Saving to file')
    data = {'observations': observations, 'rewards': rewards, 'actions': actions}
    save__path_file_name = os.path.join(folder_path, save_file_name + ".p")
    pickle.dump(data, open(save__path_file_name, "wb"))
    print('Done!')


# { 'algo': {regen_rate: {'number_of_defectors:Array of shape (N_EPISODES,MAX_N_STEPS, N_AGENTS)}} }
def process_interaction_data(folder_path, save_file_name):
    base_shape = (N_EPISODES, MAX_N_STEPS, N_AGENTS)
    coop_ids_shape = (N_EPISODES, N_AGENTS)
    obs_shape = (*base_shape, 2)

    observations = {}
    rewards = {}
    actions = {}
    cooperator_ids = {}

    for algo in ALGORITHMS:
        observations[algo] = {}
        rewards[algo] = {}
        actions[algo] = {}
        cooperator_ids[algo] = {}
        algo_path = algo
        for number_of_defectors in DEFECTORS_COUNT:
            defectors_path = "defectors=" + str(number_of_defectors)

            rewards[algo][number_of_defectors] = np.zeros(base_shape)
            rewards[algo][number_of_defectors][:] = np.nan
            actions[algo][number_of_defectors] = np.zeros(base_shape)
            actions[algo][number_of_defectors][:] = np.nan
            cooperator_ids[algo][number_of_defectors] = []

            full_path = os.path.join(folder_path, algo_path, defectors_path, '')
            if os.path.exists(full_path):

                for eps in range(N_EPISODES):
                    file_name = "run_%s_episode0.npz" % (eps)
                    data = np.load(full_path + file_name, allow_pickle=True)
                    print(algo, number_of_defectors, eps)
                    # ACTIONS
                    action_steps = min(MAX_N_STEPS, len(data['actions']))
                    actions[algo][number_of_defectors][eps, :action_steps, :] = data['actions'][:action_steps, :]
                    # REWARDS
                    rewards_steps = min(MAX_N_STEPS, len(data['rewards']))
                    rewards[algo][number_of_defectors][eps, :rewards_steps, :] = data['rewards'][:rewards_steps, :]
                    # Coop ids
                    cooperator_ids[algo][number_of_defectors].append(data['coop_ids'])

    print('Saving to file')
    data = {'observations': observations,
            'rewards': rewards,
            'actions': actions,
            'cooperator_ids': cooperator_ids}
    pickle.dump(data, open(folder_path + save_file_name + ".p", "wb"))
    print('Done!')


def run_process_evaluation_data(results_data_folder_path):
    evaluation_data_folder_path = results_data_folder_path + '/tragedy/'

    evaluation_data_save_file_name = 'evaluation_results_data'

    process_evaluation_data(evaluation_data_folder_path, evaluation_data_save_file_name)


def run_process_interaction_data(results_data_folder_path):
    interactions_save_file_name = 'interaction_results_data'

    process_interaction_data(results_data_folder_path, interactions_save_file_name)


def main(data_type='evaluation'):
    results_data_folder_path = ROOT_DATA_FOLDER + "/regen_exp"
    if data_type == 'evaluation':
        print('Processing evaluation data')
        run_process_evaluation_data(results_data_folder_path)
    elif data_type == 'interaction':
        print('Processing interaction data')
        run_process_interaction_data(results_data_folder_path)


if __name__ == '__main__':
    # Parsing in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='evaluation',
                        help='Date type to be processed (evaluation or interaction)')
    config = parser.parse_args()
    print(config.data_type)
    main(config.data_type)
