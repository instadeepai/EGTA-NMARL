import pickle
from collections import OrderedDict

ROOT_DATA_FOLDER = './results/'

SEEDS = [0, 1, 2]
ALGORITHMS = ['ia2c', 'ia2c_fp', 'ia2c_ind', 'ma2c_cu', 'ma2c_ic3', 'ma2c_dial', 'ma2c_nc']
REGEN_RATES = [0.1, 0.088, 0.077, 0.065, 0.053, 0.042, 0.03]
# [0.118750, 0.1, 0.08125, 0.0625, 0.04375, 0.025, 0.00625]
N_EPISODES = 20
MAX_N_STEPS = 100
N_AGENTS = 4
DEFECTORS_COUNT = [0, 1, 2, 3, 4]
ALGO_NAMES_ORDERD = OrderedDict({'ia2c_ind': 'IA2C',
                                 'ia2c': 'NA2C',
                                 'ia2c_fp': 'FPrint',
                                 'ma2c_cu': 'ConseNet',
                                 'ma2c_dial': 'DIAL',
                                 'ma2c_ic3': 'CommNet',
                                 'ma2c_nc': 'NeurComm'})


def load_actions(folder_path, file_name):
    file_path = folder_path + file_name
    data = pickle.load(open(file_path, 'rb'))
    actions = data['actions']

    return actions


def load_rewards(folder_path, file_name):
    file_path = folder_path + file_name
    data = pickle.load(open(file_path, 'rb'))
    rewards = data['rewards']

    return rewards


def load_cooperator_ids(folder_path, file_name):
    file_path = folder_path + file_name
    data = pickle.load(open(file_path, 'rb'))
    cooperator_ids = data['cooperator_ids']

    return cooperator_ids


def load_meta_data(folder_path, file_name):
    file_path = folder_path + file_name
    data = pickle.load(open(file_path, 'rb'))

    actions = data['actions']
    algorithms_list = list(actions.keys())
    regen_rates_list = list(actions[algorithms_list[0]].keys())
    seeds_list = list(actions[algorithms_list[0]][regen_rates_list[0]].keys())

    n_episodes, max_n_steps, n_agents = actions[algorithms_list[0]][regen_rates_list[0]][seeds_list[0]].shape

    return algorithms_list, regen_rates_list, seeds_list, n_episodes, max_n_steps, n_agents
