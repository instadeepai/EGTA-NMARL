import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from qbstyles import mpl_style

from egta.experiments.evaluation_metrics import Restraint
from egta.experiments.utils import load_meta_data, load_actions, ALGO_NAMES_ORDERD, ROOT_DATA_FOLDER

LETTER_FONT_SIZE = 17


def compute_average_restraint(actions, n_episodes, n_agents):
    restraint = OrderedDict()
    for algo in actions.keys():
        restraint[algo] = OrderedDict()
        for regen in actions[algo].keys():
            collection_over_seeds = []
            for seed in actions[algo][regen].keys():
                set_of_action = actions[algo][regen][seed]
                collection_over_seeds.append(Restraint(n_agents, n_episodes, set_of_action))
            restraint[algo][regen] = np.mean(collection_over_seeds)
            print(algo, regen, restraint[algo][regen])
    return restraint


def plot_restraint_heat_map(restraint_data, ordered_algo_dictionary=None, path="./plots/", gamma=None, letter=''):
    if ordered_algo_dictionary is None:
        ordered_algo_dictionary = OrderedDict({k: k for k in restraint_data.keys()})

    list_algorithms_names = [v for v in ordered_algo_dictionary.values()]
    list_regen_rates = restraint_data[list(ordered_algo_dictionary.keys())[0]].keys()

    data = [[val for val in restraint_data[algo].values()] for algo in ordered_algo_dictionary.keys()]

    fig, ax = plt.subplots()
    cmap = sns.diverging_palette(20, 255, as_cmap=True)

    sns.heatmap(data,
                annot=True,
                cmap=cmap,
                fmt='.2f',
                xticklabels=list_regen_rates,
                yticklabels=list_algorithms_names)
    if gamma is None:
        ax.set_title('Average Restraint (%)')
    else:
        ax.set_title('Average Restraint (%), ' + r'$\alpha=$' + str(gamma))

    plt.tight_layout()
    plt.gcf().text(0.02, 0.94, letter, fontsize=LETTER_FONT_SIZE)
    file_name = path + 'restraint_heat_map.pdf'
    if gamma is not None:
        file_name = path + 'restraint_heat_map.pdf'
    plt.savefig(file_name, format="pdf")
    plt.show()


def main():
    mpl_style(dark=False)
    plt.style.use('./egta/experiments/style/style_heatmap.mplstyle')

    evaluation_data_folder_path = os.path.join(ROOT_DATA_FOLDER, 'regen_exp', 'tragedy', '')
    evaluation_data_save_file_name = 'evaluation_results_data.p'
    algorithms_list, regen_rates_list, seeds_list, n_episodes, max_n_steps, n_agents = load_meta_data(
        evaluation_data_folder_path, evaluation_data_save_file_name)
    actions = load_actions(evaluation_data_folder_path, evaluation_data_save_file_name)

    restraint_data = compute_average_restraint(actions, n_episodes, n_agents)

    plot_restraint_heat_map(restraint_data, ALGO_NAMES_ORDERD, evaluation_data_folder_path, None,
                            letter="")


if __name__ == '__main__':
    main()
