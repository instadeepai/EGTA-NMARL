import math
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from qbstyles import mpl_style

from egta.experiments.utils import load_rewards, load_cooperator_ids, ALGO_NAMES_ORDERD, ROOT_DATA_FOLDER

# Global Variables
DEFECTOR_COLOUR = '#E31A1C'
COOPERATORE_COLOUR = '#33A02C'  # '#009900'
ALL_AGENTS_COLOUR = '#1F78B4'  # '#002db3'
CONDITION_1_COLOUR = '#00cc99'
CONDITION_2_COLOUR = 'orange'
CONDITION_3_COLOUR = '#cc00cc'
EQUILIBRIUM_AREA = '#a6cee3'
EQUILIBRIUM_AREA_GROUP = '#eda65e'
LETTER_FONT_SIZE = 17


def transform_data(rewards, cooperators_ids):
    average_data = {}
    for algo in rewards.keys():
        number_of_modes = 5
        average_data[algo] = {
            'average_eps_len': np.zeros(number_of_modes),
            'all': {
                'mean': np.zeros(number_of_modes),
                'std': np.zeros(number_of_modes)
            },
            'defectors': {
                'mean': np.zeros(number_of_modes),
                'std': np.zeros(number_of_modes)
            },
            'cooperators': {
                'mean': np.zeros(number_of_modes),
                'std': np.zeros(number_of_modes)
            }
        }
        average_data[algo]['all']['mean'][:] = np.nan
        average_data[algo]['all']['std'][:] = np.nan
        average_data[algo]['defectors']['mean'][:] = np.nan
        average_data[algo]['defectors']['std'][:] = np.nan
        average_data[algo]['cooperators']['mean'][:] = np.nan
        average_data[algo]['cooperators']['std'][:] = np.nan

        for number_of_defectors in rewards[algo].keys():
            number_episodes = 20
            all_agents = np.zeros((number_episodes, 4))
            all_agents[:] = np.nan
            cooperator_agents = np.zeros((number_episodes, 4 - number_of_defectors))
            cooperator_agents[:] = np.nan
            defector_agents = np.zeros((number_episodes, number_of_defectors))
            defector_agents[:] = np.nan
            episode_len = np.zeros((number_episodes, 1))
            if len(cooperators_ids[algo][number_of_defectors]) == number_episodes:
                for eps in range(number_episodes):
                    coop_ids = cooperators_ids[algo][number_of_defectors][eps]
                    episode_rewards = rewards[algo][number_of_defectors][eps]
                    episode_len[eps, :] = np.sum(~np.isnan(episode_rewards)) / 4
                    # get sum of rewards
                    sum_of_rewards = np.nansum(episode_rewards, axis=0)

                    c = []
                    d = []
                    # group cooperators and defectors
                    for i, reward in enumerate(sum_of_rewards):
                        if i in coop_ids:
                            c.append(reward)
                        else:
                            d.append(reward)
                    all_agents[eps, :] = sum_of_rewards
                    cooperator_agents[eps, :] = c
                    defector_agents[eps, :] = d
            else:
                print(algo, number_of_defectors)
            average_data[algo]['average_eps_len'][number_of_defectors] = np.mean(episode_len)
            average_data[algo]['all']['mean'][number_of_defectors] = np.mean(np.mean(all_agents, axis=1))
            average_data[algo]['all']['std'][number_of_defectors] = np.std(np.mean(all_agents, axis=1))
            if number_of_defectors < 4:
                average_data[algo]['cooperators']['mean'][number_of_defectors] = np.mean(
                    np.mean(cooperator_agents, axis=1))
                average_data[algo]['cooperators']['std'][number_of_defectors] = np.std(
                    np.mean(cooperator_agents, axis=1))
            if number_of_defectors > 0:
                average_data[algo]['defectors']['mean'][number_of_defectors] = np.mean(np.mean(defector_agents, axis=1))
                average_data[algo]['defectors']['std'][number_of_defectors] = np.std(np.mean(defector_agents, axis=1))

        # reverse data as we want to order from coop 0 to 4.
        average_data[algo]['average_eps_len'] = average_data[algo]['average_eps_len'][::-1]
        average_data[algo]['all']['mean'] = average_data[algo]['all']['mean'][::-1]
        average_data[algo]['all']['std'] = average_data[algo]['all']['std'][::-1]
        average_data[algo]['defectors']['mean'] = average_data[algo]['defectors']['mean'][::-1]
        average_data[algo]['defectors']['std'] = average_data[algo]['defectors']['std'][::-1]
        average_data[algo]['cooperators']['mean'] = average_data[algo]['cooperators']['mean'][::-1]
        average_data[algo]['cooperators']['std'] = average_data[algo]['cooperators']['std'][::-1]
    return average_data


def create_nsi_indicator(x, y, ax, title, xlim=None):
    # Plot the bar chart
    bar_list = ax.barh(x, y, align='edge')

    # Middle bar
    ax.plot(np.zeros(2), [0, 3], lw=2, color='black')

    # Set the bars the appropriate colour
    bar_colours = [CONDITION_1_COLOUR, CONDITION_2_COLOUR, CONDITION_3_COLOUR]
    for i in range(len(x)):
        bar_list[i].set_color(bar_colours[i])

    ax.set_yticks(x)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([0])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

    if xlim is not None: ax.set_xlim(left=-xlim, right=xlim)


def calculate_condition_3(defectors, cooperators):
    condition_3_max = defectors[0] - cooperators[0]
    for i in range(1, len(defectors)):
        R_d_i, R_c_i = defectors[i], cooperators[i]
        if R_d_i > R_c_i:
            condition_3_max = R_d_i - R_c_i

    return condition_3_max


def create_small_schelling(defectors, defectors_error_bars,
                           cooperators, cooperators_error_bars,
                           all_agents, all_agents_error_bars, algo, ax, ylim):
    ax.set_ylim(top=ylim)
    x = np.arange(len(all_agents))

    ax.errorbar(x[:-1], defectors, yerr=defectors_error_bars, fmt=DEFECTOR_COLOUR)
    ax.errorbar(x[1:], cooperators, yerr=cooperators_error_bars, fmt=COOPERATORE_COLOUR)
    ax.errorbar(x, all_agents, all_agents_error_bars, fmt=ALL_AGENTS_COLOUR)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax.set_xlabel(algo)


def plot_nsi_schelling(defectors, defectors_error_bars,
                       cooperators, cooperators_error_bars,
                       all_agents, all_agents_error_bars, equilibrium_points, title, algo, xlim, folder_path,
                       letter=None):
    # plt.rcdefaults()
    fig, ax = plt.subplots()

    # ax.set_ylim(top=ylim)

    x = np.arange(len(all_agents))

    ax.errorbar(x[:-1], defectors, yerr=defectors_error_bars, fmt=DEFECTOR_COLOUR, label='Defectors')
    ax.errorbar(x[1:], cooperators, yerr=cooperators_error_bars, fmt=COOPERATORE_COLOUR, label='Cooperators')
    ax.errorbar(x, all_agents, all_agents_error_bars, fmt=ALL_AGENTS_COLOUR, label='All agents')

    if equilibrium_points is not None:
        for i, point in enumerate(equilibrium_points['point']):
            width, height = equilibrium_points['ellipse'][i]
            equilibrium = mpatches.Ellipse(xy=point, width=width, height=height, angle=0, alpha=0.6,
                                           color=EQUILIBRIUM_AREA)
            ax.add_artist(equilibrium)
            # Add the second group equilibrium point
            if 'point_2' in equilibrium_points:
                for i, point in enumerate(equilibrium_points['point_2']):
                    width, height = equilibrium_points['ellipse_2'][i]
                    equilibrium = mpatches.Ellipse(xy=point, width=width, height=height, angle=0, alpha=0.6,
                                                   color=EQUILIBRIUM_AREA_GROUP)
                    ax.add_artist(equilibrium)

    # Condition 1: R_c(N) > R_d(0)
    R_c_N, R_d_0 = cooperators[-1], defectors[0]
    condition_1 = R_c_N - R_d_0

    # Condition 2: R_c(N) > R_c(0)
    R_c_0 = cooperators[1]
    condition_2 = R_c_N - R_c_0

    # Condition 3: R_d(i) > R_c(i)
    condition_3 = calculate_condition_3(defectors, cooperators)

    # Create SSD indicator inset subplot
    left, bottom, width, height = 0.15, .6, .2, .2
    nsi_ax = fig.add_axes([left, bottom, width, height])
    x = np.arange(3)
    y = [condition_1, condition_2, condition_3]
    y_max_local = np.max(y)
    create_nsi_indicator(x, y, nsi_ax, 'NSI Indicator', xlim)

    # Convert x-axis labels to intergers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if letter is not None:
        plt.gcf().text(0.02, 0.94, letter, fontsize=LETTER_FONT_SIZE)

    ax.set_xlabel('Number of Cooperators')
    ax.set_ylabel('Individual Payoff')
    ax.set_title(title)
    ax.legend(loc='lower right')

    plot_fold = folder_path + 'plots/'
    Path(plot_fold).mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_fold + 'ssd_schelling_' + algo + '.pdf', format='pdf', dpi=2000, bbox_inches='tight')
    plt.show()


def calculate_global_xlim_and_ylim(transformed_data, algo_names):
    x_lim = -math.inf
    y_lim = -math.inf

    for algo, algo_name in algo_names.items():
        # Remove extra Nan value.
        defectors = transformed_data[algo]['defectors']['mean'][:-1]

        cooperators = transformed_data[algo]['cooperators']['mean'][1:]

        y_lim_local = np.max([defectors, cooperators])
        if y_lim_local > y_lim: y_lim = y_lim_local

        # Condition 1: R_c(N) > R_d(0)
        R_c_N, R_d_0 = cooperators[-1], defectors[0]
        condition_1 = R_c_N - R_d_0

        # Condition 2: R_c(N) > R_c(0)
        R_c_0 = cooperators[0]
        condition_2 = R_c_N - R_c_0

        condition_3 = calculate_condition_3(defectors, cooperators)

        y = [condition_1, condition_2, condition_3]
        x_lim_local = np.max(y)
        if x_lim_local > x_lim: x_lim = x_lim_local

    return x_lim, y_lim


def plot_results(transformed_data, algo_names, xlim, ylim, gamma):
    fig, ax = plt.subplots(nrows=2, ncols=7, figsize=(20, 5))
    count = 0

    # Row 0: NSI Indicator
    # Row 1: Schelling
    for algo, algo_name in algo_names.items():
        # Remove extra Nan value.
        defectors = transformed_data[algo]['defectors']['mean'][:-1]
        defectors_error_bars = transformed_data[algo]['defectors']['std'][:-1]

        cooperators = transformed_data[algo]['cooperators']['mean'][1:]
        cooperators_error_bars = transformed_data[algo]['cooperators']['std'][1:]

        all_agents = transformed_data[algo]['all']['mean']
        all_agents_error_bars = transformed_data[algo]['all']['std']

        create_small_schelling(defectors, defectors_error_bars,
                               cooperators, cooperators_error_bars,
                               all_agents, all_agents_error_bars, algo_name['name'], ax[1][count], ylim)

        # Condition 1: R_c(N) > R_d(0)
        R_c_N, R_d_0 = cooperators[-1], defectors[0]
        c1 = R_c_N - R_d_0

        # Condition 2: R_c(N) > R_c(0)
        R_c_0 = cooperators[0]
        c2 = R_c_N - R_c_0

        c3 = calculate_condition_3(defectors, cooperators)

        x = np.arange(3)
        y = [c1, c2, c3]
        create_nsi_indicator(x, y, ax[0][count], '', xlim)

        count += 1

    fig.savefig('./plots/results_' + str(gamma) + '.pdf', format='pdf', dpi=2000)
    plt.show()


def plot_results_with_nsi_inset(transformed_data, algo_names, xlim, folder_path='./plots', gamma=None,
                                equilibrium_points=None):
    for algo, algo_name in algo_names.items():
        # Remove extra Nan value.
        defectors = transformed_data[algo]['defectors']['mean'][:-1]
        defectors_error_bars = transformed_data[algo]['defectors']['std'][:-1]

        cooperators = transformed_data[algo]['cooperators']['mean'][1:]
        cooperators_error_bars = transformed_data[algo]['cooperators']['std'][1:]

        all_agents = transformed_data[algo]['all']['mean']
        all_agents_error_bars = transformed_data[algo]['all']['std']

        title = ''
        if gamma is not None:
            title = algo_name + ', ' + r'$\alpha=$' + str(gamma)
        else:
            title = algo_name

        equilibrium = None
        if equilibrium_points is not None:
            equilibrium = equilibrium_points[algo]

        plot_nsi_schelling(defectors, defectors_error_bars,
                           cooperators, cooperators_error_bars,
                           all_agents, all_agents_error_bars, equilibrium, title, algo, xlim, folder_path)


def main():
    mpl_style(dark=False)
    plt.style.use('./egta/experiments/style/style.mplstyle')

    results_data_folder_path = os.path.join(ROOT_DATA_FOLDER, 'regen_exp', '')
    interactions_save_file_name = 'interaction_results_data.p'

    rewards = load_rewards(results_data_folder_path, interactions_save_file_name)
    cooperators_ids = load_cooperator_ids(results_data_folder_path, interactions_save_file_name)

    transformed_data = transform_data(rewards, cooperators_ids)

    xlim, ylim = calculate_global_xlim_and_ylim(transformed_data, ALGO_NAMES_ORDERD)
    xlim *= 1.1

    plot_results_with_nsi_inset(transformed_data=transformed_data,
                                algo_names=ALGO_NAMES_ORDERD, xlim=xlim,
                                folder_path=results_data_folder_path)


if __name__ == '__main__':
    main()
