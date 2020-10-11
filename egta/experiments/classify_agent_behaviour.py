import os
import pickle

from egta.experiments.evaluation_metrics import Restraint
from egta.experiments.utils import load_meta_data, load_actions, ROOT_DATA_FOLDER


def classify_agents(actions, n_episodes, n_agents, defect_threshold, coop_threshold, agent_tags_file_name):
    agents_tags = {}
    for algo in actions.keys():
        agents_tags[algo] = {}
        for regen_rate in actions[algo].keys():
            agents_tags[algo][regen_rate] = {}
            for seed in actions[algo][regen_rate].keys():
                agents_tags[algo][regen_rate][seed] = {'defectors': [], 'cooperators': []}
                set_of_action = actions[algo][regen_rate][seed]
                restraint = Restraint(n_agents, n_episodes, set_of_action)
                for v_i, v in enumerate(restraint):
                    if v < defect_threshold:
                        agents_tags[algo][regen_rate][seed]['defectors'].append(v_i)
                    else:
                        if v > coop_threshold:
                            agents_tags[algo][regen_rate][seed]['cooperators'].append(v_i)
    pickle.dump(agents_tags, open(agent_tags_file_name, "wb"))
    return agents_tags


def main():
    evaluation_data_folder_path = os.path.join(ROOT_DATA_FOLDER, 'regen_exp', 'tragedy', '')

    evaluation_data_save_file_name = 'evaluation_results_data.p'

    algorithms_list, regen_rates_list, seeds_list, n_episodes, max_n_steps, n_agents = load_meta_data(
        evaluation_data_folder_path, evaluation_data_save_file_name)

    actions = load_actions(evaluation_data_folder_path, evaluation_data_save_file_name)

    defect_threshold = 0.25
    coop_threshold = 0.35
    agent_tags_file_name = evaluation_data_folder_path + 'agent_tags.p'
    agents_tags = classify_agents(actions, n_episodes, n_agents, defect_threshold, coop_threshold,
                                  agent_tags_file_name)
    # print(agents_tags)

    for alg, data in agents_tags.items():
        print(alg, max(len(x['cooperators']) for seed in data.values() for x in seed.values()))


if __name__ == '__main__':
    main()
