"""
Main function for training and evaluating MARL algorithms in NMARL envs
@author: Tianshu Chu
"""

import logging
import argparse
import threading
import configparser
import tensorflow as tf

from .envs.tragedy_env import TragedyEnv
# from envs.cacc_env import CACCEnv
#from envs.real_net_env import RealNetEnv
#from envs.large_grid_env import LargeGridEnv
from .agents.models import IA2C, IA2C_FP, IA2C_INDEPENDENT, IA2C_CU, MA2C_NC, MA2C_IC3, MA2C_DIAL
from .utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag,
                   plot_evaluation, plot_train)

def parse_args():
    default_base_dir = '/Users/tchu/Documents/rl_test/deeprl_dist/ia2c_grid_0.9'
    default_config_dir = './config/config_ia2c_grid.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--evaluation-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(2000, 2500, 10)]),
                    help="random seeds for evaluation, split by ,")
    sp.add_argument('--demo', action='store_true', help="shows SUMO gui")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0, gui=False):
    scenario = config.get('scenario')
    if "tragedy" in scenario:
        return TragedyEnv(config, gui)
    else:
        raise ValueError()


def init_agent(env, config, total_step, seed):
    """
    params:
    -------
    env: environment.
    config: .yml config file with task parameters.
    total_step: int, number of training steps.
    seed: int, random seed for reporoducible model initialisation.

    returns:
    --------
    MARL agent model
    """
    if env.agent == 'ia2c_ind': # independent learners
        return IA2C_INDEPENDENT(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
    elif env.agent == 'ia2c': # centralised critic
        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
    elif env.agent == 'ia2c_fp': # FingerPrint
        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_nc': # Neurcomm
        return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_cu': # ConseNet
        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_ic3': # CommNet
        return MA2C_IC3(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                        total_step, config, seed=seed)
    elif env.agent == 'ma2c_dial': # DIAL
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    else:
        return None


def train(args):
    # create dirs for task (if not exist)
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    # copy and parse .yml config file
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init environemnt
    env = init_env(config['ENV_CONFIG'])
    logging.info('Task: %s, action dim: %r, agent dim: %d' % (env.scenario+"_"+env.agent,
                                                              env.n_a_ls, env.n_agent))

    # init counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init agent
    seed = config.getint('ENV_CONFIG', 'seed')
    model = init_agent(env, config['MODEL_CONFIG'], total_step, seed) # returns agent class
    logging.info('Agent model initialised. Training....')

    # Init summary writer and train agent
    summary_writer = tf.summary.FileWriter(dirs['log'], graph=model.sess.graph)
    trainer = Trainer(env, model, global_counter, summary_writer, output_path=dirs['data'], checkpoint_path=dirs['model'])
    trainer.run()
    logging.info('Training done.')

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'], final_step)


def load_agents(model, tmp_model, coop_dir, coop_ids, defectors):
    if defectors is None:
        defect_ids = []
        defectors = []
    else:
        defect_ids = [
            i for i in range(len(coop_ids) + len(defectors)) if i not in coop_ids
        ]

    if not model.load(coop_dir + '/model/'):
        logging.error("failed to load agent model")
        exit(-1)

    for my_id, (path, other_id) in zip(defect_ids, defectors):
        if not tmp_model.load(path + '/model/'):
            logging.error("failed to load agent model")
            exit(-1)
        model.load_agent(tmp_model, my_id, other_id)

    if tmp_model is not None:
        del tmp_model


def evaluate_fn(
        config_dir,
        output_dir,
        seeds,
        port,
        demo=False,
        agent_dir=None,
        coop_ids=None,
        defectors=None
    ):
    if not agent_dir:
        agent_dir = config_dir

    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return

    # load config file
    config_file = find_file(config_dir + '/data/')
    if not config_file:
        logging.error('could not find config file')
        return
    config = configparser.ConfigParser()
    config.read(config_file)

    # init environment
    env = init_env(config['ENV_CONFIG'], port=port, gui=demo)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0)

    if model is None:
        return

    if defectors is not None:
        tmp_model = init_agent(env, config['MODEL_CONFIG'], 0, 0)
    else:
        tmp_model = None

    load_agents(model, tmp_model, agent_dir, coop_ids, defectors)

    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir, gui=demo)
    evaluator.run(coop_ids=coop_ids)


def evaluate(args):
    # create sub-dirs to store output of evaluation
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        dirs = init_dir(base_dir, pathes=['demo_data'])
        output_dir = dirs['demo_data']

    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]

    # load and evaluate trained MARL model
    evaluate_fn(
        base_dir,
        args.output_dir or output_dir,
        seeds,
        1,
        args.demo,
        # These are new, here we expect args to be a DictConf which allows None access
        args.coop_dir,
        args.coop_ids,
        args.defectors,
    )


def main():
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == '__main__':
    main()