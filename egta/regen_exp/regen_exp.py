import os
import time
import hydra
import shutil
import logging
import subprocess
import configparser

import numpy as np

from ..main import train, evaluate, evaluate_fn
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

def run_bash_cmd(cmd_string):
    cmd = cmd_string.split(" ")

    process = subprocess.Popen(
        cmd_string,
        shell=True,
        stdout=subprocess.PIPE,
    )

    stdout, stderr = process.communicate()

    return stdout

def get_num_gpus():
    return int(run_bash_cmd("nvidia-smi -L | wc -l"))

def get_gpu_process_count(gpu_id):
    return int(run_bash_cmd(f"nvidia-smi --id={gpu_id} --query-compute-apps=pid --format=csv | wc -l")) - 1

def get_gpu_process_counts():
    num_gpus = get_num_gpus()
    process_counts = []

    for i in range(num_gpus):
        count = get_gpu_process_count(i)
        process_counts.append(count)

    return process_counts

def get_available_gpu():
    counts = np.array(get_gpu_process_counts())
    min_index = np.argmin(counts)
    min_usage = counts[min_index]
    gpus_with_min_usage = counts == min_usage
    num_gpus_with_min_usage = np.sum(gpus_with_min_usage)

    if num_gpus_with_min_usage == 1:
        return min_index
    else:
        indices = np.arange(counts.shape[0])
        available_gup_ids = indices[gpus_with_min_usage]
        return np.random.choice(available_gup_ids)

@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig):
    log.info(f"Running {cfg.algorithm} on env with regen_rate={cfg.regen_rate}")

    gpu_id = get_available_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id:d}"
    log.info(f"Using GPU {gpu_id:d}")

    if "defectors" in cfg:
        assert all(x in cfg for x in ("runs", "tag_file"))
        eval_episode_with_defectors(cfg)
        return

    # use the config options to construct the base and config paths
    config_path = os.path.join(cfg.config_dir, f"config_{cfg.algorithm}_{cfg.env}.ini")
    base_path = os.path.join(cfg.base_dir, cfg.env, f"seed={cfg.seed:d}", f"regen_rate={cfg.regen_rate:1.6f}", cfg.algorithm)

    # check if this experiment has already been conducted
    path_exists = os.path.exists(base_path)

    if cfg.train and path_exists and not cfg.override:
        log.warning("Trail has already been conducted, skipping... (Use the override=true option in 'egta/regen_exp/conf/config.yaml' to re-conduct previous tests).")
        return

    if cfg.train and path_exists:
        log.warning("Trail has already been conducted, overriding")

    if not cfg.train and not path_exists:
        log.error("Attempting to evalute a model which has not been trained! first train the model and then evaluate it")
        exit(-1)

    new_config_path = os.path.join(base_path, f"config.ini")

    args = OmegaConf.create({
        "base_dir": base_path,
        "config_dir": new_config_path
    })

    if "output_dir" in cfg:
        args.output_dir = cfg.output_dir

    if cfg.train:
        # make sure the tensorboard directory is empty
        if path_exists:
            for item in os.listdir(base_path):
                if item != "config.ini": # fix later, rm tree only works on directories
                    shutil.rmtree(os.path.join(base_path, item))

        make_ini(config_path, cfg, new_config_path)

        # run a trial (train the model)
        log.info("starting training for alg %s", cfg.algorithm)
        train(args)
        log.info(
            "completed training for alg %s, saved in dir %s",
            cfg.algorithm,
            base_path,
        )
    else:
        args.evaluation_seeds = ",".join(map(str, range(100, 1100, 50)))
        log.info("starting evaluation for alg %s", cfg.algorithm)
        evaluate(args)
        log.info(
            "completed evaluations for alg %s, saved in dir %s",
            cfg.algorithm,
            base_path,
        )


def eval_episode_with_defectors(cfg: DictConfig):
    import pickle
    from numpy.random import choice
    from os.path import join

    n_defect = cfg.defectors
    n_runs = cfg.runs
    n_coop = cfg.n_agent - n_defect

    config_path = join(cfg.config_dir, f"config_{cfg.algorithm}_{cfg.env}.ini")
    base_dir = join(cfg.base_dir, cfg.algorithm, f"defectors={n_defect}")

    os.makedirs(join(base_dir, "data"), exist_ok=True)
    make_ini(config_path, cfg, join(base_dir, "data", f"config.ini"))

    path = lambda rate, seed: join(
        cfg.base_dir,
        cfg.env,
        f"seed={seed:d}",
        f"regen_rate={rate:1.6f}",
        cfg.algorithm
    )

    with open(cfg.tag_file, "rb") as f:
        tags = pickle.load(f)

    coops = {
        path(rate, seed): seed_dict['cooperators']
        for rate, rate_dict in tags[cfg.algorithm].items()
        for seed, seed_dict in rate_dict.items()
        if len(seed_dict['cooperators']) >= n_coop
    }

    defectors = [
        (path(rate, seed), seed_dict['defectors'])
        for rate, rate_dict in tags[cfg.algorithm].items()
        for seed, seed_dict in rate_dict.items()
        if len(seed_dict['defectors']) > 0
    ]

    log.info("starting evaluation for alg %s with %d defectors", cfg.algorithm, n_defect)
    for n in range(n_runs):
        coop_dir = choice(list(coops.keys()))
        coop_ids = choice(coops[coop_dir], n_coop, replace=False)
        defs = [
            defectors[d]
            for d in choice(len(defectors), n_defect, replace=True)
        ]
        defs = [(d[0], choice(d[1])) for d in defs]

        evaluate_fn(
            base_dir,
            base_dir + f"/run_{n}_",
            [1234],
            1,
            False,
            coop_dir,
            coop_ids,
            defs,
        )
    log.info("completed evaluation for alg %s with %d defectors", cfg.algorithm, n_defect)


def make_ini(config_path: str, cfg: DictConfig, new_config_path: str):
    os.makedirs(os.path.dirname(new_config_path), exist_ok=True)

    # load in the base config file located at config_path
    config = configparser.ConfigParser()
    config.read(config_path)

    # fetch the learning rate to be used for this algorithm
    learning_rate = 10**cfg.optimise.learning_rate_exp
    min_learning_rate = learning_rate * 10**cfg.optimise.learning_rate_decay_exp

    # edit some properties (MODEL_CONFIG.init_lr, ENV_CONFIG.n_agent)
    config["TRAIN_CONFIG"]["total_step"] = f"{cfg.total_step:1.16E}"
    config["ENV_CONFIG"]["seed"] = f"{cfg.seed:d}"
    config["ENV_CONFIG"]["n_agent"] = f"{cfg.n_agent:d}"
    config["ENV_CONFIG"]["regen_rate"] = f"{cfg.regen_rate:1.16E}"
    config["ENV_CONFIG"]["coop_gamma"] = f"{cfg.coop_gamma:1.16E}"
    config["ENV_CONFIG"]["episode_length_sec"] = "1000"
    config["MODEL_CONFIG"]["lr_decay"] = cfg.lr_decay
    config["MODEL_CONFIG"]["optimizer"] = cfg.optimizer
    #config["MODEL_CONFIG"]["lr_init"] = f"{learning_rate:1.16E}"
    #config["MODEL_CONFIG"]["lr_min"] = f"{min_learning_rate:1.16E}"
    #config["MODEL_CONFIG"]["batch_size"] = f"{cfg.optimise.batch_size:d}"
    config["MODEL_CONFIG"]["warm_up_ratio"] = f"{(cfg.optimise.warm_up_ratio):1.16E}"

    # write the new config to disk
    with open(new_config_path, "w") as config_file:
        config.write(config_file)
