import os
import hydra
import shutil
import logging
import numpy as np
import configparser

from typing import Any
from ..main import train
from omegaconf import DictConfig, OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log = logging.getLogger(__name__)

@hydra.main(config_path="conf/config.yaml")
def sample_space(cfg: DictConfig) -> Any:
    # calculate the learning rate to be used for this trial
    learning_rate = 10**cfg.optimise.learning_rate_exp
    min_learning_rate = learning_rate * 10**cfg.optimise.learning_rate_decay_exp

    seed = np.random.randint(1000)
    # use the config options to construct the base and config paths
    config_path = os.path.join(cfg.config_dir, f"config_{cfg.algorithm}_{cfg.env}.ini")
    base_path = os.path.join(
        cfg.base_dir, cfg.env, cfg.algorithm,
        f"lr_max={learning_rate:1.4E}",
        f"lr_min={min_learning_rate:1.4E}",
        f"warm={cfg.optimise.warm_up_ratio:1.4E}",
        f"batch={cfg.optimise.batch_size:d}",
        f"seed={seed:d}",
    )

    # check if this experiment has already been conducted
    path_exists = os.path.exists(base_path)
    if cfg.override or not path_exists:
        # make sure the tensorboard directory is empty
        if path_exists:
            for item in os.listdir(base_path):
                if item != "config.ini": # fix later, rm tree only works on directories
                    shutil.rmtree(os.path.join(base_path, item))

        os.makedirs(base_path, exist_ok=True)

        # load in the base config file located at config_path
        config = configparser.ConfigParser()
        config.read(config_path)

        # edit some properties (MODEL_CONFIG.init_lr, ENV_CONFIG.n_agent)
        config["TRAIN_CONFIG"]["total_step"] = f"{cfg.total_step:1.16E}"
        config["ENV_CONFIG"]["seed"] = f"{seed:d}"
        config["ENV_CONFIG"]["n_agent"] = f"{cfg.n_agent:d}"
        config["ENV_CONFIG"]["regen_rate"] = f"{cfg.regen_rate:1.16E}"
        config["ENV_CONFIG"]["coop_gamma"] = f"{cfg.coop_gamma:1.16E}"
        config["MODEL_CONFIG"]["lr_decay"] = cfg.lr_decay
        config["MODEL_CONFIG"]["optimizer"] = cfg.optimizer
        config["MODEL_CONFIG"]["lr_init"] = f"{learning_rate:1.16E}"
        config["MODEL_CONFIG"]["lr_min"] = f"{min_learning_rate:1.16E}"
        config["MODEL_CONFIG"]["batch_size"] = f"{cfg.optimise.batch_size:d}"
        config["MODEL_CONFIG"]["warm_up_ratio"] = f"{(cfg.optimise.warm_up_ratio):1.16E}"

        # write the new config to disk
        new_config_path = os.path.join(base_path, f"config.ini")
        with open(new_config_path, "w") as config_file:
            config.write(config_file)

        # pass that directory path in with args (args.config_dir = new_config_dir)

        # construct the arguments we need to pass to the train function
        args = OmegaConf.create({
            "base_dir": base_path,
            "config_dir": new_config_path
        })

        # run a trial (train the model)
        train(args)
    else:
        log.info("Trail has already been conducted, skipping... (Use the override=true option in 'egta/hyp_optim/conf/config.yaml' to re-conduct previous tests).")

    # parse the tensorboard logs to fetch the train_reward data for this trial
    tensorboard_log_path = os.path.join(base_path, "log")
    event_acc = EventAccumulator(tensorboard_log_path)
    event_acc.Reload()
    # get wall clock, number of steps and value for a scalar 'train_reward'
    w_times, step_nums, vals = zip(*event_acc.Scalars('train_reward'))

    if cfg.metric.lower() == "max":
        metric = np.max(vals)
    elif cfg.metric.lower() == "integral":
        metric = np.trapz(vals)
    else:
        raise ValueError(f"metric {cfg.metric} not supported. Check 'egta/hyp_optim/conf/config.yaml' for available options")

    log.info(f"max lr {learning_rate}, min lr {min_learning_rate}, warm up ratio {cfg.optimise.warm_up_ratio}, batch size {cfg.optimise.batch_size} => {metric} {cfg.metric} reward")

    return metric

def main():
    sample_space()
