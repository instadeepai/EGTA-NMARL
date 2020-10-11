import os
import time
import torch
import logging
import itertools
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        os.makedirs(cur_dir, exist_ok=True)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


def plot_train(data_dirs, labels):
    pass

def plot_evaluation(data_dirs, labels):
    pass


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False
        self.num_checkpoints = 4
        self.checkpoint_step = int(total_step / self.num_checkpoints)
        self.checkpoint_count = 0

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop

    def should_checkpoint(self):
        is_checkpoint_step = (self.cur_step % self.checkpoint_step == 0) and (self.cur_step > 0)

        if is_checkpoint_step:
            self.checkpoint_count += 1

            if self.checkpoint_count < self.num_checkpoints:
                return True

        return False


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, output_path=None, checkpoint_path=None):
        self.cur_step = 0 # episode step counter
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent # type of agent, e.g. ia2c
        self.model = model # TF graph
        self.sess = self.model.sess # session inside which model is defined
        self.n_step = self.model.n_step # batch size
        self.summary_writer = summary_writer # from tf.summary.FileWriter(dirs['log'], graph=model.sess.graph) in main.py
        assert self.env.T % self.n_step == 0 # batch size divides total training steps
        self.data = []
        self.output_path = output_path # output_path=dirs['data'] in main.py
        self.env.train_mode = True
        self._init_summary() # placeholder list for train and test rewards
        self.checkpoint_path = checkpoint_path

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def _get_policy(self, ob, done, mode='train'):
        """
        return:
        -------
        policy: list, arrays representing multinomial distributions of
                actions over the given state (one per agent)
        action: numpy array, action selections, one for each agent
        """
        if self.agent.startswith('ma2c'):
            self.ps = self.env.get_fingerprint()
            policy = self.model.forward(ob, done, self.ps)
        else:
            policy = self.model.forward(ob, done)
        action = []
        for pi in policy:
            if mode == 'train':
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action.append(np.argmax(pi))
        return policy, np.array(action)

    def _get_value(self, ob, done, action):
        """
        return:
        -------
        value: list, state values for each agent depending on the local state
               as well as the actions of the agents neighbours
        """
        if self.agent.startswith('ma2c'):
            value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
        else:
            self.naction = self.env.get_neighbor_action(action)
            if not self.naction:
                self.naction = np.nan
            value = self.model.forward(ob, done, self.naction, 'v')
        return value

    def _log_episode(self, global_step, mean_reward, std_reward, sum_reward):
        log = {'agent': self.agent,
               'step': global_step,
               'test_id': -1,
               'collective_reward' : sum_reward,
               'avg_reward': mean_reward,
               'std_reward': std_reward}
        self.data.append(log)
        self._add_summary(sum_reward, global_step) # logging sum_reward is consistent with the metric used in the social influence paper
        self.summary_writer.flush()

    def explore(self, prev_ob, prev_done):
        """
        Sample a single trajectory from the environment.
        """
        ob = prev_ob
        done = prev_done
        for _ in range(self.n_step): # batch size == n_step
            # pre-decision
            policy, action = self._get_policy(ob, done) # lists of policy distributions and action selections
            # post-decision
            value = self._get_value(ob, done, action) # list of state values
            # transition
            self.env.update_fingerprint(policy) # per agent

            next_ob, reward, done, global_reward, _ = self.env.step(action) # reward + global reward are scalars...
            self.episode_rewards.append(global_reward)
            global_step = self.global_counter.next()
            self.cur_step += 1
            # collect experience
            if self.agent.startswith('ma2c'):
                self.model.add_transition(ob, self.ps, action, reward, value, done) # ps --> finger prints
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done) # naction --> neighbour actions
            # logging
            # if self.global_counter.should_log():
            #     logging.info('''Training: global step %d, episode step %d,
            #                        ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
            #                  (global_step, self.cur_step,
            #                   str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
            # terminal check must be inside batch loop for CACC env
            if done:
                break
            ob = next_ob
        if done:
            R = np.zeros(self.model.n_agent) # R--> state-value per agent, 0 if the task is over.
        else:
            _, action = self._get_policy(ob, done)
            R = self._get_value(ob, done, action)
        return ob, done, R

    def perform(self, test_ind, gui=False, episode_file=None, **kwargs):
        ob = self.env.reset(gui=gui, test_ind=test_ind)
        global_rewards, rewards, obses, actions = [], [], [], []

        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                # in on-policy learning, test policy has to be stochastic
                if (self.env.name.startswith('atsc') or self.env.name.startswith('ssd') or "tragedy" in self.env.name):
                    policy, action = self._get_policy(ob, done)
                else:
                    # for mission-critic tasks like CACC, we need deterministic policy
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)

            next_ob, reward, done, global_reward, raw_reward = self.env.step(action)

            global_rewards.append(global_reward)
            if episode_file:
                actions.append(action)
                obses.append(ob)
                rewards.append(raw_reward)

            if done:
                break
            ob = next_ob

        if episode_file:
            np.savez_compressed(
                episode_file,
                obs=obses,
                actions=actions,
                rewards=rewards,
                **kwargs
            )

        mean_reward = np.mean(np.array(global_rewards))
        std_reward = np.std(np.array(global_rewards))
        return mean_reward, std_reward

    def run(self):
        while not self.global_counter.should_stop():

            # # save model checkpoint at certain episodes
            # if self.global_counter.should_checkpoint():
            #     checkpoint = self.global_counter.checkpoint_count
            #     logging.info(f'Training: saving checkpoint {checkpoint:d} ...')
            #     self.model.save(self.checkpoint_path, checkpoint)

            ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset() # resets lstm states
            self.cur_step = 0
            self.episode_rewards = []

            while True:
                ob, done, R = self.explore(ob, done)
                dt = self.env.T - self.cur_step
                global_step = self.global_counter.cur_step
                self.model.backward(R, dt, self.summary_writer, global_step)
                # termination
                if done:
                    self.env.terminate()
                    break

            rewards = np.array(self.episode_rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            sum_reward = np.sum(self.episode_rewards)

            # TODO: add behavioural checkpointing
            # save model checkpoint at performance milestones
            # should_checkpoint, behaviour = self.behaviour_monitor.should_checkpoint()
            # if should_checkpoint:
            #     logging.info(f'Training: behaviour checkpoint: {behaviour} ...')
            #     self.model.save(self.checkpoint_path, behaviour)

            # NOTE: for CACC we have to run another testing episode after each
            # training episode since the reward and policy settings are different!
            if not (self.env.name.startswith('atsc') or self.env.name.startswith('ssd')):
                self.env.train_mode = False
                mean_reward, std_reward = self.perform(-1)
                self.env.train_mode = True
            self._log_episode(global_step, mean_reward, std_reward, sum_reward)

        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, gui=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.gui = gui

    def run(self, episode_path=None, **kwargs):
        if self.gui:
            is_record = False
        else:
            is_record = True

        if episode_path is None:
            episode_path = self.output_path

        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(
                test_ind,
                gui=self.gui,
                episode_file="{}episode{}.npz".format(
                    episode_path,
                    test_ind
                ),
                **kwargs
            )
            self.env.terminate()
            logging.info('test %i, avg reward %.2f, collective_reward %.2f' % (test_ind, reward, reward*self.env.t))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()

        self.env.reset(gui=self.gui, test_ind=test_ind)
