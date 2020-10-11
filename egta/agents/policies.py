import numpy as np
import tensorflow as tf

from .utils import *

class Policy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.identical = identical

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def prepare_loss(self, v_coef, e_coef, max_grad_norm, alpha, epsilon, optimizer):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * e_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])

        if optimizer == "nadam":
            raise NotImplementedError("There are bugs in the Nadam implementation.")
            self.optimizer = tf.keras.optimizers.Nadam(
                lr=self.lr, epsilon=epsilon,
            )
        elif optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr, epsilon=epsilon,
            )
        elif optimizer == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                    epsilon=epsilon)
        else:
            raise ValueError("not a valid argument for optimizer.")

        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        summaries = []
        summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
        summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
        summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
        summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
        summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
        summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
        self.summary = tf.summary.merge(summaries)

    def _build_actor_head(self, h, n_a=None, agent_name=None):
        name = 'pi'
        if agent_name is not None:
            name += '_' + str(agent_name)
        if n_a is None:
            n_a = self.n_a
        pi = fc(h, name, n_a, act=tf.nn.softmax)
        return pi

    def _build_critic_head(self, h, na, n_n=None, agent_name=None):
        name = 'v'
        if agent_name is not None:
            name += '_' + str(agent_name)
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            if self.identical:
                na_sparse = tf.one_hot(na, self.n_a, axis=-1)
                na_sparse = tf.reshape(na_sparse, [-1, self.n_a*n_n])
            else:
                na_sparse = []
                na_ls = tf.split(axis=1, num_or_size_splits=n_n, value=na)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(tf.squeeze(tf.one_hot(na_val, na_dim, axis=-1), axis=1))
                na_sparse = tf.concat(na_sparse, 1)
            h = tf.concat([h, na_sparse], 1)
        v = fc(h, name, 1, act=lambda x: x)
        return v


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        """
        class: an LSTM policy for a single agent
        ----------
        parameters
        ----------
        n_s, int or tupl:   number of elements in state vector (cacc & ATCS) / shape of state tensor (cleanup & harvest)
        n_a, int:           number of discrete actions allowed
        n_n, int:           number of neighbors adjancet to the agent
        n_step, int:        batch size for training
        n_fc, int:          number of units in FC layer
        n_lstm, int:        number of hidden units in LSTM cell

        """
        super().__init__(n_a, n_s, n_step, 'lstm', name, identical)
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n

        # NEW ------------------------------------------------
        self.img_dims, self.vec_dims = n_s
        n_agents, features = self.img_dims[0], self.img_dims[1]
        # n_agents, h, w, n_c = self.img_dims[0], self.img_dims[1], self.img_dims[2], self.img_dims[3]
        self.ob_im_fw = tf.placeholder(tf.float32, [1, n_agents, features]) # img portion of obs (1, 15, 15, 3)
        # self.ob_im_fw = tf.placeholder(tf.float32, [1, n_agents, h, w, n_c]) # img portion of obs (1, 15, 15, 3)
        self.ob_im_bw = tf.placeholder(tf.float32, [n_step, n_agents, features])
        # self.ob_im_bw = tf.placeholder(tf.float32, [n_step, n_agents, h, w, n_c])
        self.ob_vec_fw = tf.placeholder(tf.float32, [1, self.vec_dims]) # other state info to be passed into model other than img (e.g. finger prints)
        self.ob_vec_bw = tf.placeholder(tf.float32, [n_step, self.vec_dims])
        # NEW -------------------------------------------------

        if self.n_n:
            self.naction_fw = tf.placeholder(tf.int32, [1, n_n]) # neighbor actions forward pass
            self.naction_bw = tf.placeholder(tf.int32, [n_step, n_n]) # neighbor actions packward pass

        self.done_fw = tf.placeholder(tf.float32, [1])
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [n_lstm * 2]) # hidden states in lstm

        with tf.variable_scope(self.name):
            self.pi_fw, self.v_fw, self.new_states = self._build_net('forward')
        with tf.variable_scope(self.name, reuse=True):
            self.pi, self.v, _ = self._build_net('backward')
        self._reset()

    def backward(self, sess, obs, nactions, acts, dones, Rs, Advs, cur_lr,
                 summary_writer=None, global_step=None):
        # NEW ---------------------------------
        ob_im, ob_vec = obs[:,0], obs[:,1]
        ob_im = np.stack(ob_im, axis=0).astype(np.float32)
        ob_vec = np.stack(ob_vec, axis=0).astype(np.float32)

        if len(ob_im.shape) < 3:
            ob_im = ob_im[:, :, None]

        ins = {self.ob_im_bw: ob_im,
               self.ob_vec_bw: ob_vec,
               self.done_bw: dones,
               self.states: self.states_bw,
               self.A: acts,
               self.ADV: Advs,
               self.R: Rs,
               self.lr: cur_lr}
        # NEW ---------------------------------
        if self.n_n:
            ins[self.naction_bw] = nactions
        summary, _ = sess.run([self.summary, self._train], ins)
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step=global_step)

    def forward(self, sess, ob, done, naction=None, out_type='p'):
        # update state only when p is called
        # NEW -----------------------------------------
        ob_im, ob_vec = ob[0], ob[1]
        if len(ob_im.shape) < 2:
            ob_im = ob_im[:, None]

        ins = {self.ob_im_fw: np.array([ob_im]),
               self.ob_vec_fw : np.array([ob_vec]),
               self.done_fw: np.array([done]),
               self.states: self.states_fw}
        # NEW -----------------------------------------
        if out_type.startswith('p'):
            outs = [self.pi_fw, self.new_states]
        else:
            outs = [self.v_fw]
            if self.n_n:
                ins[self.naction_fw] = np.array([naction])
        out_values = sess.run(outs, ins)
        out_value = out_values[0]
        if out_type.startswith('p'):
            self.states_fw = out_values[-1]
        return out_value

    def _build_net(self, in_type):
        if in_type == 'forward':
            ob_im = self.ob_im_fw
            ob_vec = self.ob_vec_fw # can ignore in ia2c
            done = self.done_fw
            naction = self.naction_fw if self.n_n else None
        else:
            ob_im = self.ob_im_bw
            ob_vec = self.ob_vec_bw # can ignore in ia2c
            done = self.done_bw
            naction = self.naction_bw if self.n_n else None

        # NEW ---------------------------------------------------------
        if self.n_n > 0:
            h = []
            for i in range(self.n_n): # each agent gets the local obs from its neighbours
                scope = '{}_conv_{}'.format(self.name, i)
                # h_i = conv_to_linear(x=ob_im[:,i], scope=scope, n_out=self.n_fc)
                h_i = fc(x=ob_im[:,i], scope=scope, n_out=self.n_fc)
                h.append(h_i)
            h = tf.concat(h, axis=1)
            h = fc(h, 'fc', self.n_fc)
        else:
            h = fc(ob_im[:,0], 'fc', self.n_fc)
        # NEW ---------------------------------------------------------

        h, new_states = lstm(h, done, self.states, 'lstm')
        pi = self._build_actor_head(h)
        v = self._build_critic_head(h, naction)
        return tf.squeeze(pi), tf.squeeze(v), new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros(self.n_lstm * 2, dtype=np.float32)
        self.states_bw = np.zeros(self.n_lstm * 2, dtype=np.float32)


class FPPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        """
        n_s = dim(obs) + n_neighbours * n_actions
        """
        super().__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name,
                         na_dim_ls, identical)

    def _build_net(self, in_type):
        if in_type == 'forward':
            ob_im = self.ob_im_fw
            ob_vec = self.ob_vec_fw
            done = self.done_fw
            naction = self.naction_fw if self.n_n else None
        else:
            ob_im = self.ob_im_bw
            ob_vec = self.ob_vec_bw
            done = self.done_bw
            naction = self.naction_bw if self.n_n else None
        # NEW ----------------------------------------------------------------------
        # pass img obs through conv_2_linear and through an fc layer
        h_im = []
        for i in range(self.n_n): # each agent gets the local obs from its neighbours
            scope = '{}_conv_{}'.format(self.name, i)
            # h_i = conv_to_linear(x=ob_im[:,i], scope=scope, n_out=self.n_fc)
            h_i = fc(x=ob_im[:,i], scope=scope, n_out=self.n_fc)
            h_im.append(h_i)
        h_im = tf.concat(h_im, axis=1)
        h_im = fc(h_im, 'fc', self.n_fc)

        # separately, pass finger print vector through fc layer (if agent has neighbours)
        if self.n_n:
            h = tf.concat([h_im, ob_vec], axis=1)
        else:
            h = h_im
        # NEW --------------------------------------------------------------------------
        h, new_states = lstm(h, done, self.states, 'lstm')
        pi = self._build_actor_head(h)
        v = self._build_critic_head(h, naction)
        return tf.squeeze(pi), tf.squeeze(v), new_states


class NCMultiAgentPolicy(Policy):
    """ Inplemented as a centralized meta-DNN. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime."""
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        super().__init__(n_a, n_s, n_step, 'nc', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self._init_policy(n_agent, neighbor_mask, n_h, n_fc)

    def backward(self, sess, obs, policies, acts, dones, Rs, Advs, cur_lr,
                 summary_writer=None, global_step=None):

        if obs.shape == self.ob_bw.shape:
            summary, _ = sess.run([self.summary, self._train],
                                {self.ob_bw: obs,
                                self.policy_bw: policies,
                                self.action_bw: acts,
                                self.done_bw: dones,
                                self.states: self.states_bw,
                                self.ADV: Advs,
                                self.R: Rs,
                                self.lr: cur_lr})
            self.states_bw = np.copy(self.states_fw)
            if summary_writer is not None:
                summary_writer.add_summary(summary, global_step=global_step)

    def forward(self, sess, ob, done, policy, action=None, out_type='p'):
        # update state only when p is called
        # ins = {self.ob_fw: np.expand_dims(np.expand_dims(ob, axis=1), axis=1),
        ins = {self.ob_fw: np.expand_dims(ob, axis=1),
               self.done_fw: np.expand_dims(done, axis=1),
               self.policy_fw: np.expand_dims(policy, axis=1),
               self.states: self.states_fw}
        if out_type.startswith('p'):
            outs = [self.pi_fw, self.new_states]
        else:
            outs = [self.v_fw]
            ins[self.action_fw] = np.expand_dims(action, axis=1)
        out_values = sess.run(outs, ins)
        out_value = out_values[0]
        if out_type.startswith('p'):
            self.states_fw = out_values[-1]
        return out_value

    def prepare_loss(self, v_coef, e_coef, max_grad_norm, alpha, epsilon, optimizer):
        self.ADV = tf.placeholder(tf.float32, [self.n_agent, self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_agent, self.n_step])
        # all losses are averaged over steps but summed over agents
        if self.identical:
            A_sparse = tf.one_hot(self.action_bw, self.n_a)
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
            entropy = -tf.reduce_sum(self.pi * log_pi, axis=-1) # NxT
            prob_pi = tf.reduce_sum(log_pi * A_sparse, axis=-1) # NxT
        else:
            entropy = []
            prob_pi = []
            for i, pi_i in enumerate(self.pi):
                action_i = tf.slice(self.action_bw, [i, 0], [1, self.n_step])
                A_sparse_i = tf.one_hot(action_i, self.n_a_ls[i])
                log_pi_i = tf.log(tf.clip_by_value(pi_i, 1e-10, 1.0))
                entropy.append(tf.expand_dims(-tf.reduce_sum(pi_i * log_pi_i, axis=-1), axis=0))
                prob_pi.append(tf.expand_dims(tf.reduce_sum(log_pi_i * A_sparse_i, axis=-1), axis=0))
            entropy = tf.concat(entropy, axis=0)
            prob_pi = tf.concat(prob_pi, axis=0)
        entropy_loss = -tf.reduce_sum(tf.reduce_mean(entropy, axis=-1)) * e_coef
        policy_loss = -tf.reduce_sum(tf.reduce_mean(prob_pi * self.ADV, axis=-1))
        value_loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.R - self.v), axis=-1)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])

        if optimizer == "nadam":
            raise NotImplementedError("There are bugs in the Nadam implementation.")
            self.optimizer = tf.keras.optimizers.Nadam(
                lr=self.lr, epsilon=epsilon,
            )
        elif optimizer == "adam":
             self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr, epsilon=epsilon,
            )
        elif optimizer == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                    epsilon=epsilon)
        else:
            raise ValueError("not a valid argument for optimizer.")

        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        summaries = []
        summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
        summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
        summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
        summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
        summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
        summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
        self.summary = tf.summary.merge(summaries)

    def _build_net(self, in_type):
        if in_type == 'forward':
            ob = self.ob_fw
            policy = self.policy_fw
            action = self.action_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            policy = self.policy_bw
            action = self.action_bw
            done = self.done_bw

        # NEW: Conv and FC layers -----------------------------------------
        h = []
        for i in range(self.n_agent):
            # conv_scope = 'conv_agent_{}'.format(i)
            fc_scope = 'fc_agent_{}'.format(i)
            # h_i = conv_to_linear(x=ob[i], scope=conv_scope, n_out=self.n_fc) # local state
            # h_i = fc(x=h_i, scope=fc_scope, n_out=self.n_fc)
            h_i = fc(x=ob[i], scope=fc_scope, n_out=self.n_fc)
            h.append(h_i)
        h = tf.reshape(tf.concat(h, axis=0), shape=[self.n_agent,-1,self.n_fc])
        # NEW: Conv and FC layers -----------------------------------------

        if self.identical:
            h, new_states = lstm_comm(h, policy, done, self.neighbor_mask, self.states, 'lstm_comm')
        else:
            h, new_states = lstm_comm_hetero(h, policy, done, self.neighbor_mask, self.states,
                                             self.n_s_ls, self.n_a_ls, 'lstm_comm')
        pi_ls = []
        v_ls = []
        for i in range(self.n_agent):
            h_i = h[i] # Txn_h
            if self.identical:
                pi = self._build_actor_head(h_i, agent_name='%d' % i)
                pi_ls.append(tf.expand_dims(pi, axis=0))
                n_n = int(np.sum(self.neighbor_mask[i]))
            else:
                pi = self._build_actor_head(h_i, n_a=self.n_a_ls[i], agent_name='%d' % i)
                pi_ls.append(tf.squeeze(pi))
                self.na_dim_ls = [self.n_a_ls[j] for j in np.where(self.neighbor_mask[i] == 1)[0]]
                n_n = len(self.na_dim_ls)
            if n_n:
                naction_i = tf.transpose(tf.boolean_mask(action, self.neighbor_mask[i])) # Txn_n
            else:
                naction_i = None
            v = self._build_critic_head(h_i, naction_i, n_n=n_n, agent_name='%d' % i)
            v_ls.append(tf.expand_dims(v, axis=0))
        if self.identical:
            pi_ls = tf.squeeze(tf.concat(pi_ls, axis=0))
        return pi_ls, tf.squeeze(tf.concat(v_ls, axis=0)), new_states

    def _init_policy(self, n_agent, neighbor_mask, n_h, n_fc):
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask #n_agent x n_agent
        self.n_h = n_h # hidden units in LSTM
        # NEW ------------------------------------------------------------------------
        self.n_fc = n_fc
        _, features = self.n_s
        # _, height, width, channel = self.n_s
        self.ob_fw = tf.placeholder(tf.float32, [n_agent, 1, features]) # forward 1-step
        # self.ob_fw = tf.placeholder(tf.float32, [n_agent, 1, height, width, channel]) # forward 1-step
        # NEW ------------------------------------------------------------------------
        self.policy_fw = tf.placeholder(tf.float32, [n_agent, 1, self.n_a])
        self.action_fw = tf.placeholder(tf.int32, [n_agent, 1])
        self.done_fw = tf.placeholder(tf.float32, [1])
        # NEW ------------------------------------------------------------------------
        self.ob_bw = tf.placeholder(tf.float32, [n_agent, self.n_step, features]) # backward n-step
        # self.ob_bw = tf.placeholder(tf.float32, [n_agent, self.n_step, height, width, channel]) # backward n-step
        # NEW ------------------------------------------------------------------------
        self.policy_bw = tf.placeholder(tf.float32, [n_agent, self.n_step, self.n_a])
        self.action_bw = tf.placeholder(tf.int32, [n_agent, self.n_step])
        self.done_bw = tf.placeholder(tf.float32, [self.n_step])
        self.states = tf.placeholder(tf.float32, [n_agent, n_h * 2])

        with tf.variable_scope(self.name):
            self.pi_fw, self.v_fw, self.new_states = self._build_net('forward')
        with tf.variable_scope(self.name, reuse=True):
            self.pi, self.v, _ = self._build_net('backward')
        self._reset()

    def _reset(self):
        self.states_fw = np.zeros((self.n_agent, self.n_h * 2), dtype=np.float32)
        self.states_bw = np.zeros((self.n_agent, self.n_h * 2), dtype=np.float32)


class ConsensusPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cu', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.n_h = n_h
        self.neighbor_mask = neighbor_mask
        self._init_policy(n_agent, neighbor_mask, n_h, n_fc)

    def backward(self, sess, obs, policies, acts, dones, Rs, Advs, cur_lr,
                 summary_writer=None, global_step=None):
        super().backward(sess, obs, policies, acts, dones, Rs, Advs, cur_lr,
                         summary_writer, global_step)
        sess.run(self._consensus_update)

    def prepare_loss(self, v_coef, e_coef, max_grad_norm, alpha, epsilon, optimizer):
        super().prepare_loss(v_coef, e_coef, max_grad_norm, alpha, epsilon, optimizer)
        consensus_update = []
        for i in range(self.n_agent):
            wt_from, wt_to = self._get_critic_wts(i)
            for w1, w2 in zip(wt_from, wt_to):
                consensus_update.append(w2.assign(w1))
        self._consensus_update = tf.group(*consensus_update)

    def _build_net(self, in_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
            action = self.action_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
            action = self.action_bw
        pi_ls = []
        v_ls = []
        new_states_ls = []
        for i in range(self.n_agent):
            # NEW -----------------------------------------------------------
            # h_i = conv_to_linear(x=ob[i], scope='conv_%da' % i, n_out=self.n_fc) # local state
            # NEW -------------------------------------------------------------
            # h_i = fc(x=ob_im[i], scope=scope, n_out=self.n_fc)
            h_i = fc(ob[i], 'fc_%da' % i, self.n_h)
            # h_i = fc(h_i, 'fc_%da' % i, self.n_h)
            h_i, new_states = lstm(h_i, done, self.states[i], 'lstm_%da' % i)
            if self.identical:
                pi = self._build_actor_head(h_i, agent_name='%d' % i)
                pi_ls.append(tf.expand_dims(pi, axis=0))
                n_n = int(np.sum(self.neighbor_mask[i]))
            else:
                pi = self._build_actor_head(h_i, n_a=self.n_a_ls[i], agent_name='%da' % i)
                pi_ls.append(tf.squeeze(pi))
                self.na_dim_ls = [self.n_a_ls[j] for j in np.where(self.neighbor_mask[i] == 1)[0]]
                n_n = len(self.na_dim_ls)
            if n_n:
                naction = tf.transpose(tf.boolean_mask(action, self.neighbor_mask[i]))
            else:
                naction = None
            v = self._build_critic_head(h_i, naction, n_n=n_n, agent_name='%da' % i)
            v_ls.append(tf.expand_dims(v, axis=0))
            new_states_ls.append(tf.expand_dims(new_states, axis=0))
        if self.identical:
            pi_ls = tf.squeeze(tf.concat(pi_ls, axis=0))
        v_ls = tf.squeeze(tf.concat(v_ls, axis=0))
        new_states_ls = tf.squeeze(tf.concat(new_states_ls, axis=0))
        return pi_ls, v_ls, new_states_ls

    def _get_critic_wts(self, agent_i):
        neighbor_mask = self.neighbor_mask[agent_i]
        agents = [agent_i] + list(np.where(neighbor_mask == 1)[0])
        wt_i = []
        wt_n = []
        for i in agents:
            critic_scope = [self.name + ('/lstm_%da' % i)]
            wt = []
            for scope in critic_scope:
                wt += tf.trainable_variables(scope=scope)
            if i == agent_i:
                wt_i = wt
            wt_n.append(wt)
        mean_wt_n = []
        n_n = len(wt_n)
        n_w = len(wt_n[0])
        for i in range(n_w):
            cur_wts = []
            for j in range(n_n):
                cur_wts.append(tf.expand_dims(wt_n[j][i], axis=-1))
            cur_wts = tf.concat(cur_wts, axis=-1)
            cur_wts = tf.reduce_mean(cur_wts, axis=-1)
            mean_wt_n.append(cur_wts)
        return mean_wt_n, wt_i

# IC3 -> CommNet
class IC3MultiAgentPolicy(NCMultiAgentPolicy):
    """Reference code: https://github.com/IC3Net/IC3Net/blob/master/comm.py.
       Note in IC3, the message is generated from hidden state only, so current state
       and neigbor policies are not included in the inputs."""
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'ic3', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self._init_policy(n_agent, neighbor_mask, n_h, n_fc)

    def _build_net(self, in_type):
        if in_type == 'forward':
            ob = self.ob_fw
            action = self.action_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            action = self.action_bw
            done = self.done_bw

        # NEW: Conv and FC layers -----------------------------------------
        h = []
        for i in range(self.n_agent):
            # conv_scope = 'conv_agent_{}'.format(i)
            fc_scope = 'fc_agent_{}'.format(i)
            # h_i = conv_to_linear(x=ob[i], scope=conv_scope, n_out=self.n_fc) # local state
            # h_i = fc(x=h_i, scope=fc_scope, n_out=self.n_fc)
            h_i = fc(x=ob[i], scope=fc_scope, n_out=self.n_fc)
            h.append(h_i)
        h = tf.reshape(tf.concat(h, axis=0), shape=[self.n_agent,-1,self.n_fc])
        # NEW: Conv and FC layers -----------------------------------------

        if self.identical:
            h, new_states = lstm_ic3(h, done, self.neighbor_mask, self.states, 'lstm_ic3')
        else:
            h, new_states = lstm_ic3_hetero(h, done, self.neighbor_mask, self.states,
                                            self.n_s_ls, self.n_a_ls, 'lstm_ic3')
        pi_ls = []
        v_ls = []
        for i in range(self.n_agent):
            h_i = h[i] # Txn_h
            if self.identical:
                pi = self._build_actor_head(h_i, agent_name='%d' % i)
                pi_ls.append(tf.expand_dims(pi, axis=0))
                n_n = int(np.sum(self.neighbor_mask[i]))
            else:
                pi = self._build_actor_head(h_i, n_a=self.n_a_ls[i], agent_name='%d' % i)
                pi_ls.append(tf.squeeze(pi))
                self.na_dim_ls = [self.n_a_ls[j] for j in np.where(self.neighbor_mask[i] == 1)[0]]
                n_n = len(self.na_dim_ls)
            if n_n:
                naction_i = tf.transpose(tf.boolean_mask(action, self.neighbor_mask[i])) # Txn_n
            else:
                naction_i = None
            v = self._build_critic_head(h_i, naction_i, n_n=n_n, agent_name='%d' % i)
            v_ls.append(tf.expand_dims(v, axis=0))
        if self.identical:
            pi_ls = tf.squeeze(tf.concat(pi_ls, axis=0))
        return pi_ls, tf.squeeze(tf.concat(v_ls, axis=0)), new_states


class DIALMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'dial', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self._init_policy(n_agent, neighbor_mask, n_h, n_fc)

    def _build_net(self, in_type):
        if in_type == 'forward':
            ob = self.ob_fw
            policy = self.policy_fw
            action = self.action_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            policy = self.policy_bw
            action = self.action_bw
            done = self.done_bw

        # NEW: Conv and FC layers -----------------------------------------
        h = []
        for i in range(self.n_agent):
            # conv_scope = 'conv_agent_{}'.format(i)
            fc_scope = 'fc_agent_{}'.format(i)
            # h_i = conv_to_linear(x=ob[i], scope=conv_scope, n_out=self.n_fc) # local state
            # h_i = fc(x=h_i, scope=fc_scope, n_out=self.n_fc)
            h_i = fc(x=ob[i], scope=fc_scope, n_out=self.n_fc)
            h.append(h_i)
        h = tf.reshape(tf.concat(h, axis=0), shape=[self.n_agent,-1,self.n_fc])
        # NEW: Conv and FC layers -----------------------------------------

        if self.identical:
            h, new_states = lstm_dial(h, policy, done, self.neighbor_mask, self.states, 'lstm_comm')
        else:
            h, new_states = lstm_dial_hetero(h, policy, done, self.neighbor_mask, self.states,
                                             self.n_s_ls, self.n_a_ls, 'lstm_comm')
        pi_ls = []
        v_ls = []
        for i in range(self.n_agent):
            h_i = h[i] # Txn_h
            if self.identical:
                pi = self._build_actor_head(h_i, agent_name='%d' % i)
                pi_ls.append(tf.expand_dims(pi, axis=0))
                n_n = int(np.sum(self.neighbor_mask[i]))
            else:
                pi = self._build_actor_head(h_i, n_a=self.n_a_ls[i], agent_name='%d' % i)
                pi_ls.append(tf.squeeze(pi))
                self.na_dim_ls = [self.n_a_ls[j] for j in np.where(self.neighbor_mask[i] == 1)[0]]
                n_n = len(self.na_dim_ls)
            if n_n:
                naction_i = tf.transpose(tf.boolean_mask(action, self.neighbor_mask[i])) # Txn_n
            else:
                naction_i = None
            v = self._build_critic_head(h_i, naction_i, n_n=n_n, agent_name='%d' % i)
            v_ls.append(tf.expand_dims(v, axis=0))
        if self.identical:
            pi_ls = tf.squeeze(tf.concat(pi_ls, axis=0))
        return pi_ls, tf.squeeze(tf.concat(v_ls, axis=0)), new_states
