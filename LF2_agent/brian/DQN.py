import os
import numpy as np
import tensorflow as tf

class BasicDeepQNetwork(object):
    def __init__(
        self,
        inputs_shape,
        n_actions,
        gamma=0.99,
        optimizer=tf.train.AdamOptimizer(0.0001),
        batch_size=32,
        memory_size=10000,
        summary_path=None,
    ):  
        # params
        self.inputs_shape = inputs_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.summary_path = summary_path

        # initialize memory [s, a, r, s_, d]
        self.memory_counter = 0
        self.memory_s = np.zeros((self.memory_size,) + tuple(self.inputs_shape))
        self.memory_a = np.zeros((self.memory_size,))
        self.memory_r = np.zeros((self.memory_size,))
        self.memory_s_ = np.zeros((self.memory_size,) + tuple(self.inputs_shape))
        self.memory_d = np.zeros((self.memory_size,))

        # model
        self._build_placeholder()
        self._build_model()
        self._build_loss()
        self._build_optimize()
        self._build_replacement()

        # saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # log
        self._build_summary()

    def _build_placeholder(self):
        self.s = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None], name='a')  # input Action
        self.d = tf.placeholder(tf.float32, [None], name='d')  # done
     
    def _net(self, inputs):
        raise NotImplementedError()

    def _build_model(self):
        with tf.variable_scope('online_net'):
            self.online_net = self._net(self.s)
        with tf.variable_scope('target_net'):
            self.target_net = self._net(self.s_)

    def _build_loss(self):
        with tf.variable_scope('loss'):
            action_one_hot = tf.one_hot(self.a, self.n_actions)
            q_eval = tf.reduce_sum(self.online_net * action_one_hot, axis=1, name='q_eval')
            q_target = self.r + (1. - self.d) * self.gamma * tf.reduce_max(self.target_net, axis=1, name='q_target')
            self.q_target = tf.stop_gradient(q_target)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, q_eval), name='loss_mse')

    def _build_optimize(self):
        with tf.variable_scope('train_op'):
            clip_value = 1.
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), clip_value)
            self.train_op = self.optimizer.apply_gradients(zip(grads, trainable_variables))

    def _build_replacement(self):
        o_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        with tf.variable_scope('replacement'):
            self.replace_target_net_op = [tf.assign(t, o) for t, o in zip(t_params, o_params)]

    def _build_summary(self):
        if self.summary_path:
            self.reward_hist = tf.placeholder(tf.float32, [None], name='reward_hist')
            tf.summary.scalar('min_reward', tf.reduce_min(self.reward_hist))
            tf.summary.scalar('max_reward', tf.reduce_max(self.reward_hist))
            tf.summary.scalar('avg_reward', tf.reduce_mean(self.reward_hist))
            tf.summary.scalar('max_q', tf.reduce_max(self.q_target))
            tf.summary.scalar('loss_mse', self.loss)
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
    
    def choose_action(self, observation):
        actions_value = self.sess.run(self.online_net, feed_dict={self.s: observation[np.newaxis, :]})
        action = np.argmax(actions_value)
        return action

    def store_transition(self, s, a, r, s_, d):
        idx = self.memory_counter % self.memory_size
        self.memory_s[idx] = np.array(s)
        self.memory_a[idx] = a
        self.memory_r[idx] = float(r)
        self.memory_s_[idx] = np.array(s_)
        self.memory_d[idx] = float(d)
        self.memory_counter += 1

    def learn(self):
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        self.sess.run(self.train_op,
                      feed_dict={
                            self.s: self.memory_s[sample_index],
                            self.a: self.memory_a[sample_index],
                            self.r: self.memory_r[sample_index],
                            self.s_: self.memory_s_[sample_index],
                            self.d: self.memory_d[sample_index]
                      })

    def summary(self, step, reward_hist):
        if self.summary_path:
            sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
            result = self.sess.run(self.summary_op,
                                   feed_dict={
                                        self.reward_hist: np.array(reward_hist[-self.batch_size:]),
                                        self.s: self.memory_s[sample_index],
                                        self.a: self.memory_a[sample_index],
                                        self.r: self.memory_r[sample_index],
                                        self.s_: self.memory_s_[sample_index],
                                        self.d: self.memory_d[sample_index]
                                   })
            self.summary_writer.add_summary(result, step)

    def replace_target_net(self):
        self.sess.run(self.replace_target_net_op)
        print('replace target net')

    def save(self, checkpoint_file_path):
        dirname = os.path.dirname(checkpoint_file_path)
        if len(dirname) > 0 and not os.path.exists(dirname):
            os.makedirs(dirname)
        self.saver.save(self.sess, checkpoint_file_path)
        print('Model saved to: {}'.format(checkpoint_file_path))
    
    def load(self, checkpoint_file_path):
        self.saver.restore(self.sess, checkpoint_file_path)
        print('Model restored from: {}'.format(checkpoint_file_path))



class DeepQNetwork(BasicDeepQNetwork):
    def leaky_relu(self, x, alpha=0.01):
        return tf.maximum(tf.minimum(0.0, alpha*x), x)

    def _net(self, inputs):
        # --------- input ----------
        net = tf.identity(inputs, name='input')
        print(net.name, net.shape)
        # --------- layer1 ----------
        net = tf.layers.dense(net, 64, activation=tf.nn.relu, name='fc1')
        print(net.name, net.shape)        
        # --------- layer2 ----------
        net = tf.layers.dense(net, 32, activation=tf.nn.relu, name='fc2')
        print(net.name, net.shape)
        # --------- output ----------
        net = tf.layers.dense(net, self.n_actions, activation=None, name='output')
        print(net.name, net.shape)
        return net

