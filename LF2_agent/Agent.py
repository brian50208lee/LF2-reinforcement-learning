import numpy as np

class Agent(object):
    def __init__(self, args):
        # model parameters
        self.n_actions = 8

        # load model
        print('initial agent done')

    def choose_action(self, observation):
        action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, pre_observation, action, reward, observation, done):
        pass


class LF2_Agent(Agent):
    def cal_win_rate(self, obs):
        if not hasattr(self, 'win_rate'):
            self.win_rate = {'AVG': []}
        t_hp, m_hp, t_id = float(obs[6]), float(obs[7]), obs[16]
        if self.win_rate.get(t_id) is None:
            self.win_rate[t_id] = []
        if t_hp < m_hp:
            self.win_rate[t_id].append(1.0)
            self.win_rate['AVG'].append(1.0)
        else:
            self.win_rate[t_id].append(0.0)
            self.win_rate['AVG'].append(0.0)
        if self.episode % 10 == 0:
            for t_id, hist in self.win_rate.items():
                rate = np.array(hist[-20:], dtype='float32').mean()
                print('Target ID:{:<5} Win Rate:{:<.3}'.format(t_id, rate))

    def prepro(self, observation):
        def one_hot(idx, length):
            vector = [0.0]*length
            if idx < length:
                vector[int(idx)] = 1.0
            return vector
        # to float
        obs = np.array(observation, dtype='float32').tolist()
        # position
        t_x, m_x = obs[0], obs[1]
        t_z, m_z = obs[2], obs[3]
        t_y, m_y = obs[4], obs[5]
        # state
        t_hp, m_hp = obs[6], obs[7]
        t_mp, m_mp = obs[8], obs[9]
        t_fc, m_fc = obs[10], obs[11]
        t_st, m_st = obs[12], obs[13]
        t_fm, m_fm = obs[14], obs[15]
        # other
        t_id, pre_action = obs[16], obs[17]
        bg_bl, bg_br = 0.0, obs[18]
        bg_bt, bg_bd = obs[19], obs[20]
        # select observation
        dbl = abs(m_x - bg_bl)/500
        dbr = abs(m_x - bg_br)/500
        dx = (t_x - m_x)/500
        dz = (t_z - m_z)/100
        dy = (t_y - m_y)/100
        hp = [t_hp, m_hp]
        mp = [t_mp, m_mp]
        fc = [t_fc, m_fc]
        st = one_hot(t_st, 20) + one_hot(m_st, 20)
        fm = [t_fm/200, m_fm/200]
        t_id = one_hot(t_id, 12)
        pre_action = one_hot(pre_action, self.n_actions)
        # merge
        observation = [dbl, dbr] + [dx, dz, dy] + fc + st + fm + pre_action
        observation = np.array(observation, dtype='float32')
        return observation
        
    def __init__(self, args):
        # model parameters
        self.n_actions = 12
        self.inputs_shape = (61,)

        # learning parameters
        self.learn_start = 100
        self.learn_freq = 1
        self.replace_target_freq = 1000
        self.save_episode_freq = 10
        self.explore_rate = 1.0
        self.explore_rate_min = 0.05
        self.explore_step = 200000
        self.explore_rate_delta = -(self.explore_rate - self.explore_rate_min) / self.explore_step

        # initial model
        from brian.DQN import DeepQNetwork
        if args.train:
            self.model = DeepQNetwork(
                            inputs_shape=self.inputs_shape,
                            n_actions=self.n_actions,
                            gamma=0.99,
                            batch_size=32,
                            memory_size=10000,
                            summary_path='./LF2_agent/brian/logs/'
                        )
        else:
            self.explore_rate = 0.0
            self.model = DeepQNetwork(
                inputs_shape=self.inputs_shape,
                n_actions=self.n_actions,
                memory_size=0,
            )
        
        if args.load:
            self.model.load(args.load)

        print('initial agent done')

        # variable
        self.step = 0
        self.episode = 0
        self.episode_reward_hist = [0]



    def choose_action(self, observation):
        obs = self.prepro(observation)
        action = self.model.choose_action(obs)
        if np.random.uniform() < self.explore_rate:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, pre_observation, action, reward, observation, done):
        pre_obs = self.prepro(pre_observation)
        obs = self.prepro(observation)
        self.model.store_transition(pre_obs, action, reward, obs, done)
        # update model
        if self.step > self.learn_start:
            if self.step % self.learn_freq == 0:
                self.model.learn()
            if self.step % self.replace_target_freq == 0:
                self.model.replace_target_net()
        # update variable
        self.explore_rate += self.explore_rate_delta
        self.explore_rate = max(self.explore_rate, self.explore_rate_min)
        self.step += 1
        self.episode_reward_hist[-1] += reward
        if done:
            self.model.summary(step=self.step, reward_hist=self.episode_reward_hist)
            print('episode: {}  reward: {:<.2f}  step: {}  explore_rate: {:<.2f}'.format(
                   self.episode, self.episode_reward_hist[-1], self.step, self.explore_rate))
            if self.episode % self.save_episode_freq == 0:
                self.model.save('./LF2_agent/brian/models/{}/lf2_agent'.format(self.episode))
            self.cal_win_rate(observation)
            self.episode += 1
            self.episode_reward_hist.append(0)
            
