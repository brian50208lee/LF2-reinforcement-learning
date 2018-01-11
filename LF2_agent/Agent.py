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
    def prepro(self, observation):
        def one_hot(idx, length):
            vector = [0.0]*length
            if idx < length:
                vector[idx] = 1.0
            return vector
        observation[5] = 1.0 if observation[5] == 'true' else 0.0 # t_fc
        observation[10] = 1.0 if observation[10] == 'true' else 0.0 # m_fc
        # to float
        observation = np.array(observation, dtype='float32').tolist()
        # position
        observation[0] /= 500.0 # dx
        observation[1] /= 500.0 # dy
        observation[2] /= 100.0 # dz
        # target
        observation[3] = 0.0 # t_hp
        observation[4] = 0.0 # t_mp
        observation[5] = observation[5] # t_fc
        observation[6] = observation[6] # t_st -> one hot (20)
        observation[7] /= 500.0 # t_fm
        # my
        observation[8] = 0.0 # m_hp
        observation[9] = 0.0 # m_mp
        observation[10] = observation[10] # m_fc
        observation[11] = observation[11] # m_st -> one hot (20)
        observation[12] /= 500.0 # m_fm
        # other
        observation[13] = observation[13] # t_id -> one hot (32)
        observation[14] = observation[14] # p_action -> one hot (12)
        # one hot
        observation[14:15] = one_hot(int(observation[14]), 12)
        observation[13:14] = one_hot(int(observation[13]), 32)
        observation[11:12] = one_hot(int(observation[11]), 20)
        observation[6:7] = one_hot(int(observation[6]), 20)
        observation = np.array(observation, dtype='float32')
        return observation
        
    def __init__(self, args):
        # model parameters
        self.n_actions = 12
        self.inputs_shape = (95,)

        # learning parameters
        self.learn_start = 100
        self.learn_freq = 1
        self.replace_target_freq = 1000
        self.save_episode_freq = 10
        self.explore_rate = 1.0
        self.explore_rate_min = 0.1
        self.explore_step = 200000
        self.explore_rate_delta = -(self.explore_rate - self.explore_rate_min) / self.explore_step

        # load model
        from brian.DQN import DeepQNetwork
        self.model = DeepQNetwork(
                        inputs_shape=self.inputs_shape,
                        n_actions=self.n_actions,
                        gamma=0.95,
                        batch_size=64,
                        memory_size=10000,
                        summary_path='./LF2_agent/brian/logs/'
                    )
        
        if args.load:
            self.model.load(args.load)
        if not args.train:
            self.explore_rate = 0.0

        # variable
        self.step = 0
        self.episode = 0
        self.episode_reward_hist = [0]

        print('initial agent done')

    def choose_action(self, observation):
        observation = self.prepro(observation)
        action = self.model.choose_action(observation)
        if np.random.uniform() < self.explore_rate:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, pre_observation, action, reward, observation, done):
        pre_observation = self.prepro(pre_observation)
        observation = self.prepro(observation)
        #print(observation)
        self.model.store_transition(pre_observation, action, reward, observation, done)
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
            print('episode: {}  reward: {}  step: {}  explore_rate: {:<2f}'.format(
                   self.episode, self.episode_reward_hist[-1], self.step, self.explore_rate))
            if self.episode % self.save_episode_freq == 0:
                self.model.save('./LF2_agent/brian/models/{}/lf2_agent'.format(self.episode))
            self.episode += 1
            self.episode_reward_hist.append(0)
