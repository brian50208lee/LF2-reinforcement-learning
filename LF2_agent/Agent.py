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
            vector[idx] = 1.0
            return vector
        observation[4] = 1.0 if observation[4] == 'true' else 0.0
        observation[9] = 1.0 if observation[9] == 'true' else 0.0
        observation = np.array(observation, dtype='float32').tolist()
        # dx dz
        observation[0] /= 500.0
        observation[1] /= 500.0
        # target
        observation[2] = 0.0
        observation[3] = 0.0
        observation[6] /= 500.0
        # my
        observation[7] = 0.0
        observation[8] = 0.0
        observation[11] /= 500.0
        # one hot
        observation[10:11] = one_hot(int(observation[10]), 17)
        observation[5:6] = one_hot(int(observation[5]), 17)
        observation = np.array(observation, dtype='float32')
        return observation
        
    def __init__(self, args):
        # model parameters
        self.n_actions = 8
        self.inputs_shape = (44,)

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
                        gamma=0.99,
                        batch_size=32,
                        memory_size=5000,
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
