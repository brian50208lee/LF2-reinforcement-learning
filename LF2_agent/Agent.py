import random
import numpy as np
#from DQN import DeepQNetwork

def prepro(observation):
    def one_hot(idx, length):
        vector = [0.0]*length
        vector[idx] = 1.0
        return vector
    observation[4] = 1.0 if observation[4] == 'true' else 0.0
    observation[9] = 1.0 if observation[9] == 'true' else 0.0
    observation[10:11] = one_hot(int(observation[10]), 17)
    observation[5:6] = one_hot(int(observation[5]), 17)
    observation = np.array(observation, dtype='float32')
    return observation

class LF2_Agent:
    def __init__(self):
        # model parameters
        self.n_actions = 8
        self.inputs_shape = (44,)

        # learning parameters
        self.learn_start = 100
        self.learn_freq = 1
        self.replace_target_freq = 1000
        self.summary_freq = 1000
        self.save_episode_freq = 10
        self.explore_rate = 1.0
        self.explore_rate_min = 0.05
        self.explore_step = 10000
        self.explore_rate_delta = -(self.explore_rate - self.explore_rate_min) / self.explore_step

        # load model
        '''
        self.model = DeepQNetwork(
                        inputs_shape=self.inputs_shape,
                        n_actions=self.n_actions,
                        gamma=0.99,
                        batch_size=32,
                        memory_size=1000,
                        summary_path='LF2_agent/log/'
                    )
        '''

        # variable
        self.step = 0
        self.episode = 0
        self.episode_reward_hist = [0]

        print('initial agent done')

    def choose_action(self, observation):
        observation = prepro(observation)
        action = int(random.random()*self.n_actions)
        '''
        action = self.model.choose_action(observation)
        if np.random.uniform() < self.explore_rate:
            action = np.random.randint(0, self.n_actions)
        '''
        return action

    def store_transition(self, pre_observation, action, reward, observation, done):
        pre_observation = prepro(pre_observation)
        observation = prepro(observation)
        #self.model.store_transition(pre_observation, action, reward, observation, done)

