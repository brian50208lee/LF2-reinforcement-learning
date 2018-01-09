import random
import logging
from flask import Flask, request
from flask_cors import CORS
import numpy as np

# init
app = Flask(__name__)
CORS(app, supports_credentials=True)

# verbose off
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# np print option
np.set_printoptions(precision=2)

# load model
model = None

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

@app.route("/agent_lf2/choose_action")
def choose_action():
    observation = request.args.get('observation').split(',')
    print('====================  choose_action  ====================')
    print('observation:', observation)
    # process
    observation = prepro(observation)
    action = int(random.random()*8)
    return str(action)

@app.route("/agent_lf2/store_transition")
def store_transition():
    pre_observation = request.args.get('pre_observation').split(',')
    action = request.args.get('action')
    reward = request.args.get('reward')
    observation = request.args.get('observation').split(',')
    done = request.args.get('done')
    print('==================== store_transition ====================')
    print('pre_obs:    ', pre_observation)
    print('action:     ', action)
    print('reward:     ', reward)
    print('observation:', observation)
    print('done:       ', done)
    # process
    pre_observation = prepro(pre_observation)
    observation = prepro(observation)
    return 'success'

if __name__ == "__main__":
    app.run()