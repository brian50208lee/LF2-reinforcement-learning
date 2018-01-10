import logging
from flask import Flask, request
from flask_cors import CORS

from Agent import *

# param
verbose = False

# init
app = Flask(__name__)
CORS(app, supports_credentials=True)

# verbose off
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# agent
agent = Agent()
#agent = LF2_Agent()

@app.route("/agent_lf2/choose_action")
def choose_action():
    observation = request.args.get('observation').split(',')
    if verbose:
        print('====================  choose_action  ====================')
        print('observation:', observation)
    action = agent.choose_action(observation)
    return str(action)

@app.route("/agent_lf2/store_transition")
def store_transition():
    pre_observation = request.args.get('pre_observation').split(',')
    action = int(request.args.get('action'))
    reward = int(request.args.get('reward'))
    observation = request.args.get('observation').split(',')
    done = request.args.get('done') == 'true'
    if verbose:
        print('==================== store_transition ====================')
        print('pre_obs:    ', pre_observation)
        print('action:     ', action)
        print('reward:     ', reward)
        print('observation:', observation)
        print('done:       ', done)
    agent.store_transition(pre_observation, action, reward, observation, done)
    return 'success'

if __name__ == "__main__":
    app.run()
    