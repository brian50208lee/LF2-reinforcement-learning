import logging
import argparse
from flask import Flask, request
from flask_cors import CORS

from Agent import *

# params
parser = argparse.ArgumentParser(description="LF2 Agent Server")
parser.add_argument('--verbose', default=False, action='store_true', help='print when received request')
parser.add_argument('--train', default=False, action='store_true', help='train model')
parser.add_argument('--load', default=None, help='model path')
args = parser.parse_args()

# init
app = Flask(__name__)
CORS(app, supports_credentials=True)

# verbose off
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# agent
#agent = Agent(args)
agent = LF2_Agent(args)

@app.route("/agent_lf2/choose_action")
def choose_action():
    observation = request.args.get('observation').split(',')
    if args.verbose:
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
    if args.verbose:
        print('==================== store_transition ====================')
        print('pre_obs:    ', pre_observation)
        print('action:     ', action)
        print('reward:     ', reward)
        print('observation:', observation)
        print('done:       ', done)
    if args.train:
        agent.store_transition(pre_observation, action, reward, observation, done)
    return 'success'

if __name__ == "__main__":
    app.run()
    