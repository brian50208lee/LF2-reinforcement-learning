import random
import logging
from flask import Flask, request
from flask_cors import CORS

# init
app = Flask(__name__)
CORS(app, supports_credentials=True)

# verbose off
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# load model
model = None

@app.route("/agent_lf2/choose_action")
def choose_action():
    observation = request.args.get('observation').split(',')
    print('====================  choose_action  ====================')
    print('observation:', observation)
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
    return 'success'

if __name__ == "__main__":
    app.run()