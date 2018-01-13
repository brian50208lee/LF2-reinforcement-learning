define(function() {
	function AIscript(self, match, controller){
		// states
		var cc = 0; //counter
		var action_map = ['do_nothing', 'left', 'right', 'up', 'down', 'def', 'jump', 'att']; //number to direction mapping
		var hold_action_map = ['left', 'right', 'up', 'down']
		var target; //target opponent to chase after
		// dqn
		var pre_observation = null;
		var observation = null;
		var action = 0;
		var next_action = 0;
		var pre_t_hp = 500;
		var pre_m_hp = 500;
		// AI script
		this.TU = function() {
			// load target
			if(cc % 100 === 0) {
				load_target();
			}
			// destroy items every 10
			if (cc % 10 === 0) {
				match.destroy_weapons();
				self.health.mp = 0;
				target.health.mp = 0;
			}
			cc++;
			// next state
			pre_observation = observation;
			observation = get_observation();
			action = next_action;
			next_action = choose_action(observation);
			if (next_action != 0) {
				if (next_action < action_map.length) {
					controller.keypress(action_map[next_action]);
				} else {
					controller.keypress(hold_action_map[next_action - action_map.length], 1, 1);
				}
			}
			done = is_done();
			reward = get_reward();
			// store transition
			store_transition(pre_observation, action, reward, observation, done);
			// game over restart
			if (done) {
				pre_observation = null;
				observation = null;
				match.destroy();
				match.manager.start_demo();
			}
		}

		function get_observation() {
			// position
			var x = [target.ps.x, self.ps.x];
			var z = [target.ps.z, self.ps.z];
			var y = [target.ps.y, self.ps.y];
			// state
			var hp = [target.health.hp, self.health.hp];
			var mp = [target.health.mp, self.health.mp];
			var fc = [target.AI.facing() ? 1 : 0, self.AI.facing() ? 1 : 0];
			var st = [target.state(), self.state()];
			var fm = [target.frame.N, self.frame.N];
			// other
			var other = [target.id, action, match.background.width, match.background.zboundary[0], match.background.zboundary[1]]
			// observation vector
			observation = []
			observation = observation.concat(x, z, y, hp, mp, fc, st, fm, other)
			observation = observation.join() // join by ','
			return observation
		}

		function is_done () {
			return self.health.hp <= 0 || target.health.hp <= 0 || cc > 3000;
		}

		function get_reward() {
			reward = 0.0;
			if (target.health.hp - pre_t_hp < 0) {
				reward = 1.0;
			} else if (self.health.hp - pre_m_hp < 0) {
				reward = -1.0;
			} else if (self.ps.x == 0 || self.ps.x == match.background.width) {
				reward = -0.5;
			}
			pre_t_hp = target.health.hp;
			pre_m_hp = self.health.hp;
			return reward;
		}

		function choose_action(observation) {
		    var action = '0';
		    $.ajax({
		        async: false, // wait for response
		        url: 'http://localhost:5000/agent_lf2/choose_action',
		        type: 'GET',
		        data: {observation: observation},
		        success: function (response) { action = response; },
		        error: function (error) { console.log('error'); }
		    });
		    return parseInt(action);
		}

		function store_transition(pre_observation, action, reward, observation, done) {
		    if (pre_observation == null) { return 'skip'; }
		    if (action == null) { return 'skip'; }
		    if (reward == null) { return 'skip'; }
		    if (observation == null) { return 'skip'; }
		    if (done == null) { return 'skip'; }
		    var result = 'error';
		    $.ajax({
		        async: false, // wait for response
		        url: 'http://localhost:5000/agent_lf2/store_transition',
		        type: 'GET',
		        data: {
		            pre_observation: pre_observation,
		            action: action,
		            reward: reward,
		            observation: observation,
		            done: done,
		        },
		        success: function (response) { result = response; }, // 'success'
		        error: function (error) { console.log(error); }
		    });
		    return result;
		}

		function load_target() {
			var targets = [];
			var game_objects = match.scene.live; //list of living object in scene
			for (var i in game_objects) {
				var obj = game_objects[i];
				if( obj.type==='character' && obj.team!==self.team) {
					var dx = obj.ps.x-self.ps.x;
					var dz = obj.ps.z-self.ps.z;
					targets.push({
						dist:Math.sqrt(dx*dx+dz*dz),
						obj:obj
					});
				}
			}
			targets.sort(function(a,b){
				return a.dist-b.dist; //sort according to distance
			});
			if( rand(2)===0) {
				target = targets[rand(targets.length)].obj; //select a random opponent as target
			} else {
				target = targets[0].obj; //select the closest opponent
			}
		}
		
		function rand(i) {	//extremely important: one must get a pseudo random number from match for the sake of determinism
			return Math.floor(match.random()*i); //return a random integer from 0 to i-1 (inclusive)
		}
	}

	AIscript.type = AIscript.prototype.type = 'AIscript'; //must define a type
	return AIscript;
});
