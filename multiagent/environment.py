import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, render_callback=None, shared_viewer=True, 
                 update_callback=None, show_visual_range=False, cam_range=1):

        self.world = world
        self.agents = self.world.policy_agents
        self.agent_types = [agent.type for agent in self.agents]
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.render_callback = render_callback

        # customized 
        self.update_callback = update_callback
        self.show_visual_range = show_visual_range
        self.cam_range = cam_range

        # environment parameters
        # action space is Discrete or Box (both u and c)
        self.discrete_action_space = world.config.get("discrete_action_space", False)
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = world.config.get("discrete_action_input", False)
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.config.get("discrete_action", False)
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            # action space 
            act_space = self.make_action_space(agent, self.world)
            self.action_space.append(act_space)
            # observation space
            obs_sample = observation_callback(agent, self.world)
            obs_space = self.make_observation_space(obs_sample) 
            self.observation_space.append(obs_space)
            # misc
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def make_action_space(self, agent, world):
        """ make per agent action space
        current actin space include: move, comm
        """
        total_action_space = []
        # physical action space
        if self.discrete_action_space:
            u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
        else:
            u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
        if agent.movable:
            total_action_space.append(u_action_space)
        # communication action space
        if self.discrete_action_space:
            c_action_space = spaces.Discrete(world.dim_c)
        else:
            c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
        if not agent.silent:
            total_action_space.append(c_action_space)
        # total action space
        if len(total_action_space) > 1:
            # NOTE: use Dict for flexibility 
            act_space = spaces.Dict({
                name: space for name, space in zip(
                    ["move", "comm"], total_action_space)
            })
            return act_space 
        else:
            return total_action_space[0]

    def make_observation_space(self, obs_sample):
        """ make per agent observation space from a given sample 
            compatible with original mpe and mpe_hierarchy
        """
        if isinstance(obs_sample, np.ndarray):
            # for original mpe
            obs_dim = len(obs_sample)
            obs_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        elif isinstance(obs_sample, dict):
            # for mpe_hierarchy 
            obs_space = {}
            for k, v in obs_sample.items():
                if k == "masks":
                    n = len(v)
                    space = spaces.MultiBinary(n)
                elif k == "states":
                    assert len(v) > 0
                    n, state_dim = len(v), len(v[0])
                    space = spaces.Tuple((
                        spaces.Box(low=-np.inf, high=+np.inf, shape=(state_dim,), dtype=np.float32) 
                        for _ in range(n)
                    ))
                obs_space[k] = space 
            obs_space = spaces.Dict(spaces=obs_space)
        else:
            # default to an empty obs space 
            obs_sapce = spaces.Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32)
        return obs_space 

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        self._update_world()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            info_n['n'].append(self._get_info(agent))
        # return obs_n
        return obs_n, info_n

    # update world assets / states 
    def _update_world(self):
        if self.update_callback is not None:
            self.update_callback(self.world)

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        """ action: action np array or dict of action (np array) 
        """     
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if agent.movable and not agent.silent:
            move_act = action["move"]
            comm_act = action["comm"]
        elif agent.movable:
            move_act = action 
        elif not agent.silent:
            comm_act = action 
        else:
            raise Exception("Agent set action failed...")

        if agent.movable:   # physical action
            if self.discrete_action_input:  # for hand control
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete (non one-hot) action
                if move_act == 1: agent.action.u[0] = -1.0
                if move_act == 2: agent.action.u[0] = +1.0
                if move_act == 3: agent.action.u[1] = -1.0
                if move_act == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:  
                    # for softened discrete action
                    d = np.argmax(move_act)
                    move_act[:] = 0.0
                    move_act[d] = 1.0
                if self.discrete_action_space:
                    # move act is 5 dim if discrete
                    agent.action.u[0] += move_act[1] - move_act[2]
                    agent.action.u[1] += move_act[3] - move_act[4]
                else:
                    agent.action.u = move_act

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

        if not agent.silent:    # communication action
            if self.force_discrete_action:  
                # for softened discrete action
                d = np.argmax(comm_act)
                comm_act[:] = 0.0
                comm_act[d] = 1.0
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[comm_act] = 1.0
            else:
                agent.action.c = comm_act

    # reset rendering assets
    def _reset_render(self):
        # self.render_geoms = None
        # self.render_geoms_xform = None
        self.render_dict = None  # use per-scenario render callback

    # render environment
    def render(self, mode='human'):   
        """ return list of rgb arrays as renedered images
        """         
        if mode == 'human':
            self.display_comm()
            
        # create viewers (if necessary)
        for i in range(len(self.viewers)):
            if self.viewers[i] is None:
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # actual rendering 
        if self.render_callback is not None:
            return self.render_callback(self, mode=mode)        
            
        self.setup_geometry()
        results = self.exec_rendering(mode=mode)
        return results

    # create geoms and transforms for basic agents and landmarks
    def setup_geometry(self):
        # lazy import 
        from multiagent import rendering
        if self.render_dict is not None:
            return 
        self.render_dict = {}

        # make geometries and transforms
        for entity in self.world.entities:
            name = entity.name
            geom = rendering.make_circle(entity.size)
            xform = rendering.Transform()

            # agent on top, other entity to background 
            alpha = 1.0 if "agent" in name else 0.5
            geom.set_color(*entity.color, alpha=alpha)   

            geom.add_attr(xform)
            self.render_dict[name] = {
                "geom": geom, 
                "xform": xform, 
                "attach_ent": entity
            }

            # LABEL: display type & numbering 
            prefix = "A" if "agent" in entity.name else "L"
            idx = int(name.split(" ")[-1])
            x = entity.state.p_pos[0] 
            y = entity.state.p_pos[1] 
            label_geom = rendering.Text("{}{}".format(prefix,idx), position=(x,y), font_size=30)
            label_xform = rendering.Transform()
            label_geom.add_attr(label_xform)
            self.render_dict[name+"_label"] = {
                "geom": label_geom, 
                "xform": label_xform, 
                "attach_ent": entity
            }
        
        # add geoms to viewer
        for viewer in self.viewers:
            viewer.geoms = []
            for k, d in self.render_dict.items():
                viewer.add_geom(d["geom"])

    # lay out objects in the scene 
    def exec_rendering(self, mode="human"):
        results = []
        for i in range(len(self.viewers)):
            # update bounds to center around agent
            cam_range = self.cam_range   # 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            
            # set view centered arond agent
            # self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            self.viewers[i].set_bounds(-cam_range, cam_range, -cam_range, cam_range)
            
            # update geometry positions
            for k, v in self.render_dict.items():
                xform_pos = v["attach_ent"].state.p_pos
                v["xform"].set_translation(*xform_pos)

            # render to display or array
            results.append(
                self.viewers[i].render(return_rgb_array = mode=='rgb_array')
            )
        return results

    # display communication message
    def display_comm(self):
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        message = ''
        for agent in self.world.agents:
            comm = []
            for other in self.world.agents:
                if other is agent: continue
                if np.all(other.state.c == 0):
                    word = '_'
                else:
                    word = alphabet[np.argmax(other.state.c)]
                # message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
        print(message)

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


########################################################################################

# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
