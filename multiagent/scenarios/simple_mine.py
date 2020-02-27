import numpy as np
from multiagent.core import World, entity2idx, SkilledAgent, Landmark, Mine
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self, **kwargs):
        self.before_make_world(**kwargs)

        world = World()
        world.np_random = self.np_random
        # cache kwargs in case needed in Env wrapper 
        world.config = kwargs
        # set any world properties first
        world.dim_c = 2
        num_agents = 6  # 3
        world.num_agents = num_agents
        num_landmarks = 2
        num_mines = 8
        # add agents
        world.agents = [SkilledAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = False
            agent.size = 0.15

            # temporary skill allocations
            agent.vision_range = np.random.uniform(1, agent.skill_points)
            agent.max_speed = agent.skill_points - agent.vision_range
            self.change_entity_attribute(agent, **kwargs)
            
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.2
            self.change_entity_attribute(landmark, **kwargs)

        # add mines 
        world.mines = [Mine() for i in range(num_mines)]
        for i, mine in enumerate(world.mines):
            mine.name = 'mine %d' % i
            mine.collide = False 
            mine.movable = False 
            mine.size = 0.25
            self.change_entity_attribute(landmark, **kwargs)
            
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        # world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(0, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set mine attributes 
        for mine in world.mines:
            mine.color = np.array([1.0, 1.0, 0.0])
            mine.total_mine = np.random.randint(mine.min_total_mine, mine.max_total_mine)
            mine.state.p_pos = np.random.uniform(-1, +1, world.dim_p) * 10
            mine.state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p) * 10
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.has_mine = False 
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p) * 10
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        # not sure if needed here ???
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    def reward(self, agent, world):
        # Agents are rewarded based on total number of mines mined 
        rew = 0 
        if agent.has_mine:
            # 1 load of mine == 1 reward 
            agent.has_mine = False 
            rew += 1 

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew 

    def observation(self, agent, world):
        """ get positions of all entities in this agent's reference frame
        returns: dict of entity states & masks wrt agent's reference 
        state convention: [entity_type, entity_relative_pos, entity_vel, communication]
        """ 
        states, masks = [], []

        # get self state 
        self_type = agent.__class__.__name__
        self_state = [[entity2idx[self_type]], agent.state.p_pos, agent.state.p_vel]
        if "Agent" in self_type:    # only agent has communication
            self_state += [agent.state.c]
        self_state = np.concatenate(self_state)
        states.append(self_state) 
        masks.append(1)

        # get each entity to agent state 
        for e in world.entities: 
            if e is agent: 
                continue 
            e_pos = e.state.p_pos - agent.state.p_pos
            e_type = e.__class__.__name__
            e_state = [[entity2idx[e_type]], e_pos, e.state.p_vel]
            if "Agent" in e_type:
                e_state += [e.state.c]
            e_state = np.concatenate(e_state)
            states.append(e_state)
            masks.append(1 if np.sqrt(np.sum(np.square(e_pos))) <= agent.vision_range else 0)

        return {"states": states, "masks": masks}

    def update_world(self, world):
        """ advance world assets / states 
        """
        for mine in world.mines:
            # agent mining if at mine 
            for agent in world.agents:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - mine.state.p_pos)))
                if dist < agent.size + mine.size:
                    mine.total_mine -= 1 
                    agent.has_mine = True 
            # respawn mine if depleted 
            if mine.total_mine <= 0:
                mine.total_mine = np.random.randint(1, mine.max_total_mine)
                mine.state.p_pos = np.random.uniform(-1, +1, world.dim_p) * 10
                mine.state.p_vel = np.zeros(world.dim_p)

 




