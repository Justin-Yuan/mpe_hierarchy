import numpy as np
from multiagent.core import World, SkilledAgent, Landmark
from multiagent.scenario import BaseScenario
from multiagent.utils import bound_reward, ObsDecorator


class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        self.before_make_world(**kwargs)

        world = World()
        world.np_random = self.np_random
        # cache kwargs in case needed in Env wrapper 
        world.config = kwargs
        # set any world properties first
        world.collaborative = True
        # all entity positions are scaled/extended by size 
        world.size = kwargs.get("world_size", 1)
        world.dim_c = kwargs.get("dim_c", 2)
        num_agents = kwargs.get("num_agents", 3)
        num_landmarks = kwargs.get("num_landmarks", 3)


        # NOTE: for debug, use stacked  agent observations 
        use_stacked_obs = kwargs.get("use_stacked_obs", False)
        stacked_obs_num = kwargs.get("stacked_obs_num", 1)
        if use_stacked_obs:
            obs_decorator = ObsDecorator(stacked_obs_num=stacked_obs_num)
            self.observation = obs_decorator.obs_stacker(self.observation)


        # add agents
        world.agents = [SkilledAgent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = kwargs.get("agent_silence", True)
            agent.size = 0.025
            # self.change_entity_attribute(agent, world, **kwargs)
        
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.075
            # self.change_entity_attribute(landmark, world, **kwargs)

        # make initial conditions
        self.reset_world(world, **kwargs)
        return world

    def reset_world(self, world, **kwargs):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            
        # set random initial states
        for agent in world.agents:
            pos_min, pos_max = kwargs.get("agent_pos_init", [0,-1,1])[1:]
            agent.state.p_pos = self.np_random.uniform(pos_min, pos_max, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            pos_min, pos_max = kwargs.get("landmark_pos_init", [0,-1,1])[1:]
            landmark.state.p_pos = self.np_random.uniform(pos_min, pos_max, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

        # # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # rew += bound_reward(agent, world)   # bound reward already negative
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    
