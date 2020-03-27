import numpy as np
from multiagent.core import World, Agent, SkilledAgent, Landmark
from multiagent.scenario import BaseScenario
from multiagent.utils import bound_reward
import random


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
        # num_good_agents = kwargs.get("num_good_agents", 1)
        # num_adversaries = kwargs.get("num_adversaries", 1)
        # num_agents = num_adversaries + num_good_agents
        num_landmarks = kwargs.get("num_landmarks", 2)
        num_balls = kwargs.get("num_balls", 1) 

        # add agents
        world.agents = [SkilledAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = kwargs.get("agent_silence", True)
            agent.size = 0.025
            # agent.adversary = True if i < num_adversaries else False
            # agent.type = "adversary" if agent.adversary else "agent"
            # agent.size = 0.075 if agent.adversary else 0.05
            # agent.accel = 3.0 if agent.adversary else 4.0
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            self.change_entity_attribute(agent, world, **kwargs)

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            # landmark.size = 0.075
            self.change_entity_attribute(landmark, world, **kwargs)

        # add balls 
        world.balls = [Landmark() for i in range(num_balls)]
        for i, landmark in enumerate(world.balls):
            landmark.name = 'ball %d' % i
            landmark.collide = True
            landmark.movable = True
            landmark.size = 0.2
            self.change_entity_attribute(landmark, world, **kwargs)

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

        # random properties for balls
        for i, landmark in enumerate(world.balls):
            landmark.color = np.array([0.15, 0.15, 0.65])

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

        for i, landmark in enumerate(world.balls):
            pos_min, pos_max = kwargs.get("ball_pos_init", [0,-1,1])[1:]
            landmark.state.p_pos = self.np_random.uniform(pos_min, pos_max, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark
    #     return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    # def agent_reward(self, agent, world):
    #     # the distance to the goal
    #     return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))

    # def adversary_reward(self, agent, world):
    #     # keep the nearest good agents away from the goal
    #     agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if not a.adversary]
    #     pos_rew = min(agent_dist)
    #     #nearest_agent = world.good_agents[np.argmin(agent_dist)]
    #     #neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
    #     neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
    #     #neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
    #     return pos_rew - neg_rew


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

               
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
        if not agent.adversary:
            return np.concatenate([agent.state.p_vel] + [agent.goal_a.state.p_pos - agent.state.p_pos] + [agent.color] + entity_pos + entity_color + other_pos)
        else:
            #other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)
