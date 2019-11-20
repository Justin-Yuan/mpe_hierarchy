import numpy as np
from multiagent.core import World, SkilledAgent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        self.before_make_world(**kwargs)
        # cache kwargs in case needed in Env wrapper 
        self.config = kwargs

        world = World()
        world.np_random = self.np_random
        # set any world properties first
        world.collaborative = True
        world.size = kwargs.get("world_size", 1)
        world.dim_c = kwargs.get("dim_c", 2)
        num_agents = kwargs.get("num_agents", 3)
        num_landmarks = kwargs.get("num_landmarks", 3)

        # add agents
        world.agents = [SkilledAgent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.size = 0.025

            # temporary skill allocations
            # agent.vision_range = np.random.uniform(1, agent.skill_points)  
            # agent.max_speed = agent.skill_points - agent.vision_range

            # vision range is how much further can agent see outside of its own area 
            min_vis, max_vis = 5*agent.size, 0.5*world.size
            vis_ratio = np.random.randint(1, agent.skill_points) / agent.skill_points
            agent.vision_range = min_vis + (max_vis - min_vis) * vis_ratio
            # acceleration, applied to scale action force  
            min_accel, max_accel = 0, 1
            if agent.accel is None:
                agent.accel = 1.0
            agent.accel *= min_accel + (max_accel - min_accel) * (1-vis_ratio)

            self.change_entity_attribute(agent, **kwargs)
        
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.075
            self.change_entity_attribute(landmark, **kwargs)

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
        return rew

    def observation(self, agent, world):
        """ agent obs with partial observation 
        """
        entity_pos = []
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            # relative position 
            e_pos = entity.state.p_pos - agent.state.p_pos
            # zero mask out entities not in agent signt 
            e_mask = 1 if np.sqrt(np.sum(np.square(e_pos))) <= agent.vision_range + entity.size else 0
            entity_pos.append(e_pos * e_mask)
            # entity colors
            entity_color.append(entity.color * e_mask)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            e_pos = other.state.p_pos - agent.state.p_pos
            e_mask = 1 if np.sqrt(np.sum(np.square(e_pos))) <= agent.vision_range + other.size else 0
            comm.append(other.state.c)
            other_pos.append(e_pos * e_mask)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

        # plus agent's own attributs 
        self_attr = np.array([agent.size, agent.vision_range, agent.accel])
        return np.concatenate([self_attr] + [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)




