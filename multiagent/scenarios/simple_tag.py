import numpy as np
from multiagent.core import World, SkilledAgent, Landmark
from multiagent.scenario import BaseScenario
from multiagent.utils import bound_reward


class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        self.before_make_world(**kwargs)
        
        world = World()
        world.np_random = self.np_random
        # cache kwargs in case needed in Env wrapper 
        world.config = kwargs
        # set any world properties first
        world.collaborative = False
        # all entity positions are scaled/extended by size 
        world.size = kwargs.get("world_size", 1)
        world.dim_c = kwargs.get("dim_c", 2)
        # other configs 
        world.shape_rewards = kwargs.get("shape_rewards", False)
        world.per_adv_rewards = kwargs.get("per_adv_rewards", False)

        num_good_agents = kwargs.get("num_good_agents", 1)
        num_adversaries = kwargs.get("num_adversaries", 3)
        num_agents = num_adversaries + num_good_agents
        num_landmarks = kwargs.get("num_landmarks", 2)

        # add agents
        world.agents = [SkilledAgent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = kwargs.get("agent_silence", True)
            agent.adversary = True if i < num_adversaries else False
            # agent.size = 0.075 if agent.adversary else 0.05
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3

            self.change_entity_attribute(agent, world, **kwargs)
        
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.075
            landmark.boundary = False
            self.change_entity_attribute(landmark, world, **kwargs)

        # make initial conditions
        self.reset_world(world, **kwargs)
        return world

    def reset_world(self, world, **kwargs):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])

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
            if not landmark.boundary:
                pos_min, pos_max = kwargs.get("landmark_pos_init", [0,-1,1])[1:]
                landmark.state.p_pos = self.np_random.uniform(0.9*pos_min, 0.9*pos_max, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        # shape = False
        adversaries = self.adversaries(world)
        # if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
        if world.shape_rewards:
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10
                    # rew -= 1

        # # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        rew += bound_reward(agent, world)
        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        # shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
        if not world.per_adv_rewards:
            if world.shape_rewards:
                for adv in adversaries:
                    rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
            if agent.collide:
                for ag in agents:
                    for adv in adversaries:
                        if self.is_collision(ag, adv):
                            rew += 10
                            # rew += 1
        else:
            if world.shape_rewards:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents])
            if agent.collide:
                for ag in agents:
                    if self.is_collision(ag, agent):
                        rew += 10
        return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)


    def setup_geometry(self, env):
        """ create geoms and transforms for basic agents and landmarks
        """ 
        # lazy import 
        from multiagent import rendering

        if getattr(env, "render_dict", None) is not None:
            return 
        env.render_dict = {}

        # make geometries and transforms
        for entity in env.world.entities:
            name = entity.name
            geom = rendering.make_circle(entity.size)
            xform = rendering.Transform()

            # agent on top, other entity to background 
            alpha = 0.6 if "agent" in name else 0.5
            geom.set_color(*entity.color, alpha=alpha)   

            geom.add_attr(xform)
            env.render_dict[name] = {
                "geom": geom, 
                "xform": xform, 
                "attach_ent": entity
            }

            # VIS: show visual range/receptor field
            if 'agent' in entity.name and env.show_visual_range:
                vis_geom = rendering.make_circle(entity.vision_range)
                vis_geom.set_color(*entity.color, alpha=0.2)
                vis_xform = rendering.Transform()
                vis_geom.add_attr(vis_xform)
                env.render_dict[name+"_vis"] = {
                    "geom": vis_geom, 
                    "xform": vis_xform, 
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
            env.render_dict[name+"_label"] = {
                "geom": label_geom, 
                "xform": label_xform, 
                "attach_ent": entity
            }
                    
        
        # add geoms to viewer
        for viewer in env.viewers:
            viewer.geoms = []
            for k, d in env.render_dict.items():
                viewer.add_geom(d["geom"])


