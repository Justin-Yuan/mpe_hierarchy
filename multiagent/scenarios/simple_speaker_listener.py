import numpy as np
import pyglet
from pyglet import gl

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
        world.collaborative = True
        # all entity positions are scaled/extended by size 
        world.size = kwargs.get("world_size", 1)
        world.dim_c = kwargs.get("dim_c", 3)
        num_agents = kwargs.get("num_agents", 2)
        num_landmarks = kwargs.get("num_landmarks", 3)

        # ablation settings 
        world.use_oracle_dist = kwargs.get("use_oracle_dist", False)
        world.use_oracle_pos = kwargs.get("use_oracle_pos", False)
        world.use_oracle_speaker = kwargs.get("use_oracle_speaker", False)
        world.use_oracle_speaker_goal = kwargs.get("use_oracle_speaker_goal", False)

        # add agents
        world.agents = [SkilledAgent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
            agent.movable = True 
            agent.silent = False 
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04

        # make initial conditions
        self.reset_world(world, **kwargs)
        return world

    def reset_world(self, world, **kwargs):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])  

        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        
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
        # returns data for benchmarking purposes
        return self.reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)
        
        ############################################################
        # use oracle on distance to goal position
        if world.use_oracle_dist:
            if not agent.movable:
                return np.concatenate([goal_color])
            if agent.silent:
                a = world.agents[0]
                dist_vec = a.goal_b.state.p_pos - a.goal_a.state.p_pos
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos]+ entity_pos + dist_vec)

        # use oracle on goal position
        if world.use_oracle_pos:
            if not agent.movable:
                return np.concatenate([goal_color])
            if agent.silent:
                goal_pos = world.agents[0].goal_b.state.p_pos
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + goal_pos)

        # uses oracle message for speaker 
        if world.use_oracle_speaker:
            if not agent.movable:
                a = world.agents[0]
                dist_vec = a.goal_b.state.p_pos - a.goal_a.state.p_pos
                return np.concatenate([dist_vec])
            if agent.silent:
                goal_pos = world.agents[0].goal_b.state.p_pos
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + comm)

        # uses oracle message for speaker 
        if world.use_oracle_speaker_goal:
            if not agent.movable:
                goal_pos = world.agents[0].goal_b.state.p_pos
                return np.concatenate([goal_pos])
            if agent.silent:
                goal_pos = world.agents[0].goal_b.state.p_pos
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + comm)

        ############################################################

        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
            # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + entity_color + comm)


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

            # GOAL: highlight goal entities
            if "agent" in entity.name and entity.goal_b is not None:
                for goal in [entity.goal_a, entity.goal_b]:
                    goal_geom = rendering.make_circle(goal.size*1.5)
                    goal_geom.set_color(*goal.color, alpha=0.2)
                    goal_xform = rendering.Transform()
                    goal_geom.add_attr(goal_xform)
                    env.render_dict[goal.name+"_highlight"] = {
                        "geom": goal_geom, 
                        "xform": goal_xform, 
                        "attach_ent": goal
                    }

            # LABEL: display comm message
            if "agent" in entity.name and entity.goal_b is not None:
            # if "agent" in entity.name and entity.goal_b is None:
                x = entity.state.p_pos[0] + 50
                y = entity.state.p_pos[1] + 50
                comm_geom = rendering.Text("_", position=(x,y), font_size=36)
                comm_xform = rendering.Transform()
                comm_geom.add_attr(comm_xform)
                env.render_dict[name+"_comm"] = {
                    "geom": comm_geom, 
                    "xform": comm_xform, 
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

            
    def exec_rendering(self, env, mode="human"):
        """ lay out objects in the scene 
        """
        results = []
        for i in range(len(env.viewers)):
            # update bounds to center around agent
            cam_range = env.cam_range   # 1
            if env.shared_viewer:
                pos = np.zeros(env.world.dim_p)
            else:
                pos = env.agents[i].state.p_pos
            
            # set view centered arond agent
            # self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            env.viewers[i].set_bounds(-cam_range, cam_range, -cam_range, cam_range)
            
            # update geometry positions
            for k, v in env.render_dict.items():    
                # update comm message
                if "comm" in k:
                    ent = v["attach_ent"]
                    comm_msg = str(ent.state.c)
                    v["geom"].set_text(comm_msg)
                   
                xform_pos = v["attach_ent"].state.p_pos
                v["xform"].set_translation(*xform_pos)
               
            # render to display or array
            results.append(
                env.viewers[i].render(return_rgb_array = mode=='rgb_array')
            )
        return results
             