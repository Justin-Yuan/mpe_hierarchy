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
        world.collaborative = True
        # all entity positions are scaled/extended by size 
        world.size = kwargs.get("world_size", 1)
        world.dim_c = kwargs.get("dim_c", 10)
        num_agents = kwargs.get("num_agents", 2)
        num_landmarks = kwargs.get("num_landmarks", 3)

        # add agents
        world.agents = [SkilledAgent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = kwargs.get("agent_silence", False)
            agent.size = 0.025
            # self.change_entity_attribute(agent, world, **kwargs)

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
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
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])  

        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75,0.25,0.25]) 
        world.landmarks[1].color = np.array([0.25,0.75,0.25]) 
        world.landmarks[2].color = np.array([0.25,0.25,0.75]) 
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color                
        world.agents[1].goal_a.color = world.agents[1].goal_b.color 

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

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        # agent reward depend on distance btw its goal agent and goal landmark
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color 

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
            if other is agent: continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
            
    
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