import numpy as np
from gym.utils import seeding
import ipdb

from multiagent.utils import CHANGE_FN_REGISTRY as fn_registry


# defines scenario upon which the world is built
class BaseScenario(object):

    def before_make_world(self, **kwargs):
        seed = kwargs.get("seed", None)
        self.np_random, seed = seeding.np_random(seed)

    # create elements of the world
    def make_world(self, **kwargs):
        raise NotImplementedError

    # create initial conditions of the world
    def reset_world(self, world, **kwargs):
        raise NotImplementedError()

    def change_entity_attribute(self, entity, world, **kwargs):
        """ dynamically set entity properties 
        """
        ent_name, ent_idx = entity.name.split(" ")
        ent_config = ent_name + "_config"
        ent_config = kwargs.get(ent_config, None)

        if ent_config is None:
            # no customized config
            return 
        elif isinstance(ent_config, list):
            # separate config for each agent 
            change_fn = ent_config[ent_idx]["change_fn"]
            fn_registry[change_fn](entity, world, **ent_config[ent_idx])
        else:
            # shared config 
            change_fn = ent_config["change_fn"]
            fn_registry[change_fn](entity, world, **ent_config)


    def reset_world(self, world, **kwargs):
        raise NotImplementedError


    def reward(self, agent, world):
        raise NotImplementedError


    def observation(self, agent, world):
        raise NotImplementedError


    def render(self, env, mode="human"):
        """ default rendering callback function 
        - takes in env instance since rendering should belong to environment scope
        """
        self.setup_geometry(env)
        return self.exec_rendering(env, mode=mode)


    def setup_geometry(self, env):
        """ create geoms and transforms for basic agents and landmarks
        """ 
        # lazy import, in case remote has no screen
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
            alpha = 1.0 if "agent" in name else 0.5
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
                xform_pos = v["attach_ent"].state.p_pos
                v["xform"].set_translation(*xform_pos)

            # render to display or array
            results.append(
                env.viewers[i].render(return_rgb_array = mode=='rgb_array')
            )
        return results
             