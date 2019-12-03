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

        
