import numpy as np
from gym.utils import seeding
import ipdb

# defines scenario upon which the world is built
class BaseScenario(object):

    def before_make_world(self, **kwargs):
        self.seeding(kwargs['seed'] if 'seed' in kwargs else None)

    # create elements of the world
    def make_world(self):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

    def seeding(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
        
    def change_entity_attribute(self, entity, **kwargs):
        # if 'agent' in entity.name and 'agent_config' in kwargs:
        #     change_fn = kwargs['agent_config']['change_fn']
        #     change_fn(entity)
        # elif 'landmard' in entity.name and 'landmard_config' in kwargs:
        #     change_fn = kwargs['landmark_config']['change_fn']
        #     change_fn(entity)
        # elif 'food' in entity.name and 'food_config' in kwargs:
        #     change_fn = kwargs['food_config']['change_fn']
        #     change_fn(entity)
        # elif 'forest' in entity.name and 'forest_config' in kwargs:
        #     change_fn = kwargs['forest_config']['change_fn']
        #     change_fn(entity)
        # else:
        #     pass
        ent_name = entity.name.split(" ")[0]
        ent_config = ent_name + "_config"
        if ent_config in kwargs:
            change_fn = kwargs[ent_config]['change_fn']
            change_fn(entity)
        
