import numpy as np 



##########################################################################################
####################################     Entity change fn   ##################################
##########################################################################################

CHANGE_FN_REGISTRY = {}

def register_change_fn(name):
    """ register function with argument name """
    def wrap(f):
        CHANGE_FN_REGISTRY[name] = f
        def wrapped_f(*args, **kwargs):
            f(*args, **kwargs)
        return wrapped_f
    return wrap 


@register_change_fn("agent_set_skills_local")
def set_agent_skills_local(agent, world, **kwargs):
    """ sample new skill sets per episode 
    """
    # vision range is how much further can agent see outside of its own area 
    min_vis, max_vis = 5*agent.size, 0.5*world.size
    vis_ratio = np.random.randint(1, agent.skill_points) / agent.skill_points
    agent.vision_range = min_vis + (max_vis - min_vis) * vis_ratio
    # acceleration, applied to scale action force  
    min_accel, max_accel = 0, 1
    if agent.accel is None:
        agent.accel = 1.0
    agent.accel *= min_accel + (max_accel - min_accel) * (1-vis_ratio)


@register_change_fn("agent_set_skills_global")
def set_agent_skills_global(agent, world, vis_ratios=[0.2, 0.5, 0.8], **kwargs):
    """ fixed skill sets defined in env config 
    """
    idx = int(agent.name.split(" ")[-1])
    idx = idx % len(vis_ratios)
    vis_ratio = vis_ratios[idx]

    min_vis, max_vis = 5*agent.size, 0.5*world.size
    agent.vision_range = min_vis + (max_vis - min_vis) * vis_ratio

    min_accel, max_accel = 0, 1
    if agent.accel is None:
        agent.accel = 1.0
    agent.accel *= min_accel + (max_accel - min_accel) * (1-vis_ratio)

