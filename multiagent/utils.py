import numpy as np 
from copy import deepcopy

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
    min_accel, max_accel = 3, 8
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

    min_accel, max_accel = 3, 8
    if agent.accel is None:
        agent.accel = 1.0
    agent.accel *= min_accel + (max_accel - min_accel) * (1-vis_ratio)


@register_change_fn("agent_set_vision_global")
def set_agent_vision_global(agent, world, vis_ratios=[0.2, 0.5, 0.8], **kwargs):
    """ fixed skill sets defined in env config 
    """
    idx = int(agent.name.split(" ")[-1])
    idx = idx % len(vis_ratios)
    vis_ratio = vis_ratios[idx]

    min_vis, max_vis = 5*agent.size, world.size
    agent.vision_range = min_vis + (max_vis - min_vis) * vis_ratio



##########################################################################################
####################################     reward shaping   ##################################
##########################################################################################


def bound(x):
    x = x 
    if x < 0.9:
        return 0
    if x < 1.0:
        return (x - 0.9) * 10
    return min(np.exp(2 * x - 2), 10)


def bound_reward(agent, world):
    """ location bound reward, agent is discouraged to go out of world bound 
    agents are penalized for exiting the screen, so that they can be caught by the adversaries
    """
    rew = 0 
    for p in range(world.dim_p):
        x = abs(agent.state.p_pos[p])
        rew -= bound(x / world.size)    # normalize
    return rew 


def influence_reward(agent, world):
    """ influence-based reward, directional
    reference: https://arxiv.org/pdf/1810.08647.pdf
    """
    rew = 0 
    return rew 


def mutual_information_reward(agetn, world):
    """ reward based on agent action/message mutual information 
    """
    rew = 0 
    return rew 


def hierarchy_reward(agent, world):
    """ reward to encourage emergence of agent hierarchy 
    """
    rew = 0 
    return rew


##########################################################################################
####################################     Environemtn Utils   ##################################
##########################################################################################


def mask_agent_obs(vecs, agent, entity, world):
    """ filter agent observation (list of vecs) by distance metric 
    """
    if not world.use_mask:
        return vecs 
    new_vecs = deepcopy(vecs)
    dist = np.linalg.norm(entity.state.p_pos - agent.state.p_pos)
    if dist <= agent.vision_range + entity.size:
        new_vecs = [np.ones_like(v) * world.mask_value for v in new_vecs]
    return new_vecs
    

##########################################################################################
####################################     observation change fn   ##################################
##########################################################################################

class ObsDecorator(object):
    """ contain different wrapper on normal obs_fn 
        (that returns current time step obs)
    """
    def __init__(self, stacked_obs_num=1, **kwargs):
        self.stacked_obs_num = stacked_obs_num

    def obs_stacker(self, obs_fn):
        """ stack obs from multiple time steps 
        """
        def wrapped_f(*args):
            obs = obs_fn(*args) # obs at current time step, (O,)
            agent = args[1]
            # only done initially 
            if not hasattr(agent, "obs_list"):
                agent.obs_list = [obs] * self.self.stacked_obs_num
            # push current obserevation in 
            agent.obs_list.pop(0)
            agent.obs_list.append(obs)  # [(O,)]*k
            # return stacked obs (concatenated)
            return np.concatenate(agent.obs_list, -1)   # (O*k,)
        return wrapped_f

        