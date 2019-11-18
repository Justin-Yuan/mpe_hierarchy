#!/usr/bin/env python
import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import yaml 

import cv2 
from cv2 import VideoWriter
import numpy as np 
from pdb import set_trace

from multiagent.policy import Policy
from multiagent.core import idx2entity
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

##################################################################################

class OracleMine(Policy):
    def __init__(self, env, agent_index):
        super(OracleMine, self).__init__()
        self.env = env 

    def action(self, obs):
        closest_mine, closest_dist  = None, None  
        
        for state in obs["states"]:
            type_idx = state[0]
            if idx2entity[type_idx] == "Mine":
                e_pos = state[1:1+self.env.world.dim_p]
                e_dist = np.sqrt(np.sum(np.square(e_pos)))
                if closest_mine is None or e_dist  < closest_dist:
                    closest_mine = e_pos
                    closest_dist = e_dist

        u = closest_mine / closest_dist
        # return np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return u 


class OracleHunt(Policy):
    def __init__(self, env, agent_index):
        super(OracleHunt, self).__init__()
        self.env = env 
        self.agent = env.world.agents[agent_index]
        
    def action(self, obs):
        closest_target, closest_dist  = None, None  

        for agent in self.env.world.agents:
            if agent is self.agent:
                continue 
            e_pos = agent.state.p_pos - self.agent.state.p_pos 
            e_dist = np.sqrt(np.sum(np.square(e_pos)))
            # set chasing/evading direction 
            if self.agent.adversary and not agent.adversary:
                if closest_target is None or e_dist < closest_dist:
                    closest_target = e_pos
                    closest_dist = e_dist
            elif not self.agent.adversary and agent.adversary:
                if closest_target is None:
                    closest_target = np.zeros_like(e_pos)
                closest_target += -e_pos / e_dist

        if self.agent.adversary:
            u = closest_target / closest_dist
        else:
            u = closest_target / np.sqrt(np.sum(np.square(closest_target)))
            pos_norm = np.sqrt(np.sum(np.square(self.agent.state.p_pos)))
            u += -self.agent.state.p_pos * 0.2
        # return np.concatenate([u, np.zeros(self.env.world.dim_c)]) 
        return u 



fourcc = cv2.VideoWriter_fourcc(*'XVID')

def save_video(frames, fps=20, output="out.avi"):
    assert len(frames) > 0
    height, width, _ = frames[0].shape
    writer = VideoWriter(output, fourcc, fps, (width, height), True)
    for f in frames:
        writer.write(f)
    writer.release()

def savegif(arrs, output="demo.gif"):
    import imageio
    imageio.mimsave(output, arrs)


def load_config(config_file):
    if len(config_file) < 1 or not os.path.exists(config_file):
        return {}
    with open(config_file, "r") as f:
        config = yaml.load(f)
    return config 



##################################################################################

def main(args, **kwargs):
    """ entry function 
    """
    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world(**kwargs)
    # create multiagent environment
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    world_update_callback = getattr(scenario, "update_world", None)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, 
        info_callback=None, shared_viewer=args.share_viewer, 
        update_callback=world_update_callback, show_visual_range=True, cam_range=args.cam_range)
    # render call to create viewer window (necessary only for interactive policies)

    game = args.game
    if game == "mine": 
        p_builder = OracleMine 
        env.discrete_action_space = False 
    elif game == "hunt":
        p_builder = OracleHunt
        env.discrete_action_space = False 
    else:
        p_builder = InteractivePolicy

    frames = []
    frame = env.render(mode="rgb_array")[0]
    frames.append(frame)
    # create interactive policies for each agent
    policies = [p_builder(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    count = 0 
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        frame = env.render(mode="rgb_array")[0]
        frames.append(frame)
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
        count += 1
        print(count)
        print(reward_n)
        if args.length > 0 and count > args.length: break 

    # save demo video
    # save_video(frames, output=args.output) 
    savegif(frames, output=args.output) 


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    parser.add_argument('-g', '--game', default='interactive', help='type of policy to try')
    parser.add_argument('-o', '--output', default="demo.gif", help='output name of demo video')
    parser.add_argument('-l', '--length', default=100, type=int, help='length of demo video')
    parser.add_argument('--share_viewer', action='store_true', default=False, help="if to share rendering viewer")
    parser.add_argument('-c', '--config', default="", help='Path of the environment config file')
    parser.add_argument('--cam_range', default=10, type=int, help='viewer size when rendering')
    args = parser.parse_args()
    # run main 
    config = load_config(args.config)
    main(args, **config)



##################################################################################
# # test 
# import mpe
# import time
# import numpy as np


# def savegif(arrs):
#     import imageio
#     imageio.mimsave('test.gif', arrs)


# def agent_fn(entity):
#     entity.initial_mass = 100.0
    

# kwargs = {
#     'seed': 123, 
#     'agent_config': {'change_fn': agent_fn}
# }
# env = mpe.make_env("simple_spread", **kwargs)

# rgb_arrs = []
# for i in range(100):
#     act = np.zeros(5)
#     act[3] = 1
#     env.step([act for i in range(3)])
#     rgb_arr = env.render(mode='rgb_array')
#     rgb_arrs.append(rgb_arr[0])


# savegif(rgb_arrs)

