import copy
import numpy as np
import hex_envs
from simulator.hexsimulator.blocks import Block # type: ignore
from hex_envs.tools.converter import feasible_action_idxs,int2act
from hex_envs.tools.reward import cummulative_discounted_reward # type: ignore

def hindsightexperiencereplay(env):
    
    return episodeHER