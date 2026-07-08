import gymnasium as gym
import numpy as np
import torch

def tuple2dict(action,n_robots,n_block_types,max_blocks,vertical=False):
    """
    Convert a tuple action to a dictionary action.
    """
    action_dict = {}

    discrete_action = action[0]
    action_dict['support_block'],action_dict['robot_id'],action_dict['block_type'] = np.unravel_index(discrete_action.cpu(),(max_blocks,n_robots,n_block_types))
    
    cont_action = action[1].cpu().numpy()
  
    action_dict['center_offset'] = cont_action[:,:3]
    if not vertical:
        action_dict['direction'] = cont_action[:,3:6]
    else:
        action_dict['direction'] = np.zeros((cont_action.shape[0],3))
        action_dict['direction'][:,-1]=-1
    action_dict['rotation_offset'] = cont_action[:,-6:]
    return action_dict
def tuple2dict_flat(action,n_robots,n_block_types,max_blocks,vertical=False,normalized=False,action_boundary=None):
    """
    Convert a tuple action to a dictionary action.
    """
    action_dict = {}

    discrete_action = action[0]

    action_dict['support_block'],action_dict['robot_id'],action_dict['block_type'] = np.unravel_index(discrete_action.cpu(),(max_blocks,n_robots,n_block_types))
    
    cont_action = action[1].detach().float().cpu().numpy()
    if normalized:
        cont_action = (cont_action+action_boundary.mean(-1))*(action_boundary[:,1] - action_boundary[:,0])/2
    if not vertical:
        action_dict['center_offset'] = cont_action[:3]
        action_dict['direction'] = cont_action[3:6]
    else:
        action_dict['center_offset'] = np.zeros((discrete_action.shape[0],3))
        action_dict['center_offset'][:,:2] = cont_action[:,:2]
        action_dict['direction'] = np.zeros((discrete_action.shape[0],3))
        action_dict['direction'][:,-1]=-1
    action_dict['rotation_offset'] = cont_action[:,-2:]
    return action_dict
def tuple2dict_global(action,n_block_types,vertical=True,normalized=False,action_boundary=None):
    """
    Convert a tuple action to a dictionary action.
    """
    action_dict = {}

    discrete_action = action[0]

    action_dict['block_type'] = discrete_action.cpu()
    
    cont_action = action[1].detach().cpu().numpy()
    if normalized:
        cont_action = (cont_action+action_boundary.mean(-1))*(action_boundary[:,1] - action_boundary[:,0])/2
    if not vertical:
        action_dict['pos'] = cont_action[:3]
        action_dict['direction'] = cont_action[3:6]
        
    else:
        action_dict['pos'] = np.zeros((discrete_action.shape[0],3))
        action_dict['pos'][:,:2] = cont_action[:,:2]
        action_dict['pos'][:,-1] = action_boundary[0,1]
        action_dict['direction'] = np.zeros((discrete_action.shape[0],3))
        action_dict['direction'][:,-1]=-1
    action_dict['rotation_offset'] = cont_action[:,-2:]
    return action_dict
def tuple2dict_col(action,n_block_types,max_blocks,perpendicular=False,normalized=False,action_boundary=None):
    """
    Convert a tuple action to a dictionary action.
    """
    action_dict = {}

    discrete_action = action[0]

    action_dict['block_type'] = np.unravel_index(discrete_action.cpu(),(n_block_types))
    
    cont_action = action[1].detach().cpu().numpy()
    if normalized:
        cont_action = (cont_action+action_boundary.mean(-1))*(action_boundary[:,1] - action_boundary[:,0])/2
    if not perpendicular:
        action_dict['center_offset'] = cont_action[:3]
        action_dict['direction'] = cont_action[3:6]
        action_dict['rotation_angle'] = np.atan2(cont_action[:,-2],cont_action[:,-1])
    else:
        action_dict['center_offset'] = np.zeros((discrete_action.shape[0],3))
        action_dict['center_offset'][:,:2] = cont_action[:,:2]
        action_dict['rotation_angle'] = np.arctan2(cont_action[:,-2],cont_action[:,-1])
    return action_dict
def dict2tuple_flat(action_dict,n_robots,n_block_types,max_blocks,vertical =True,normalized = False,action_boundary=None,device='cpu',floatType=torch.float32,intType=torch.int32):
    """
    Convert a dictionary action to a tuple action.
    """
    discrete_action = action_dict['support_block']*n_robots*n_block_types + action_dict['robot_id']*n_block_types + action_dict['block_type']
    discrete_action = torch.tensor(discrete_action,device=device,dtype=intType).view(-1,1)
    if not vertical:
        cont_action = np.hstack([action_dict['center_offset'],action_dict['direction'],action_dict['rotation_offset']])
    else:
        cont_action = np.hstack([action_dict['center_offset'],action_dict['rotation_offset']])
    cont_action = torch.tensor(cont_action,device=device,dtype=floatType)
    if normalized:
        #even if normalized: dont modify the rotation angle
        cont_action[:,:-2] = (cont_action[:,:-2] - action_boundary.mean(-1)[:-2])*2/(action_boundary[:-2,1] - action_boundary[:-2,0])
    return (discrete_action,cont_action)