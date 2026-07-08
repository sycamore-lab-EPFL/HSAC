
from matplotlib import pyplot as plt
import numpy as np
import hexsimulator.graphics as gr
def state_to_RGB(state,axis='yx-right-down'):
    occ_color = np.array([255,255,255])
    ground_color = np.array([128,128,128])
    background_color = np.array([0,0,0])
    eta_color = np.array([255,255,0])
    hold_color = np.array([64,64,255])
    image = np.ones((state['occ'].shape[0],state['occ'].shape[1]*2,3),dtype = np.uint8)
    image[:] =background_color[None,None,:]

    occ = tosquare(np.array(np.nonzero(state.get('occ'))).T)
    image[occ[:,0],occ[:,1],:]=occ_color[None,:]

    ground = tosquare(np.array(np.nonzero(state.get('grounds'))).T)
    image[ground[:,0],ground[:,1],:]=ground_color[None,:]

    eta = tosquare(np.array(np.nonzero(state.get('put_eta')>=0)).T)
    image[eta[:,0],eta[:,1],:]=eta_color[None,:]
    any_hold = state.get('hold')
    any_hold = any_hold.reshape(any_hold.shape[0],any_hold.shape[1],-1,2).sum(-2)
    hold = tosquare(np.array(np.nonzero(any_hold)).T)
    image[hold[:,0],hold[:,1],:]=hold_color[None,:]
    if 'down' in axis:
        image = image[:,-1::-1,:]
    if 'yx' in axis:
        image = np.transpose(image,(1,0,2))
    return image

def tosquare(hexarray):
    hexarray[:,1]*=2
    hexarray[:,1]+=1-hexarray[:,2]
    hexarray = hexarray[:,:2]
    return hexarray
if __name__ == '__main__':
    from hex_envs.static_timed_env import TimedDiscreteGeneric
    env = TimedDiscreteGeneric()
    
    gym_option = {'target_type':'random_gap',
                  'gap_range':[1,10]}
    obs,info = env.reset(options=gym_option)
    """image = state_to_RGB(obs)
    plt.imshow(np.transpose(image,(1,0,2)))
    plt.show()"""
    state,reward,terminal,truncated,info = env.step({'action_type':0,'block_type':0,'location':np.array([5,0])})
    state,reward,terminal,truncated,info =env.step({'action_type':1,'block_type':0,'location':np.array([5,0])})
    state,reward,terminal,truncated,info =env.step({'action_type':0,'block_type':0,'location':np.array([18,0])})  # agent policy that uses the observation and info
    """image = state_to_RGB(state)
    plt.imshow(image)
    plt.show()"""
    state,reward,terminal,truncated,info =env.step({'action_type':1,'block_type':0,'location':np.array([5,0])})
    image = state_to_RGB(state)
    plt.imshow(image)
    plt.show()
    env.sim.draw_state_debug()
    pass
