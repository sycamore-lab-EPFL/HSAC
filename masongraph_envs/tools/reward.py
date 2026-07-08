import numpy as np
def cummulative_discounted_reward(reward, gamma,terminal_dcr=0):
    cmr = np.zeros(len(reward))
    cmr[-1] = reward[-1]+gamma*terminal_dcr
    if len(reward)>1:
        for i in range(2,len(reward)+1):
            cmr[-i] = gamma*cmr[-i+1]+reward[-i]
    return cmr
def lambda_return(reward,value,gamma,lam,terminal_lamr=0):
    lamr = np.zeros(len(reward))
    lamr[-1] = gamma*terminal_lamr+reward[-1]
    if len(reward)>1:
        for i in range(2,len(reward)+1):
            lamr[-i] = gamma*lam*lamr[-i+1]+reward[-i]+gamma*(1-lam)*value[-i+1]
    return lamr