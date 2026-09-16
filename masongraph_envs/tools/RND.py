import copy
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hex_envs
from simulator.hexsimulator.blocks import Block # type: ignore
from hex_envs.tools.converter import feasible_action_idxs,int2act,stat_obs
from hex_envs.tools.reward import cummulative_discounted_reward
import gymnasium as gym
class SimpleConvNetBN(nn.Module):
    def __init__(self,
                 obs_space,
                 n_outputs,
                 n_channels,
                 n_internal_layer,
                 n_hidden_neurons,
                 device='cuda',
                 ):
        super().__init__()
        assert  all([isinstance(obs_space[i],gym.spaces.Box) for i in obs_space])

        self.obs_space = obs_space
        in_dim = sum([obs_space[i].shape[2] for i in obs_space if i != 'mask'])
        grid_shape = np.array(obs_space['occ'].shape[:2])
        self.device = device
        self.pad = nn.ConstantPad2d(1,-1)
        self.convin = nn.Conv2d(in_dim,n_channels,3,stride=1,device=self.device)
        self.convinternal1 = nn.ModuleList([nn.Conv2d(n_channels,n_channels,3,stride=1,device=self.device,padding='same') for i in range(n_internal_layer)])
        self.convinternal2 = nn.ModuleList([nn.Conv2d(n_channels,n_channels,3,stride=1,device=self.device,padding='same') for i in range(n_internal_layer)])
        self.bn_in = nn.BatchNorm2d(n_channels,device=self.device)
        self.bn_1 = nn.ModuleList([nn.BatchNorm2d(n_channels,device=self.device) for i in range(n_internal_layer)])
        self.bn_2 = nn.ModuleList([nn.BatchNorm2d(n_channels,device=self.device) for i in range(n_internal_layer)])
        self.bn_out = nn.BatchNorm2d(1,device=self.device)
        self.bn_val = nn.BatchNorm1d(1,device=self.device)
        self.convout = nn.Conv2d(n_channels,1,3,stride=1,device=self.device,padding='same')

        self.val_hidden = nn.Linear(grid_shape.prod(),n_hidden_neurons,device=self.device)
        self.val_out = nn.Linear(n_hidden_neurons,n_outputs,device = self.device)
        torch.nn.init.uniform(self.val_out.weight,-1,1)
    def forward(self,obs,inference=False):
        with torch.inference_mode(inference):
            inputs,mask = hex_envs.tools.converter.concat_obs(obs)
            inputs = torch.tensor(inputs, dtype = torch.float32, device=self.device)
            if torch.any(torch.isnan(inputs)):
                assert False, 'nans in the input'
            inputs= inputs.permute(0,3,1,2)
            #add a padding of -1 so the model has a way of knowing that there is a limit to the frame
            
            #inputs= inputs.permute(0,4,1,2,3)
            # rep = torch.cat([F.relu(self.convinup(inputs[...,0])),
            #                  F.relu(self.convindown(inputs[...,1]))],1)
            rep = F.relu(self.bn_in(self.convin(self.pad(inputs))))
            for conv1,conv2,bn1,bn2 in zip(self.convinternal1,self.convinternal2,self.bn_1,self.bn_2):
                rep = F.relu(bn2(conv2(F.relu(bn1(conv1(rep)))))+rep) #residual block
            
            rep_val = F.relu(self.bn_out(self.convout(rep))).flatten(1)
                    
            
            rep_val = F.relu(self.val_hidden(rep_val))
            val = self.val_out(rep_val)
            return val
class SimpleConvNet(nn.Module):
    def __init__(self,
                 obs_space,
                 n_outputs,
                 n_channels,
                 n_internal_layer,
                 n_hidden_neurons,
                 device='cuda',
                 ):
        super().__init__()
        assert  all([isinstance(obs_space[i],gym.spaces.Box) for i in obs_space])

        self.obs_space = obs_space
        in_dim = sum([obs_space[i].shape[2] for i in obs_space if i != 'mask'])
        grid_shape = np.array(obs_space['occ'].shape[:2])
        self.device = device
        #self.pad = nn.ConstantPad2d(1,-1)
        self.convin = nn.Conv2d(in_dim,n_channels,3,stride=1,device=self.device,padding='same')
        self.convinternal1 = nn.ModuleList([nn.Conv2d(n_channels,n_channels,3,stride=1,device=self.device,padding='same') for i in range(n_internal_layer)])
        self.convinternal2 = nn.ModuleList([nn.Conv2d(n_channels,n_channels,3,stride=1,device=self.device,padding='same') for i in range(n_internal_layer)])
        self.convout = nn.Conv2d(n_channels,1,3,stride=1,device=self.device,padding='same')

        self.val_hidden = torch.nn.Sequential(nn.Linear(grid_shape.prod(),n_hidden_neurons,device=self.device),
                                           nn.ReLU(),
                                           nn.Linear(n_hidden_neurons,n_hidden_neurons,device=self.device),
                                           nn.ReLU(),
                                           nn.Linear(n_hidden_neurons,n_hidden_neurons,device=self.device),)
        self.val_out = nn.Linear(n_hidden_neurons,n_outputs,device = self.device)
        torch.nn.init.uniform(self.val_out.weight,-1,1)
        self.running_var_obs=None
        self.running_mean_obs=None
    def forward(self,obs,inference=False):
        with torch.inference_mode(inference):
            inputs,mask = hex_envs.tools.converter.concat_obs(obs)
            inputs_prenorm = torch.tensor(inputs, dtype = torch.float32, device=self.device)
            if self.running_var_obs is None:
                return torch.zeros(inputs.shape[0],device=self.device)
            inputs = (inputs_prenorm-self.running_mean_obs)/torch.sqrt(self.running_var_obs+1e-9)
            inputs = torch.clip(inputs,-5,5)
            if torch.any(torch.isnan(inputs)):
                assert False, 'nans in the input'
            inputs= inputs.permute(0,3,1,2)
            #add a padding of -1 so the model has a way of knowing that there is a limit to the frame
            
            #inputs= inputs.permute(0,4,1,2,3)
            # rep = torch.cat([F.relu(self.convinup(inputs[...,0])),
            #                  F.relu(self.convindown(inputs[...,1]))],1)
            rep = F.relu(self.convin(inputs))
            for conv1,conv2 in zip(self.convinternal1,self.convinternal2):
                rep = F.relu(conv2(F.relu(conv1(rep))))+rep #residual block
            
            rep_val = F.relu(self.convout(rep)).flatten(1)
                    
            
            rep_val = F.relu(self.val_hidden(rep_val))
            val = self.val_out(rep_val)
            return val
class RNDModule():
    def __init__(self,obs_space,running_method:Literal['exp','window']='exp',alpha=0.01,gamma = 0.9) -> None:
        self.device = 'cuda'
        self.random_net = SimpleConvNet(obs_space,1,10,20,10,self.device)
        self.estimator =  SimpleConvNet(obs_space,1,10,20,10,self.device)
        self.optimizer = torch.optim.Adam(self.estimator.parameters(),lr=1e-4)
        self.eps = 1e-3
        self.gamma = gamma
        self.obs_var = 0
        self.obs_mean = 0
        if running_method == 'exp':        
            self.running_var = torch.zeros(1,device=self.device)
            self.running_mean = torch.zeros(1,device=self.device)
            self.alpha = alpha
            self.init = True
        else:
            raise NotImplementedError
    def compute_intrinsic_r_states(self,states):
        self.random_net.eval()
        self.estimator.eval()
        r_rnd = self.random_net(states,inference=True)
        r_est= self.estimator(states,inference=True)
        loss = F.mse_loss(r_est,r_rnd,reduction='none')
        if not self.init:
            normalized_loss = loss/torch.sqrt(self.running_var+self.eps)
            self.random_net.train()
            self.estimator.train()
            return normalized_loss.detach()
        else:
            self.random_net.train()
            self.estimator.train()
            return loss.detach()
    def compute_intrinsic_r(self,episode,new_init_state,n_steps = 1,next_cdr_est=None):
        states = [trans['ns'] for trans in episode[:-1]] +[new_init_state]
        effective_alpha = 1-np.power(1-self.alpha,len(episode))
        if self.init:
            self.obs_mean,self.obs_var = stat_obs(states,self.obs_mean,self.obs_var,1)
        else:
            self.obs_mean,self.obs_var = stat_obs(states,self.obs_mean,self.obs_var,1)
        self.random_net.running_mean_obs = torch.tensor(self.obs_mean,device=self.device,dtype = torch.float32)
        self.estimator.running_mean_obs = torch.tensor(self.obs_mean,dtype = torch.float32,device=self.device)

        self.random_net.running_var_obs = torch.tensor(self.obs_var,dtype = torch.float32,device=self.device)
        self.estimator.running_var_obs = torch.tensor(self.obs_var,dtype = torch.float32,device=self.device)

        val_rnd = self.random_net(states)
        for i in range(n_steps):
            self.optimizer.zero_grad()
            val_est= self.estimator(states)
            
            loss = F.mse_loss(val_est,val_rnd,reduction='none')
            if i == 0:
                #update the normalization statistics
                
                if self.init:
                    self.running_mean = loss.mean()
                    self.running_var = loss.var()
                    self.init = False
                else:
                    self.running_mean = effective_alpha * loss.mean() + self.running_mean*(1-effective_alpha)
                    self.running_var = effective_alpha * torch.square(loss - self.running_mean).mean()  + self.running_var*(1-effective_alpha)
                #compute the intrinsic reward and the discounted intrinc reward 
                normalized_loss = (loss)/torch.sqrt(self.running_var+self.eps)

                dcr_i = cummulative_discounted_reward(normalized_loss.detach().cpu().numpy(),self.gamma,
                                                      terminal_dcr= next_cdr_est or (self.running_mean/(self.eps+(1-self.gamma)*torch.sqrt(self.running_var))).detach().cpu().numpy())
                assert not np.isnan(dcr_i).any()
                #dcr_i *= (1-self.gamma) #normalize the magnitude of the reward, otherwise a change in gamma has multiple effects
                for step in range(len(episode)):
                    episode[step]['ri']=normalized_loss[step]
                    episode[step]['dcri']=dcr_i[step]
            loss.mean().backward()
            self.optimizer.step()
        return episode, loss, self.running_mean.detach().cpu().numpy(),self.running_var.detach().cpu().numpy()

if __name__ == '__main__':
    env = gym.make("static-env2d/generic-v0",render_mode = None)
    rnd = RNDModule(env.observation_space)
    
    state,*_ = env.reset()
    states = [{'ns':state} for i in range(10)]
    for i in range(1000):
        episode, loss, running_mean,running_var =rnd.compute_intrinsic_r(states,state)
        print(f'{episode[0]["ri"]=}')
