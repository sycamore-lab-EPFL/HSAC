import pickle
from time import perf_counter
from warnings import warn

from matplotlib import pyplot as plt
from .her_buffer import HERReplayBufferGNN
from .prefilled_buffer import PrefilledReplayBufferGNN
from sac.policy_edges import TransformerGFTFPolicy
from sac.valuefunction_edges import TransformerGFTFValue
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch_geometric
from .qfunctions_edges import TransformerQ
from .buffer import ReplayBufferGNN
from time import perf_counter
import wandb

from .helper import *
from .agent import SAC


class SACvec(SAC):
    def __init__(self, state_metadata, action_metadata, config, attach='central'):
        state_metadata = (['state_vec'],[])
        super().__init__(state_metadata, action_metadata, config, attach=attach)
    def init_policy(self, state_metadata, action_metadata, config):
        if hasattr(config,'prefilled_buffer') and config.prefilled_buffer:
            raise NotImplementedError("Prefilled buffer with vector not implemented")
        elif self.her:
            raise NotImplementedError("HER with vector not implemented")
        else:
            self.buffer = ReplayBufferGNN(config.n_batch,config.her,max_blocks= config.max_blocks if hasattr(config,'max_blocks') else 0,
                                        max_size = config.buffer_size,
                                        training_ratio = config.training_ratio if hasattr(config, 'training_ratio') else 1,
                                        prioritized = config.prioritized,
                                        alpha = config.prioritized_alpha,
                                        anneal_steps = config.prioritized_anneal_steps,
                                        replace = config.prioritized_replace,
                                        n_actions_per_block= action_metadata['discrete'][0],
                                        base_state=self.bs,)

        self.policy = PolicyVec(state_metadata, action_metadata, config)
        self.qfunction = [QFunctionVec(state_metadata, action_metadata, config) for _ in range(2)]
        self.vfunction = ValueVec(state_metadata, config)
        self.target_vfunction = ValueVec(state_metadata, config)
        self.optimizer_pol_params = {'params':  list(self.vfunction.parameters()) +
                                        list(self.policy.parameters()) +
                                         [self.log_entropy_weightc,self.log_entropy_weightd],
                                 'lr': config.learning_rate,'weight_decay':config.weight_decay, 'betas':config.betas}
        self.optimizer_q_params = {'params':list(self.qfunction[0].parameters()) +
                                            list(self.qfunction[1].parameters()),
                                              'lr': config.learning_rate*config.vf_coef, 'weight_decay':config.weight_decay, 'betas':config.betas}
        self.optimizer_pol = torch.optim.Adam([self.optimizer_pol_params])
        self.optimizer_q = torch.optim.Adam([self.optimizer_q_params])
        if hasattr(config,"lr_milestones"):
            self.scheduler = [torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pol,milestones= config.lr_milestones),
                              torch.optim.lr_scheduler.MultiStepLR(self.optimizer_q,milestones= config.lr_milestones)]
        else:
            self.scheduler =None
        if hasattr(config,'prefilled_buffer') and config.prefilled_buffer:
            self.prefill = True
            self.behaviour_cloning(batch_size=config.policy_update_batch_size,update_iter=config.n_updates_prefill)
        self.prefill = False
    def save(self, checkpoint_path,print_info=None):
        if print_info is None:
            print_info = f"Untested policy; Accuracy on the prioritized replay buffer: {self.saved_accuracy}"
        d =  {'config':wandb.config.as_dict(),
              'pol_state_dict':self.policy.state_dict(),
              'qf0_state_dict':self.qfunction[0].state_dict(),
              'qf1_state_dict':self.qfunction[1].state_dict(),
              'vf_state_dict':self.vfunction.state_dict(),
              'print_info': print_info}
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
class PolicyVec(nn.Module):
    def __init__(self, state_metadata,action_metadata, config):
        super().__init__()
        n_neurons_full = config.hidden_channels
        n_layers = config.n_layers
        sp = config.state_dim
        self.action_metadata = action_metadata
        self.n_ad = action_metadata['discrete'][0]
        ap = action_metadata['continuous']
        self.layers = nn.ModuleList([nn.Linear(sp if i==0 else n_neurons_full, n_neurons_full,device=device) for i in range(n_layers)])
        self.mean_layer = nn.Linear(n_neurons_full, ap*self.n_ad,device=device)
        self.std_layer = nn.Linear(n_neurons_full, ap*self.n_ad,device=device)
        #self.std_layer = nn.Linear(n_neurons_full, 1,device=device)

        self.d_layer = nn.Linear(n_neurons_full, self.n_ad,device=device)
        self.deterministic = False
        self.independent = True
    def forward(self, state):
        x = state.x_dict['state_vector']
        for l in self.layers:
            x = F.gelu(l(x))
        mean = self.mean_layer(x)
        std = F.softplus(self.std_layer(x)) + 1e-5
        dists_unbounded = torch.distributions.Normal(mean.reshape(mean.shape[0],self.n_ad,-1), std.reshape(mean.shape[0],self.n_ad,-1))
        #dists_unbounded = torch.distributions.Normal(mean.reshape(mean.shape[0],self.n_modes,-1), std.reshape(mean.shape[0],1,1))
        dists = torch.distributions.TransformedDistribution(dists_unbounded, torch.distributions.transforms.TanhTransform())
        logits = self.d_layer(x)
        #logits = torch.zeros((x.shape[0], self.n_modes),device=device)
        select_dist = torch.distributions.Categorical(logits=logits)
        #select_dist = torch.distributions.Categorical(logits=torch.ones((x.shape[0], 1),device=device))
        return select_dist, dists,dists_unbounded
    def act(self,states,total_prob=True,return_var=False):
        select_dist,dists,dists_unbounded = self.forward(states)
        if self.deterministic:
            actions = torch.argmax(select_dist.probs, dim=-1).reshape(-1)
            selected_action_params = F.tanh(dists_unbounded.loc[torch.arange(actions.shape[0]),actions])
        else:
            actions = select_dist.sample()
            action_params = dists.sample()
            selected_action_params = action_params[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
        if return_var:
            var = dists_unbounded.variance[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
            return (actions,selected_action_params), var
        return (actions,selected_action_params)
class ValueVec(nn.Module):
    def __init__(self, state_metadata, config):
        super().__init__()
        n_neurons_full = config.hidden_channels
        n_layers = config.n_layers
        sp = config.state_dim
        self.layers = nn.ModuleList([nn.Linear(sp if i==0 else n_neurons_full, n_neurons_full,device=device) for i in range(n_layers)])
        self.out_layer = nn.Linear(n_neurons_full, 1,device=device)
    def forward(self, state_vec):
        x = state_vec.x_dict['state_vector']
        for l in self.layers:
            x = F.gelu(l(x))
        V = self.out_layer(x)
        return V
class QFunctionVec(nn.Module):
    def __init__(self, state_metadata, action_metadata, config):
        super().__init__()
        n_neurons_full = config.hidden_channels
        self.n_ad = config.n_modes**action_metadata['continuous']
        n_layers = config.n_layers
        sp = config.state_dim
        ap = action_metadata['continuous']
        self.bounded_value = False
        self.layers = nn.ModuleList([nn.Linear(sp+ap*self.n_ad if i==0 else n_neurons_full, n_neurons_full,device=device) for i in range(n_layers)])
        self.out_layer = nn.Linear(n_neurons_full, 1,device=device)
    def forward(self, state,action=None):
        if action is not None:
            # modify inputs to include action
            if action[1].dim()==2:
                x = state.x_dict['state_vector']
                a = action[1]
                a_all = torch.zeros((x.shape[0], self.n_ad,a.shape[1]),device=device)
                a_all[torch.arange(x.shape[0]),action[0],:] = a
                xa = torch.cat([x,a_all.flatten(1)],axis=1)
            elif action[1].dim()==3:
                x =state.x_dict['state_vector'].repeat_interleave(self.n_ad,0)

                a_all = torch.zeros((action[1].shape[0]*self.n_ad, self.n_ad,action[1].shape[2]),device=device)

                a_all[torch.arange(x.shape[0],device=device),torch. arange(self.n_ad).tile(action[1].shape[0])] = action[1].reshape(-1,action[1].shape[2])
                xa = torch.cat([x,a_all.flatten(1)],dim=-1)
            elif action[1].dim()==4:
                assert action[0] is None, "cannot provide action index when action is given as a full tensor"
                x = state.x_dict['state_vector'].tile(action[1].shape[1],action[1].shape[2],1,1).permute(2,0,1,3).reshape(-1, x.shape[-1])
                a = action[1].reshape(-1, action[1].shape[-1])
                xa = torch.cat([x,a],dim=-1)
        else:
            x = state.x_dict['state_vector']
            
            a = state.x_dict['action_continuous']
            a_all = torch.zeros((x.shape[0], self.n_ad,a.shape[1]),device=device)
            a_all[torch.arange(x.shape[0],device=device),state.x_dict['actiond_long'],:] = a
            xa = torch.cat([x,a_all.flatten(1)],axis=1)
        for l in self.layers:
            xa = F.gelu(l(xa))
        Q = self.out_layer(xa)
        if action is not None:
            #if action[1].dim()==2:
                #Q = Q[torch.arange(Q.shape[0],device=device),action[0]]
            if action[1].dim()==3:
                Q = Q.reshape(state.num_graphs, self.n_ad,1)
            elif action[1].dim()==4:
                Q = Q.reshape(state.num_graphs, action[1].shape[1], self.n_ad,1)
        #else:
        #    Q = Q[torch.arange(Q.shape[0],device=device),state.x_dict['actiond_long'],None]
        return Q