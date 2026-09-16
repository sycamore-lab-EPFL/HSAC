import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric
from torch_geometric.nn import GAT, to_hetero
from torch_geometric.nn import aggr,LayerNorm
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from torch_geometric.nn import TransformerConv
from torch.nn import GELU
import wandb
from .helper import device, floatType, intType
import numpy as np
class TransformerConvEdge(nn.Module):
    def __init__(self, n_layers, n_channels, heads=1, dropout=0.0,layer_norm = False,layer_norm_mode='node', **kwargs):
        super(TransformerConvEdge, self).__init__()
        #self.convs = nn.ModuleList([TransformerConv(-1, n_channels, edge_dim=-1, heads=heads,beta=False, dropout=dropout,flow ='target_to_source', **kwargs) for _ in range(n_layers)])
        self.convs = nn.ModuleList([TransformerConv(-1, n_channels, edge_dim=-1, heads=heads,beta=True, dropout=dropout, **kwargs) for _ in range(n_layers)])
        
        if layer_norm:
            self.lns = nn.ModuleList([LayerNorm(n_channels,affine=True,mode=layer_norm_mode) for _ in range(n_layers)])
        else:
            self.lns = [[] for i in range(n_layers)]
        self.layer_norm = layer_norm
    def forward(self, x, edge_index, edge_attr,batch):
        for conv,ln in zip(self.convs,self.lns):
            x = conv(x, edge_index, edge_attr)
            x = F.gelu(x)
            if self.layer_norm:
                x = ln(x,batch)
        return x
class TransformerGFTFSharedEncoder(nn.Module):
    def __init__(self, state_metadata,action_metadata,config,attach='block'):
        super().__init__()
        n_layers_full = 2
        n_neurons_full = config.hidden_channels
        #n_inpart = 1
        self.an = attach
        self.action_scale = torch.tensor((action_metadata['boundary'][:,1]-action_metadata['boundary'][:,0])/2,dtype=floatType,device=device).reshape(1,1,-1)
        self.action_mean =  torch.tensor(action_metadata['boundary'].mean(-1),dtype=floatType,device=device).reshape(1,1,-1)
        #self.action_mean[0,0,-6:] = torch.tensor([1,0,0,0,1,0],device=device,dtype=floatType) #set rotation to identity
        self.var = torch.tensor(config.exploration_noise,dtype=floatType,device=device).reshape(1,1,-1)*self.action_scale #torch.tensor(action_metadata['variance'],dtype=floatType,device=device).reshape(1,1,-1)
        self.deterministic = False
        self.action_metadata = action_metadata
        #self.embedding_parts = torch.nn.Linear(n_inpart,config.hidden_channels//2,device=device,bias=False)

        #self.embedding_state = torch.nn.Embedding(4,config.hidden_channels//2,device=device)

        #self.bnin = torch.nn.BatchNorm1d(config['n_neurons'],device=self.device)

        self.convs = TransformerConvEdge(config.n_layers,config.hidden_channels,heads=1,dropout=config.dropout
                                                    )
        self.convs= torch_geometric.nn.to_hetero(self.convs,state_metadata,aggr='mean').to(device)
        self.convActorDiscrete = TransformerConvEdge(1,config.hidden_channels,heads=1,dropout=config.dropout
                                                    )
        self.convActorContinuous = TransformerConvEdge(1,config.hidden_channels,heads=1,dropout=config.dropout
                                                    )
        self.convActorDiscrete= torch_geometric.nn.to_hetero(self.convActorDiscrete,state_metadata,aggr='mean').to(device)
        self.convActorContinuous= torch_geometric.nn.to_hetero(self.convActorContinuous,state_metadata,aggr='mean').to(device)
        k=len(state_metadata[0])
        self.aggr = aggr.MultiAggregation([aggr.MaxAggregation(),aggr.MinAggregation(),aggr.MeanAggregation()])#aggr.MedianAggregation().to(self.device)
        #self.bn_val = torch.nn.LayerNorm(config.hidden_channels*k*3,elementwise_affine=True,device=device)
        self.innet = torch.nn.Linear(3*k*config.hidden_channels,n_neurons_full,device=device)
        self.full_net = torch.nn.ModuleList([nn.Linear(n_neurons_full,n_neurons_full,device = device) for i in range(n_layers_full)])
        self.outnet = torch.nn.Linear(n_neurons_full,1,device=device)
        """self.bnin = torch.nn.ModuleDict({'block':  torch.nn.BatchNorm1d(14,affine=False,device=device),
                                         'torque': torch.nn.BatchNorm1d(3,affine=False,device=device),
                                         'force':  torch.nn.BatchNorm1d(3,affine=False,device=device),
                                         'cover':  torch.nn.BatchNorm1d(2,affine=False,device=device)})"""
        """self.lnin = torch.nn.ModuleDict({'block':torch.nn.LayerNorm(14,elementwise_affine=False,bias=False),
                                         'torque':torch.nn.LayerNorm(3,elementwise_affine=False,bias=False),
                                         'force':torch.nn.LayerNorm(3,elementwise_affine=False,bias=False),
                                         'cover':torch.nn.LayerNorm(3,elementwise_affine=False,bias=False),})"""
                                    
        #self.bnContinuous =torch.nn.LayerNorm(config.hidden_channels,elementwise_affine=True,device=device)
        #self.bnDiscrete =torch.nn.LayerNorm(config.hidden_channels,elementwise_affine=True,device=device)
        self.actionDiscrete = torch.nn.Linear(config.hidden_channels,np.prod(action_metadata['discrete']),device=device)
        self.actionContinuous = nn.Linear(config.hidden_channels,np.prod(action_metadata['discrete'])*action_metadata['continuous'],device=device)
        self.repnorm = {key:torch.nn.LayerNorm(config.hidden_channels,elementwise_affine=True,device=device) for key in state_metadata[0]}
        self.state_metadata = state_metadata
        self.action_metadata = action_metadata
        self.max_part = config.max_actiond#action_metadata['discrete'][0]
    def forward(self,inputs,log_bn=False):
        self.convs.to(device)
        self.convActorDiscrete.to(device)
        self.convActorContinuous.to(device)
        known_edges = {key: inputs.edge_index_dict[key] for key in self.state_metadata[1]}
        #rep = {key:self.bnin[key](inputs.x_dict[key]) for key in self.state_metadata[0]}
        rep = self.convs(inputs.x_dict,known_edges,edge_attr=inputs.edge_attr_dict,batch=inputs.batch_dict)

        #rep = {key:self.repnorm[key](rep[key]) for key in self.state_metadata[0]}
        #concat all output
        repD = self.convActorDiscrete(rep,known_edges,edge_attr=inputs.edge_attr_dict, batch=inputs.batch_dict)
        rep_actionsDn = repD[self.an]#self.bnDiscrete(repD[self.an])
        rep_actionsD = self.actionDiscrete(rep_actionsDn).reshape(-1,*self.action_metadata['discrete'])
        repC = self.convActorContinuous(rep,known_edges,edge_attr=inputs.edge_attr_dict, batch=inputs.batch_dict)
        
        rep_actionsCn = repC[self.an]#self.bnContinuous(repC[self.an])
        rep_actionsC = self.actionContinuous(rep_actionsCn).reshape(-1,*self.action_metadata['discrete'], self.action_metadata['continuous'])


        select_probs_batched = torch.zeros((inputs[self.an].batch.max()+1, self.max_part,*self.action_metadata['discrete']), device=device, dtype=floatType)
        actionsC_batched = torch.zeros((inputs[self.an].batch.max()+1, self.max_part,*self.action_metadata['discrete'], self.action_metadata['continuous']), device=device, dtype=floatType)
        

        select_probs = torch_geometric.utils.softmax(rep_actionsD,ptr=inputs[self.an].ptr)
        
        actionsC_scaled = self.action_scale*rep_actionsC+self.action_mean
        actionsC_batched[inputs[self.an].batch,inputs[self.an].part_id] = actionsC_scaled
        actionsC_batched = actionsC_batched.reshape(actionsC_batched.shape[0],-1,self.action_metadata['continuous'])
        dists = torch.distributions.Normal(loc=actionsC_batched, scale=self.var)
        select_probs_batched[inputs[self.an].batch, inputs[self.an].part_id] = select_probs.reshape(-1,*self.action_metadata['discrete'])
        select_dist = torch.distributions.Categorical(probs=select_probs_batched.reshape(select_probs_batched.shape[0],-1))
        rep = torch.cat([self.aggr(rep[key],ptr=inputs.ptr_dict[key])
                        for key in self.state_metadata[0]],axis = 1)
        #batchnorm work well in general, but as the batches are not shuffled it can cause problems
        if log_bn:
            wandb.log({"rep_mean_error":rep.mean(0).detach().cpu()-self.bn_val.running_mean.detach().cpu(),
                       "rep_var_error":rep.var(0).detach().cpu()-self.bn_val.running_var.detach().cpu(),
                       })
        repn  = rep#self.bn_val(rep)
        repV = F.gelu(self.innet(repn))
        for layer in self.full_net:
            repV = F.gelu(layer(repV))
        V =self.outnet(repV)
        return select_dist,dists, V
    def act(self,states,total_prob=False):
        select_dist,dists, V = self.forward(states)
        if (torch.abs(V)> 10).any():
            select_dist,dists, V = self.forward(states)
        if self.deterministic:
            actions = torch.argmax(select_dist.probs, dim=-1).reshape(-1)
            selected_action_params = dists.loc[torch.arange(actions.shape[0]),actions]
        else:
            actions = select_dist.sample()
            action_params =  dists.sample()
            selected_action_params = action_params[torch.arange(dists.loc.shape[0]),actions,:]
        return (actions,selected_action_params), self.get_logprob(select_dist,dists,actions,selected_action_params,total_prob=total_prob), V
    def get_logprob(self,select_dist, dists, actions, action_params, total_prob=False):
        """
        Get log probability of actions given the distribution parameters.
        """
        action_params_dummy = action_params.unsqueeze(1).expand(-1,select_dist.probs.shape[1],-1)
        if total_prob:
            action_logprob = select_dist.log_prob(actions) + dists.log_prob(action_params_dummy)[torch.arange(dists.loc.shape[0]),actions,:].sum(-1)
            return action_logprob.unsqueeze(-1)  # Return log probability as a tensor with shape (batch_size, 1)
        else:
            action_logprobD = select_dist.log_prob(actions).unsqueeze(-1) 
            action_logprobC = dists.log_prob(action_params_dummy)[torch.arange(dists.loc.shape[0]),actions,:]
            return torch.hstack([action_logprobD,action_logprobC])
    def get_entropy(self,select_dist, dists,eps=1e-9):
        """
        Get entropy of the distribution.
        """
        action_entropy = select_dist.entropy() #+ (select_dist.probs.unsqueeze(-1)*dists.entropy()).sum((1,2))
        #dist_entropy is constant for all actions, so we can use the first one
        #action_entropy = select_dist.entropy()
        return action_entropy