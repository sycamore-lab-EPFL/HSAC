import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric
from torch_geometric.nn import GAT, to_hetero
from torch_geometric.nn import aggr
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from torch_geometric.nn import TransformerConv
from torch.nn import GELU
import wandb
from .helper import device, floatType, intType
import numpy as np
from hppo.policy_edges import TransformerConvEdge

class PopArtLinear(nn.Module):
    """
    PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets)
    Wrapper around nn.Linear that normalizes targets dynamically while maintaining unnormalized outputs.
    """
    def __init__(self, in_features, out_features, beta=0.0003, device=None, dtype=None):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(in_features, out_features, device=device, dtype=dtype)
        self.register_buffer('mu', torch.zeros(1, out_features, device=device, dtype=dtype))
        self.register_buffer('sigma', torch.ones(1, out_features, device=device, dtype=dtype))

    def forward(self, x):
        normalized = self.linear(x)
        unnormalized = normalized * self.sigma + self.mu
        return unnormalized

    def normalize(self, unnormalized_targets):
        """Returns targets scaled by current stats, for computing loss."""
        return (unnormalized_targets - self.mu) / self.sigma

    @torch.no_grad()
    def update_stats(self, targets,weight=None):
        """Update running stats and precisely adjust weights/biases to preserve outputs."""
        if weight is not None:
            batch_mu = (targets * weight).sum(dim=0, keepdim=True) / weight.sum()
            batch_sigma = torch.sqrt((weight * (targets - batch_mu)**2).sum(dim=0, keepdim=True) / weight.sum() + 1e-8)
        else:
            batch_mu = targets.mean(dim=0, keepdim=True)
            batch_sigma = torch.sqrt(targets.var(dim=0, unbiased=False, keepdim=True) + 1e-8)

        new_mu = (1 - self.beta) * self.mu + self.beta * batch_mu
        new_sigma = (1 - self.beta) * self.sigma + self.beta * batch_sigma

        # Preserve unnormalized outputs: 
        # (W*x + b)*sigma + mu = (W_new*x + b_new)*new_sigma + new_mu
        self.linear.weight.data.copy_(self.linear.weight.data * (self.sigma / new_sigma).view(-1, 1))
        self.linear.bias.data.copy_(((self.linear.bias.data.view(1, -1) * self.sigma + self.mu - new_mu) / new_sigma).view(-1))

        self.mu.copy_(new_mu)
        self.sigma.copy_(new_sigma)


class TransformerGFTFValue(nn.Module):
    def __init__(self, state_metadata,action_metadata,config):
        super().__init__()
        n_layers_full = 2
        n_neurons_full = config.hidden_channels

        self.convs = TransformerConvEdge(config.n_layers,config.hidden_channels,heads=config.heads,dropout=config.dropout,
                                        )
        self.convs= torch_geometric.nn.to_hetero(self.convs,state_metadata,aggr='mean').to(device,dtype=floatType )
        
        k=len(state_metadata[0])
        self.aggr = aggr.MultiAggregation([aggr.MaxAggregation(),aggr.MinAggregation(),aggr.MeanAggregation()])#aggr.MedianAggregation().to(self.device)
        self.bn_val = torch.nn.LayerNorm(config.hidden_channels*k*3,elementwise_affine=True,device=device,dtype=floatType)
        self.innet = torch.nn.Linear(3*k*config.hidden_channels*config.heads,n_neurons_full,device=device,dtype=floatType)
        self.full_net = torch.nn.ModuleList([nn.Linear(n_neurons_full,n_neurons_full,device = device,dtype=floatType) for i in range(n_layers_full)])
        self.outnet = torch.nn.Linear(n_neurons_full,1,device=device,dtype=floatType)
        
        self.repnorm = {key:torch.nn.LayerNorm(config.hidden_channels*config.heads,elementwise_affine=True,device=device,dtype=floatType) for key in state_metadata[0]}
        self.state_metadata = state_metadata
        self.use_popart = getattr(config, 'use_popart', False)
        
        if hasattr(config,'use_popart') and config.use_popart:
            self.bounded_value = False
            self.outnet = PopArtLinear(n_neurons_full, 1, device=device, dtype=floatType)
        elif hasattr(config,'value_bounds'):
            self.bounded_value = True
            self.value_bounds = config.value_bounds
            self.value_mean = 0.5*(self.value_bounds[0]+self.value_bounds[1])
            self.value_halfspan = 0.5*(self.value_bounds[1]-self.value_bounds[0])
        else:
            self.bounded_value = False
            self.value_mean = nn.Parameter(torch.tensor([0.0],device=device,dtype=floatType,requires_grad=True))
            self.value_logstd = nn.Parameter(torch.tensor([0.0],device=device,requires_grad=True,dtype=floatType))
    #@torch.compile(dynamic=True)
    def _forward_inner(self,x,edges,known_edges,batch,ptr):
        rep = self.convs(x,known_edges,edge_attr=edges,batch=batch)
        #rep = {key:self.repnorm[key](rep[key]) for key in self.state_metadata[0]}
        #concat all output
        rep = torch.cat([self.aggr(rep[key],ptr=ptr[key])
                        for key in self.state_metadata[0]],axis = 1)
        
        repn  = rep#self.bn_val(rep)
        repV = F.gelu(self.innet(repn))
        for layer in self.full_net:
            repV = F.gelu(layer(repV))
        
        if self.use_popart:
            V = self.outnet(repV)
        elif self.bounded_value:
            V =self.value_halfspan*F.tanh(self.outnet(repV))+self.value_mean
        else:
            V_out = self.outnet(repV)
            #V_norm = (V_out - V_out.mean())/(V_out.std(correction=0)+1e-5)
            V = F.relu(self.value_logstd.exp()+1e-5)*(V_out+self.value_mean)
        return V
    def forward(self,inputs):
        # Don't move to device in forward - should be done in __init__
        known_edges = {key: inputs.edge_index_dict[key] for key in self.state_metadata[1]}
        edges = {key: inputs.edge_attr_dict[key] for key in self.state_metadata[1]}
        x = {key: inputs.x_dict[key] for key in self.state_metadata[0]}
        batch = {key: inputs[key].batch for key in self.state_metadata[0]}
        ptr = {key: inputs[key].ptr for key in self.state_metadata[0]}
        return self._forward_inner(x,edges,known_edges,batch,ptr)
        