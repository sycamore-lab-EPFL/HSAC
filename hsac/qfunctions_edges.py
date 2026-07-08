from ppo.policy_edges import TransformerConvEdge
from torch import nn
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import aggr
import wandb
from .helper import floatType, intType, device
import numpy as np
class TransformerQ(nn.Module):
    def __init__(self, state_metadata,action_metadata,config,action_attach='block'):
        super().__init__()
        n_layers_full = 2
        n_neurons_full = config.hidden_channels
        self.bs = action_attach
        #n_inpart = 1
        if action_attach=='action_discrete':
            state_metadata = (list(state_metadata[0])+['action_continuous'],
                                list(state_metadata[1])+[('action_discrete','aa','action_continuous'),
                                                         ('block','bac','action_continuous'),
                                                         ('action_continuous','is','action_continuous'),
                                                    ])
        else:
            state_metadata = (list(state_metadata[0])+['action_discrete','action_continuous'],
                            list(state_metadata[1])+[(self.bs,'bad','action_discrete'),
                                                    ('action_discrete','aa','action_continuous'),
                                                    (self.bs,'bac','action_continuous'),
                                                    ('action_discrete','is','action_discrete'),
                                                        ('action_continuous','is','action_continuous'),
                                                    ])
        self.deterministic = False
        self.action_metadata = action_metadata

        self.convs = TransformerConvEdge(config.n_layers,config.hidden_channels,heads=config.heads if hasattr(config,'heads') else 1,dropout=config.dropout,layer_norm=config.layer_norm,
                                        layer_norm_mode=config.layer_norm_mode)
        self.convs= torch_geometric.nn.to_hetero(self.convs,state_metadata,aggr='mean').to(device)
        
        """self.out = nn.Sequential(nn.Linear(config.hidden_channels,config.hidden_channels,device=device),
                                 nn.ReLU(),
                                nn.Linear(config.hidden_channels,1,device=device))"""
        self.out = nn.Linear(config.hidden_channels*config.heads,1,device=device)
        self.batchnorm_out = torch.nn.BatchNorm1d(1,device=device)
        if hasattr(config,'value_bounds'):
            self.bounded_value = True
            self.value_bounds = config.value_bounds
            self.value_mean = 0.5*(self.value_bounds[0]+self.value_bounds[1])
            self.value_halfspan = 0.5*(self.value_bounds[1]-self.value_bounds[0])
        else:
            self.bounded_value = False
        self.state_metadata = state_metadata
        self.max_actiond = config.max_actiond#action_metadata['discrete'][0]
    def forward(self,inputs,action=None):
        x = inputs.x_dict
        edges = {key: inputs.edge_attr_dict[key] for key in self.state_metadata[1]}
        known_edges = {key: inputs.edge_index_dict[key] for key in self.state_metadata[1]}
        batch = inputs.batch_dict
        if action is not None:
            # modify inputs to include action
            
            if action[1].dim()==2:
                bid = action[0] //self.action_metadata['discrete'][0]  + inputs[self.bs].ptr[:-1]
                aid = action[0] % self.action_metadata['discrete'][0]
                x['action_discrete'] = F.one_hot(aid,num_classes=self.action_metadata['discrete'][0]).to(floatType)
                x['action_continuous'] = action[1]
                batch['action_discrete'] = torch.arange(len(aid),device=device)
                batch['action_continuous'] = torch.arange(len(aid),device=device)
                known_edges[self.bs,'bad','action_discrete']= torch.vstack([bid,torch.arange(len(aid),device=device)])
                #known_edges['action_discrete','rev_bad',self.bs] = inputs[self.bs,'bad','action_discrete'].edge_index[[1,0],:]
                known_edges[self.bs,'bac','action_continuous'] = torch.vstack([bid,torch.arange(len(aid),device=device)])
                #known_edges['action_continuous','rev_bac',self.bs] = inputs[self.bs,'bac','action_continuous'].edge_index[[1,0],:]
                known_edges['action_discrete','aa','action_continuous'] = torch.tile(torch.arange(len(aid),device=device),(2,1))
                #known_edges['action_continuous','rev_aa','action_discrete'] = inputs['action_discrete','aa','action_continuous'].edge_index[[1,0],:]
                known_edges['action_discrete','is','action_discrete'] = torch.tile(torch.arange(x['action_discrete'].shape[0],device=device),(2,1))
                known_edges['action_continuous','is','action_continuous'] = torch.tile(torch.arange(x['action_continuous'].shape[0],device=device),(2,1))
                edges['action_continuous','is','action_continuous'] = x['action_continuous']
            elif action[1].dim()==3:
                assert action[0] is None, "cannot provide action index when action is given as a full tensor"
                #create a new node for each possible discrete action
                x['action_discrete'] = torch.eye(self.action_metadata['discrete'][0],device=device,dtype=floatType).repeat(x[self.bs].shape[0],1)
                nad = x['action_discrete'].shape[0]
                x['action_continuous'] = torch.vstack([action[1][i,:inputs[self.bs].ptr[i+1]*self.action_metadata['discrete'][0]-inputs[self.bs].ptr[i]*self.action_metadata['discrete'][0]] for i in range(action[1].shape[0])])
                nac = x['action_continuous'].shape[0]
                batch['action_discrete'] = torch.repeat_interleave(batch[self.bs],repeats=self.action_metadata['discrete'][0])
                batch['action_continuous'] = torch.repeat_interleave(batch[self.bs],repeats=self.action_metadata['discrete'][0])
                known_edges[self.bs,'bad','action_discrete']= torch.stack([ torch.repeat_interleave(torch.arange(x[self.bs].shape[0],device=device),self.action_metadata['discrete'][0],0).reshape(-1),
                                                                            torch.arange(nad,device=device)])
                #known_edges['action_discrete','rev_bad',self.bs] = inputs[self.bs,'bad','action_discrete'].edge_index[[1,0],:]
                known_edges[self.bs,'bac','action_continuous'] = torch.stack([ torch.repeat_interleave(torch.arange(x[self.bs].shape[0],device=device),self.action_metadata['discrete'][0],0).reshape(-1),
                                                                               torch.arange(nac,device=device)])
                #known_edges['action_continuous','rev_bac',self.bs] = inputs[self.bs,'bac','action_continuous'].edge_index[[1,0],:]
                known_edges['action_discrete','aa','action_continuous'] = torch.tile(torch.arange(x[self.bs].shape[0]*self.action_metadata['discrete'][0],device=device),(2,1))
                edges[self.bs,'bad','action_discrete'] = torch.ones((known_edges[self.bs,'bad','action_discrete'].shape[1],1),dtype=floatType,device=device)
                edges[self.bs,'bac','action_continuous'] = torch.ones((known_edges[self.bs,'bac','action_continuous'].shape[1],1),dtype=floatType,device=device)
                edges['action_discrete','aa','action_continuous'] = torch.ones((known_edges['action_discrete','aa','action_continuous'].shape[1],1),dtype=floatType,device=device)
                known_edges['action_discrete','is','action_discrete'] = torch.tile(torch.arange(x['action_discrete'].shape[0],device=device),(2,1))
                known_edges['action_continuous','is','action_continuous'] = torch.tile(torch.arange(x['action_continuous'].shape[0],device=device),(2,1))
                edges['action_discrete','is','action_discrete'] = torch.ones((known_edges['action_discrete','is','action_discrete'].shape[1],1),dtype=floatType,device=device)
                edges['action_continuous','is','action_continuous'] = x['action_continuous']#torch.ones((known_edges['action_continuous','is','action_continuous'].shape[1],1),dtype=floatType,device=device)
                #known_edges['action_continuous','rev_aa','action_discrete'] = inputs['action_discrete','aa','action_continuous'].edge_index[[1,0],:]
            elif action[1].dim()==4:
                #get the Q value for diffenrent action parameters for each discrete action
                assert action[0] is None, "cannot provide action index when action is given as a full tensor"
                #create a new node for each possible discrete action
                x['action_discrete'] = torch.eye(self.action_metadata['discrete'][0],device=device,dtype=floatType).repeat(x[self.bs].shape[0],1)
                #TODO: add all continuous action nodes, find the right discrete action and block nodes to connect to them
                x['action_continuous'] = torch.vstack([action[1][i,:inputs[self.bs].ptr[i+1]*self.action_metadata['discrete'][0]-inputs[self.bs].ptr[i]*self.action_metadata['discrete'][0]] for i in range(action[1].shape[0])]).reshape(-1,action[1].shape[-1])
                nac = x['action_continuous'].shape[0]
                batch['action_discrete'] = torch.repeat_interleave(batch[self.bs],repeats=self.action_metadata['discrete'][0])
                batch['action_continuous'] = torch.repeat_interleave(batch[self.bs],repeats=self.action_metadata['discrete'][0]*action[1].shape[2])
                known_edges[self.bs,'bad','action_discrete']= torch.stack([ torch.tile(torch.arange(x[self.bs].shape[0],device=device),(1,self.action_metadata['discrete'][0])).squeeze(),
                                                                            torch.arange(x[self.bs].shape[0]*self.action_metadata['discrete'][0],device=device)])
                known_edges[self.bs,'bac','action_continuous'] = torch.stack([  torch.repeat_interleave(torch.arange(x[self.bs].shape[0],device=device),self.action_metadata['discrete'][0]*action[1].shape[2],0).squeeze(),
                                                                                torch.arange(nac,device=device)])

                known_edges['action_discrete','aa','action_continuous'] = torch.stack([  torch.repeat_interleave(torch.arange(x[self.bs].shape[0]*self.action_metadata['discrete'][0],device=device),action[1].shape[2],0).squeeze(),
                                                                                       torch.arange(nac,device=device)])

                edges[self.bs,'bad','action_discrete'] = torch.ones((known_edges[self.bs,'bad','action_discrete'].shape[1],1),dtype=floatType,device=device)
                edges[self.bs,'bac','action_continuous'] = torch.ones((known_edges[self.bs,'bac','action_continuous'].shape[1],1),dtype=floatType,device=device)
                edges['action_discrete','aa','action_continuous'] = torch.ones((known_edges['action_discrete','aa','action_continuous'].shape[1],1),dtype=floatType,device=device)
                known_edges['action_discrete','is','action_discrete'] = torch.tile(torch.arange(x['action_discrete'].shape[0],device=device),(2,1))
                known_edges['action_continuous','is','action_continuous'] = torch.tile(torch.arange(nac,device=device),(2,1))
                edges['action_discrete','is','action_discrete'] = torch.ones((known_edges['action_discrete','is','action_discrete'].shape[1],1),dtype=floatType,device=device)
                edges['action_continuous','is','action_continuous'] = x['action_continuous']#torch.ones((known_edges['action_continuous','is','action_continuous'].shape[1],1),dtype=floatType,device=device)
                #known_edges['action_continuous','rev_aa','action_discrete'] = inputs['action_discrete','aa','action_continuous'].edge_index[[1,0],:]
        self.convs.to(device)
        rep = self.convs(x,known_edges,edge_attr=edges,batch=batch)
        if self.bounded_value:
            Q = self.value_halfspan*F.tanh(self.out(rep['action_continuous']))+self.value_mean
        else:
            Q = self.out(rep['action_continuous'])
        #Q = self.batchnorm_out(Q).squeeze(-1)
        if action is not None and action[1].dim()==3:
            Q_placeholder = torch.zeros((*action[1].shape[:2],1),dtype=floatType,device=device)
            Q_placeholder[batch['action_continuous'],
                          torch.cat([torch.arange(inputs[self.bs].ptr[i+1]*self.action_metadata['discrete'][0]-inputs[self.bs].ptr[i]*self.action_metadata['discrete'][0]) for i in range(action[1].shape[0])])] = Q
            return Q_placeholder
        elif action is not None and action[1].dim()==4:
            Q_placeholder = torch.zeros((*action[1].shape[:2],action[1].shape[2],1),dtype=floatType,device=device)
            Q_placeholder[batch['action_continuous'],
                          torch.repeat_interleave(torch.cat([torch.arange(inputs[self.bs].ptr[i+1]*self.action_metadata['discrete'][0]
                                                                          -inputs[self.bs].ptr[i]*self.action_metadata['discrete'][0]) for i in range(action[1].shape[0])]),
                                                 action[-1].shape[2],0),
                          torch.tile(torch.arange(action[-1].shape[2],device=device),(1,Q.shape[0]//action[1].shape[2])).squeeze()] = Q
            return Q_placeholder
        return Q
