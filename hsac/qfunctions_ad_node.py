from ppo.policy_edges import TransformerConvEdge
from sac.valuefunction_edges import PopArtLinear
from torch import nn
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import aggr
import wandb
from .helper import floatType, intType, device
import numpy as np
class TransformerAttachedQ(nn.Module):
    def __init__(self, state_metadata,action_metadata,config,action_attach='block'):
        super().__init__()
        n_layers_full = 2
        n_neurons_full = config.hidden_channels
        self.bs = action_attach
        assert self.bs == 'action_discrete'
        #n_inpart = 1
        state_metadata = (list(state_metadata[0])+['action_continuous'],
                                list(state_metadata[1])+[('action_discrete','aa','action_continuous'),
                                                         ('block','bac','action_continuous'),
                                                         ('action_continuous','is','action_continuous'),
                                                    ])
        self.deterministic = False
        self.action_metadata = action_metadata

        self.convs = TransformerConvEdge(config.n_layers,config.hidden_channels,heads=config.heads if hasattr(config,'heads') else 1,dropout=config.dropout,layer_norm=config.layer_norm,
                                        layer_norm_mode=config.layer_norm_mode)
        self.convs= torch_geometric.nn.to_hetero(self.convs,state_metadata,aggr='mean').to(device,dtype=floatType)
        
        """self.out = nn.Sequential(nn.Linear(config.hidden_channels,config.hidden_channels,device=device),
                                 nn.ReLU(),
                                nn.Linear(config.hidden_channels,1,device=device))"""
        
        if hasattr(config,'use_popart') and config.use_popart:
            self.bounded_value = False
            self.use_popart = True
            self.out = PopArtLinear(config.hidden_channels*(config.heads if hasattr(config,'heads') else 1), 1, device=device, dtype=floatType)
        elif hasattr(config,'value_bounds'):
            self.use_popart = False
            self.bounded_value = True
            self.value_bounds = config.value_bounds
            self.value_mean = 0.5*(self.value_bounds[0]+self.value_bounds[1])
            self.value_halfspan = 0.5*(self.value_bounds[1]-self.value_bounds[0])
            self.out = nn.Linear(config.hidden_channels*(config.heads if hasattr(config,'heads') else 1),1,device=device,dtype=floatType)
        else:
            self.use_popart = False
            self.bounded_value = False
            self.value_mean = nn.Parameter(torch.tensor([0.0],device=device,requires_grad=True))
            self.value_logstd = nn.Parameter(torch.tensor([0.0],device=device,requires_grad=True))
            self.out = nn.Linear(config.hidden_channels*(config.heads if hasattr(config,'heads') else 1),1,device=device,dtype=floatType)
        self.state_metadata = state_metadata
        self.max_actiond = action_metadata['discrete'][0]
    def attach_single_action(self,inputs,action:tuple):
        
        #x = {key: inputs.x_dict[key].contiguous() for key in self.state_metadata[0]}
        #edges = {key: inputs.edge_attr_dict[key].contiguous() for key in self.state_metadata[1]}
        #known_edges = {key: inputs.edge_index_dict[key].contiguous() for key in self.state_metadata[1]}
        #batch = inputs.batch_dict.copy()
        assert action[1].dim()==2, "continuous action should be given as a 2D tensor of shape (batch_size, continuous_action_dim)"
        new_batch_list = []
        for i,graph in enumerate(inputs.to_data_list()):
            #discard all discrete action nodes except the chosen one
            new_graph = graph.clone()
            aid = action[0][i]
            new_graph['action_discrete'].x = new_graph['action_discrete'].x[aid]
            new_graph['action_discrete'].part_id = new_graph['action_discrete'].part_id[aid]
            bid = new_graph['block','bad','action_discrete'].edge_index[0,aid]
            ac = action[1][i]
            new_graph['action_continuous'].x = ac.unsqueeze(0)

            new_graph['block','bad','action_discrete'].edge_index = torch.tensor([[bid,0]],device=device,dtype=int).T
            new_graph['block','bad','action_discrete'].edge_attr = torch.ones((1,1),dtype=floatType,device=device)
            
            new_graph['action_discrete','is','action_discrete'].edge_index = torch.zeros((2,1),device=device,dtype=int)
            new_graph['action_discrete','is','action_discrete'].edge_attr = new_graph['action_discrete'].x
            
            new_graph['block','bac','action_continuous'].edge_index = torch.vstack([bid,torch.arange(len(aid),device=device)])
            new_graph['block','bac','action_continuous'].edge_attr = torch.ones((aid.shape[0],1),dtype=floatType,device=device)

            new_graph['action_discrete','aa','action_continuous'].edge_index = torch.tile(torch.arange(len(aid),device=device),(2,1))
            new_graph['action_discrete','aa','action_continuous'].edge_attr = torch.ones((aid.shape[0],1),dtype=floatType,device=device)

            new_graph['action_continuous','is','action_continuous'].edge_index = torch.zeros((2,1),device=device,dtype=int)
            new_graph['action_continuous','is','action_continuous'].edge_attr = new_graph['action_continuous'].x
            new_graph.validate()
            new_batch_list.append(new_graph)
        new_batch = torch_geometric.data.Batch.from_data_list(new_batch_list)
        #new_batch.validate()
        return new_batch

    def attach_all_actions(self,inputs,action:torch.tensor):
        new_batch_list = []
        if action.dim()==3:
            for i,graph in enumerate(inputs.to_data_list()):
                #discard all discrete action nodes except the chosen one
                new_graph = graph.clone()

                ###change this line if not last block 
                #actioni = action[i,:new_graph['block'].x.shape[0]]
                actioni = action[i]
                
                #action discrete is already correctly given as a one-hot vector in the input action tensor
                #new_graph['action_discrete'].x = F.eye(self.max_actiond,device=device,dtype=floatType).repeat(new_graph[self.bs].shape[0],1).contiguous()
                #new_graph['block','bad','action_discrete'].edge_index = torch.vstack([[bid,0]])
                #new_graph['block','bad','action_discrete'].edge_attr = torch.ones((1,1),dtype=floatType,device=device)
                #new_graph['action_discrete','is','action_discrete'].edge_index = torch.zeros((2,1),device=device,dtype=int)
                #new_graph['action_discrete','is','action_discrete'].edge_attr = F.one_hot(aid,num_classes=self.max_actiond).unsqueeze(0).float()
                
                new_graph['action_continuous'].x = action[i]
                new_graph['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(actioni.shape[0],device=device),(2,1))
                new_graph['action_continuous','is','action_continuous'].edge_attr = action[i]
                #action continuous is attached to  the same block as the action discrete
                new_graph['block','bac','action_continuous'].edge_index = new_graph['block','bad','action_discrete'].edge_index
                new_graph['block','bac','action_continuous'].edge_attr = new_graph['block','bad','action_discrete'].edge_attr

                #same number of action continuous nodes as discrete action nodes, each connected to the corresponding discrete action node

                new_graph['action_discrete','aa','action_continuous'].edge_index = torch.tile(torch.arange(actioni.shape[0],device=device),(2,1))
                new_graph['action_discrete','aa','action_continuous'].edge_attr = torch.ones((actioni.shape[0],1),dtype=floatType,device=device)
                new_graph.validate()
                new_batch_list.append(new_graph)
            new_batch = torch_geometric.data.Batch.from_data_list(new_batch_list)
            new_batch.validate()
        if action.dim()==4:
            raise NotImplementedError("Batching with multiple action steps is not implemented yet")
            for i,graph in enumerate(inputs.to_data_list()):
                #discard all discrete action nodes except the chosen one
                new_graph = graph.clone()
                actioni = action[i].reshape(-1,action.shape[-1])
                #copy the same discrete action action.shape[2] times, each connected to the same block but a different continuous action node
                new_graph['action_discrete'].x = F.eye(self.max_actiond,device=device,dtype=floatType).repeat(new_graph['block'].shape[0],1).contiguous()
                new_graph['block','bad','action_discrete'].edge_index = torch.vstack([torch.arange(new_graph['block'].shape[0],device=device),torch.arange(actioni.shape[1]*actioni.shape[2],device=device)])
                new_graph['block','bad','action_discrete'].edge_attr = torch.ones((actioni.shape[0],1),dtype=floatType,device=device)
                new_graph['action_discrete','is','action_discrete'].edge_index = torch.tile(torch.arange(actioni.shape[1],device=device),(2,1))
                new_graph['action_discrete','is','action_discrete'].edge_attr = new_graph['action_discrete'].x
                #new_graph['action_discrete'].x = F.eye(self.max_actiond,device=device,dtype=floatType).repeat(new_graph[self.bs].shape[0],1).contiguous()
                #new_graph['block','bad','action_discrete'].edge_index = torch.vstack([[bid,0]])
                #new_graph['block','bad','action_discrete'].edge_attr = torch.ones((1,1),dtype=floatType,device=device)
                #new_graph['action_discrete','is','action_discrete'].edge_index = torch.zeros((2,1),device=device,dtype=int)
                #new_graph['action_discrete','is','action_discrete'].edge_attr = F.one_hot(aid,num_classes=self.max_actiond).unsqueeze(0).float()
                
                new_graph['action_continuous'].x = actioni.reshape(-1,actioni.shape[-1])
                new_graph['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(new_graph['action_discrete'].x.shape[0],device=device),(2,1))
                new_graph['action_continuous','is','action_continuous'].edge_attr = actioni.reshape(-1,actioni.shape[-1])
                #action continuous is attached to  the same block as the action discrete
                new_graph['block','bac','action_continuous'].edge_index = new_graph['block','bad','action_discrete'].edge_index
                new_graph['block','bac','action_continuous'].edge_attr = new_graph['block','bad','action_discrete'].edge_attr

                #same number of action continuous nodes as discrete action nodes, each connected to the corresponding discrete action node

                new_graph['action_discrete','aa','action_continuous'].edge_index = torch.tile(torch.arange(actioni.shape[0],device=device),(2,1))
                new_graph['action_discrete','aa','action_continuous'].edge_attr = torch.ones((actioni.shape[0],1),dtype=floatType,device=device)

                new_batch_list.append(new_graph)
            new_batch = torch_geometric.data.Batch.from_data_list(new_batch_list)
            new_batch.validate()
        return new_batch
    #@torch.compile(dynamic=True)
    def _forward_inner(self,x,edges,known_edges,batch):
        rep = self.convs(x,known_edges,edge_attr=edges,batch=batch)
        if self.use_popart:
            Q = self.out(rep['action_continuous'])
        elif self.bounded_value:
            Q = self.value_halfspan*F.tanh(self.out(rep['action_continuous']))+self.value_mean
        else:
            Q_out = self.out(rep['action_continuous'])
            #Q_norm = (Q_out - Q_out.mean())/(Q_out.std(correction=0)+1e-5)
            Q = (self.value_logstd.exp()+1e-5)*(Q_out+self.value_mean)
        return Q
    def forward(self,inputs):
        x = {key: inputs.x_dict[key].contiguous() for key in self.state_metadata[0]}
        edges = {key: inputs.edge_attr_dict[key].contiguous() for key in self.state_metadata[1]}
        known_edges = {key: inputs.edge_index_dict[key].contiguous() for key in self.state_metadata[1]}
        batch = inputs.batch_dict
        #dictnew = {k:{"x":v, "batch":batch[k]} for k,v in x.items()}
        #dictnew.update({k:{'edge_index': v, 'edge_attr':edges[k]} for k,v in known_edges.items()})
        #new_batch = torch_geometric.data.HeteroData(dictnew)
        #new_batch.validate()
        """for k,v in x.items():
            print(f"{k} node features shape: {v.shape}")
        for k,v in known_edges.items():
            print(f"{k} edge index shape: {v.shape}, edge attr shape: {edges[k].shape}")"""
        return self._forward_inner(x,edges,known_edges,batch)
