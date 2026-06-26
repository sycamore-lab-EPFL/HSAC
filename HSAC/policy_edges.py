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
#from torch_scatter import gather_csr, scatter, segment_csr
from .helper import device, floatType, intType
import numpy as np
from ppo.policy_edges import TransformerConvEdge

class TransformerGFTFPolicy(nn.Module):
    def __init__(self, state_metadata,action_metadata,config,action_attach = 'block'):
        super().__init__()
        self.bs = action_attach
        n_layers_full = 2
        n_neurons_full = config.hidden_channels
        #n_inpart = 1
        self.deterministic = False
        self.max_std = config.exploration_noise
        if hasattr(config,'safety_range'):
            self.safety = config.safety_range
        else:
            self.safety = 2*self.max_std
        self.action_metadata = action_metadata
        #self.embedding_parts = torch.nn.Linear(n_inpart,config.hidden_channels//2,device=device,bias=False)

        #self.embedding_state = torch.nn.Embedding(4,config.hidden_channels//2,device=device)

        #self.bnin = torch.nn.BatchNorm1d(config['n_neurons'],device=self.device)

        self.convs = TransformerConvEdge(config.n_layers,config.hidden_channels,heads=config.heads,dropout=config.dropout
                                                    )
        self.convs= torch_geometric.nn.to_hetero(self.convs,state_metadata,aggr='mean').to(device)
        self.convActorDiscrete = TransformerConvEdge(2,config.hidden_channels,heads=config.heads,dropout=config.dropout
                                                    )
        self.convActorContinuous = TransformerConvEdge(2,config.hidden_channels,heads=config.heads,dropout=config.dropout
                                                    )
        self.convActorDiscrete= torch_geometric.nn.to_hetero(self.convActorDiscrete,state_metadata,aggr='mean').to(device)
        self.convActorContinuous= torch_geometric.nn.to_hetero(self.convActorContinuous,state_metadata,aggr='mean').to(device)
        """self.bnin = torch.nn.ModuleDict({'block':  torch.nn.BatchNorm1d(14,affine=False,device=device),
                                         'torque': torch.nn.BatchNorm1d(3,affine=False,device=device),
                                         'force':  torch.nn.BatchNorm1d(3,affine=False,device=device),
                                         'cover':  torch.nn.BatchNorm1d(2,affine=False,device=device)})"""
        """self.lnin = torch.nn.ModuleDict({'block':torch.nn.LayerNorm(14,elementwise_affine=False,bias=False),
                                         'torque':torch.nn.LayerNorm(3,elementwise_affine=False,bias=False),
                                         'force':torch.nn.LayerNorm(3,elementwise_affine=False,bias=False),
                                         'cover':torch.nn.LayerNorm(3,elementwise_affine=False,bias=False),})"""
                                    
        self.bnContinuous =torch.nn.BatchNorm1d(config.hidden_channels,device=device)
        self.bnDiscrete =torch.nn.BatchNorm1d(config.hidden_channels,device=device)
        self.actionDiscrete = nn.Linear(config.hidden_channels*config.heads,np.prod(action_metadata['discrete']),device=device)
        self.actionContinuous = nn.Linear(config.hidden_channels*config.heads,np.prod(action_metadata['discrete'])*action_metadata['continuous'],device=device)
        """self.actionDiscrete = nn.Sequential(nn.Linear(config.hidden_channels,config.hidden_channels,device=device),
                                            nn.GELU(),
                                            nn.Linear(config.hidden_channels,np.prod(action_metadata['discrete']),device=device))
        self.actionContinuous = nn.Sequential(nn.Linear(config.hidden_channels,config.hidden_channels,device=device),
                                              nn.GELU(),
                                              nn.Linear(config.hidden_channels,np.prod(action_metadata['discrete'])*action_metadata['continuous'],device=device))"""
        
        if config.independent_var:
            self.variancenn = nn.Linear(config.hidden_channels*config.heads,np.prod(action_metadata['discrete'])*action_metadata['continuous'],device=device)
        else:
            self.variancenn = nn.Linear(config.hidden_channels*config.heads,np.prod(action_metadata['discrete'])*action_metadata['continuous']*
                                                               (action_metadata['continuous']+1)//2,device=device)
        self.independent = config.independent_var
        self.repnorm = {key:torch.nn.LayerNorm(config.hidden_channels*config.heads,elementwise_affine=True,device=device) for key in state_metadata[0]}
        self.state_metadata = state_metadata
        self.action_metadata = action_metadata
        self.max_actiond = config.max_actiond#action_metadata['discrete'][0]
    def forward(self,inputs,log_bn=False):
        nad = int(self.max_actiond*np.prod(self.action_metadata['discrete']))
        known_edges = {key: inputs.edge_index_dict[key] for key in self.state_metadata[1]}
        #rep = {key:self.bnin[key](inputs.x_dict[key]) for key in self.state_metadata[0]}
        rep = self.convs(inputs.x_dict,known_edges,edge_attr=inputs.edge_attr_dict,batch=inputs.batch_dict)

        #rep = {key:self.repnorm[key](rep[key]) for key in self.state_metadata[0]}
        #concat all output
        repD = self.convActorDiscrete(rep,known_edges,edge_attr=inputs.edge_attr_dict,batch=inputs.batch_dict)

        rep_actionsDn = repD[self.bs]#self.bnDiscrete(repD[self.bs])
        rep_actionsD = self.actionDiscrete(rep_actionsDn).reshape(-1,nad)
        repC = self.convActorContinuous(rep,known_edges,edge_attr=inputs.edge_attr_dict,batch=inputs.batch_dict)
        if self.bs == 'action_discrete':
            rep_actionsCn = repC[self.bs]#self.bnContinuous(repC[self.bs])
        else:
            rep_actionsCn = repC[self.bs]#self.bnContinuous(repC[self.bs])
        rep_actionsC = self.actionContinuous(rep_actionsCn).reshape(-1,nad, self.action_metadata['continuous'])

        select_probs_batched = torch.zeros((inputs[self.bs].batch.max()+1, nad), device=device, dtype=floatType)
        #select_probs_batched = torch.full((inputs[self.bs].batch.max()+1, self.max_actiond,*self.action_metadata['discrete']),-1e9, device=device, dtype=floatType)
        actionsC_batched = torch.zeros((inputs[self.bs].batch.max()+1, nad, self.action_metadata['continuous']), device=device, dtype=floatType)
        self.repaction = rep_actionsD
        select_probs = graph_softmax(rep_actionsD,ptr=inputs[self.bs].ptr)
        #cap the action to be within the action bounds with a prob of 95%
        #actionsC_scaled = (1-self.safety)*F.tanh(rep_actionsC)
        actionsC_scaled = rep_actionsC

        actionsC_batched[inputs[self.bs].batch,inputs[self.bs].part_id,:] = actionsC_scaled
        actionsC_batched = actionsC_batched.reshape(actionsC_batched.shape[0],-1,self.action_metadata['continuous'])
        if self.independent:
            #var_actionsC = self.max_std*F.sigmoid(self.variancenn(rep_actionsCn)).reshape(-1,*self.action_metadata['discrete'], self.action_metadata['continuous'])
            var_actionsC = F.softplus(self.variancenn(rep_actionsCn)).reshape(-1,nad, self.action_metadata['continuous'])+1e-6
            var_batched = torch.ones((inputs[self.bs].batch.max()+1, self.max_actiond,nad, self.action_metadata['continuous']), device=device, dtype=floatType)
            var_batched[inputs[self.bs].batch,inputs[self.bs].part_id,:] = var_actionsC
            var_batched = var_batched.reshape(var_batched.shape[0],-1,self.action_metadata['continuous'])
            dists_unbounded = torch.distributions.Normal(loc=actionsC_batched, scale=var_batched)
        else:
            var_actionsC = self.variancenn(rep_actionsCn).reshape(-1,nad, self.action_metadata['continuous']*(self.action_metadata['continuous']+1)//2)
            var_batched =torch.eye(self.action_metadata['continuous'],device=device).repeat(((inputs[self.bs].batch.max()+1)* nad,1))
            var_batched = var_batched.reshape((inputs[self.bs].batch.max()+1, self.max_actiond,nad, self.action_metadata['continuous'],self.action_metadata['continuous']))
            
            tril_indices = torch.tril_indices(row=self.action_metadata['continuous'], col=self.action_metadata['continuous'], offset=0).repeat(1,inputs[self.bs].x.shape[0]*np.prod(self.action_metadata['discrete']))
            var_batched[inputs[self.bs].batch.repeat_interleave(self.action_metadata['continuous']*
                                                               (self.action_metadata['continuous']+1)//2),
                        inputs[self.bs].part_id.repeat_interleave(self.action_metadata['continuous']*
                                                               (self.action_metadata['continuous']+1)//2),
                        :,tril_indices[0],tril_indices[1]] = F.tanh(var_actionsC.reshape(-1,*self.action_metadata['discrete']))
            diag_idx = torch.arange(self.action_metadata['continuous'],device=device)
            rest_idxs = torch.tril_indices(row=self.action_metadata['continuous'], col=self.action_metadata['continuous'], offset=-1)
            var_batched[...,diag_idx,diag_idx] = self.max_std*F.sigmoid(var_batched[...,diag_idx,diag_idx])+1e-6
            var_batched[...,rest_idxs[0],rest_idxs[1]] = F.tanh(var_batched[...,rest_idxs[0],rest_idxs[1]])*var_batched[...,rest_idxs[1],rest_idxs[1]].detach()*var_batched[...,rest_idxs[0],rest_idxs[0]].detach()
            var_batched = var_batched.reshape(actionsC_batched.shape[0],-1,self.action_metadata['continuous'],self.action_metadata['continuous'])
            dists_unbounded = torch.distributions.MultivariateNormal(loc=actionsC_batched, scale_tril=var_batched)
        
        select_probs_batched[inputs[self.bs].batch, inputs[self.bs].part_id] = select_probs.reshape(-1,nad)
        select_dist = torch.distributions.Categorical(probs=select_probs_batched)

        dists = torch.distributions.TransformedDistribution(dists_unbounded,torch.distributions.transforms.TanhTransform())
        return select_dist,dists,dists_unbounded
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
    def get_logprob(self,select_dist, dists, actions, action_params, total_prob=False):
        """
        Get log probability of actions given the distribution parameters.
        """
        action_params_dummy = action_params.unsqueeze(1).expand(-1,select_dist.probs.shape[1],-1)
        if self.independent:
            action_logprob = select_dist.log_prob(actions) + dists.log_prob(action_params_dummy)[torch.arange(actions.shape[0]),actions,:].sum(-1)
            return action_logprob.unsqueeze(-1)  # Return log probability as a tensor with shape (batch_size, 1)
        else:
            raise NotImplementedError("Not implemented for multivariate case.")
            if total_prob:
                action_logprob = select_dist.log_prob(actions) + dists.log_prob(action_params_dummy)[torch.arange(dists.loc.shape[0]),actions]
                return action_logprob.unsqueeze(-1)  # Return log probability as a tensor with shape (batch_size, 1)
            else:
                action_logprobD = select_dist.log_prob(actions).unsqueeze(-1) 
                action_logprobC = dists.log_prob(action_params_dummy)[torch.arange(dists.loc.shape[0]),actions]
                return torch.hstack([action_logprobD,action_logprobC.unsqueeze(-1)])
def graph_softmax(src, ptr=None):
    """Computes a softmax over node features in a graph.

    Args:
        src (Tensor): Node feature tensor.
        index (LongTensor): Node indices for each feature.
        ptr (LongTensor, optional): Pointer tensor for batching.
    Returns:
        Tensor: Softmax-normalized node features.
    """
    src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr).max(-1,keepdim=True).values
    out = (src - src_max).exp()
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr).sum(-1,keepdim=True)
    
    return out / (out_sum + 1e-16)