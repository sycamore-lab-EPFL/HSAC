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
from hppo.policy_edges import TransformerConvEdge

class TransformerAttachedPolicy(nn.Module):
    def __init__(self, state_metadata,action_metadata,config,action_attach = 'block'):
        super().__init__()
        assert action_attach == 'action_discrete'
        #remove the continuous action part
        state_metadata = copy.deepcopy(state_metadata)
        self.bs = action_attach
        n_layers_full = 2
        n_neurons_full = config.hidden_channels
        #n_inpart = 1
        self.deterministic = False
        self.max_std = config.exploration_noise
        self.action_metadata = action_metadata
        #self.embedding_parts = torch.nn.Linear(n_inpart,config.hidden_channels//2,device=device,bias=False)

        #self.embedding_state = torch.nn.Embedding(4,config.hidden_channels//2,device=device)

        #self.bnin = torch.nn.BatchNorm1d(config['n_neurons'],device=self.device)

        self.convs = TransformerConvEdge(config.n_layers,config.hidden_channels,heads=config.heads,dropout=config.dropout,
                                        )
        self.convs= torch_geometric.nn.to_hetero(self.convs,state_metadata,aggr='mean').to(device)
        self.convActorDiscrete = TransformerConvEdge(2,config.hidden_channels,heads=config.heads,dropout=config.dropout,
                                                    )
        self.convActorContinuous = TransformerConvEdge(2,config.hidden_channels,heads=config.heads,dropout=config.dropout,
                                                    )
        self.convActorDiscrete= torch_geometric.nn.to_hetero(self.convActorDiscrete,state_metadata,aggr='mean').to(device)
        self.convActorContinuous= torch_geometric.nn.to_hetero(self.convActorContinuous,state_metadata,aggr='mean').to(device)
       
                                    
        self.actionDiscrete = nn.Linear(config.hidden_channels*config.heads,1,device=device)
        self.actionContinuous = nn.Linear(config.hidden_channels*config.heads,action_metadata['continuous'],device=device)
       
        if config.independent_var:
            self.variancenn = nn.Linear(config.hidden_channels*config.heads,action_metadata['continuous'],device=device)
        else:
            self.variancenn = nn.Linear(config.hidden_channels*config.heads,action_metadata['continuous']*
                                                               (action_metadata['continuous']+1)//2,device=device)
        self.independent = config.independent_var
        self.state_metadata = state_metadata
        self.action_metadata = action_metadata
        self.max_actiond = config.max_actiond#action_metadata['discrete'][0]

    #@torch.compile(dynamic=True)
    def _forward_inner(self,x,edges,known_edges,ptr,batch,partid,nad,nbatch):
        
        rep = self.convs(x,known_edges,edge_attr=edges,batch=batch)

        #concat all output
        repD = self.convActorDiscrete(rep,known_edges,edge_attr=edges,batch=batch)

        rep_actionsDn = repD[self.bs]
        rep_actionsD = self.actionDiscrete(rep_actionsDn)
        repC = self.convActorContinuous(rep,known_edges,edge_attr=edges,batch=batch)
            
        rep_actionsCn = repC[self.bs]
     
        rep_actionsC = self.actionContinuous(rep_actionsCn)

        select_probs_batched = torch.zeros((nbatch, nad), device=device)

        actionsC_batched = torch.zeros((nbatch, nad, self.action_metadata['continuous']), device=device)
        self.repaction = rep_actionsD
        select_probs = torch_geometric.utils.softmax(rep_actionsD,ptr = ptr['action_discrete'])
        #select_probs = graph_softmax(rep_actionsD,ptr=inputs[self.bs].ptr)
        actionsC_scaled = rep_actionsC

        actionsC_batched[batch[self.bs],partid.flatten(),:] = actionsC_scaled.to(actionsC_batched.dtype)
        actionsC_batched = actionsC_batched.reshape(actionsC_batched.shape[0],-1,self.action_metadata['continuous'])
        if self.independent:
            var_actionsC = F.softplus(self.variancenn(rep_actionsCn))+1e-6
            var_batched = torch.ones((nbatch, self.max_actiond, self.action_metadata['continuous']), device=device)
            var_batched[batch[self.bs],partid.flatten(),:] = var_actionsC
            #var_batched = var_batched.reshape(var_batched.shape[0],-1,self.action_metadata['continuous'])
            dists_unbounded = torch.distributions.Normal(loc=actionsC_batched, scale=var_batched)
        else:
            var_actionsC = self.variancenn(rep_actionsCn).reshape(-1,nad, self.action_metadata['continuous']*(self.action_metadata['continuous']+1)//2)
            var_batched =torch.eye(self.action_metadata['continuous'],device=device,dtype=var_actionsC.dtype).repeat((nbatch* nad,1))
            var_batched = var_batched.reshape(nbatch, self.max_actiond,nad, self.action_metadata['continuous'],self.action_metadata['continuous'])
            
            tril_indices = torch.tril_indices(row=self.action_metadata['continuous'], col=self.action_metadata['continuous'], offset=0).repeat(1,np.prod(self.action_metadata['discrete']))
            var_batched[batch[self.bs].repeat_interleave(self.action_metadata['continuous']*
                                                               (self.action_metadata['continuous']+1)//2),
                        partid.repeat_interleave(self.action_metadata['continuous']*
                                                               (self.action_metadata['continuous']+1)//2),
                        :,tril_indices[0],tril_indices[1]] = F.tanh(var_actionsC.reshape(-1,*self.action_metadata['discrete']))
            diag_idx = torch.arange(self.action_metadata['continuous'],device=device)
            rest_idxs = torch.tril_indices(row=self.action_metadata['continuous'], col=self.action_metadata['continuous'], offset=-1)
            var_batched[...,diag_idx,diag_idx] = self.max_std*F.sigmoid(var_batched[...,diag_idx,diag_idx])+1e-6
            var_batched[...,rest_idxs[0],rest_idxs[1]] = F.tanh(var_batched[...,rest_idxs[0],rest_idxs[1]])*var_batched[...,rest_idxs[1],rest_idxs[1]].detach()*var_batched[...,rest_idxs[0],rest_idxs[0]].detach()
            var_batched = var_batched.reshape(actionsC_batched.shape[0],-1,self.action_metadata['continuous'],self.action_metadata['continuous'])
            dists_unbounded = torch.distributions.MultivariateNormal(loc=actionsC_batched, scale_tril=var_batched)
        
        select_probs_batched[batch[self.bs], partid.flatten()] = select_probs.flatten()
        select_dist = torch.distributions.Categorical(probs=select_probs_batched)

        return select_dist,dists_unbounded
    def forward(self,inputs,insert_noise=0.0):

        inputs = inputs.contiguous()
        known_edges = {key: inputs.edge_index_dict[key].contiguous() for key in self.state_metadata[1]}

        if insert_noise > 0.0:
            noise = torch.randn_like(inputs.edge_attr_dict['block','bb','block'])*insert_noise
            inputs.edge_attr_dict['block','bb','block'] += noise
        return self._forward_inner(inputs.x_dict,
                                   inputs.edge_attr_dict,
                                   known_edges,
                                   inputs.ptr_dict,
                                   inputs.batch_dict,
                                   inputs[self.bs].part_id,
                                   int(np.prod(self.action_metadata['discrete'])),
                                   len(inputs.ptr_dict[self.bs])-1)
    def bound_action_params(self,action_params):
        boundaction_params = torch.zeros_like(action_params)
        #The two first continuous parameters are bounded between -1 and 1 using tanh
        boundaction_params[...,:2] = F.tanh(action_params[..., :2])
        #The last two continuous parameters are normalized to have norm 1
        norms = torch.norm(action_params[...,2:],dim=-1,keepdim=True)+1e-6
        boundaction_params[...,2:] = action_params[...,2:]/norms
        return boundaction_params
    def act(self,states,total_prob=True,return_var=False,insert_noise=0.0):
        select_dist,dists_unbounded = self.forward(states,insert_noise=insert_noise)
        if self.deterministic:
            actions = torch.argmax(select_dist.probs, dim=-1).reshape(-1)
            selected_uaction_params = dists_unbounded.loc[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
            selected_action_params = self.bound_action_params(selected_uaction_params)
        else:
            actions = select_dist.sample()
            uaction_params = dists_unbounded.sample()
            action_params_bounded = self.bound_action_params(uaction_params)
            selected_action_params = action_params_bounded[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
        if return_var:
            var = dists_unbounded.variance[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
            return (actions,selected_action_params), var
        return (actions,selected_action_params)
    def actu(self,states,return_var=False,insert_noise=0.0,offset=0):
        select_dist,dists_unbounded = self.forward(states,insert_noise=insert_noise)
        if self.deterministic:
            actions = torch.argmax(select_dist.probs[:,offset:], dim=-1).reshape(-1)+offset
            selected_uaction_params = dists_unbounded.loc[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
        else:
            actions = select_dist.sample()
            uaction_params = dists_unbounded.sample()
            selected_uaction_params = uaction_params[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
        if return_var:
            var = dists_unbounded.variance[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
            return (actions,selected_uaction_params), var
        return (actions,selected_uaction_params)
    def get_logprobc(self,select_dist, dists_unbounded, uaction_params):
        if self.independent:
            uac_logprob = dists_unbounded.log_prob(uaction_params) # (batch_size,N,m)
            tanhac_logprob = uac_logprob[...,:-2].sum(-1) - torch.log(1-F.tanh(uaction_params[...,:2]).pow(2)+1e-6).sum(dim=-1) # (batch_size)
            angle = uaction_params[...,-2:]/(torch.norm(uaction_params[...,-2:],dim=-1,keepdim=True)+1e-9) #(batch_size,N, 2)
            aloc = dists_unbounded.loc[...,-2:]
            ascale = dists_unbounded.scale[...,-2:]
            avar = dists_unbounded.variance[...,-2:]
            angle_var = 1/(angle/ascale).pow(2).sum(dim=-1)
            angle_std = angle_var.sqrt()
            angle_loc = (angle*aloc/ascale.pow(2)).sum(dim=-1)*angle_var
            # Took me a day to figure this out, good luck understanding it in the future
            # Hint: integrate the bivariate normal over the radius, remember to use rdr in polar coordinates, and remember that (mu^TAd)^2/d^TAd \neq mu^T A mu
            angle_prob = ((angle_loc*(1-torch.erf(-np.sqrt(1/2)*angle_loc/angle_var.sqrt()))*np.sqrt(np.pi/2) +
                          angle_std*torch.exp(-0.5*(angle_loc.pow(2)/angle_var)))*
                          angle_std/(2*torch.pi*(avar.prod(dim=-1)).sqrt())*
                          #can be simplified as angle_loc^2/angle_var######||||||||||||||||||||||||||||||||||||||||||||||
                          torch.exp(-0.5*((aloc.pow(2)/avar).sum(dim=-1) - (angle*aloc/avar).sum(dim=-1).pow(2)*angle_var))
                          )
            angle_logprob = torch.log(angle_prob+1e-9)
            ac_logprob = tanhac_logprob + angle_logprob
            ac_logprobE = torch.sum(select_dist.probs*ac_logprob,dim=1).unsqueeze(-1)
            return ac_logprobE # Return log probability as a tensor with shape (batch_size,N, 1)
        else:
            raise NotImplementedError("Not implemented for multivariate case.")
    def get_Elogprobd(self,select_dist):
        if self.independent:
            new_action_logprobd = -select_dist.entropy().unsqueeze(-1)
            return new_action_logprobd  # Return log probability as a tensor with shape (batch_size, 1)
        else:
            raise NotImplementedError("Not implemented for multivariate case.")
    def get_logprob(self,select_dist, dists_unbounded, actions, uaction_params, total_prob=True):
        """
        Get log probability of actions given the distribution parameters.
        """
        uaction_params_dummy = uaction_params.unsqueeze(1).expand(-1,select_dist.probs.shape[1],-1)
        if self.independent:
            ad_logprob = select_dist.log_prob(actions)
            uac_logprob = dists_unbounded.log_prob(uaction_params_dummy)[torch.arange(actions.shape[0]),actions,:-2] # (batch_size,m)
            tanhac_logprob = uac_logprob.sum(-1) - torch.log(1-F.tanh(uac_logprob).pow(2)+1e-6).sum(dim=-1) # (batch_size)
            angle = uaction_params[...,-2:] #(batch_size,N, 2)
            aloc = dists_unbounded.loc[torch.arange(actions.shape[0]),actions,-2:]
            ascale = dists_unbounded.scale[torch.arange(actions.shape[0]),actions,-2:]
            avar = dists_unbounded.variance[torch.arange(actions.shape[0]),actions,-2:]
            angle_var = 1/(angle/ascale).pow(2).sum(dim=-1)
            angle_std = angle_var.sqrt()
            angle_loc = (angle*aloc/ascale.pow(2)).sum(dim=-1)*angle_var
            # Took me a day to figure this out, good luck understanding it in the future
            # Hint: integrate the bivariate normal over the radius, remember to use rdr in polar coordinates, and remember that (mu^TAd)^2/d^TAd \neq mu^T A mu
            angle_prob = ((angle_loc*(1-torch.erf(-np.sqrt(1/2)*angle_loc/angle_var.sqrt()))*np.sqrt(np.pi/2) +
                          angle_std*torch.exp(-0.5*(angle_loc.pow(2)/angle_var)))*
                          angle_std/(2*torch.pi*(avar.prod(dim=-1)).sqrt())*
                          torch.exp(-0.5*((aloc.pow(2)/avar).sum(dim=-1) - (angle*aloc/avar).sum(dim=-1).pow(2)*angle_var))
                          )
            angle_logprob = torch.log(angle_prob+1e-6)
            ac_logprob = tanhac_logprob + angle_logprob
            logprobs =  torch.vstack([ad_logprob.unsqueeze(0),ac_logprob.unsqueeze(0)])
            if total_prob:
                return logprobs.sum(dim=0).unsqueeze(-1) # Return log probability as a tensor with shape (batch_size, 1)
            else:
                return logprobs.unsqueeze(-1)  # Return log probability as a tensor with shape (2,batch_size, 1)
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
    # Validate inputs to prevent segfaults from torch_scatter
    if ptr is None or src.numel() == 0:
        return torch.softmax(src, dim=0)
    
    # Check ptr tensor validity
    assert ptr.ndim == 1, f"ptr must be 1D, got {ptr.ndim}D"
    assert ptr[0] == 0, f"ptr must start with 0, got {ptr[0]}"
    assert ptr[-1] == src.shape[0], f"ptr[-1]={ptr[-1]} must equal src.shape[0]={src.shape[0]}"
    assert (ptr[1:] >= ptr[:-1]).all(), "ptr must be monotonically non-decreasing"
    
    src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr).max(-1,keepdim=True).values
    out = (src - src_max).exp()
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr).sum(-1,keepdim=True)
    
    return out / (out_sum + 1e-16)