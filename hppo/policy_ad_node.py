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

class TransformerAttachedSharedEncoder(nn.Module):
    def __init__(self, state_metadata,action_metadata,config,action_attach = 'block'):
        super().__init__()
        assert action_attach == 'action_discrete'
        self.bs = action_attach
        n_layers_full = 2
        n_neurons_full = config.hidden_channels
        #n_inpart = 1
        self.deterministic = False
        self.logstd = torch.tensor([np.log(config.exploration_noise_init)]*action_metadata['continuous'],dtype=floatType,device=device,
                                   requires_grad=True)
        self.minstd = torch.tensor([config.exploration_noise]*action_metadata['continuous'],dtype=floatType,device=device)
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
       
                                    
        self.actionDiscrete = nn.Linear(config.hidden_channels*config.heads,1,device=device)
        self.actionContinuous = nn.Linear(config.hidden_channels*config.heads,action_metadata['continuous'],device=device)
       

        k=len(state_metadata[0])
        self.aggr = aggr.MultiAggregation([aggr.MaxAggregation(),aggr.MinAggregation(),aggr.MeanAggregation()])#aggr.MedianAggregation().to(self.device)
        #self.bn_val = torch.nn.LayerNorm(config.hidden_channels*k*3,elementwise_affine=True,device=device)
        self.innet = torch.nn.Linear(3*k*config.hidden_channels*config.heads,n_neurons_full,device=device)
        self.full_net = torch.nn.ModuleList([nn.Linear(n_neurons_full,n_neurons_full,device = device) for i in range(n_layers_full)])
        self.outnet = torch.nn.Linear(n_neurons_full,1,device=device)


        self.state_metadata = state_metadata
        self.action_metadata = action_metadata
        self.max_actiond = config.max_actiond#action_metadata['discrete'][0]
    def forward(self,inputs):
        nad = np.prod(self.action_metadata['discrete'])
        self.convs.to(device)
        self.convActorDiscrete.to(device)
        self.convActorContinuous.to(device)
        known_edges = {key: inputs.edge_index_dict[key] for key in self.state_metadata[1]}
        rep = self.convs(inputs.x_dict,known_edges,edge_attr=inputs.edge_attr_dict,batch=inputs.batch_dict)

        #concat all output
        repD = self.convActorDiscrete(rep,known_edges,edge_attr=inputs.edge_attr_dict,batch=inputs.batch_dict)

        rep_actionsDn = repD[self.bs]
        rep_actionsD = self.actionDiscrete(rep_actionsDn)
        repC = self.convActorContinuous(rep,known_edges,edge_attr=inputs.edge_attr_dict,batch=inputs.batch_dict)
            
        rep_actionsCn = repC[self.bs]
     
        rep_actionsC = self.actionContinuous(rep_actionsCn)

        select_probs_batched = torch.zeros((inputs[self.bs].batch.max()+1, nad), device=device, dtype=floatType)
        #select_probs_batched = torch.full((inputs[self.bs].batch.max()+1, self.max_actiond,*self.action_metadata['discrete']),-1e9, device=device, dtype=floatType)
        actionsC_batched = torch.zeros((inputs[self.bs].batch.max()+1, nad, self.action_metadata['continuous']), device=device, dtype=floatType)
        self.repaction = rep_actionsD

        select_probs = torch_geometric.utils.softmax(rep_actionsD,ptr = inputs[self.bs].ptr)
        #select_probs = graph_softmax(rep_actionsD,ptr=inputs[self.bs].ptr)
        #cap the action to be within the action bounds with a prob of 95%
        #actionsC_scaled = (1-self.safety)*F.tanh(rep_actionsC)
        actionsC_scaled = rep_actionsC

        actionsC_batched[inputs[self.bs].batch,inputs[self.bs].part_id,:] = actionsC_scaled
        actionsC_batched = actionsC_batched.reshape(actionsC_batched.shape[0],-1,self.action_metadata['continuous'])
        #actionsC_batched[...,-2] += 1
        dists_unbounded = torch.distributions.Normal(loc=actionsC_batched, scale=(torch.exp(self.logstd)+self.minstd).unsqueeze(0).unsqueeze(0))
        
        select_probs_batched[inputs[self.bs].batch, inputs[self.bs].part_id] = select_probs.flatten()
        select_dist = torch.distributions.Categorical(probs=select_probs_batched)

        repV = torch.cat([self.aggr(rep[key],ptr=inputs.ptr_dict[key])
                        for key in self.state_metadata[0]],axis = 1)

        repV = F.gelu(self.innet(repV))
        for layer in self.full_net:
            repV = F.gelu(layer(repV))
        V =self.outnet(repV)
        return select_dist,dists_unbounded,V
    def bound_action_params(self, action_params):
        #The two first continuous parameters are bounded between -1 and 1 using tanh
        action_params_bounded = torch.zeros_like(action_params)
        action_params_bounded[...,:2] = F.tanh(action_params[..., :2])
        #The last two continuous parameters are normalized to have norm 1
        norms = torch.norm(action_params_bounded[...,2:],dim=-1,keepdim=True)+1e-6
        action_params_bounded[...,2:] = action_params_bounded[...,2:]/norms
        return action_params_bounded
    def act(self,states,return_var=False):
        select_dist,dists_unbounded,V = self.forward(states)
        if self.deterministic:
            actions = torch.argmax(select_dist.probs, dim=-1).reshape(-1)
            action_params = dists_unbounded.loc
            selected_action_params = action_params[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
            action_params_bounded = self.bound_action_params(selected_action_params)
        else:
            actions = select_dist.sample()
            action_params = dists_unbounded.sample()
            
            selected_action_params = action_params[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
            action_params_bounded = self.bound_action_params(selected_action_params)
        if return_var:
            var = dists_unbounded.variance[torch.arange(dists_unbounded.loc.shape[0]),actions,:]
            return (actions,action_params_bounded,selected_action_params),self.get_logprob(select_dist,dists_unbounded,actions,selected_action_params),V, var
        return (actions,action_params_bounded,selected_action_params),self.get_logprob(select_dist,dists_unbounded,actions,selected_action_params),V
    def get_logprobc(self,select_dist, dists_unbounded, uaction_params):
        uac_logprob = dists_unbounded.log_prob(uaction_params) # (batch_size,N,m)
        tanhac_logprob = uac_logprob[...,:-2].sum(-1) - torch.log(1-F.tanh(uaction_params[...,:2]).pow(2)+1e-6).sum(dim=-1) # (batch_size)
        angle = uaction_params[...,-2:]/(torch.norm(uaction_params[...,-2:],dim=-1,keepdim=True)+1e-9) #(batch_size,N, 2)
        aloc = dists_unbounded.loc[...,-2:]
        ascale = dists_unbounded.scale[...,-2:]
        avar = dists_unbounded.variance[...,-2:]
        angle_var = 1/(angle/ascale).pow(2).sum(dim=-1)
        angle_std = angle_var.sqrt()
        angle_loc = (angle*aloc/ascale.pow(2)).sum(dim=-1)*angle_var
        angle_dist = torch.distributions.Normal(loc=torch.zeros_like(angle_loc), scale=torch.ones_like(angle_var.sqrt()))
        # Took me a day to figure this out, good luck understanding it in the future
        # Hint: integrate the bivariate normal over the radius, remember to use rdr in polar coordinates, and remember that (mu^TAd)^2/d^TAd \neq mu^T A mu
        angle_prob = ((angle_loc*(1-torch.erf(-np.sqrt(1/2)*angle_loc/angle_var.sqrt()))*np.sqrt(np.pi/2) +
                        angle_std*torch.exp(-0.5*(angle_loc.pow(2)/angle_var)))*
                        angle_std/(2*torch.pi*(avar.prod(dim=-1)).sqrt())*
                        torch.exp(-0.5*((aloc.pow(2)/avar).sum(dim=-1) - (angle*aloc/avar).sum(dim=-1).pow(2)*angle_var))
                        )
        angle_logprob = torch.log(angle_prob+1e-9)
        ac_logprob = tanhac_logprob + angle_logprob
        ac_logprobE = torch.sum(select_dist.probs*ac_logprob,dim=1)
        return ac_logprobE # Return log probability as a tensor with shape (batch_size,N, 1)
    def get_Elogprobd(self,select_dist, actions):
        if self.independent:
            new_action_logprobd = -select_dist.entropy().unsqueeze(-1)
            return new_action_logprobd  # Return log probability as a tensor with shape (batch_size, 1)
        else:
            raise NotImplementedError("Not implemented for multivariate case.")
    def get_logprob(self,select_dist, dists_unbounded, actions, uaction_params):
        """
        Get log probability of actions given the distribution parameters.
        """
        uaction_params_dummy = uaction_params.unsqueeze(1).expand(-1,select_dist.probs.shape[1],-1)
            
        ad_logprob = select_dist.log_prob(actions)
        uac_logprob = dists_unbounded.log_prob(uaction_params_dummy)[torch.arange(actions.shape[0]),actions] # (batch_size,m)
        tanhac_logprob = uac_logprob[...,:-2].sum(-1) - torch.log(1-F.tanh(uaction_params[...,:2]).pow(2)+1e-6).sum(dim=-1) # (batch_size)
        angle = uaction_params_dummy[torch.arange(actions.shape[0]),actions,-2:] #(batch_size, 2)
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
        angle_prob[angle_prob<=0] = 1e-9
        angle_logprob = torch.log(angle_prob+1e-9)
        ac_logprob = tanhac_logprob + angle_logprob
        logprobs =  torch.hstack([ad_logprob.unsqueeze(-1),ac_logprob.unsqueeze(-1)]).unsqueeze(-1)
        
        return logprobs  # Return log probability as a tensor with shape (batch_size,2, 1)
    
    def get_entropy(self,select_dist, dists,eps=1e-9):
        """
        Get entropy of the distribution.
        """
        action_entropy = select_dist.entropy() #+ (select_dist.probs.unsqueeze(-1)*dists.entropy()).sum((1,2))
        #dist_entropy is constant for all actions, so we can use the first one
        #action_entropy = select_dist.entropy()
        return action_entropy
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
