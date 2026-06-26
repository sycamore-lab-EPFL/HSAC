import numpy as np
from sac.buffer import ReplayBufferGNN


import torch
import torch_geometric
import wandb
import copy
from .helper import floatType, intType, device
from torch.cuda import Stream
import torch.nn.functional as F
class HERReplayBufferGNN(ReplayBufferGNN):
    def __init__(self, batch_size,her,
                 storing_device='cpu',
                 max_loss=100,
                 max_blocks=30,
                 n_actions_per_block=1,
                 max_size=1000000,
                 prioritized = False,
                 max_td_error = 1e6,
                 training_ratio = 1,
                 replace = True,
                 alpha=0.6,
                 anneal_steps=20000,
                 rep='noigraph',
                 gamma=0.99):
        super().__init__(batch_size,her,storing_device,max_loss,max_blocks,n_actions_per_block,max_size,
                        prioritized,max_td_error,training_ratio,replace,alpha,anneal_steps,gamma=gamma)
        self.her = ReplayBufferGNN(batch_size,her,storing_device,max_loss,max_blocks,n_actions_per_block,max_size,
                        prioritized,max_td_error,training_ratio,replace,alpha,anneal_steps,gamma=gamma)
        self.rep = rep
        import masongraph_envs # import here to initialize static variables with the config first
    def reset_new_samples(self):
        self.num_new_samples = 0
        self.her.num_new_samples = 0
    @property
    def all_states(self):
        self.num_new_samples = self.her.num_new_samples + self.num_new_samples
        return super().all_states + self.her.all_states
    @property
    def next_states(self):
        return super().next_states + self.her.next_states
    def add_traj(self):
        self.truncate()
        
        self.finished_traj += self.trajs
        self.num_new_samples = len(self.trajs)
        self.trajs = []
        self.last_states = None
        self.traj_start_id = torch.zeros_like(self.traj_start_id)

    def add_results(self, rewards, terminated,truncated,rotationvec=None):
        if isinstance(rewards,np.ndarray):
            rewards = torch.tensor(rewards, device=self.storing_device, dtype=floatType)
        if isinstance(terminated,np.ndarray):
            terminated = torch.tensor(terminated, device=self.storing_device, dtype=bool)
        if isinstance(truncated,np.ndarray):
            truncated = torch.tensor(truncated, device=self.storing_device, dtype=bool)
        self.last_states['reward'].x = rewards.unsqueeze(-1)
        self.last_states['terminal'].x = terminated.unsqueeze(-1)
        self.last_states['truncated'].x = truncated.unsqueeze(-1)
        if rotationvec is not None:
            self.last_states['rotationvec'].x = torch.tensor(rotationvec, device=self.storing_device, dtype=floatType)
        self.trajs.append(self.last_states)
        terminated_idx = torch.where(terminated)[0]
        self.compute_discounted_rewards(terminated_idx)
        self.compute_hindsight(terminated_idx)
        self.traj_start_id[terminated_idx] = len(self.trajs)
        if len(self.trajs) > self.max_size:
            self.trajs.pop(0)
    def compute_hindsight(self, terminated_idx,remove_blocks=None):
        if remove_blocks is None:
            remove_blocks = "ft" in self.rep
        for ti in terminated_idx:
            traj = self.trajs[self.traj_start_id[ti]:]
            if len(traj) > 1:
                last_valid_state = traj[-1].index_select([ti])[0]
                total_covered = last_valid_state['covered']
                covered_edges = last_valid_state['covered','cc','covered']
                if total_covered.x.shape[0]==0:
                    continue
                if remove_blocks:
                    blockstokeep,previdx =torch.unique(last_valid_state['block','bf','force'].edge_index[0],return_inverse=True) # torch.unique(last_valid_state['block'].x)
                breakidx = len(traj)
                for aidx,olds in enumerate(traj[:-1]):
                    news = copy.deepcopy(olds.index_select([ti])[0])
                    news['cover'].pos = total_covered.pos[(total_covered.aidx>=aidx).squeeze()]
                    news['cover'].x = total_covered.x[(total_covered.aidx>=aidx).squeeze()]
                    if news['cover'].x.shape[0]==0:
                        breakidx = aidx
                        news['cover','cc','cover'].edge_index = torch.empty((2,0),dtype=int,device=self.storing_device)
                        news['cover','cc','cover'].edge_attr =  torch.empty((0,2),dtype=floatType,device=self.storing_device)
                        news['block','bc','cover'].edge_index = torch.empty((2,0),dtype=int,device=self.storing_device)
                        news['block','bc','cover'].edge_attr = torch.empty((0,2),dtype=floatType,device=self.storing_device)
                        news['cover','rev_bc','block'].edge_index = torch.empty((2,0),dtype=int,device=self.storing_device)
                        news['cover','rev_bc','block'].edge_attr = torch.empty((0,2),dtype=floatType,device=self.storing_device)
                        news['cover','is','cover'].edge_index = torch.empty((2,0),dtype=int,device=self.storing_device)          
                        news['cover','is','cover'].edge_attr = torch.empty((0,1),dtype=floatType,device=self.storing_device)
                        break
                    news['cover','cc','cover'].edge_index = covered_edges.edge_index[:,covered_edges.aidx.squeeze()>=aidx]-(total_covered.aidx<aidx).sum()
                    news['cover','cc','cover'].edge_attr = covered_edges.edge_attr[covered_edges.aidx.squeeze()>=aidx]
                    news['block','bc','cover'].edge_index = torch.tensor([[i,j] for i in range(news['block'].x.shape[0]) for j in range(news['cover'].x.shape[0])],dtype=int,device=self.storing_device).T
                    news['block','bc','cover'].edge_attr = torch.vstack([news['block'].invtransform[ni[0],:2,:2]@(news['block'].centroid[ni[0],:2]-news['cover'].pos[ni[1]]) for ni in news['block','bc','cover'].edge_index.T])
                    news['cover','rev_bc','block'].edge_index = news['block','bc','cover'].edge_index[[1,0],:]
                    news['cover','rev_bc','block'].edge_attr = -news['block','bc','cover'].edge_attr
                    news['cover','is','cover'].edge_index = torch.tile(torch.arange(news['cover'].x.shape[0],device=self.storing_device),(2,1))          
                    news['cover','is','cover'].edge_attr = torch.ones((news['cover'].x.shape[0],1),dtype=floatType,device=self.storing_device)
                    if remove_blocks:
                        keeps = blockstokeep[blockstokeep<news['block'].x.shape[0]]
                        for k in news['block']:
                            news['block'][k] = news['block'][k][keeps]
                        #Breaks if more than 1 action per block
                        assert news['action_discrete'].x.shape[1]==1
                        assert (keeps==news['actiond_long'].x).any()
                        news['actiond_long'].x = torch.where(keeps==news['actiond_long'].x)[0]
                        news['block'].part_id = torch.arange(news['block'].x.shape[0],device=self.storing_device)
                        assert news['actiond_long'].x.shape[0]==1
                        for edgetype in news.metadata()[1]:
                            if edgetype[0]=='block':
                                reindex_edges(news[edgetype],keeps,source=True)
                            if edgetype[2]=='block':
                                reindex_edges(news[edgetype],keeps,source=False)
                    #news.validate()
                    self.her.all_states_.append(news.to(self.her.storing_device))
                    self.num_new_samples +=1
                    if len(self.her.all_states_) > self.her.max_size:
                        self.her.all_states_.pop(0)
                        self.her.next_states_.pop(0)
                #write the last state as terminal
                self.her.all_states_[-1]['terminal'].x[0] = True
                for aidx,oldns in enumerate(traj[1:breakidx+1]):
                    newns = copy.deepcopy(oldns.to_data_list()[ti])
                    newns['cover'].pos = total_covered.pos[(total_covered.aidx>=aidx+1).squeeze()]
                    newns['cover'].x = total_covered.x[(total_covered.aidx>=aidx+1).squeeze()]
                    if aidx==breakidx:
                        break
                    newns['cover','cc','cover'].edge_index = covered_edges.edge_index[:,covered_edges.aidx.squeeze()>=aidx+1]-(total_covered.aidx<aidx+1).sum()
                    newns['cover','cc','cover'].edge_attr = covered_edges.edge_attr[covered_edges.aidx.squeeze()>=aidx+1]
                    if newns['cover','cc','cover'].edge_index.shape[1]>0:
                        newns['block','bc','cover'].edge_index = torch.tensor([[i,j] for i in range(newns['block'].x.shape[0]) for j in range(newns['cover'].x.shape[0])],dtype=int,device=self.storing_device).T
                        newns['block','bc','cover'].edge_attr = torch.vstack([newns['block'].invtransform[ni[0],:2,:2]@(newns['block'].centroid[ni[0],:2]-newns['cover'].pos[ni[1]]) for ni in newns['block','bc','cover'].edge_index.T])
                        newns['cover','rev_bc','block'].edge_index = newns['block','bc','cover'].edge_index[[1,0],:]
                        newns['cover','rev_bc','block'].edge_attr = -newns['block','bc','cover'].edge_attr
                        newns['cover','is','cover'].edge_index = torch.tile(torch.arange(newns['cover'].x.shape[0],device=self.storing_device),(2,1))          
                        newns['cover','is','cover'].edge_attr = torch.ones((newns['cover'].x.shape[0],1),dtype=floatType,device=self.storing_device)
                    else:
                        newns['block','bc','cover'].edge_index = torch.empty((2,0),dtype=int,device=self.storing_device)
                        newns['block','bc','cover'].edge_attr = torch.empty((0,2),dtype=floatType,device=self.storing_device)
                        newns['cover','rev_bc','block'].edge_index = torch.empty((2,0),dtype=int,device=self.storing_device)
                        newns['cover','rev_bc','block'].edge_attr = torch.empty((0,2),dtype=floatType,device=self.storing_device)
                        newns['cover','is','cover'].edge_index = torch.empty((2,0),dtype=int,device=self.storing_device)          
                        newns['cover','is','cover'].edge_attr = torch.empty((0,1),dtype=floatType,device=self.storing_device)
                    if remove_blocks:
                        keeps = blockstokeep[blockstokeep<newns['block'].x.shape[0]]
                        for k in newns['block']:
                            newns['block'][k] = newns['block'][k][keeps]
                        newns['block'].part_id = torch.arange(newns['block'].x.shape[0],device=self.storing_device)
                        newns['actiond_long'].x = torch.where(keeps==newns['actiond_long'].x)[0]
                        for edgetype in newns.metadata()[1]:
                            if edgetype[0]=='block':
                                reindex_edges(newns[edgetype],keeps,source=True)
                            if edgetype[2]=='block':
                                reindex_edges(newns[edgetype],keeps,source=False)
                    #newns.validate()
                    self.her.next_states_.append(newns)

def reindex_edges(edge, nodes_to_keep,source=True):
    k = 0 if source else 1
    edge_mask = edge.edge_index[k,None]==nodes_to_keep[:,None]
    edge_to_keep = edge_mask.any(dim=0)
    new_idx = torch.where(edge_mask)
    idxe = torch.where(edge_to_keep)[0]
    edge.edge_index = edge.edge_index[:,edge_to_keep]
    edge.edge_index[k] = new_idx[0]
    edge.edge_attr = edge.edge_attr[edge_to_keep]
    #return edge

