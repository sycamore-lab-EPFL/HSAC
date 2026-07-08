import time
import numpy as np
from ppo.buffer import RolloutBufferGNN
import torch
import torch_geometric
import wandb
from .helper import floatType, intType, device
from torch.cuda import Stream
import torch.nn.functional as F
class ReplayBufferGNN(RolloutBufferGNN):
    def __init__(self, batch_size,her,
                 storing_device=torch.device('cpu'),
                 max_loss=100,
                 max_blocks=0,
                 n_actions_per_block=1,
                 n_selectable_blocks=1,
                 max_size=1000000,
                 prioritized = False,
                 max_td_error = 1e6,
                 training_ratio = None,
                 replace = True,
                 alpha=0.6,
                 anneal_steps=20000,
                 gamma=0.99,
                 gae_lambda=0.95,
                 base_state = 'block',
                 bs_mode = 'last'):
        super().__init__(gamma,gae_lambda, batch_size,her,storing_device=storing_device,max_loss=max_loss,base_state=base_state)
        self.max_blocks = max_blocks
        self.n_actions = n_actions_per_block
        self.bs_mode = bs_mode
        self.max_size = max_size
        self.prioritized = prioritized
        self.max_td_error = max_td_error
        self.storing_device = storing_device
        self.replace = replace
        self.alpha = alpha
        self.all_states_ = []
        self.next_states_ = []
        self.finished_traj = []
        self.num_new_samples = 0
        #self.num_samples = n_samples if n_samples is not None else max_size
        self.training_ratio = training_ratio if training_ratio is not None else 1.0
        self.anneal_amplitude = alpha/anneal_steps if anneal_steps>0 else 0
    def add_inputs(self,states,actions):
        states = states.clone().to(self.storing_device,non_blocking=True)
        assert torch.all(actions[1]<= 1) and torch.all(actions[1]>= -1), "Continuous actions should be in (-1,1)"
        #states['logprob'].x = action_logprob.to(self.storing_device,non_blocking=True)
        if self.bs != 'action_discrete':
            bid = actions[0].to(self.storing_device,non_blocking=True) //self.n_actions + states[self.bs].ptr[:-1].to(self.storing_device,non_blocking=True)
            aid = actions[0].to(self.storing_device,non_blocking=True) % self.n_actions
            assert torch.all(bid < states[self.bs].ptr[1:]), f"block id {bid} exceeds max {states[self.bs].ptr[1:]}"
            assert torch.all(aid < self.n_actions), f"action id {aid} exceeds max {self.n_actions}"
            states['actiond_long'].x = actions[0].to(self.storing_device,non_blocking=True)
            states['action_discrete'].x = F.one_hot(aid,num_classes=self.n_actions).to(dtype=floatType,device=self.storing_device)
            states['action_discrete'].batch = torch.arange(len(aid),device=self.storing_device)
            states['action_discrete'].ptr = torch.arange(len(aid)+1,device=self.storing_device)
            states['action_continuous'].batch = torch.arange(len(actions[1]),device=self.storing_device)
            states['action_continuous'].ptr = torch.arange(len(actions[1])+1,device=self.storing_device)
            states['action_continuous'].x = actions[1].to(self.storing_device,non_blocking=True,dtype=floatType)
            states['uaction_continuous'].batch = torch.arange(len(actions[2]),device=self.storing_device)
            states['uaction_continuous'].ptr = torch.arange(len(actions[2])+1,device=self.storing_device)
            states['uaction_continuous'].x = actions[2].to(self.storing_device,non_blocking=True,dtype=floatType)
            states[self.bs,'bad','action_discrete'].edge_index = torch.vstack([bid,torch.arange(len(aid),device=self.storing_device)])
            states[self.bs,'bad','action_discrete'].edge_attr = torch.ones((len(aid),1),dtype=floatType,device=self.storing_device)
            #states['action_discrete','rev_bad',self.bs].edge_index = states[self.bs,'bad','action_discrete'].edge_index[[1,0],:]
            states[self.bs,'bac','action_continuous'].edge_index = torch.vstack([bid,torch.arange(len(aid),device=self.storing_device)])
            states[self.bs,'bac','action_continuous'].edge_attr = torch.ones((len(aid),1),dtype=floatType,device=self.storing_device)
            #states['action_continuous','rev_bac',self.bs].edge_index = states[self.bs,'bac','action_continuous'].edge_index[[1,0],:]
            states['action_discrete','aa','action_continuous'].edge_index = torch.tile(torch.arange(len(aid),device=self.storing_device),(2,1))
            #states['action_continuous','rev_aa','action_discrete'].edge_index = states['action_discrete','aa','action_continuous'].edge_index[[1,0],:]
            states['action_discrete','is','action_discrete'].edge_index = torch.tile(torch.arange(len(aid),device=self.storing_device),(2,1))          
            states['action_discrete','is','action_discrete'].edge_attr = torch.ones((len(aid),1),dtype=floatType,device=self.storing_device)
            states['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(len(aid),device=self.storing_device),(2,1))          
            states['action_continuous','is','action_continuous'].edge_attr = actions[1].to(self.storing_device,non_blocking=True,dtype=floatType)
        elif self.bs_mode == 'last':
            bid = states['block','bad','action_discrete'].edge_index[0,actions[0].to(self.storing_device)+states['action_discrete'].ptr[:-1]]
            aid = actions[0].to(self.storing_device)
            assert torch.all(aid < self.n_actions), f"action id {aid} exceeds max {self.n_actions}"
            states['actiond_long'].x = actions[0].to(self.storing_device,non_blocking=True)
            states['action_continuous'].batch = torch.arange(len(actions[1]),device=self.storing_device)
            states['action_continuous'].ptr = torch.arange(len(actions[1])+1,device=self.storing_device)
            states['action_continuous'].x = actions[1].to(self.storing_device,non_blocking=True,dtype=floatType)
            states['uaction_continuous'].batch = torch.arange(len(actions[2]),device=self.storing_device)
            states['uaction_continuous'].ptr = torch.arange(len(actions[2])+1,device=self.storing_device)
            states['uaction_continuous'].x = actions[2].to(self.storing_device,non_blocking=True,dtype=floatType)
            states['block','bac','action_continuous'].edge_index = torch.vstack([bid,torch.arange(len(aid),device=self.storing_device)])
            states['block','bac','action_continuous'].edge_attr = torch.ones((len(aid),1),dtype=floatType,device=self.storing_device)
            states['action_discrete','aa','action_continuous'].edge_index = torch.vstack([aid+states['action_discrete'].ptr[:-1],torch.arange(len(aid),device=self.storing_device)])
            states['action_discrete','aa','action_continuous'].edge_attr = torch.ones((len(aid),1),dtype=floatType,device=self.storing_device)
            
            states['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(len(aid),device=self.storing_device),(2,1))          
            states['action_continuous','is','action_continuous'].edge_attr = actions[1].to(self.storing_device,non_blocking=True,dtype=floatType)
        elif self.bs_mode == 'all':
            bid = actions[0].to(self.storing_device,non_blocking=True) //self.n_actions + states[self.bs].ptr[:-1].to(self.storing_device,non_blocking=True)
            aid = actions[0].to(self.storing_device,non_blocking=True) % self.n_actions
            assert torch.all(aid < self.n_actions), f"action id {aid} exceeds max {self.n_actions}"
            states['actiond_long'].x = actions[0].to(self.storing_device,non_blocking=True)
            states['action_continuous'].batch = torch.arange(len(actions[1]),device=self.storing_device)
            states['action_continuous'].ptr = torch.arange(len(actions[1])+1,device=self.storing_device)
            states['action_continuous'].x = actions[1].to(self.storing_device,non_blocking=True,dtype=floatType)
            states['uaction_continuous'].batch = torch.arange(len(actions[2]),device=self.storing_device)
            states['uaction_continuous'].ptr = torch.arange(len(actions[2])+1,device=self.storing_device)
            states['uaction_continuous'].x = actions[2].to(self.storing_device,non_blocking=True,dtype=floatType)
            states['block','bac','action_continuous'].edge_index = torch.vstack([bid,torch.arange(len(aid),device=self.storing_device)])
            states['block','bac','action_continuous'].edge_attr = torch.ones((len(aid),1),dtype=floatType,device=self.storing_device)
            states['action_discrete','aa','action_continuous'].edge_index = torch.vstack([aid+states['action_discrete'].ptr[:-1],torch.arange(len(aid),device=self.storing_device)])
            states['action_discrete','aa','action_continuous'].edge_attr = torch.ones((len(aid),1),dtype=floatType,device=self.storing_device)
            
            states['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(len(aid),device=self.storing_device),(2,1))          
            states['action_continuous','is','action_continuous'].edge_attr = actions[1].to(self.storing_device,non_blocking=True,dtype=floatType)
        # add to buffer
        #states.index_select([0])[0]['action_discrete','aa','action_continuous']
        self.last_states = states
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
            self.last_states['rotationvec'].x = torch.tensor(rotationvec, device=device, dtype=floatType)
        self.trajs.append(self.last_states)
        terminated_idx = torch.where(terminated)[0]
        self.compute_discounted_rewards(terminated_idx)
        self.traj_start_id[terminated_idx] = len(self.trajs)
        if len(self.trajs) > self.max_size:
            self.trajs.pop(0)
            
    def truncate(self):
        """Use the value of the last state as dcr """
        #maybe this is a bug?
        if self.last_states is None:
            return
        batch_inds = torch.where(~self.last_states['terminal'].x)[0]
        self.last_states['truncated'].x[batch_inds] = True
        for b in batch_inds:
            #self.trajs[-1]['truncated'].x[batch_inds] = True
            self.trajs[-1]['dcr'].x[b]=  self.gamma * self.last_states['value_est'].x[b]
            #self.adv_list.append(torch.zeros(1, device=device, dtype=floatType).squeeze())
            #self.dcr_list.append(self.trajs[-1]['dcr'].x[b])
            for i in reversed(range(self.traj_start_id[b],len(self.trajs)-1)):
                #self.trajs[i]['truncated'].x[batch_inds] = True
                self.trajs[i]['dcr'].x[b] = self.trajs[i]['reward'].x[b] + self.trajs[i+1]['dcr'].x[b]*self.gamma
                self.trajs[i]['adv'].x[b] = self.trajs[i+1]['adv'].x[b]*self.gamma*self.gae_lambda + self.trajs[i]['reward'].x[b] + self.gamma *self.trajs[i+1]['value_est'].x[b]-self.trajs[i]['value_est'].x[b]
                self.adv_list.append(self.trajs[i]['adv'].x[b])
                self.dcr_list.append(self.trajs[i]['dcr'].x[b])
    def add_traj(self):
        self.truncate()
        self.finished_traj += self.trajs
        self.num_new_samples = len(self.trajs)
        self.trajs = []
        self.last_states = None
        self.traj_start_id = torch.zeros_like(self.traj_start_id)
    def reset_new_samples(self):
        self.num_new_samples = 0
    @property
    def all_states(self):
        if len(self.finished_traj)>0:
            for i,state in enumerate(self.finished_traj):
                new_states = state.to(self.storing_device,non_blocking=True).index_select(~state['truncated'].x)
                self.all_states_ += new_states
                self.num_new_samples += len(new_states)
            if len(self.all_states_) > self.max_size:
                self.all_states_ = self.all_states_[-self.max_size:]
            for i,next_state in enumerate(self.finished_traj[1:]):
                self.next_states_ += next_state.to(self.storing_device,non_blocking=True).index_select(~self.finished_traj[i]['truncated'].x)
            assert torch.all(self.finished_traj[-1]['truncated'].x | self.finished_traj[-1]['terminal'].x), "Last traj should be finished"
            #add a dummy next state for the last state
            self.next_states_ += self.finished_traj[-1].to(self.storing_device,non_blocking=True).index_select(~self.finished_traj[-1]['truncated'].x) 
            if len(self.next_states_) > self.max_size:
                self.next_states_ = self.next_states_[-self.max_size:]
            self.num_training_data = len(self.all_states_)
           
            self.finished_traj = []
        return self.all_states_
    @property
    def next_states(self):
        if len(self.trajs)>0:
            #clear the trajs
            self.all_states
        return self.next_states_
    def anneal(self):
        self.alpha = max(0,self.alpha - self.anneal_amplitude)
    def select(self):
        if len(self.all_states) == 0:
            return torch.tensor([],device=device,dtype=torch.long)
        tde = torch.stack([state['tde'].x for state in self.all_states],dim=0).squeeze()
        order = torch.argsort(tde, descending=False)
        probs_unormalized = torch.pow(torch.arange(1,len(order)+1,device=self.storing_device,dtype=floatType),self.alpha)
        probs = probs_unormalized / probs_unormalized.sum()
        if not self.replace:
            n = int(min(self.num_new_samples*self.training_ratio,len(order)))
        else:
            n = int(self.num_new_samples*self.training_ratio)
        indices = torch.multinomial(probs,n,replacement=self.replace)
        selected_indices = order[indices]
        #if len(order)>10*n:
            #unbias the tde distribution only when the replacement is negligible, otherwise the distribution is already biased by the replacement
        self.update_info(selected_indices,'weight',1.0/(probs[indices]*len(order)))
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]
        return selected_indices
    def update_info(self,indices,info,value):
        for idx,v in zip(indices,value):
            self.all_states[idx][info].x=v.reshape(1,-1).to(self.storing_device,non_blocking=True)
    def build_dl(self, batchsize,target_vfunction=None):
        t0 = time.perf_counter()
        if self.prioritized:
            t0s = time.perf_counter()
            indices = self.select()
            t1f = time.perf_counter()
        else:
            n = int(self.num_new_samples*self.training_ratio)
            indices = torch.randint(len(self.all_states), (n,), device=device)
        
        Vs = []
         # compute V for all next states
        if target_vfunction is not None:
            terminal = torch.cat([self.all_states[i]['terminal'].x for i in indices],dim=0).squeeze()
            indices_val = indices[~terminal]
            if indices_val.numel() >0:
                t0dlv = time.perf_counter()
                dl_val = torch_geometric.loader.DataLoader(self.next_states, batch_size=batchsize, sampler=indices_val, pin_memory=self.moveto, 
                                                        num_workers=0 if self.moveto else 0,)
                t1dlv = time.perf_counter()
                t0v = time.perf_counter()
                with torch.no_grad():
                    target_vfunction.eval()
                    for next_state in dl_val:
                        #next_state.validate()
                        V = target_vfunction(next_state.to(device,non_blocking=True))
                        Vs.append(V)
                t1v = time.perf_counter()
                Vs = torch.cat(Vs,dim=0)
                t0u = time.perf_counter()
                self.update_info(indices_val,'value_est',Vs)
                t1u = time.perf_counter()
                mean_val = Vs.mean().item()
            else:
                mean_val = 0.0
        else:
            mean_val = None
        t0dl = time.perf_counter()
         # build dataloader
        dl = torch_geometric.loader.DataLoader(self.all_states, batch_size=batchsize, sampler=indices, pin_memory=self.moveto,
                                               num_workers=0 if self.moveto else 0)
        t1dl = time.perf_counter()
         # clear the buffer
        tf = time.perf_counter()
        #print(f"Built replay buffer dataloader with {len(indices)} samples in {tf - t0:.3f} seconds.")
        #print(f"time breakdown: select {(t1f - t0s)/(tf-t0):.3f}, dl val {(t1dlv - t0dlv)/(tf-t0):.3f}, dl {(t1dl-t0dl)/(tf-t0)}, vfunc {(t1v - t0v)/(tf-t0):.3f}, update {(t1u - t0u)/(tf-t0):.3f}")
        return dl,indices,len(self.all_states),mean_val