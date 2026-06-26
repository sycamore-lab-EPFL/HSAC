import numpy as np
import torch
import torch_geometric
import wandb
from .helper import floatType, intType, device
from torch.cuda import Stream
import torch.nn.functional as F
class RolloutBufferGNN:
    def __init__(self, gamma,gae_lambda, batch_size,her,
                 storing_device=torch.device('cuda'),
                 max_loss=100,n_actions=1,base_state = 'block'):
        self.bs = base_state
        self.gamma = gamma
        self.positive_reward = False
        self.n_actions = n_actions
        self.her = her
        self.gae_lambda = gae_lambda
        self.storing_device = storing_device
        self.mode = "gnn"
        if storing_device != device:
            self.tos = Stream(device)
            self.moveto = True
        else:
            self.moveto = False
        self.traj_start_id = torch.zeros((batch_size),dtype=intType, device=self.storing_device)
        self.std_adv = True
        self.history = {}
        self.clear()
        self.max_loss = max_loss
        self.adv_list = []
        self.dcr_list = []
        self.dcr_mean  = 0
        self.n_sample_dcr = 0
        self.dcr_var = 0
    def set_policy(self, policy):
        self.policy = policy
    def modify_entropy(self,failed_inds,success_inds):
        increasable = self.entropy_weights[failed_inds]<self.max_entropy
        self.entropy_weights[failed_inds[increasable]]+=self.entropy_weight_increase
        self.entropy_weights[success_inds]=self.base_entropy_weight

    def clear(self, num_env=0):
        self.trajs = []
        self.adv_list = []
        self.dcr_list = []
        self.last_states = None
        self.traj_start_id.fill_(0)
        self.num_training_data = 0
    def add_inputs(self,states,actions,action_logprob,V,link_action=False):
        states['value_est'].x = V
        states['value_est'].batch = torch.arange(len(V), device=device)
        states['value_est'].ptr = torch.arange(len(V)+1, device=device)
        states['logprob'].x = action_logprob
        
        if self.bs != 'action_discrete':
            bid = actions[0].to(self.storing_device) //self.n_actions + states[self.bs].ptr[:-1].to(self.storing_device)
            aid = actions[0].to(self.storing_device) % self.n_actions
            assert torch.all(bid < states[self.bs].ptr[1:]), f"block id {bid} exceeds max {states[self.bs].ptr[1:]}"
            assert torch.all(aid < self.n_actions), f"action id {aid} exceeds max {self.n_actions}"
            states['actiond_long'].x = actions[0].to(self.storing_device)
            states['action_discrete'].x = F.one_hot(aid,num_classes=self.n_actions).to(floatType)
            states['action_discrete'].batch = torch.arange(len(aid),device=self.storing_device)
            states['action_discrete'].ptr = torch.arange(len(aid)+1,device=self.storing_device)
            states['action_continuous'].batch = torch.arange(len(actions[1]),device=self.storing_device)
            states['action_continuous'].ptr = torch.arange(len(actions[1])+1,device=self.storing_device)
            states['action_continuous'].x = actions[1].to(self.storing_device)
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
            states['action_continuous','is','action_continuous'].edge_attr = actions[1].to(self.storing_device)
        else:
            bid = states['block','bad','action_discrete'].edge_index[0,actions[0].to(self.storing_device)+states['action_discrete'].ptr[:-1]]
            aid = actions[0].to(self.storing_device)
            assert torch.all(aid < self.n_actions), f"action id {aid} exceeds max {self.n_actions}"
            states['actiond_long'].x = actions[0].to(self.storing_device)
            states['action_continuous'].batch = torch.arange(actions[1].shape[0],device=self.storing_device)
            states['action_continuous'].ptr = torch.arange(actions[1].shape[0]+1,device=self.storing_device)
            states['action_continuous'].x = actions[1].to(self.storing_device)
            
            states['uaction_continuous'].batch = torch.arange(len(actions[2]),device=self.storing_device)
            states['uaction_continuous'].ptr = torch.arange(len(actions[2])+1,device=self.storing_device)
            states['uaction_continuous'].x = actions[2].to(self.storing_device)

            states['block','bac','action_continuous'].edge_index = torch.vstack([bid,torch.arange(len(aid),device=self.storing_device)])
            states['block','bac','action_continuous'].edge_attr = torch.ones((len(aid),1),dtype=floatType,device=self.storing_device)
            states['action_discrete','aa','action_continuous'].edge_index = torch.vstack([aid+states['action_discrete'].ptr[:-1],torch.arange(len(aid),device=self.storing_device)])
            states['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(len(aid),device=self.storing_device),(2,1))          
            states['action_continuous','is','action_continuous'].edge_attr = actions[1].to(self.storing_device)
        #states.validate()
        self.last_states = states
    def add_results(self, rewards, terminated,rotationvec=None):
        if isinstance(rewards,np.ndarray):
            rewards = torch.tensor(rewards, device=device, dtype=floatType)
        if isinstance(terminated,np.ndarray):
            terminated = torch.tensor(terminated, device=device, dtype=bool)
        self.last_states['reward'].x = rewards.unsqueeze(-1)
        self.last_states['terminal'].x = terminated.unsqueeze(-1)
        if rotationvec is not None:
            self.last_states['rotationvec'].x = torch.tensor(rotationvec, device=device, dtype=floatType)
        self.trajs.append(self.last_states)
        terminated_idx = torch.where(terminated)[0]
        self.compute_discounted_rewards(terminated_idx)
        self.traj_start_id[terminated] = len(self.trajs)
    def compute_discounted_rewards(self,batch_inds):
        r = []  # Just for logging
        for b in batch_inds:
            r.append( self.trajs[-1]['reward'].x[b].item())
            self.trajs[-1]['dcr'].x[b] = self.trajs[-1]['reward'].x[b]
            self.trajs[-1]['adv'].x[b] = self.trajs[-1]['dcr'].x[b] - self.trajs[-1]['value_est'].x[b]
            self.adv_list.append(self.trajs[-1]['adv'].x[b])
            self.dcr_list.append(self.trajs[-1]['dcr'].x[b])
            for i in reversed(range(self.traj_start_id[b],len(self.trajs)-1)):
                self.trajs[i]['dcr'].x[b] = self.trajs[i+1]['dcr'].x[b] * self.gamma + self.trajs[i]['reward'].x[b]
                r[-1] += self.trajs[i]['reward'].x[b].item()
                self.trajs[i]['adv'].x[b] = self.trajs[i+1]['adv'].x[b]*self.gamma*self.gae_lambda + self.trajs[i]['reward'].x[b] + self.gamma *self.trajs[i+1]['value_est'].x[b]-self.trajs[i]['value_est'].x[b]
                self.adv_list.append(self.trajs[i]['adv'].x[b])
                self.dcr_list.append(self.trajs[i]['dcr'].x[b])
        if batch_inds.numel() > 0:
            wandb.log({"dcr": torch.stack([self.trajs[self.traj_start_id[b]]['dcr'].x[b] for b in batch_inds]).mean(),
                       "return": np.mean(r)})
    def truncate(self):
        """Use the value of the last state as dcr """
        batch_inds = torch.where(~self.last_states['terminal'].x)[0]
        for b in batch_inds:
            self.trajs[-1]['dcr'].x[b]=  self.gamma * self.last_states['value_est'].x[b]
            #self.adv_list.append(torch.zeros(1, device=device, dtype=floatType).squeeze())
            #self.dcr_list.append(self.trajs[-1]['dcr'].x[b])
            for i in reversed(range(self.traj_start_id[b],len(self.trajs)-1)):
                self.trajs[i]['dcr'].x[b] = self.last_states['value_est'].x[b]
                self.trajs[i]['adv'].x[b] = self.trajs[i+1]['adv'].x[b]*self.gamma*self.gae_lambda + self.trajs[i]['reward'].x[b] + self.gamma *self.trajs[i+1]['value_est'].x[b]-self.trajs[i]['value_est'].x[b]
                self.adv_list.append(self.trajs[i]['adv'].x[b])
                self.dcr_list.append(self.trajs[i]['dcr'].x[b])
        if len(batch_inds) > 0:
            if len(batch_inds) ==self.trajs[-1]['terminal'].x.shape[0]:
                self.trajs = self.trajs[:-1]  # Remove the last trajectory if no batch_inds
            else:
                self.trajs[-1] = torch_geometric.data.Batch.from_data_list(self.trajs[-1].index_select(self.trajs[-1]['terminal'].x))
            
    def build_dl(self, batchsize,normalize=True):
        all_states = []
        for state in self.trajs:
            all_states += state.to_data_list()
        if normalize:
            adv_tensor = torch.stack(self.adv_list).to(dtype=floatType)
            adv_mean = adv_tensor.mean()
            adv_std = adv_tensor.std()
            dcr_tensor = torch.stack(self.dcr_list).to(dtype=floatType)
            dcr_mean = dcr_tensor.mean()
            dcr_std = dcr_tensor.std()
        else:
            adv_mean = 0
            adv_std = 1
            dcr_mean = 0
            dcr_std = 1
        self.num_training_data = len(all_states)
        dl = torch_geometric.loader.DataLoader(all_states, batch_size=batchsize, shuffle=True, pin_memory=self.moveto)
        return dl,adv_mean,adv_std,dcr_mean,dcr_std,len(all_states)

    def build_dataset_her(self, batchsize):
        return None
