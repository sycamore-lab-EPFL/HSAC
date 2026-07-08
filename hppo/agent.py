import pickle
from time import perf_counter
from warnings import warn
from ppo.policy_edges import TransformerGFTFSharedEncoder
from ppo.policy_ad_node import TransformerAttachedSharedEncoder
import torch
from torch.distributions import Categorical
import numpy as np
from torch import nn
import torch_geometric
from .policy import GATGFTFSharedEncoder
from .buffer import RolloutBufferGNN
from time import perf_counter
import wandb

from .helper import *



class PPO:
    def __init__(self, state_metadata,action_metadata,config,attach='block', n_bin=10):
        self.gamma = config.gamma
        self.bs = attach
        self.rep = config.rep
        self.eps_clip = config.eps_clip
        self.entropy_weightc = config.ent_coefc
        self.entropy_weightd = config.ent_coefd
        self.value_weight = config.vf_coef
        self.normalize_value = config.normalize_value if hasattr(config,'normalize_value') else True
        self.fitrot = config.get('fitrot',False)
        self.saved_accuracy = 0
        self.episode = 0
        self.num_failed = torch.zeros(n_bin,dtype=intType,device=device)
        self.num_success = torch.zeros(n_bin,dtype=intType,device=device)
        self.MseLoss = nn.MSELoss(reduction='none')
        self.init_policy(state_metadata,action_metadata, config,attach=attach)
        if hasattr(config,"lr_milestones"):
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones= config.lr_milestones)
        else:
            self.scheduler =None
        # timer
        self.reset_timer()
        self.time_step = 0
        
    def reset_timer(self):
        self.sim_time = 0
        self.training_time = 0
        self.inference_time = 0
        self.batch_graph_time = 0
        self.build_graph_time = 0
        self.build_ds_time = 0
    def init_policy(self, state_metadata, action_metadata, config, attach='block'):
        self.buffer = RolloutBufferGNN(self.gamma, config.gae_lambda,config.n_batch,config.her,n_actions=action_metadata['discrete'][0],base_state=attach)
        if attach=='action_discrete':
            self.policy = TransformerAttachedSharedEncoder(state_metadata, action_metadata, config, action_attach=attach).to(device)
            self.policy_old = TransformerAttachedSharedEncoder(state_metadata, action_metadata, config, action_attach=attach).to(device)
            #does not work, might cause problems
            self.policy_old.load_state_dict(self.policy.state_dict())
        elif self.rep in ['noigraph'] or 'bframe' in self.rep:
            self.policy = TransformerGFTFSharedEncoder(state_metadata, action_metadata, config, attach=attach).to(device)
            self.policy_old = TransformerGFTFSharedEncoder(state_metadata, action_metadata, config,  attach=attach).to(device)
            #does not work, might cause problems
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy = GATGFTFSharedEncoder(state_metadata, action_metadata, config, attach=attach).to(device)
            self.policy_old = GATGFTFSharedEncoder(state_metadata, action_metadata, config, attach=attach).to(device)
            #does not work, might cause problems
            self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer_params = {'params': list(self.policy.parameters())+[self.policy.logstd],'lr': config.learning_rate,'weight_decay':config.weight_decay, 'betas':config.betas}
        self.optimizer = torch.optim.Adam([self.optimizer_params])

    def init_stats(self,states,reset_weights=True):
        start_timer = perf_counter()
        self.policy_old.train()
        with torch.no_grad():
            actions,action_logprob, V = self.policy_old.act(states)
        if reset_weights:
            self.policy.load_state_dict(self.policy_old.state_dict())
        self.inference_time += perf_counter() - start_timer
        # add to buffer
        self.buffer.add_inputs(states,actions,action_logprob,V)
        return actions
    def select_action(self, states):
        # inference
        start_timer = perf_counter()
        self.policy_old.eval()
        with torch.no_grad():
            actions,action_logprob, V = self.policy_old.act(states)
        self.inference_time += perf_counter() - start_timer

        # add to buffer
        self.buffer.add_inputs(states,actions,action_logprob,V)
        return actions
    def compute_policy(self, states):
        # inference
        start_timer = perf_counter()
        self.policy_old.eval()
        with torch.no_grad():
            actions,action_logprob, V = self.policy_old.act(states)
        self.inference_time += perf_counter() - start_timer
        return actions, action_logprob, V
    def reset_optim(self):
        if self.scheduler is not None:
            self.optimizer_params['lr'],*_ = self.scheduler.get_last_lr()
        self.optimizer = torch.optim.Adam([self.optimizer_params])

    def set_deterministic_policy(self, status = False):
        self.policy_old.deterministic = status
        self.policy.deterministic = status

    def update_policy(self, batch_size=None, update_iter=5,clear_buffer=True,burn_in=False):
        # extract training dataset from buffer
        start_timer = perf_counter()
        #dataset = self.buffer.build_batch_dataset(batch_size=batch_size, shuffle = shuffle)
        dl,mean_adv,std_adv,mean_dcr,std_dcr,num_training_data = self.buffer.build_dl(batch_size)
        self.build_ds_time += perf_counter()-start_timer
        """if self.buffer.num_step_anneal >0:
            self.buffer.num_step_anneal-=1
            self.buffer.beta+=self.buffer.slope"""

        # record
        loss = []
        entropyd = []
        entropyc = []

        # update policy
        start_timer = perf_counter()
        self.policy.train(True)
        self.reset_optim()

        for epoch_index in range(update_iter):
            self.sur_loss = 0
            self.val_loss = 0
            self.optimizer.zero_grad()
            for i, batch in enumerate(dl):
                if batch.num_graphs <2:
                    continue
                loss_all, sur_loss,val_loss, entropyd_batch,entropyc_batch = self.train_minibatch(batch,mean_adv,std_adv,mean_dcr,std_dcr,burn_in=burn_in)
                self.sur_loss +=sur_loss
                self.val_loss += val_loss
                loss.append(loss_all)
                entropyd.append(entropyd_batch)   
                entropyc.append(entropyc_batch)
        self.sur_loss /= len(dl.dataset)
        self.val_loss /= len(dl.dataset)
        self.training_time += perf_counter() - start_timer
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        if self.scheduler is not None:
            self.scheduler.step()
        # clear buffer
        if clear_buffer:
            self.buffer.clear()

        # avoid empty list
        if len(entropyc) == 0:
            entropyc = [0]
            entropyd = [0]
            loss = [0]
        return np.sum(loss), np.mean(entropyc),np.mean(entropyd), num_training_data
    def train_minibatch(self,
                        data,mean_adv,std_adv,mean_dcr,std_dcr,burn_in=False):
        
        select_dist, udists, state_values = self.policy(data)
        logprobs = self.policy.get_logprob(select_dist, udists, data['actiond_long'].x, data['uaction_continuous'].x) 
        # Finding the ratio (pi_theta / pi_theta__old)
        # Clamp log difference to prevent numerical instability
        log_diff = logprobs - data['logprob'].x.detach()
        log_diff = torch.clamp(log_diff, -20.0, 20.0)  # Prevent exp overflow
        ratios = torch.exp(log_diff)
        
        print(f"ratios: {ratios.mean().cpu().item()} {ratios.max().cpu().item()} {ratios.min().cpu().item()}")
        wandb.log({'ratios':ratios.mean().cpu().item(),
                   'ratios_max':ratios.max().cpu().item(),
                   'ratios_min':ratios.min().cpu().item()},step=self.time_step)
        advantages = (data['adv'].x- mean_adv)/ (std_adv + 1)
        # Finding Surrogate Loss
        surr1 = ratios * advantages.unsqueeze(1)
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.unsqueeze(1)
        surrogate_loss = -torch.min(surr1, surr2).sum(dim=1)

        # final loss of clipped objective PPO
        if self.normalize_value:
            value_loss = self.MseLoss((state_values-mean_dcr)/(std_dcr+1), (data['dcr'].x-mean_dcr)/(std_dcr+1)).mean()
        else:
            value_loss = self.MseLoss(state_values, data['dcr'].x).mean()
        dist_entropy = self.policy.get_entropy(select_dist, udists)
        #sample an action and bound it to get the entropy
        uactions = udists.rsample()
        dist_entropy_sampled = -self.policy.get_logprobc(select_dist, udists, uactions)
        #do not use the bounded entropy due to stability issue
        loss =  self.value_weight* value_loss.mean() + surrogate_loss.mean() - dist_entropy.mean()* self.entropy_weightd - self.entropy_weightc * dist_entropy_sampled.mean()
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            warn(f"NaN or Inf detected in loss, skipping this batch")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # take gradient step        
        entropy_mean = dist_entropy.mean().item()
        variance = udists.variance.mean().item()
        loss_all = loss.item()
        self.optimizer.zero_grad()
        loss.mean().backward()

        grad_conv_last = torch.nn.utils.clip_grad_norm_(self.policy.convs.convs[-1].parameters(), torch.inf, norm_type='inf')
        grad_conv_all = torch.nn.utils.clip_grad_norm_(self.policy.convs.convs[:-1].parameters(), torch.inf, norm_type='inf')
        grad_valnet0 = torch.nn.utils.clip_grad_norm_(self.policy.innet.parameters(), torch.inf, norm_type='inf')
        grad_valnet1 = torch.nn.utils.clip_grad_norm_(self.policy.full_net.parameters(), torch.inf, norm_type='inf')
        if not burn_in:
            self.optimizer.step()
        #print(scale)
        wandb.log({'grad_conv_last':grad_conv_last,
                   'grad_conv_all':grad_conv_all,
                   'grad_valnet0':grad_valnet0,
                   'grad_valnet1':grad_valnet1},step=self.time_step)
        return loss_all, surrogate_loss.mean().item(), value_loss.mean().item(), entropy_mean, dist_entropy_sampled.mean().cpu().item()
    def save(self, checkpoint_path,print_info=None):
        if print_info is None:
            print_info = f"Untested policy; Accuracy on the prioritized replay buffer: {self.saved_accuracy}"
        d =  {'config':wandb.config.as_dict(),
              'state_dict':self.policy_old.state_dict(),
              'print_info': print_info}
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)