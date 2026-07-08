import pickle
from time import perf_counter
from warnings import warn

from matplotlib import pyplot as plt

from .her_buffer import HERReplayBufferGNN
from .prefilled_buffer import PrefilledReplayBufferGNN
from sac.policy_edges import TransformerGFTFPolicy
from sac.policy_ad_node import TransformerAttachedPolicy
from sac.qfunctions_ad_node import TransformerAttachedQ
from sac.valuefunction_edges import TransformerGFTFValue
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch_geometric
from .qfunctions_edges import TransformerQ
from .buffer import ReplayBufferGNN
from time import perf_counter
import wandb

from .helper import *





class SAC:
    def __init__(self, state_metadata,action_metadata,config, n_bin=10,attach='block',bs_mode='all'):
        self.bs = attach
        self.bs_mode = bs_mode
        self.gamma = torch.tensor([config.gamma],dtype=floatType,device=device)
        self.compile_backend = config.compile_backend if hasattr(config,'compile_backend') else None
        self.log_entropy_weightc = torch.tensor([np.log(config.ent_coefc)],dtype=floatType,device=device,requires_grad=True)
        self.log_entropy_weightd = torch.tensor([np.log(config.ent_coefd)],dtype=floatType,device=device,requires_grad=True)
        self.total_target_entropy = config.total_entropy if hasattr(config,'total_entropy') else False
        if self.total_target_entropy:
            self.target_entropy = config.target_entropy if hasattr(config,'target_entropy') else - (action_metadata['continuous'] + 1)
        else:
            self.target_entropyc = config.target_entropyc if hasattr(config,'target_entropyc') else -action_metadata['continuous']
            self.target_entropyd = config.target_entropyd if hasattr(config,'target_entropyd') else 1
        self.saved_accuracy = 0
        self.episode = 0
        self.tau = config.tau
        self.value_bounds = config.value_bounds if hasattr(config,'value_bounds') else [-np.inf,np.inf]
        self.num_failed = torch.zeros(n_bin,dtype=intType,device=device)
        self.num_success = torch.zeros(n_bin,dtype=intType,device=device)
        self.total_prob = config.total_probability
        self.fss = 'fss' in config.rep
        self.reset_timer()
        self.time_step = 0
        self.her = config.her if hasattr(config,'her') else False
        self.pessimistic = config.pessimistic if hasattr(config,'pessimistic') else False
        self.pessimistic_steps = config.pessimistic_steps if hasattr(config,'pessimistic_steps') else 1
        # timer
        self.deterministic_rollout = config.deterministic_rollout if hasattr(config,'deterministic_rollout') else False
        self.init_policy(state_metadata,action_metadata, config)
    @property
    def entropy_weight(self):
        if self.total_target_entropy:
            return self.log_entropy_weightc.exp()
        else:
            raise AttributeError("Use entropy_weightc and entropy_weightd for separate entropies")
    @property
    def entropy_weightc(self):
        return self.log_entropy_weightc.exp()
    @property
    def entropy_weightd(self):
        return self.log_entropy_weightd.exp()
    def reset_timer(self):
        self.sim_time = 0
        self.training_time = 0
        self.inference_time = 0
        self.batch_graph_time = 0
        self.build_graph_time = 0
        self.build_ds_time = 0
    def init_policy(self, state_metadata, action_metadata, config):
        if hasattr(config,'prefilled_buffer') and config.prefilled_buffer:
            self.buffer = PrefilledReplayBufferGNN(config.n_batch,config.her,max_blocks=config.max_blocks,
                                      max_size = config.buffer_size,
                                      n_samples = config.n_samples,
                                      prioritized = config.prioritized,
                                      alpha = config.prioritized_alpha,
                                      anneal_steps = config.prioritized_anneal_steps,
                                      replace = config.prioritized_replace,
                                      rep = config.rep,
                                      prefill_weight = config.prefill_weight if hasattr(config,'prefill_weight') else 10,
                                      n_actions_per_block= action_metadata['discrete'][0],
                                      gamma = config.gamma)
        elif self.her:
            self.buffer = HERReplayBufferGNN(config.n_batch,config.her,max_blocks=config.max_actiond,
                                        max_size = config.buffer_size,
                                        training_ratio = config.training_ratio if hasattr(config, 'training_ratio') else 1,
                                        prioritized = config.prioritized,
                                        alpha = config.prioritized_alpha,
                                        anneal_steps = config.prioritized_anneal_steps,
                                        replace = config.prioritized_replace,
                                        n_actions_per_block= action_metadata['discrete'][0],
                                        gamma = config.gamma,
                                        rep = config.rep)
        else:
            self.buffer = ReplayBufferGNN(config.n_batch,config.her,max_blocks= config.max_blocks if hasattr(config,'max_blocks') else 0,
                                        max_size = config.buffer_size,
                                        training_ratio = config.training_ratio if hasattr(config, 'training_ratio') else 1,
                                        prioritized = config.prioritized,
                                        alpha = config.prioritized_alpha,
                                        anneal_steps = config.prioritized_anneal_steps,
                                        replace = config.prioritized_replace,
                                        n_actions_per_block= action_metadata['discrete'][0],
                                        base_state=self.bs,bs_mode =self.bs_mode, )
        if self.fss:
            self.policy = TransformerPolicy(state_metadata, action_metadata, config)
            #random bullshit
            self.qfunction = [TransformerPolicy(state_metadata, action_metadata, config) for _ in range(2)]
            self.vfunction = TransformerPolicy(state_metadata, action_metadata, config)
            self.target_vfunction = TransformerPolicy(state_metadata, action_metadata, config)
        else:
            if self.bs == 'action_discrete':
                self.policy = TransformerAttachedPolicy(state_metadata, action_metadata, config,action_attach='action_discrete')
                self.qfunction = [TransformerAttachedQ(state_metadata, action_metadata, config,action_attach='action_discrete') for _ in range(2)]
            else:
                self.policy = TransformerGFTFPolicy(state_metadata, action_metadata, config,action_attach=self.bs)
                self.qfunction = [TransformerQ(state_metadata, action_metadata, config,action_attach=self.bs) for _ in range(2)]
            self.vfunction = TransformerGFTFValue(state_metadata, action_metadata, config)
            self.target_vfunction = TransformerGFTFValue(state_metadata, action_metadata, config)
        if self.total_target_entropy:
             self.optimizer_pol_params = {'params':  list(self.vfunction.parameters()) +
                                        list(self.policy.parameters()) +
                                         [self.log_entropy_weightc],
                                 'lr': config.learning_rate,'weight_decay':config.weight_decay, 'betas':config.betas}
        else:
            self.optimizer_pol_params = {'params':  list(self.vfunction.parameters()) +
                                            list(self.policy.parameters()) +
                                            [self.log_entropy_weightc,self.log_entropy_weightd],
                                    'lr': config.learning_rate,'weight_decay':config.weight_decay, 'betas':config.betas}
        self.optimizer_q_params = {'params':list(self.qfunction[0].parameters()) +
                                            list(self.qfunction[1].parameters()),
                                              'lr': config.learning_rate*config.vf_coef, 'weight_decay':config.weight_decay, 'betas':config.betas}
        self.optimizer_pol = torch.optim.Adam([self.optimizer_pol_params], fused=True)
        self.optimizer_q = torch.optim.Adam([self.optimizer_q_params], fused=True)
        if hasattr(config,"lr_milestones"):
            self.scheduler = [torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pol,milestones= config.lr_milestones),
                              torch.optim.lr_scheduler.MultiStepLR(self.optimizer_q,milestones= config.lr_milestones)]
        else:
            self.scheduler =None
        if hasattr(config,'prefilled_buffer') and config.prefilled_buffer:
            self.prefill = True
            self.behaviour_cloning(batch_size=config.policy_update_batch_size,update_iter=config.n_updates_prefill)
        self.prefill = False
    def init_stats(self,states,backend=None):
        start_timer = perf_counter()
        if self.fss:
            gpubatch = (states[0].to(device,non_blocking=True),
                        states[1].to(device,non_blocking=True))
        else:
            gpubatch = states.to(device,non_blocking=True)
        #with torch.no_grad():
            #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        select_dist,dist_unbounded = self.policy(gpubatch)
        actiond = select_dist.sample()
        actioncus = dist_unbounded.rsample()
        actioncs = self.policy.bound_action_params(actioncus)
        if self.bs == 'action_discrete':
            gpubatch = self.qfunction[0].attach_all_actions(gpubatch,actioncs)
        q0 = self.qfunction[0](gpubatch)
        q1 = self.qfunction[1](gpubatch)
        V = self.vfunction(gpubatch)
        Vt = self.target_vfunction(gpubatch)


        self.inference_time += perf_counter() - start_timer
        
        for target_param, param in zip(self.target_vfunction.parameters(), self.vfunction.parameters()):
            target_param.data.copy_(param.data)

        self.policy.to(device,dtype=floatType)
        self.vfunction.to(device,dtype=floatType)
        self.target_vfunction.to(device,dtype=floatType)
        self.qfunction[0].to(device,dtype=floatType)
        self.qfunction[1].to(device,dtype=floatType)
        if self.compile_backend is not None:
            #compile_opts = {"epilogue_fusion": False, "max_autotune": False}
            self.policy.compile(backend=self.compile_backend, dynamic=True, options=compile_opts)
            self.vfunction.compile(backend=self.compile_backend, dynamic=True, options=compile_opts)
            self.target_vfunction.compile(backend=self.compile_backend, dynamic=True, options=compile_opts)
            self.qfunction[0].compile(backend=self.compile_backend, dynamic=True, options=compile_opts)
            self.qfunction[1].compile(backend=self.compile_backend, dynamic=True, options=compile_opts)
        # add to buffer
        #self.buffer.add_inputs(states,actions,action_logprob)

    def select_action(self, states,input_noise=0.0,offset=0):
        # inference
        start_timer = perf_counter()
        self.policy.eval()
        if self.deterministic_rollout:
            self.set_deterministic_policy(True)
        with torch.no_grad():
            uactions = self.policy.actu(states,insert_noise=input_noise,offset=offset)
        cactions = self.policy.bound_action_params(uactions[1])
        actions_info = (uactions[0], cactions, uactions[1])
        actions = (uactions[0], cactions)
        self.inference_time += perf_counter() - start_timer
        # add to buffer
        self.buffer.add_inputs(states,actions_info)
        return actions
    def compute_policy(self, states):
        # inference
        start_timer = perf_counter()
        with torch.no_grad():
            actions,action_logprob = self.policy.act(states,total_prob=self.total_prob)
        self.inference_time += perf_counter() - start_timer
        return actions, action_logprob
    def reset_optim(self):
        if self.scheduler is not None:
            self.optimizer_pol_params['lr'],*_ = self.scheduler[0].get_last_lr()
            self.optimizer_q_params['lr'],*_ = self.scheduler[1].get_last_lr()
        self.optimizer_pol = torch.optim.Adam([self.optimizer_pol_params])
        self.optimizer_q = torch.optim.Adam([self.optimizer_q_params])
    def set_deterministic_policy(self, status = False):
        self.policy.deterministic = status

    def update_policy(self, batch_size=None, update_iter=1):
        # extract training dataset from buffer
        
        """if self.buffer.num_step_anneal >0:
            self.buffer.num_step_anneal-=1
            self.buffer.beta+=self.buffer.slope"""

        # record
        loss = []
        entropy = []
        var = []
        entropydb = 0
        # update policy
        
        self.policy.train(True)
        self.qfunction[0].train(True)
        self.qfunction[1].train(True)
        self.vfunction.train(True)
        #self.reset_optim()
        
        for epoch_index in range(update_iter):
            self.pol_loss = 0
            self.q0loss = 0
            self.q1loss = 0
            self.val_loss = 0
            idxstart = 0
            dlstart_timer = perf_counter()
            dl,indices,num_training_data,mean_val = self.buffer.build_dl(batch_size,target_vfunction =self.target_vfunction)
            self.build_ds_time += perf_counter()-dlstart_timer
            start_timer = perf_counter()
            num_samples = 0
            for i, batch in enumerate(dl):
                try:
                    print(f'batch: {i}, num_graphs: {batch.num_graphs}', flush=True)

                    gpubatch = batch.to(device,non_blocking=True)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        num_samples += gpubatch.num_graphs

                        print(f"  Q-function training...", flush=True)
                        q0loss,q1loss = self.train_q_minibatch(gpubatch)
                        print(f"  Q trained. Loss shape: {q0loss.shape}", flush=True)

                        if hasattr(wandb.config,'freeze_pol_epochs'):
                            freeze_pol = self.episode < wandb.config.freeze_pol_epochs
                        else:
                            freeze_pol = False

                        print(f"  Policy training...", flush=True)
                    
                        loss_all, pol_loss,val_loss,entropyb,var_d,entropydb = self.train_pol_minibatch(batch.to(device),freeze_pol=freeze_pol)
                    print(f"  Policy trained. Loss: {loss_all:.4f}", flush=True)
                except RuntimeError as e:
                    print(f"ERROR in batch {i}: {e}", flush=True)
                    raise
                self.pol_loss += pol_loss
                self.val_loss += val_loss
                self.q0loss += q0loss.mean().item()
                self.q1loss += q1loss.mean().item()
                entropy.append(entropyb)
                var.append(var_d)
                loss.append(loss_all+q0loss.sum().item()+q0loss.sum().item())
                if self.buffer.prioritized:
                    self.buffer.update_info(indices[idxstart:idxstart+batch['tde'].x.shape[0]], 'tde', torch.min(q0loss,q1loss).detach())
                    idxstart += batch['tde'].x.shape[0]
            # Copy new weights into target value
            effective_tau = 1-np.power(1-self.tau, num_samples)
            for target_param, param in zip(self.target_vfunction.parameters(), self.vfunction.parameters()):
                target_param.data.copy_((1 - effective_tau) * target_param.data + effective_tau * param.data)
            if hasattr(self.vfunction, 'use_popart') and self.vfunction.use_popart:
                self.target_vfunction.outnet.mu.data.copy_(self.vfunction.outnet.mu.data)
                self.target_vfunction.outnet.sigma.data.copy_(self.vfunction.outnet.sigma.data)
            ##keep the same normalization
            #if not self.vfunction.bounded_value:
            #        self.target_vfunction.value_mean.data.copy_(self.vfunction.value_mean.data)
            #        self.target_vfunction.value_scale.data.copy_(self.vfunction.value_scale.data)
            if self.scheduler is not None:
                [s.step() for s in self.scheduler]
            self.training_time += perf_counter() - start_timer
            
            self.pol_loss /= len(dl)
            self.val_loss /= len(dl)
            self.q0loss /= len(dl)
            self.q1loss /= len(dl)
        
            
            self.buffer.anneal()
        if self.episode % 50 ==0:
            try:
                wandb.log({"Qplot": wandb.Image(self.plot_Q(gpubatch,n_samples=64))},step=self.time_step)
                plt.close('all')
            except torch.OutOfMemoryError:
                print("Could not plot Q")
                plt.close('all')
        for epoch_sil_index in range(wandb.config.n_sil_epochs):
            dlstart_timer = perf_counter()
            dl,indices,num_training_data,_ = self.buffer.build_dl(batch_size)
            self.build_ds_time += perf_counter()-dlstart_timer
            start_timer = perf_counter()
            for i, batch in enumerate(dl):
                #if batch.num_graphs <2:
                    #continue
                try:
                    self.self_imitate_minibatch(batch.to(device))
                except torch.OutOfMemoryError:
                    print("OOM in SIL minibatch")
                    continue
            self.training_time += perf_counter() - start_timer
        
        # avoid empty list
        if len(entropy) == 0:
            entropy = [0]
            loss = [0]

        tde=  torch.stack([state['tde'].x for state in self.buffer.all_states],dim=0).squeeze()
        if self.buffer.prioritized:
            wandb.log({'unseen_samples': torch.sum(tde>self.buffer.max_td_error/2).cpu().item(),
                                'alpha':self.buffer.alpha},step=self.time_step)
        #reset the new sample count
        self.buffer.reset_new_samples()
        return np.mean(loss), np.mean(entropy), num_training_data,np.mean(var), mean_val,entropydb
    def train_q_minibatch(self,data):
        data.validate()
        try:
            target_values = data['value_est'].x
            target_q = data['reward'].x + self.gamma * (~data['terminal'].x) * target_values
            
            if hasattr(self.qfunction[0], 'use_popart') and self.qfunction[0].use_popart:
                self.qfunction[0].out.update_stats(target_q,weight=data['weight'].x)
                target_q_norm = self.qfunction[0].out.normalize(target_q).detach()
                
                # Compute q_values AFTER stats update to avoid inplace variable version errors
                q_values = torch.stack([qf(data) for qf in self.qfunction])
                
                # Optimize normalized outputs
                q_loss = [F.mse_loss(qf.out.normalize(qv), target_q_norm, reduction='none') 
                          for qf, qv in zip(self.qfunction, q_values)]
            else:
                # Compute q_values
                q_values = torch.stack([qf(data) for qf in self.qfunction])
                q_loss = [F.mse_loss(qv, target_q.detach(), reduction='none') for qv in q_values]

            lossq = q_loss[0]*data['weight'].x + q_loss[1]*data['weight'].x
            self.optimizer_q.zero_grad(set_to_none=True)
            lossq.mean().backward()
            self.optimizer_q.step()
            wandb.log({"q_minibatch": q_values.min(0).values.mean().item(),
                       "reward_minibatch": data['reward'].x.mean().item(),
                    }, step=self.time_step)
            return q_loss[0],q_loss[1]
        except torch.OutOfMemoryError:
            print("OOM in Q minibatch")
            return torch.tensor([0.0],device=device),torch.tensor([0.0],device=device)
    def train_pol_minibatch(self,data,freeze_pol=False):
        try:
            select_dist,dist_unbounded = self.policy(data)
            actiond = select_dist.sample()
            actioncus = dist_unbounded.rsample()
            actioncs = self.policy.bound_action_params(actioncus)
            #actionc = actioncs[torch.arange(dists.loc.shape[0]),actiond,:]
            
            #use the explicit expectation to reduce variance
            #new_action_logprob = select_dist.log_prob(actiond) + dists.log_prob(actioncs)[torch.arange(dists.loc.shape[0]),actiond,:].sum(-1)
            if self.policy.independent:
                var = (dist_unbounded.variance*select_dist.probs.unsqueeze(-1)).sum(1).mean()
                #new_action_logprobd = -select_dist.entropy().unsqueeze(-1)
                
                #new_action_logprobc = torch.sum(select_dist.probs*
                #                                (dist_unbounded.log_prob(actioncus) - torch.log(1-actioncs.pow(2) + 1e-9)).sum(-1),dim=(1)).unsqueeze(-1)
                new_action_logprobd = self.policy.get_Elogprobd(select_dist)
                new_action_logprobc = self.policy.get_logprobc(select_dist,dist_unbounded,actioncus)
               #new_action_logprobc = torch.sum(select_dist.probs[:,:,None]*dists.log_prob(actioncs),dim=(1,2)).unsqueeze(-1)
            else:
                raise NotImplementedError("Non-independent SAC not implemented")
                #new_action_logprob = -select_dist.entropy() + torch.sum(select_dist.probs*dists.log_prob(actioncs),dim=(-1))
                #var = (torch.linalg.det(dists.covariance_matrix)*select_dist.probs).sum(1).mean()
            #new_action_logprob = new_action_logprob.unsqueeze(-1)
            # Move state_values evaluation AFTER update_stats to avoid inplace graph errors
            new_state_actions = self.qfunction[0].attach_all_actions(data,action=actioncs)
            new_q_values = torch.stack([qf(new_state_actions) for qf in self.qfunction])
            
            ptr = new_state_actions['action_continuous'].ptr
            min_q = torch.min(new_q_values,axis=0).values
            estimated_value = torch.stack([(min_q[ptr[i]:ptr[i+1]]
                                              *select_dist.probs[i,:ptr[i+1]-ptr[i],None]).sum(0)
                                                for i in range(data.num_graphs)])
            target_v_unnormalized = estimated_value.detach() \
                                    - self.entropy_weightc.detach() * new_action_logprobc.detach() \
                                    - self.entropy_weightd.detach() * new_action_logprobd.detach()
            
            if hasattr(self.vfunction, 'use_popart') and self.vfunction.use_popart:
                self.vfunction.outnet.update_stats(target_v_unnormalized,weight=data['weight'].x)
                state_values = self.vfunction(data)
                target_v_normalized = self.vfunction.outnet.normalize(target_v_unnormalized)
                state_values_normalized = self.vfunction.outnet.normalize(state_values)
                v_loss = (F.mse_loss(state_values_normalized, target_v_normalized, reduction='none') * data['weight'].x).mean()
            else:
                state_values = self.vfunction(data)
                v_loss = (F.mse_loss(state_values, target_v_unnormalized, reduction='none') * data['weight'].x).mean()
            
            pol_loss = ((  self.entropy_weightc.detach()*new_action_logprobc 
                        + self.entropy_weightd.detach()*new_action_logprobd
                        - estimated_value)*data['weight'].x
                        #+ dist_unbounded.variance[-2:].mean()
                        #- torch.min(new_q_values,axis=0).values.sum(1)
                        ).mean()
            if self.total_target_entropy:
                l_w = self.entropy_weight * (-((new_action_logprobc + new_action_logprobd).detach() - self.target_entropy)*data['weight'].x).sum()
            else:
                l_w = (  self.entropy_weightd * ((-new_action_logprobd.detach() - self.target_entropyd)*data['weight'].x).sum()
                    + self.entropy_weightc * ((-new_action_logprobc.detach() - self.target_entropyc)*data['weight'].x).sum())

            loss = v_loss + pol_loss + l_w
            if not freeze_pol:
                self.optimizer_pol.zero_grad(set_to_none=True)
                loss.mean().backward()
                self.optimizer_pol.step()

            return (loss).mean().item(), pol_loss.item(), v_loss.item(), -new_action_logprobc.mean().item(),var.mean().item(),select_dist.entropy().mean().item()
        except torch.OutOfMemoryError:
            print("OOM in policy minibatch")
            return 0.0,0.0,0.0,0.0,0.0,0.0
    def behaviour_cloning(self,batch_size=16,update_iter=1):
         # extract training dataset from buffer
        
        """if self.buffer.num_step_anneal >0:
            self.buffer.num_step_anneal-=1
            self.buffer.beta+=self.buffer.slope"""

        # record
        loss = []
        entropy = []
        var = []
        # update policy
        
        self.policy.train(True)
        self.qfunction[0].train(True)
        self.qfunction[1].train(True)
        self.vfunction.train(True)
        #self.reset_optim()
        num_training_data= 0
        for epoch_index in range(update_iter):
            print(f"BC epoch {epoch_index}/{update_iter}")
            self.pol_loss = 0
            self.q0loss = 0
            self.q1loss = 0
            self.val_loss = 0
            idxstart = 0
            dlstart_timer = perf_counter()
            dl,indices,num_training_data = self.buffer.prefilled.build_dl(batch_size)
            self.build_ds_time += perf_counter()-dlstart_timer
            start_timer = perf_counter()
            for i, batch in enumerate(dl):
                #if batch.num_graphs <2:
                    #continue
                try:
                    if epoch_index==0 and i==0:
                        # use the first batch to init the value function
                        self.target_vfunction(batch.to(device,non_blocking=True))
                    q0loss,q1loss=self.bc_minibatch(batch.to(device,non_blocking=True),std=wandb.config.exploration_noise_init)
                    if self.buffer.prioritized:
                        self.buffer.update_info(indices[idxstart:idxstart+batch['tde'].x.shape[0]], 'tde', torch.max(q0loss,q1loss).detach())
                        idxstart += batch['tde'].x.shape[0]
                except torch.OutOfMemoryError:
                    print("OOM in BC minibatch")
                    continue
            self.training_time += perf_counter() - start_timer
        # avoid empty list
        if len(entropy) == 0:
            entropy = [0]
            loss = [0]
        #hard copy of the weights
        for target_param, param in zip(self.target_vfunction.parameters(), self.vfunction.parameters()):
            target_param.data.copy_(param.data)
        self.reset_optim()
        return np.mean(loss), np.mean(entropy), num_training_data,np.mean(var)
    def bc_minibatch(self,
                        data,std=0.1):
        #target_values = data['value_est'].x
        
        q_values = torch.stack([qf(data) for qf in self.qfunction])
        target_q = data['dcr'].x #data['reward'].x + self.gamma * (~data['terminal'].x) * target_values.detach()
        q_loss = [F.huber_loss(qv, target_q,reduction='none') for qv in q_values]
        lossq = q_loss[0].mean() + q_loss[1].mean()
        if self.pessimistic:
            
            actioncs =  torch.rand((data.num_graphs,self.policy.max_part,self.pessimistic_steps,self.policy.action_metadata['continuous']),device=device)*2 -1
            new_q_values = torch.stack([qf(data,action=(None,actioncs)) for qf in self.qfunction])
            target_qp = torch.zeros_like(new_q_values)
            close_mask = ((torch.arange(actioncs.shape[1],device=device)[None] ==data['actiond_long'].x[:,None])[:,:,None]
                          & (torch.all((torch.abs(actioncs - data['action_continuous'].x[:,None,None,:])<std),-1)))
            q_lossp_uni = F.huber_loss(new_q_values,target_qp.detach(),reduction='none')
            q_lossp_uni[:,close_mask]=0
            q_lossp_uni = q_lossp_uni.mean()
                #once with policy sampling on the other discrete actions
            select_dist,dists_unbounded = self.policy(data)
            raise NotImplementedError("Pessimistic BC not implemented")
            #actioncs =  torch.stack([dist.sample() for i in range(self.pessimistic_steps)],dim=2)
            actioncs = self.policy.bound_action_params(dists_unbounded.loc).unsqueeze(2)
            actioncs[torch.arange(actioncs.shape[0],device=device),data['actiond_long'].x]=data['action_continuous'].x.unsqueeze(1)
            new_q_values = torch.stack([qf(data,action=(None,actioncs)) for qf in self.qfunction])
            target_qp = torch.zeros_like(new_q_values)
            target_qp[:,torch.arange(new_q_values.shape[1],device=device),data['actiond_long'].x] = q_values[...,None]
            q_lossp_pol = F.huber_loss(new_q_values,target_qp.detach(),reduction='mean')
            #q_lossp_pol[:,torch.arange(new_q_values.shape[1],device=device),data['actiond_long'].x] *= 200
            #q_lossp_pol = q_lossp_pol.mean()
            lossq += (q_lossp_pol+q_lossp_uni) 
        self.optimizer_q.zero_grad()
        lossq.mean().backward()

        self.optimizer_q.step()
        select_dist,dists_unbounded = self.policy(data)
        state_values = self.vfunction(data)
        #self.qfunction[0].train()
        #self.qfunction[1].train()
        v_loss = F.huber_loss(state_values,target_q).mean()
        pold_loss = F.nll_loss(select_dist.logits, data['actiond_long'].x.squeeze(-1), reduction='mean')
        polc_loss = F.mse_loss(F.tanh(dists_unbounded.loc[torch.arange(dists_unbounded.loc.shape[0]),data['actiond_long'].x.squeeze(-1),:]),data['action_continuous'].x)
        polstd_loss = F.mse_loss(dists_unbounded.stddev[torch.arange(dists_unbounded.loc.shape[0]),data['actiond_long'].x.squeeze(-1),:],
                                   torch.full(data['action_continuous'].x.shape,std,device=device), reduction='mean')
        lossp = v_loss + pold_loss+polc_loss+polstd_loss
        self.optimizer_pol.zero_grad()
        lossp.mean().backward()
        
        self.optimizer_pol.step()
        wandb.log({"v_loss_bc": v_loss.cpu().item(),
                   "pold_loss_bc": pold_loss.cpu().item(),
                   "polc_loss_bc": polc_loss.cpu().item(),
                   "polstd_loss_bc": polstd_loss.cpu().item(),
                   "q0loss_bc": q_loss[0].mean().item(),
                   "q1loss_bc": q_loss[1].mean().item(),
                   "pessimistic_uni_loss_bc": q_lossp_uni.cpu().item() if self.pessimistic else 0,
                   #"pessimistic_pol_loss_bc": q_lossp_pol.cpu().item() if self.pessimistic else 0,
                   }, step=self.time_step)
        self.time_step +=1
        return q_loss[0],q_loss[1]
    def self_imitate_minibatch(self,data,max_norm =5.0):
        
        q_values = torch.stack([qf(data) for qf in self.qfunction])
        target_q = data['dcr'].x
        q_loss = [torch.square(torch.clip((target_q-qv),0)) for qv in q_values]
        lossq =  q_loss[0].mean() + q_loss[1].mean()
                
        self.optimizer_q.zero_grad()
        lossq.mean().backward()
        torch.nn.utils.clip_grad_norm_(list(self.qfunction[0].parameters())+list(self.qfunction[1].parameters()), max_norm, norm_type='inf')
        self.optimizer_q.step()
        select_dist,unbounded_dists = self.policy(data)

        state_values = self.vfunction(data)

        #self.qfunction[0].train()
        #self.qfunction[1].train()
        v_loss = torch.square(torch.clip((target_q-state_values),0))
        advantage_pos = torch.clip((target_q - state_values),0).detach()
        pol_loss = -self.policy.get_logprob(select_dist,unbounded_dists,data['actiond_long'].x,data['uaction_continuous'].x,True)*advantage_pos
        lossp = v_loss + pol_loss
        self.optimizer_pol.zero_grad()
        lossp.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm, norm_type='inf')
        self.optimizer_pol.step()
        wandb.log({"v_loss_sil": v_loss.detach().mean().cpu().item(),
                "pol_loss_sil": pol_loss.detach().mean().cpu().item(),
                "q_loss_sil": lossq.detach().cpu().item(),
                }, step=self.time_step)
        return 
    def save(self, checkpoint_path,print_info=None):
        if print_info is None:
            print_info = f"Untested policy; Accuracy on the prioritized replay buffer: {self.saved_accuracy}"
        d =  {'config':wandb.config.as_dict(),
              'pol_state_dict':self.policy.state_dict(),
              'qf0_state_dict':self.qfunction[0].state_dict(),
              'qf1_state_dict':self.qfunction[1].state_dict(),
              'vf_state_dict':self.vfunction.state_dict(),
              'print_info': print_info}
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def plot_Q(self,state,n_samples=128,idx_batch=0):
        
        test_state = torch_geometric.data.Batch.from_data_list(state.index_select([idx_batch]))
        select_dist,dist_unbounded = self.policy(test_state)
        discrete_action = select_dist.probs[idx_batch].argmax()
        cont_action_loc = dist_unbounded.loc
        cont_action_scale = dist_unbounded.scale
        fig,axs = plt.subplots(1,cont_action_loc.shape[2],figsize=(25,5))
        for axis in range(cont_action_loc.shape[2]-2):
            test_array = np.linspace(-1,1,n_samples)
            q = torch.zeros((n_samples,2),device=device,dtype=floatType)
            ptr = np.append(np.arange(0,n_samples,state.num_graphs), n_samples)
            for idx_start,idx_end in zip(ptr[:-1],ptr[1:]):
                test_state_b = torch_geometric.data.Batch.from_data_list(test_state.index_select([idx_batch])*(idx_end-idx_start))
                test_action = self.policy.bound_action_params(cont_action_loc[idx_batch,discrete_action].repeat((idx_end-idx_start,1)))
                test_action[:,axis] = torch.tensor(test_array[idx_start:idx_end],device=device,dtype=floatType)
                test_state_action = self.qfunction[0].attach_single_action(test_state_b,action=(discrete_action.repeat(idx_end-idx_start).contiguous()[:,None],test_action))
                q[idx_start:idx_end,0] = self.qfunction[0](test_state_action).squeeze()
                q[idx_start:idx_end,1] = self.qfunction[1](test_state_action).squeeze()
            axs[axis].plot(test_array,q[:,0].detach().cpu(),label='Q1')
            axs[axis].plot(test_array,q[:,1].detach().cpu(),label='Q2')
            axs[axis].axvline(x=F.tanh(cont_action_loc[idx_batch,discrete_action,axis]).item(),color='k',linestyle='--',label='Current action mean')
            axs[axis].axvline(x=(F.tanh(cont_action_loc[idx_batch,discrete_action,axis].detach().cpu()-cont_action_scale[idx_batch,discrete_action,axis]).item()),color='k',linestyle=':',label='Current action - 1 stddev')
            axs[axis].axvline(x=(F.tanh(cont_action_loc[idx_batch,discrete_action,axis].detach().cpu()+cont_action_scale[idx_batch,discrete_action,axis]).item()),color='k',linestyle=':',label='Current action + 1 stddev')
            axs[axis].set_xlabel('Action value along axis '+str(axis))
            axs[axis].set_ylabel('Q value')
            axs[axis].set_title('Q value vs action along axis '+str(axis)+f", ad={discrete_action.item()}")
            axs[axis].legend()
            axs[axis].grid()

        #values for the last two axes, that are normalized and not bounded by the tanh
        axis = cont_action_loc.shape[2]-2
        test_range = np.linspace(-np.pi,np.pi,n_samples)
        test_array = np.vstack([np.cos(test_range),np.sin(test_range)]).T
        q = torch.zeros((n_samples,2),device=device,dtype=floatType)
        ptr = np.append(np.arange(0,n_samples,state.num_graphs), n_samples)
        for idx_start,idx_end in zip(ptr[:-1],ptr[1:]):
            test_state_b = torch_geometric.data.Batch.from_data_list(test_state.index_select([idx_batch])*(idx_end-idx_start))
            test_action = self.policy.bound_action_params(cont_action_loc[idx_batch,discrete_action].repeat((idx_end-idx_start,1)))
            test_action[:,-2:] = torch.tensor(test_array[idx_start:idx_end],device=device,dtype=floatType)
            test_state_action = self.qfunction[0].attach_single_action(test_state_b,action=(discrete_action.repeat(idx_end-idx_start).contiguous()[:,None],test_action))
            q[idx_start:idx_end,0] = self.qfunction[0](test_state_action).squeeze()
            q[idx_start:idx_end,1] = self.qfunction[1](test_state_action).squeeze()
        axs[axis].plot(test_range,q[:,0].detach().cpu(),label='Q1')
        axs[axis].plot(test_range,q[:,1].detach().cpu(),label='Q2')
        #sample 10*n_sample actions to draw a histogram of the distribution
        samples = dist_unbounded.sample((10*n_samples,))
        samples_rot = samples[:,0,discrete_action,-2:]
        angles = torch.atan2(samples_rot[:,1],samples_rot[:,0])
        counts, edges = np.histogram(angles.cpu().numpy(), bins=n_samples, range=(-np.pi, np.pi))
        max_count = counts.max() if counts.max() > 0 else 1
        counts_norm = counts / max_count
        counts_scaled = counts_norm * (q.max().detach().cpu().numpy() - q.min().detach().cpu().numpy())
        axs[axis].bar(
                        edges[:-1], 
                        counts_scaled, 
                        width=(edges[1] - edges[0]), 
                        bottom=q.min().detach().cpu(), 
                        align='edge', 
                        color='k', 
                        label='Action distribution'
                    )
        #current_mean = torch.atan2(cont_action_loc[idx_batch,discrete_action,-2],cont_action_loc[idx_batch,discrete_action,-1])
        #axs[axis].axvline(x=current_mean.item(),color='k',linestyle='--',label='Current action mean')
        #axs[axis].axvline(x=(F.tanh(cont_action_loc[0,discrete_action,axis].detach().cpu()-cont_action_scale[0,discrete_action,axis]).item()),color='k',linestyle=':',label='Current action - 1 stddev')
        #axs[axis].axvline(x=(F.tanh(cont_action_loc[0,discrete_action,axis].detach().cpu()+cont_action_scale[0,discrete_action,axis]).item()),color='k',linestyle=':',label='Current action + 1 stddev')
        axs[axis].set_xlabel('Action value along rotation')
        axs[axis].set_ylabel('Q value')
        axs[axis].set_title(f"Q value vs action along rotation, ad={discrete_action.item()}")
        axs[axis].legend()
        axs[axis].grid()
        if self.bs == 'action_discrete':
            normed_cont_action =  self.policy.bound_action_params(cont_action_loc[idx_batch].unsqueeze(0))
            test_state_action = self.qfunction[0].attach_all_actions(test_state,action=normed_cont_action)
            nadtot = test_state_action['action_discrete'].x.shape[0]
            x = np.arange(nadtot)  # the label locations
            qd = torch.zeros((nadtot,2),device=device,dtype=torch.float32)
            qd[:,0] = self.qfunction[0](test_state_action).flatten()
            qd[:,1] = self.qfunction[1](test_state_action).flatten()
        else:
            x = np.arange(test_state[self.bs].ptr[1].cpu()*np.prod(self.policy.action_metadata['discrete']))  # the label locations
            qd = torch.zeros((test_state[self.bs].ptr[1]*np.prod(self.policy.action_metadata['discrete']),2),device=device,dtype=floatType)
            qd[:,0] = self.qfunction[0](test_state,(None,F.tanh(cont_action_loc)))[0,:test_state[self.bs].ptr[1]*self.policy.action_metadata['discrete'][0],0]
            qd[:,1] = self.qfunction[1](test_state,(None,F.tanh(cont_action_loc)))[0,:test_state[self.bs].ptr[1]*self.policy.action_metadata['discrete'][0],0]
        pold = select_dist.probs[0].detach().cpu()
        
        width = 1/4  # the width of the bars
        
        rects0 = axs[-1].bar(x,qd[:,0].detach().cpu(), width, label='Q1',color=plt.cm.tab10(0))
        rects1 = axs[-1].bar(x+width, qd[:,1].detach().cpu(), width, label='Q2',color=plt.cm.tab10(1))
        if self.qfunction[0].bounded_value:
            rects2 = axs[-1].bar(x+2*width, pold[:qd.shape[0]]*self.qfunction[0].value_bounds[1], width, label='P(ad|s)',color='k')
        else:
            rects2 = axs[-1].bar(x+2*width, pold[:qd.shape[0]]*qd.max().detach().cpu(), width, label='P(ad|s)',color='k')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[-1].set_ylabel('Q(s,E[a|s])')
        axs[-1].set_title('Q values by discrete action')
        axs[-1].set_xticks(x[::5] + width/2, np.arange(0,qd.shape[0],5))
        axs[-1].legend()
        if self.qfunction[0].bounded_value:
            axs[-1].set_ylim(self.qfunction[0].value_bounds[0], self.qfunction[0].value_bounds[1])
        plt.tight_layout()
        return fig