import math
import os
import pickle
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
import polyscope as ps
from .helper import intType,floatType, device
from copy import deepcopy
from .agent import SAC
def init_stats(sac_agent:SAC,env,max_steps = 20):
    # 0. init
    states, info = env.reset()
    rewards = torch.zeros(env.n_batch, dtype=intType, device=device)
    actions = torch.ones(env.n_batch, dtype=intType, device=device) * (-1)
    num_actions = torch.zeros(env.n_batch, dtype=intType, device=device)
    sac_agent.buffer.clear(env.n_batch)
    finished_once = np.zeros(env.n_batch, dtype=bool)
    i = 0
    n_trajectory = 0
    sac_agent.policy.eval()
    al = []
    while not finished_once.all() and i < max_steps:
        #choose action
        i += 1
        _ = sac_agent.init_stats(states)
        if sac_agent.bs == 'action_discrete':
            actioncu = torch.cat([F.tanh(torch.normal(torch.zeros(1,env.action_metadata()['continuous']),torch.ones(1,env.action_metadata()['continuous']))) for n in range(env.n_batch)],axis=0)
            actionc = torch.zeros_like(actioncu)
            actionc[:,:2] = F.tanh(actioncu[:,:2])
            actionc[:,2:] = actioncu[:,2:]/actioncu[:,2:].norm(dim=1,keepdim=True)
            actions =  (torch.cat([torch.randint(torch.sum(states[sac_agent.bs].batch==n),(1,1)) for n in range(env.n_batch)]).flatten(),
                        actionc,actioncu
                        ) 
        else:
            actions =  (torch.cat([torch.randint(torch.sum(states[sac_agent.bs].batch==n)*np.prod(env.action_metadata()['discrete']),(1,1)) for n in range(env.n_batch)]).flatten(),
                        torch.cat([F.tanh(torch.normal(torch.zeros(1,env.action_metadata()['continuous']),torch.ones(1,env.action_metadata()['continuous']))) for n in range(env.n_batch)],axis=0))          
        #env step
        sac_agent.buffer.add_inputs(states,(actions[0].to(sac_agent.buffer.storing_device),actions[1].to(sac_agent.buffer.storing_device),actions[2].to(sac_agent.buffer.storing_device)) )
        states, rewards, terminated, truncated, info = env.step(actions,render=False)
        finished_once = finished_once | terminated | truncated
        sac_agent.buffer.add_results(rewards, terminated,truncated)
        """if np.sum(rewards != 0) == env.n_batch:
            if ui:
                env.render(states)
            break"""
    sac_agent.policy.train()
    #sac_agent.vfunction.load_state_dict(sac_agent.target_vfunction.state_dict())
    #sac_agent.buffer.clear()
    wandb.log({"epoch": sac_agent.episode,
               "Reward": rewards.mean()}, step=sac_agent.time_step)
    return rewards
def rollout_asyn(sac_agent:SAC, env, render = True,max_imgs=200):
    # 0. init
    states, info = env.reset()
    env.render()
    rewards = torch.zeros(env.n_batch, dtype=intType, device=device)
    actions = torch.ones(env.n_batch, dtype=intType, device=device) * (-1)
    num_actions = torch.zeros(env.n_batch, dtype=intType, device=device)
    #sac_agent.buffer.clear(env.n_batch)
    finished_once = np.zeros(env.n_batch, dtype=bool)
    if render:
        try:
            render_size = env.render_size
        except:
            render_size = (480,480)
        video = np.ones((max_imgs, *render_size,4),dtype=np.uint8)
    i = 0
    is_finished = False
    n_trajectory = 0
    t0 = perf_counter()
    areas =[]
    reward_list = []
    n_blocks = []
    while not finished_once.all():
        #choose action
        
        actions = sac_agent.select_action(states.to(device))
        #env step
        states, rewards, terminated, truncated, info = env.step(actions,render=render and not is_finished)

        reward_list.append(rewards)
        n_blocks.append(states[sac_agent.bs].x.shape[0])
        finished_once = finished_once | (terminated & env.autoreset) | truncated
        areas += info.get('areas',[0])
        if info.get('image') is not None and i+info.get('image').shape[0] <= max_imgs and not is_finished:
            video[i:i+info.get('image').shape[0],:,:,:info.get('image').shape[-1]]=info['image']
            i = i + info.get('image').shape[0]
        else:
            is_finished = True
        is_finished = is_finished or (terminated[0] & env.autoreset)
        if info.get('rotationvec',None) is not None:
            sac_agent.buffer.add_results(rewards, terminated,truncated,info['rotationvec'])
        else:
            sac_agent.buffer.add_results(rewards, terminated,truncated)
        sac_agent.time_step = sac_agent.time_step + env.n_batch
        n_trajectory = n_trajectory + np.sum(terminated)
    sac_agent.buffer.add_traj()
    
    t1 = perf_counter()
    sac_agent.sim_time += t1 - t0
    wandb.log({"epoch": sac_agent.episode,
               "Reward": np.mean(np.concatenate(reward_list)),
               "Area": np.min(areas),
               "number of blocks": np.mean(n_blocks),
               "sim time": sac_agent.sim_time}, step=sac_agent.time_step)
    if render:
        if i >0:
            wandb.log({"video": wandb.Video(np.transpose(video[:i,:,:,:3],(0,3,1,2)), fps=env.fps if hasattr(env,'fps') else 2, format="gif")}, step=sac_agent.time_step)
    return rewards
def save_policy(sac_agent,accuracy = 0):
    #delta_accuracy = wandb.config.save_delta_accuracy
    #if accuracy > sac_agent.saved_accuracy + delta_accuracy:
        #sac_agent.saved_accuracy = accuracy
    sac_agent.saved_accuracy =accuracy
    name = f"{wandb.config.model_name}_{int(sac_agent.saved_accuracy * 100)}"
    folder_path = "./models/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder
    model_path = f"{folder_path}/{name}.pol"
    sac_agent.save(model_path)
    if wandb.config.online:
        policy_artifact = wandb.Artifact(f"{wandb.config.model_name}_TEST_pol",
                                            type='model')
        policy_artifact.add_file(local_path=model_path, name=name)
        wandb.log_artifact(policy_artifact)
def update_stats(sac_agent, curriculum_inds, accuracy = None):
    # update number of fails and successes
    rewards = sac_agent.buffer.rewards
    avg_reward = rewards.mean().item()
    failed_inds = curriculum_inds[(rewards < 0)]
    success_inds = curriculum_inds[(rewards > 0)]

    sac_agent.num_success[success_inds] += 1
    sac_agent.num_failed[success_inds] = 0
    sac_agent.num_failed[failed_inds] = sac_agent.num_failed[failed_inds] + 1
    sac_agent.buffer.modify_entropy(failed_inds,success_inds)
    if accuracy is None:
        accuracy = (avg_reward+1)/2

    # print status
    sac_agent.print_iter = sac_agent.print_iter + 1
    print_epoch = wandb.config.print_epochs
    if sac_agent.print_iter % print_epoch == print_epoch - 1:
        print("Episode : {} \t\t Accuracy : {} \t\t Reward : {}".format( sac_agent.episode,
                                                                          round(accuracy, 2),
                                                                          round(avg_reward, 2)))
        print("Failed Curriculum: {}".format(curriculum_inds[sac_agent.buffer.rewards < 0]))

def policy_update(sac_agent:SAC):
    loss, entropy, num_training_data,var,mean_val,entropyd = sac_agent.update_policy(
        batch_size=wandb.config.policy_update_batch_size,
        update_iter=wandb.config.n_updates_per_step)

    sac_agent.uniform_training_data = num_training_data
    sac_agent.uniform_entropy = entropy
    wandb.log({"train time": sac_agent.training_time,
               "datasetbuild time":sac_agent.build_ds_time,
                "loss": loss,
                "entropy": entropy,
                "entropy_d": entropyd,
                "train data": num_training_data,
                "value loss":sac_agent.val_loss,
                "surrogate loss":sac_agent.pol_loss,
                "Q0 loss": sac_agent.q0loss,
                "Q1 loss": sac_agent.q1loss,
                "Variance":var,
                "Mean target V":mean_val,
                "entropy weightc": sac_agent.entropy_weightc.item(),
                "entropy weightd": sac_agent.entropy_weightd.item(),
                }, step=sac_agent.time_step)
def train(sac_agent, env, render_freq = 100, init_stats_fist = True):
    print(f"Start training")
    # update variables
    sac_agent.print_iter = 0
    if init_stats_fist:
        init_stats(sac_agent,env)
    # training
    #sac_agent.buffer.clear()
    while sac_agent.episode <= wandb.config.n_episodes:
        print(f"Episode {sac_agent.episode} / {wandb.config.n_episodes}")
        print(f"Rollout")
        rollout_asyn(sac_agent, env, render = sac_agent.episode%render_freq == 0)
        
        # compute accuracy

        # update number of fails

        # save model
        if sac_agent.episode % wandb.config.save_model_freq == 0:
            accuracy = 1#(sac_agent.buffer.rewards.mean() + 1) / 2
            print(f"Saving model")
            save_policy(sac_agent, accuracy)

        
        # policy update
        print(f"Policy update")
        policy_update(sac_agent)
        sac_agent.episode = sac_agent.episode + 1