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
from .agent import PPO
def init_stats(ppo_agent:PPO,env,max_steps = 20):
    # 0. init
    states, info = env.reset()
    rewards = torch.zeros(env.n_batch, dtype=intType, device=device)
    actions = torch.ones(env.n_batch, dtype=intType, device=device) * (-1)
    num_actions = torch.zeros(env.n_batch, dtype=intType, device=device)
    ppo_agent.buffer.clear(env.n_batch)
    finished_once = np.zeros(env.n_batch, dtype=bool)
    i = 0
    n_trajectory = 0
    while not finished_once.all() and i < max_steps:
        #choose action
        i += 1
        _ = ppo_agent.init_stats(states.to(device),reset_weights=True)
        if ppo_agent.bs == 'action_discrete':
            actions =  (torch.cat([torch.randint(torch.sum(states[ppo_agent.bs].batch==n),(1,1)) for n in range(env.n_batch)]).flatten(),
                        torch.cat([F.tanh(torch.normal(torch.zeros(1,env.action_metadata()['continuous']),torch.ones(1,env.action_metadata()['continuous']))) for n in range(env.n_batch)],axis=0)) 
        else:
            actions =  (torch.cat([torch.randint(torch.sum(states[ppo_agent.bs].batch==n)*np.prod(env.action_metadata()['discrete']),(1,1)) for n in range(env.n_batch)]).flatten(),
                        torch.cat([F.tanh(torch.normal(torch.zeros(1,env.action_metadata()['continuous']),torch.ones(1,env.action_metadata()['continuous']))) for n in range(env.n_batch)],axis=0))          
        #env step
        states, rewards, terminated, truncated, info = env.step(actions,render=False)
        finished_once = finished_once | terminated
        if info.get('rotationvec',None) is not None:
            ppo_agent.buffer.add_results(rewards, terminated,info['rotationvec'])
        else:
            ppo_agent.buffer.add_results(rewards, terminated)
        """if np.sum(rewards != 0) == env.n_batch:
            if ui:
                env.render(states)
            break"""
    ppo_agent.buffer.clear()
    ppo_agent.policy.load_state_dict(ppo_agent.policy_old.state_dict())
    wandb.log({"epoch": ppo_agent.episode,
               "Reward": rewards.mean()}, step=ppo_agent.time_step)
    ppo_agent.policy_old.eval()
    return rewards
def rollout_asyn(ppo_agent:PPO, env, render = True,max_imgs=200):
    # 0. init
    states, info = env.reset()
    rewards = torch.zeros(env.n_batch, dtype=intType, device=device)
    actions = torch.ones(env.n_batch, dtype=intType, device=device) * (-1)
    num_actions = torch.zeros(env.n_batch, dtype=intType, device=device)
    ppo_agent.buffer.clear(env.n_batch)
    finished_once = np.zeros(env.n_batch, dtype=bool)
    if render:
        try:
            render_size = env.render_size
        except:
            render_size = (480,480)
        video = np.zeros((max_imgs, *render_size,4),dtype=np.uint8)
    i = 0
    is_finished = False
    n_trajectory = 0
    t0 = perf_counter()
    areas =[]
    while not finished_once.all():
        #choose action
        
        actions = ppo_agent.select_action(states.to(device))
        #env step
        states, rewards, terminated, truncated, info = env.step(actions,render=render and not is_finished)
        finished_once = finished_once | terminated
        areas += info.get('areas',[])
        if info.get('image') is not None and i+info.get('image').shape[0] <= max_imgs and not is_finished:
            video[i:i+info.get('image').shape[0],:,:,:info.get('image').shape[-1]]=info['image']
            i = i + info.get('image').shape[0]
        else:
            is_finished = True
        is_finished = is_finished or terminated[0]
        if info.get('action',None) is not None:
            ppo_agent.buffer.add_results(rewards, terminated,info['rotationvec'])
        else:
            ppo_agent.buffer.add_results(rewards, terminated)
        ppo_agent.time_step = ppo_agent.time_step + env.n_batch
        n_trajectory = n_trajectory + np.sum(terminated)
    ppo_agent.buffer.truncate()
    t1 = perf_counter()
    ppo_agent.sim_time += t1 - t0
    wandb.log({"epoch": ppo_agent.episode,
               "Reward": torch.mean(torch.cat([t['reward'].x for t in ppo_agent.buffer.trajs])).cpu(),
               "Area": np.min(areas),
               "dcr":torch.mean(torch.cat([t['dcr'].x for t in ppo_agent.buffer.trajs])).cpu(),
               "Value_error_rollout":torch.abs(torch.vstack([t['value_est'].x-t['dcr'].x for t in ppo_agent.buffer.trajs])).mean().cpu(),
               "sim time": ppo_agent.sim_time}, step=ppo_agent.time_step)
    
    if render:
        if i >0:
            wandb.log({"video": wandb.Video(np.transpose(video[:i,:,:,:3],(0,3,1,2)), fps=env.fps if hasattr(env,'fps') else 2, format="gif")}, step=ppo_agent.time_step)
    return rewards
def save_policy(ppo_agent,accuracy = 0):
    #delta_accuracy = wandb.config.save_delta_accuracy
    #if accuracy > ppo_agent.saved_accuracy + delta_accuracy:
        #ppo_agent.saved_accuracy = accuracy
    ppo_agent.saved_accuracy =accuracy
    name = f"{wandb.config.model_name}_{int(ppo_agent.saved_accuracy * 100)}"
    folder_path = "./models/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder
    model_path = f"{folder_path}/{name}.pol"
    ppo_agent.save(model_path)
    if wandb.config.online:
        policy_artifact = wandb.Artifact(f"{wandb.config.model_name}_TEST_pol",
                                            type='model')
        policy_artifact.add_file(local_path=model_path, name=name)
        wandb.log_artifact(policy_artifact)
def update_stats(ppo_agent, curriculum_inds, accuracy = None):
    # update number of fails and successes
    rewards = ppo_agent.buffer.rewards
    avg_reward = rewards.mean().item()
    failed_inds = curriculum_inds[(rewards < 0)]
    success_inds = curriculum_inds[(rewards > 0)]

    ppo_agent.num_success[success_inds] += 1
    ppo_agent.num_failed[success_inds] = 0
    ppo_agent.num_failed[failed_inds] = ppo_agent.num_failed[failed_inds] + 1
    ppo_agent.buffer.modify_entropy(failed_inds,success_inds)
    if accuracy is None:
        accuracy = (avg_reward+1)/2

    # print status
    ppo_agent.print_iter = ppo_agent.print_iter + 1
    print_epoch = wandb.config.print_epochs
    if ppo_agent.print_iter % print_epoch == print_epoch - 1:
        print()
        print("Episode : {} \t\t Accuracy : {} \t\t Reward : {}".format( ppo_agent.episode,
                                                                          round(accuracy, 2),
                                                                          round(avg_reward, 2)))
        print("Failed Curriculum: {}".format(curriculum_inds[ppo_agent.buffer.rewards < 0]))

def policy_update(ppo_agent:PPO):
    loss, entropyc,entropyd, num_training_data = ppo_agent.update_policy(
        batch_size=wandb.config.policy_update_batch_size,
        update_iter=wandb.config.n_updates_per_step)

    ppo_agent.uniform_training_data = num_training_data
    ppo_agent.uniform_entropy = entropyd
    wandb.log({"train time": ppo_agent.training_time,
               "datasetbuild time":ppo_agent.build_ds_time,
                "loss": loss,
                "entropy": entropyc,
                "entropy_d": entropyd,
                "train data": num_training_data,
                "value loss":ppo_agent.val_loss,
                "surrogate loss":ppo_agent.sur_loss,}, step=ppo_agent.time_step)
def train(ppo_agent, env, render_freq = 100):    
    print(f"Start training")
    # update variables
    ppo_agent.print_iter = 0
    for i in range(2):
        init_stats(ppo_agent,env,max_steps=20)
        break
    # training
    while ppo_agent.episode <= wandb.config.n_episodes:
        print(f"Episode {ppo_agent.episode} / {wandb.config.n_episodes}")
        print(f"Rollout")
        rollout_asyn(ppo_agent, env, render = ppo_agent.episode%render_freq == 0)
        
        # compute accuracy

        # update number of fails

        # save model
        if ppo_agent.episode % wandb.config.save_model_freq == 0:
            accuracy = 1#(ppo_agent.buffer.rewards.mean() + 1) / 2
            print(f"Saving model")
            save_policy(ppo_agent, accuracy)

        
        # policy update
        print(f"Policy update")
        policy_update(ppo_agent)
        ppo_agent.episode = ppo_agent.episode + 1