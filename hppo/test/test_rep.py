import hppo
import masongraph_envs
import wandb
import torch
config = {'model_name':'debug',
          'online':True,
          "n_batch":16,
          "gamma":0.9,
          "n_robots":1,
          "max_blocks":50,
          "min_area":[10,10],
          "max_area":[1000,1000],
          "block_list":"brick",
          "scales":[1],
          "friction_coef":0.5,
          "ground_gen":"around",
          "ground_width":300,
          "hidden_channels":64,

          "n_layers":7,
          "learning_rate":0.0001,
          "betas":(0.9, 0.999),
          "n_episodes":10000,
          "save_model_freq":100,
          "n_updates_per_step":3,
          "policy_update_batch_size":128,
          "eps_clip":0.2,
          "ent_coef":0.001,
          "vf_coef":0.5,
          "max_grad_norm":0.5,
          "gae_lambda":0.95,
          "her":False}
wandb.init(mode='online' if config['online'] else 'offline',project="RoboticConstructionTestRep", config=config)
env = masongraph_envs.MasonGraphCover(n_batch=config['n_batch'],
                                      render_mode='rgb_array',
                                      n_robots=config['n_robots'],
                                      scales=config['scales'],
                                      max_area=config['max_area'],
                                      min_area=config['min_area'],
                                      ground_width=config['ground_width'],autoreset=False)
agent = hppo.PPO(env.state_metadata(),env.action_metadata(),wandb.config)
#hppo.rollout_asyn(agent, env, render = False)
states, info = env.reset()
agent.buffer.clear(env.n_batch)

actions = agent.select_action(states)
#env step
states, rewards, terminated, truncated, info = env.step(actions,render=False)
agent.buffer.add_results(rewards, terminated)

actions = agent.select_action(states)
#env step
states, rewards, terminated, truncated, info = env.step(actions,render=False)
agent.buffer.add_results(rewards, terminated)

agent.buffer.truncate()


for b in range(config['n_batch']):
    for i in range(len(agent.buffer.trajs)):
        agent.buffer.trajs[i]['dcr'].x[b] = torch.rand(1)
#agent.buffer.trajs[0]['dcr'].x = torch.tensor([i.area for i in env.sim.to_cover],device='cuda')
agent.buffer.adv_list =[ torch.tensor([0.5,0.6,0.4,0.5],device='cuda')]
#agent.buffer.adv_list = [agent.buffer.trajs[0]['dcr'].x]
print("Start optim")
for i in range(config['n_episodes']):
    loss, entropy, num_training_data = agent.update_policy(
        batch_size=wandb.config.policy_update_batch_size,
        update_iter=1,clear_buffer=False)
    wandb.log({"train time": agent.training_time,
               "datasetbuild time":agent.build_ds_time,
                "loss": loss,
                "entropy": entropy,
                "train data": num_training_data,
                "value loss":agent.val_loss,
                "surrogate loss":agent.sur_loss,}, step=agent.time_step)
    agent.time_step+=1
