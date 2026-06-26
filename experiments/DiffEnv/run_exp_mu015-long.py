import json
import torch
import numpy as np
import wandb
import time
import sac
import warp as wp
config_file = "scripts/experiments/config.json"

with open(config_file) as json_file:
    print(f"load parameter from {config_file}")
    config = json.load(json_file)
config['state_rep'] = 'bframegraph_noobj'
config['max_actiond']= len(config['scales'])
config['friction_coef']=0.15
config['scales']=[20,10, 5]
config['max_actiond']= len(config['scales'])
config['max_dist']=12.01
config['min_dist']=12
config['target_entropyd']=0.5
config['ground_angle']=30
config['model_name']='HSAC_mu015'

wandb.init(mode='online' if config['online'] else 'offline',project="RoboticConstruction", config=config)
wp.config.mode = 'release'
wp.set_device('cuda')  # Set to 'cpu' if you want to run on CPU
import masongraph_envs

env = masongraph_envs.ColumnMasonGraph(n_batch=config['n_batch'],
                                        render_mode='rgb_array',
                                        scales=config['scales'],
                                        max_dist=config['max_dist'],
                                        min_dist=config['min_dist'],
                                        max_blocks=config['max_blocks'],
                                        friction_coef=config['friction_coef'],
                                        glue_strength=config['glue_strength'],
                                        ground_gen=config['ground_gen'],
                                        ground_angle=config['ground_angle'],
                                        tol=config['tol'],
                                        rep=config['rep'],
                                        reset_on_fall=config['reset_on_fall'],
                                        additional_contact_reward=config['additional_contact_reward'],
                                        touch_target=config['touch_target'],
                                        block_list=config['block_list'],
                                        physics_on=config['physics_on'],
                                        autoreset=config['autoreset'],
                                        terminal_on_invalid=config['terminal_on_invalid'],
                                        continuous_reward=config['continuous_reward'],
                                        her=config['her'])

print(f"Warp mode: {wp.get_device()}, {wp.config.mode}")
agent = sac.SAC(env.state_metadata(),env.action_metadata(),wandb.config,attach='action_discrete')
print("START TIMER")
t0 = time.perf_counter()
sac.train(agent,env,render_freq=100)
t1 = time.perf_counter()
print(f"Training took {t1-t0} seconds")