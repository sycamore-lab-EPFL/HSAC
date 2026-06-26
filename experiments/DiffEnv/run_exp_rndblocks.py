import json
import torch
import numpy as np
import trimesh
import wandb
import time
import sac
import warp as wp
config_file = "scripts/experiments/config.json"

with open(config_file) as json_file:
    print(f"load parameter from {config_file}")
    config = json.load(json_file)
#for the blocks, ensure that
# supface = 0
# bface = 9
rng = np.random.default_rng(0)

block_type = []
for b in range(10):
    if b == 0:
        #keep the trap. ground
        d = np.tan(np.radians(5))*1.2
        corners = np.array([[0, 0, 0.8],[d,1.2,0.8],[2.4,0,0.8],[2.4-d,1.2,0.8],
                            [2.4,0,0], [2.4-d,1.2,0], [0,0,0],[d,1.2,0]])
    else:
        corners_flat = np.array([[0, 0, 0.8],[0,1.2,0.8],[2.4,0,0.8],[2.4,1.2,0.8],
                            [2.4,0,0], [2.4,1.2,0], [0,0,0],[0,1.2,0]])
        deltac = rng.random(corners_flat.shape)*0.6-0.2
        corners = corners_flat+deltac
    trimesh_obj = trimesh.Trimesh(vertices=corners,
                                faces=np.array([[2,8,1],[8,7,1],[7,5,1],[5,3,1],
                                                [6,5,7],[8,6,7],[4,6,8],[2,4,8],
                                                [4,3,5],[6,4,5],[2,1,3], [4,2,3]])-1)

    centroid = trimesh_obj.centroid
    trimesh_obj.apply_translation(-centroid)
    face_centers = trimesh_obj.triangles.mean(axis=1)
    outward = face_centers
    dot = np.sum(trimesh_obj.face_normals * outward, axis=1)
    assert np.all(dot > 0), "Some normals point inward!"
    assert trimesh_obj.is_watertight
    assert trimesh_obj.is_volume
    assert trimesh_obj.volume>1e-4
    trimesh_obj.center_mass
    block_type.append(trimesh_obj)
config['max_actiond']= len(block_type)
config['model_name']='rnd_blocks'
config['online']=True
wandb.init(mode='online' if config['online'] else 'offline',project="RoboticConstruction", config=config)
wp.config.mode = 'release'
wp.set_device('cuda')  # Set to 'cpu' if you want to run on CPU
import masongraph_envs

env = masongraph_envs.ColumnMasonGraph(n_batch=config['n_batch'],
                                        render_mode='rgb_array',
                                        scales=None,
                                        block_list = block_type,
                                        max_dist=config['max_dist'],
                                        min_dist=config['min_dist'],
                                        max_blocks=config['max_blocks'],
                                        friction_coef=config['friction_coef'],
                                        glue_strength=config['glue_strength'],
                                        ground_gen=config['ground_gen'],
                                        tol=config['tol'],
                                        rep=config['rep'],
                                        reset_on_fall=config['reset_on_fall'],
                                        additional_contact_reward=config['additional_contact_reward'],
                                        touch_target=config['touch_target'],
                                        physics_on=config['physics_on'],
                                        
                                        autoreset=config['autoreset'],
                                        terminal_on_invalid=config['terminal_on_invalid'],
                                        continuous_reward=config['continuous_reward'],
                                        her=config['her'])

print(f"Warp mode: {wp.get_device()}, {wp.config.mode}")
agent = sac.SAC(env.state_metadata(),env.action_metadata(),wandb.config,attach='action_discrete',bs_mode='last')
print("START TIMER")
t0 = time.perf_counter()
sac.train(agent,env,render_freq=100)
t1 = time.perf_counter()
print(f"Training took {t1-t0} seconds")