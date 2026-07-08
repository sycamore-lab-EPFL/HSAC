
import sac
import wandb
import os
import pickle
import numpy as np
def load_artifact(artifact_name,batchsize=None):
    """
    Load a pretrained model artifact from wandb.
    
    Args:
        artifact_name (str): The name of the artifact to load.
        
    Returns:
        str: The path to the downloaded model file.
    """
    artifact = wandb.use_artifact(artifact_name, type='model')
    artifact_dir = artifact.download()
    pretrained_file = artifact_dir+"/"+os.listdir(artifact_dir)[0]
    with open(pretrained_file, 'rb') as handle:
        artifact = pickle.load(handle)
        config = artifact['config']
        if batchsize is not None:
            config['n_batch'] = batchsize
    wandb.config.update(config,allow_val_change=True)
    import masongraph_envs
    if config['ground_gen'] == 'aligned':
        env = masongraph_envs.ColumnMasonGraph(n_batch=config['n_batch'],
                                              render_mode='rgb_array',
                                              scales=config['scales'],
                                              max_dist=config['max_dist'],
                                              min_dist=config['min_dist'],
                                              max_blocks=20,#config['max_blocks'],
                                              friction_coef=config['friction_coef'],
                                              glue_strength=config['glue_strength'],
                                              ground_gen=config['ground_gen'],
                                              ground_angle=config.get('ground_angle',None),
                                              tol=1e-2,
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
    else:
        env = masongraph_envs.StraightMasonGraphCover(n_batch=config['n_batch'],
                                                    render_mode='rgb_array',
                                                    scales=config['scales'],
                                                    max_area=config['max_area'],
                                                    ground_width=config['ground_width'],
                                                    min_area=config['min_area'],
                                                    max_blocks=config['max_blocks'],
                                                    autoreset=config['autoreset'],
                                                    friction_coef=config['friction_coef'],
                                                    glue_strength=config['glue_strength'],
                                                    tol=1e-2,
                                                    rep=config['rep'],
                                                    reset_on_fall=config['reset_on_fall'],
                                                    additional_contact_reward=config['additional_contact_reward'],
                                                    touch_target=config['touch_target'],
                                                    block_list=config['block_list'],
                                                    physics_on=config['physics_on'],
                                                    terminal_on_invalid=config['terminal_on_invalid'])
    wandb.config.update({"prefilled_buffer":False},allow_val_change=True)
    if not hasattr(wandb.config,'heads'):
        wandb.config.update({"heads":1},allow_val_change=True)
    if config['ground_gen'] == 'aligned':
        agent = sac.SAC(env.state_metadata(),env.action_metadata(),wandb.config,attach='action_discrete')
    else:
        agent = sac.SAC(env.state_metadata(),env.action_metadata(),wandb.config)
    agent.policy.load_state_dict(artifact['pol_state_dict'])
    agent.qfunction[0].load_state_dict(artifact['qf0_state_dict'])
    agent.qfunction[1].load_state_dict(artifact['qf1_state_dict'])
    agent.vfunction.load_state_dict(artifact['vf_state_dict'])
    agent.target_vfunction.load_state_dict(artifact['vf_state_dict'])
    print(f"Loaded pretrained model, info: {artifact['print_info']}")

    return agent,env
def load_artifact_mujoco(artifact_name):
    """
    Load a pretrained model artifact from wandb.
    
    Args:
        artifact_name (str): The name of the artifact to load.
        
    Returns:
        str: The path to the downloaded model file.
    """
    artifact = wandb.use_artifact(artifact_name, type='model')
    artifact_dir = artifact.download()
    pretrained_file = artifact_dir+"/"+os.listdir(artifact_dir)[0]
    with open(pretrained_file, 'rb') as handle:
        artifact = pickle.load(handle)
        config = artifact['config']
    wandb.config.update(config,allow_val_change=True)
    import mujocograph
    env = mujocograph.MujocoWrapper(n_batch=config['n_batch'],
                                env_name=config['env_name'],
                                n_modes=config['n_modes'],
                                render_mode='rgb_array',)                                              
    agent = sac.SACvec(env.state_metadata(),env.action_metadata(),wandb.config,attach = 'central')
    agent.policy.load_state_dict(artifact['pol_state_dict'])
    agent.qfunction[0].load_state_dict(artifact['qf0_state_dict'])
    agent.qfunction[1].load_state_dict(artifact['qf1_state_dict'])
    agent.vfunction.load_state_dict(artifact['vf_state_dict'])
    agent.target_vfunction.load_state_dict(artifact['vf_state_dict'])
    print(f"Loaded pretrained model, info: {artifact['print_info']}")
    return agent,env