import masongraph_envs
import ppo
import wandb
import os
import pickle

def load_artifact(artifact_name):
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
    env = masongraph_envs.MasonGraphCover(n_batch=config['n_batch'],
                                      render_mode='rgb_array',
                                      n_robots=config['n_robots'],
                                      scales=config['scales'],
                                      max_area=config['max_area'],
                                      ground_width=config['ground_width'],
                                      min_area=config['min_area'],
                                      max_blocks=config['max_blocks'],
                                      vertical=True,
                                      autoreset=False)
    agent = ppo.PPO(env.state_metadata(),env.action_metadata(),wandb.config)
    agent.policy_old.load_state_dict(artifact['state_dict'])
    return agent,env