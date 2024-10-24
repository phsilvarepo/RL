#!/usr/bin/env python3

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from yumi_rl.yumi_task import ReachingYumi

# Define the policy architecture as specified in the configuration file
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # Use the architecture specified in the YAML file
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )

        # Adjust the policy layer to have the correct dimensions according to the checkpoint
        self.policy_layer = nn.Linear(64, self.num_actions)  # Expected size: [64, num_actions]

        # Adjust the value layer to match the checkpoint dimensions
        self.value_layer = nn.Linear(64, 1)  # Expected size: [64, 1]

        # Log standard deviation parameter
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.net_container(inputs["states"])
        return x, self.log_std_parameter, {}

# Initialize the environment and wrap it
env = ReachingYumi()
env = wrap_env(env)
device = env.device

# Instantiate the agent's policy with the trained architecture
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)

# Load the trained model
state_dict = torch.load("/home/rics/IsaacLab/logs/skrl/franka_cabinet_direct/yumi_cuboid/checkpoints/best_agent.pt")

# Extract the policy part of the state_dict
policy_dict = state_dict['policy']

# Load the state dictionary into the policy instance
models_ppo["policy"].load_state_dict(policy_dict, strict=False)


# Configure and instantiate the PPO agent with the appropriate configuration
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 0

# Instantiate the PPO agent
agent = PPO(
    models=models_ppo,
    memory=None,
    cfg=cfg_ppo,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device
)

# Load the trained model
state_dict = torch.load("/home/rics/IsaacLab/logs/skrl/franka_cabinet_direct/yumi_cuboid/checkpoints/best_agent.pt")

# Extract the policy part of the state_dict
policy_dict = state_dict['policy']

# Load the state dictionary into the policy instance
models_ppo["policy"].load_state_dict(policy_dict)

# Configure the trainer for evaluation
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Start evaluation
trainer.eval()
