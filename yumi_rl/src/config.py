#!/usr/bin/env python3

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from yumi_rl.yumi_task import ReachingYumi

# Define the policy architecture as specified in the configuration file
class Policy(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-20.0, max_log_std=2.0, role="policy")
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(64, self.num_actions)
        self.value_layer = nn.Linear(64, 1)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        #print("States being passed to the policy:", inputs["states"])

        net = self.net_container(inputs["states"])
        if role == "policy":
            output = self.policy_layer(net)
            return output, self.log_std_parameter, {}
        elif role == "value":
            output = self.value_layer(net)
            return output, {}

# Initialize the environment and wrap it
env = ReachingYumi()
env = wrap_env(env)
device = env.device

# Instantiate the agent's policy using your custom architecture
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)

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
agent.load("/home/rics/IsaacLab/logs/skrl/franka_cabinet_direct/yumi_final/checkpoints/best_agent.pt")

# Configure the trainer for evaluation
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Start evaluation
trainer.eval()
