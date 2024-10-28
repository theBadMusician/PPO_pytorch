import torch
from PPOAgent import PPOAgent
from SimpleGoalEnv import SimpleGoalEnv

# Define environment and configuration for PPO training
env = SimpleGoalEnv(device="cuda")
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_cells": 256,
    "lr": 3e-4,
    "max_grad_norm": 1.0,
    "frames_per_batch": 1000,
    "total_frames": 50000,
    "sub_batch_size": 64,
    "num_epochs": 10,
    "clip_epsilon": 0.2,
    "gamma": 0.99,
    "lmbda": 0.95,
    "entropy_eps": 1e-4,
}

# Create PPO agent and train
agent = PPOAgent(env, config)
agent.train()